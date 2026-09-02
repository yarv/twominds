"""Generation phase: ask every model the roster N times via Inspect.

The whole sweep is a **single** ``inspect_ai.eval`` call over one task per
model (each task pinned to its model and named ``twominds:<model>``) — Inspect
runs them concurrently in one process with its own per-provider connection pool
and one shared progress display (``model_concurrency`` just caps how many run
at once via ``max_tasks``). The returned logs come back in task order, so each
is written to its own ``<run_dir>/logs/<model>/`` directory (disambiguating
two rungs that share one underlying model id) in **both** ``.eval``
(canonical) and ``.json`` (human-readable) form. ``analyze.load_responses``
reads the ``.eval`` back.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .models import DEFAULT_JUDGE_CONCURRENCY, ModelSpec
from .questions import Question

load_dotenv()


# Store key for the fused-sample response list (one sample = one question,
# its N generations fanned out inside the solver).
GEN_RESPONSES_KEY = "twominds:responses"


def _fanout_solver(n: int):
    """Generate N independent samples of the question inside ONE sample, so a
    per-sample scorer (the cross-sample judge) sees all N answers together."""
    from inspect_ai.solver import solver

    @solver
    def _solve():
        async def solve(state, generate):
            import anyio

            from inspect_ai.model import get_model

            model = get_model()  # the task's pinned generation model
            results: list = [None] * n

            async def one(i: int):
                out = await model.generate(state.input)
                results[i] = out.completion or ""

            async with anyio.create_task_group() as tg:
                for i in range(n):
                    tg.start_soon(one, i)
            state.store.set(GEN_RESPONSES_KEY, results)
            state.output.completion = results[0] or ""
            return state

        return solve

    return _solve()


def inline_judge_scorer(
    judge_name: str,
    judge_reasoning,
    *,
    max_connections: int = DEFAULT_JUDGE_CONCURRENCY,
    max_response_chars: int = 8000,
):
    """The cross-sample judge as an Inspect scorer on fused samples.

    Runs the moment a question's N answers are in — inside the generation
    eval, so judge progress, retries, and token usage all live in the one
    Inspect display/log. Score metadata carries the verdict + the judge
    identity so ``analyze`` can harvest matching verdicts instead of re-judging.
    """
    from inspect_ai.scorer import Score, mean, scorer

    from .judge import get_judge_model, judge_bundle, judge_identity

    identity = judge_identity(judge_name, judge_reasoning)

    @scorer(metrics=[mean()])
    def twominds_judge():
        model = get_judge_model(
            judge_name, judge_reasoning, max_connections=max_connections
        )

        async def score(state, target):
            meta = state.metadata or {}
            responses = state.store.get(GEN_RESPONSES_KEY) or []
            try:
                jr = await judge_bundle(
                    model,
                    meta.get("prompt") or state.input_text,
                    responses,
                    max_response_chars=max_response_chars,
                )
            except Exception as e:
                # The generations are already in — a judge timeout/provider
                # error must not fail (or hang) the sample. Returning a score
                # WITHOUT judge_result just defers this bundle to the
                # analyze-phase judge (harvesting skips it).
                return Score(
                    value=0.0,
                    answer=f"(judge error: {type(e).__name__}; re-judged in analyze)",
                    metadata={"judge_error": str(e)[:500]},
                )
            return Score(
                value=1.0 if jr.contradiction else 0.0,
                answer=jr.rationale[:200],
                metadata={"judge_result": jr.to_dict(), "judge_identity": identity},
            )

        return score

    return twominds_judge()


def build_task(
    questions: list[Question],
    name: str = "twominds",
    model=None,
    n_per_sample: Optional[int] = None,
    judge_scorer=None,
):
    """Build an Inspect Task: the questions as samples + a bare generate() solver.

    ``model`` pins the task to one configured model, so a multi-model sweep can
    name each task after its model.

    ``n_per_sample`` switches to the fused shape: one sample per question with
    the N generations fanned out inside the solver (instead of Inspect epochs),
    which is what lets ``judge_scorer`` — the cross-sample judge — run as a
    normal per-sample scorer, overlapped with the rest of the generation."""
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser
    from inspect_ai.solver import generate

    samples = []
    for q in questions:
        if q.system:
            inp = [
                ChatMessageSystem(content=q.system),
                ChatMessageUser(content=q.prompt),
            ]
        else:
            inp = q.prompt
        samples.append(
            Sample(
                input=inp,
                id=q.id,
                metadata={"group": q.group, "prompt": q.prompt},
            )
        )

    return Task(
        dataset=MemoryDataset(samples),
        solver=_fanout_solver(n_per_sample) if n_per_sample else generate(),
        scorer=judge_scorer,
        name=name,
        model=model,
    )


def write_manifest(
    run_dir: Path,
    model_specs: list[ModelSpec],
    questions: list[Question],
    *,
    n: int,
    temperature: float,
    max_tokens: int,
    judge: str,
) -> None:
    """Persist what was run so the analysis phase is fully decoupled from Inspect."""
    run_dir.mkdir(parents=True, exist_ok=True)
    questions_meta = {
        q.id: {"prompt": q.prompt, "group": q.group, "system": q.system}
        for q in questions
    }
    (run_dir / "questions.json").write_text(json.dumps(questions_meta, indent=2))
    manifest = {
        "models": {
            m.name: {
                "inspect_model": m.inspect_model,
                "reasoning_effort": m.reasoning_effort,
                "display": m.display or m.name,
            }
            for m in model_specs
        },
        "question_ids": [q.id for q in questions],
        "n": n,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "judge": judge,
    }
    (run_dir / "run_config.json").write_text(json.dumps(manifest, indent=2))


def _build_models(
    model_specs: list[ModelSpec],
    *,
    temperature: float,
    max_tokens: int,
    timeout: int,
    attempt_timeout: int,
    max_connections: Optional[int],
):
    """One Inspect ``Model`` per spec, carrying its temperature / reasoning effort."""
    from inspect_ai.model import GenerateConfig, get_model

    models = []
    for spec in model_specs:
        cfg = GenerateConfig(
            max_tokens=max_tokens, timeout=timeout, attempt_timeout=attempt_timeout
        )
        if spec.reasoning_effort in (None, "none", "minimal"):
            cfg.temperature = temperature
        # else: reasoning models pin temperature to 1 internally — which is the
        # method's setting anyway; passing the param only triggers a provider
        # warning mid-run.
        if spec.reasoning_effort is not None:
            cfg.reasoning_effort = spec.reasoning_effort
        if max_connections is not None:
            cfg.max_connections = max_connections
        models.append(get_model(spec.inspect_model, config=cfg))
    return models


def run_generation(
    model_specs: list[ModelSpec],
    questions: list[Question],
    *,
    n: int,
    run_dir: Path,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    display: str = "rich",
    retry_on_error: int = 2,
    max_connections: Optional[int] = None,
    timeout: int = 300,
    attempt_timeout: int = 120,
    model_concurrency: int = 3,
    judge_inline: Optional[dict] = None,
) -> dict[str, str]:
    """Run the whole generation sweep in one Inspect call. Returns {model: log_dir}.

    A single ``inspect_ai.eval`` call over one task per model (each pinned to
    its model, named ``twominds:<model>``) — Inspect schedules them
    concurrently in one process with its own connection pool and one shared
    progress display. ``model_concurrency`` maps straight to Inspect's
    ``max_tasks`` (how many models run at once; each model is also internally
    concurrent across its N×Q samples). ``max_connections`` caps each model's
    in-flight requests (None = the provider default, ~10 for OpenAI);
    effective API concurrency is ~``model_concurrency × max_connections`` —
    mind provider rate limits (Inspect's adaptive concurrency backs off on
    429s).

    ``attempt_timeout`` caps each request *attempt* (seconds): a hung HTTP call
    is abandoned at 120s and retried immediately inside the same request,
    instead of burning the whole request budget. ``timeout`` caps the entire
    request including those retries.

    ``eval`` returns one ``EvalLog`` per model in model order; each is written to
    ``logs/<spec.name>/<spec.name>.{eval,json}`` (``.eval`` canonical for
    ``analyze``; ``.json`` for human reading).

    A model whose eval did NOT succeed (bad id, auth failure, provider 4xx) still
    gets its log written for debugging, but a ``RuntimeError`` naming every
    failed model is raised at the end — an errored log holds cancelled samples
    with empty completions, which would otherwise flow silently into the judge
    as "responses".

    ``judge_inline`` (kwargs of :func:`inline_judge_scorer`) fuses the judge
    into the sweep: samples switch to one-per-question with the N generations
    fanned out in-solver, and the cross-sample judge runs as that sample's
    scorer the moment its answers are in — judge progress and usage share the
    sweep's Inspect display, and ``analyze`` later harvests the verdicts from
    the logs instead of re-judging.
    """
    from inspect_ai import eval as inspect_eval
    from inspect_ai.log import write_eval_log

    logs_root = Path(run_dir) / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    raw_dir = logs_root / ".raw"  # Inspect's incremental writes; re-placed below

    models = _build_models(
        model_specs,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        attempt_timeout=attempt_timeout,
        max_connections=max_connections,
    )
    # One task per model, each pinned to its configured model and named after
    # the spec. Still ONE eval call. One shared scorer instance keeps the judge
    # on one connection pool.
    scorer = inline_judge_scorer(**judge_inline) if judge_inline else None
    tasks = [
        build_task(
            questions,
            name=f"twominds:{spec.name}",
            model=model,
            n_per_sample=n if judge_inline else None,
            judge_scorer=scorer,
        )
        for spec, model in zip(model_specs, models)
    ]
    logs = inspect_eval(
        tasks,
        epochs=1 if judge_inline else n,
        log_dir=str(raw_dir),
        log_format="eval",
        display=display,
        retry_on_error=retry_on_error,
        score=judge_inline is not None,
        max_tasks=max(1, model_concurrency),
    )
    if len(logs) != len(model_specs):  # eval returns one log per task, in order
        raise RuntimeError(
            f"expected {len(model_specs)} eval logs, got {len(logs)}; "
            "cannot map logs back to model specs"
        )

    out: dict[str, str] = {}
    failures: list[str] = []
    for spec, log in zip(model_specs, logs):
        safe = spec.name.replace("/", "_")  # guard: a slash would nest the dir
        model_log_dir = logs_root / safe
        model_log_dir.mkdir(parents=True, exist_ok=True)
        write_eval_log(log, str(model_log_dir / f"{safe}.eval"), format="eval")
        write_eval_log(log, str(model_log_dir / f"{safe}.json"), format="json")
        out[spec.name] = str(model_log_dir)
        if log.status != "success":
            # the error's .message is the upstream API error; str(log.error)
            # would drag a full embedded traceback into the CLI message
            detail = (
                getattr(log.error, "message", None) or str(log.error or "")
            ) or "no error detail"
            failures.append(
                f"{spec.name} ({spec.inspect_model}), status={log.status}: "
                f"{detail[:600]}"
            )
    shutil.rmtree(raw_dir, ignore_errors=True)
    if failures:
        raise RuntimeError("generation failed for:\n  " + "\n  ".join(failures))
    return out
