"""Generation phase: ask every model the roster N times via Inspect.

Inspect does *generation only* here (no scorer). The whole sweep is a **single**
``inspect_ai.eval`` call over one task per rung (each task pinned to its model and
named ``twominds:<rung>``) — Inspect runs them concurrently in one
process with its own per-provider connection pool and one shared progress display
(``model_concurrency`` just caps how many run at once via ``max_tasks``). The
returned logs come back in task order, so each is written to its own
``<run_dir>/logs/<model>/`` directory (disambiguating the two gpt-5.2 rungs, which
share one underlying model id) in **both** ``.eval`` (canonical) and ``.json``
(human-readable) form. ``analyze.load_responses`` reads the ``.eval`` back.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .models import ModelSpec
from .questions import Question

load_dotenv()


def build_task(questions: list[Question], name: str = "twominds", model=None):
    """Build an Inspect Task: the questions as samples + a bare generate() solver.

    ``model`` pins the task to one configured model, so a multi-rung sweep can
    name each task after its rung — two rungs sharing an underlying model id
    (gpt-5.2 vs gpt-5.2-thinking) are then distinguishable in the console."""
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
        samples.append(Sample(input=inp, id=q.id, metadata={"group": q.group}))

    return Task(
        dataset=MemoryDataset(samples), solver=generate(), name=name, model=model
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
        q.id: {
            "prompt": q.prompt,
            "group": q.group,
            "bucket": q.bucket,
            "system": q.system,
            "family": q.family,
            "variant": q.variant,
        }
        for q in questions
    }
    (run_dir / "questions.json").write_text(json.dumps(questions_meta, indent=2))

    # Persist the cross-variant family metadata for any family referenced by the
    # selected questions, so the analysis phase stays decoupled from the source
    # YAML (matching how questions.json decouples it from the question roster).
    from .questions import load_families

    referenced = {q.family for q in questions if q.family}
    if referenced:
        all_fams = load_families()
        fam_meta = {
            fid: {
                "prompt": f.prompt,
                "scalar": f.scalar,
                "title": f.title,
                "description": f.description,
            }
            for fid, f in all_fams.items()
            if fid in referenced
        }
        (run_dir / "families.json").write_text(json.dumps(fam_meta, indent=2))
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

    from .run_meta import build_meta, write_meta_safe

    write_meta_safe(
        run_dir,
        build_meta(
            "variance",
            label=run_dir.name,
            models=[m.name for m in model_specs],
            n_questions=len(questions),
            n=n,
        ),
    )


def _build_models(
    model_specs: list[ModelSpec],
    *,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_connections: Optional[int],
):
    """One Inspect ``Model`` per spec, carrying its temperature / reasoning effort."""
    from inspect_ai.model import GenerateConfig, get_model

    models = []
    for spec in model_specs:
        cfg = GenerateConfig(max_tokens=max_tokens, timeout=timeout)
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
    temperature: float = 1.0,
    max_tokens: int = 2048,
    run_dir: Optional[Path] = None,
    display: str = "rich",
    retry_on_error: int = 2,
    max_connections: Optional[int] = None,
    timeout: int = 180,
    model_concurrency: int = 1,
    log_dirs: Optional[dict[str, Path]] = None,
    on_model_done: Optional[callable] = None,
) -> dict[str, str]:
    """Run the whole generation sweep in one Inspect call. Returns {model: log_dir}.

    A single ``inspect_ai.eval`` call over one task per rung (each pinned to its
    model, named ``twominds:<rung>`` so same-id rungs are tellable apart
    in the console) — Inspect schedules them concurrently in one process with its
    own connection pool and one shared progress display, so there is no process
    pool, no display juggling, and no racing. ``model_concurrency`` maps straight to Inspect's ``max_tasks`` (how many
    models run at once; 1 = one at a time, but each model is still internally
    concurrent across its N×Q samples). Effective API concurrency is
    ~``model_concurrency × max_connections`` — mind provider rate limits.

    ``timeout`` caps each request (seconds): without it a single hung HTTP call
    blocks on the OpenAI client default (~10 min) before ``retry_on_error`` can
    retry it, freezing the sweep on the last straggler sample. 180s is ample for
    these short answers even on the reasoning rungs.

    ``eval`` returns one ``EvalLog`` per model in model order; each is written to
    ``logs/<spec.name>/<spec.name>.{eval,json}`` (``.eval`` canonical for
    ``analyze``; ``.json`` for human reading), keeping the per-model on-disk layout
    that disambiguates same-id rungs (gpt-5.2 vs gpt-5.2-thinking). ``log_dirs``
    overrides the destination per model (the store's gen dirs); with it,
    ``run_dir`` may be omitted. ``on_model_done(spec.name)`` fires per model once
    its logs are written and its eval succeeded (the store marks the generation
    complete there).

    A model whose eval did NOT succeed (bad id, auth failure, provider 4xx) still
    gets its log written for debugging, but ``on_model_done`` is skipped and a
    ``RuntimeError`` naming every failed model is raised at the end — an errored
    log holds cancelled samples with empty completions, which would otherwise flow
    silently into the judge as "responses".
    """
    import tempfile

    from inspect_ai import eval as inspect_eval
    from inspect_ai.log import write_eval_log

    if run_dir is None and log_dirs is None:
        raise ValueError("run_generation needs run_dir and/or log_dirs")
    logs_root: Optional[Path] = None
    if run_dir is not None:
        logs_root = Path(run_dir) / "logs"
        logs_root.mkdir(parents=True, exist_ok=True)
        raw_dir = logs_root / ".raw"  # Inspect's incremental writes; re-placed below
    else:
        raw_dir = Path(tempfile.mkdtemp(prefix="variance_raw_"))

    models = _build_models(
        model_specs,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_connections=max_connections,
    )
    # One task per rung, each pinned to its configured model and named after
    # the spec — so console panels distinguish two rungs that share one
    # underlying model id (gpt-5.2 vs gpt-5.2-thinking). Still ONE eval call.
    tasks = [
        build_task(questions, name=f"twominds:{spec.name}", model=model)
        for spec, model in zip(model_specs, models)
    ]
    logs = inspect_eval(
        tasks,
        epochs=n,
        log_dir=str(raw_dir),
        log_format="eval",
        display=display,
        retry_on_error=retry_on_error,
        score=False,
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
        if log_dirs is not None and spec.name in log_dirs:
            model_log_dir = Path(log_dirs[spec.name])
        else:
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
        elif on_model_done is not None:
            on_model_done(spec.name)
    shutil.rmtree(raw_dir, ignore_errors=True)
    if failures:
        raise RuntimeError("generation failed for:\n  " + "\n  ".join(failures))
    return out
