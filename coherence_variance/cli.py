"""Response-variance / coherence experiment — canonical CLI entry point.

Ask each model a fixed set of free-form questions N times each (temperature
1.0) and study the variance across re-samples with a cross-sample LLM judge +
embedding clustering.

Both phases are Inspect ``eval``s (generation = one eval over all models; the judge
= one eval over the bundles), each log written in both ``.eval`` + ``.json`` form.
Phases leave artefacts on disk between each, so they are independently re-runnable:

    generate  ->  <run>/logs/<model>/<model>.{eval,json}, questions.json, run_config.json
    analyze   ->  <run>/judge_logs/{responses,families}.{eval,json}, analysis.json
    report    ->  <run>/report.html

Examples
--------
    # plan + rough cost, no API calls
    uv run python variance_experiment.py run --groups values --models gpt-4.1 --n 3 --dry-run

    # tiny smoke run end to end
    uv run python variance_experiment.py run --groups values --models gpt-4.1 --n 3

    # full default sweep (3 models, 96 questions incl. framing families, N=20)
    uv run python variance_experiment.py run --n 20
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import List, Optional

import typer

from coherence_variance import analyze as analyze_mod
from coherence_variance import cost as cost_mod
from coherence_variance import generate as generate_mod
from coherence_variance import plan as plan_mod
from coherence_variance import questions as questions_mod
from coherence_variance import report as report_mod
from coherence_variance import store as store_mod
from coherence_variance.embed import BACKENDS
from coherence_variance.models import (
    DEFAULT_JUDGE,
    DEFAULT_JUDGE_REASONING,
    DEFAULT_MODELS,
    next_name,
    resolve_models,
)

app = typer.Typer(
    add_completion=False,
    help=__doc__,
    no_args_is_help=True,
    # user-facing errors print as one clean message via main(); set
    # COHERENCE_DEBUG=1 for full tracebacks.
    pretty_exceptions_enable=False,
)


def main() -> None:
    """CLI entry point. Expected failures (a model that errored, a missing
    config file) print as one clean message; COHERENCE_DEBUG=1 re-enables the
    full traceback for debugging."""
    try:
        app()
    except (RuntimeError, FileNotFoundError) as e:
        if os.environ.get("COHERENCE_DEBUG"):
            raise
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        typer.secho(
            "(set COHERENCE_DEBUG=1 for the full traceback)",
            fg=typer.colors.RED,
            err=True,
        )
        raise SystemExit(1) from e


_RESULTS_ROOT = Path("results/variance")


def _csv(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _select_questions(
    groups: Optional[str],
    ids: Optional[str],
    all_questions: bool,
    families: Optional[str] = None,
    roster: Optional[str] = None,
    folders: Optional[str] = None,
):
    # --all-questions = every bucket; otherwise --folders
    # (default: tier_1 + prompt_robustness).
    buckets = list(questions_mod.BUCKETS) if all_questions else _csv(folders)
    return questions_mod.select_questions(
        groups=_csv(groups),
        ids=_csv(ids),
        buckets=buckets,
        families=_csv(families),
        roster=roster,
    )


def _default_run_dir() -> Path:
    return _RESULTS_ROOT / time.strftime("%Y%m%d_%H%M%S")


# ---- shared options ----
ModelsOpt = typer.Option(
    ",".join(DEFAULT_MODELS), "--models", "-m", help="comma-separated roster names"
)
GroupsOpt = typer.Option(
    None,
    "--groups",
    "-g",
    help="comma-separated question groups, matched across every selected bucket "
    "(default: all groups in the selected buckets)",
)
IdsOpt = typer.Option(
    None, "--ids", help="comma-separated explicit question ids (overrides groups)"
)
FamiliesOpt = typer.Option(
    None,
    "--families",
    help="comma-separated cross-variant family ids (selects every variant of each; "
    "overrides groups). Enables the framing-invariance / families_report analysis.",
)
AllQOpt = typer.Option(
    False,
    "--all-questions",
    help="select every bucket (tier_1 + tier_2 + prompt_robustness)",
)
RosterOpt = typer.Option(
    None,
    "--roster",
    help="named question roster from questions/_rosters.yaml "
    "(frozen id-list, overrides groups/buckets; none shipped by default)",
)
FoldersOpt = typer.Option(
    None,
    "--folders",
    help="comma-separated nature buckets to select: "
    "tier_1|tier_2|prompt_robustness (default: tier_1,prompt_robustness)",
)
NOpt = typer.Option(20, "--n", "-n", help="samples per question")
TempOpt = typer.Option(1.0, "--temperature", "-t", help="sampling temperature")
MaxTokOpt = typer.Option(2048, "--max-tokens", help="max output tokens per response")
ModelConcurrencyOpt = typer.Option(
    1,
    "--model-concurrency",
    help="how many models generate at once (Inspect max_tasks; default 1 = one at "
    "a time, but each model is still internally concurrent across its samples). "
    "Effective API concurrency is ~model_concurrency × max_connections, so watch "
    "provider rate limits / OpenRouter budget; 3-4 is a sane same-provider ceiling.",
)
JudgeOpt = typer.Option(
    DEFAULT_JUDGE, "--judge", help="Inspect model string for the coherence judge"
)
JudgeReasonOpt = typer.Option(
    DEFAULT_JUDGE_REASONING, "--judge-reasoning", help="judge reasoning effort"
)
BackendsOpt = typer.Option(
    ["openai-3-small"],
    "--embedding-backend",
    "-b",
    help=f"repeatable; one of {BACKENDS}, or 'none' to skip embedding "
    "clustering entirely (judge-only analysis)",
)


def _resolve_backends(backends: List[str]) -> list[str]:
    """Normalise --embedding-backend values; ``none`` disables embeddings."""
    vals = [b.strip() for b in backends if b and b.strip()]
    unknown = [b for b in vals if b != "none" and b not in BACKENDS]
    if unknown:
        raise typer.BadParameter(
            f"unknown embedding backend(s) {unknown}; choose from {BACKENDS} or 'none'"
        )
    if "none" in vals:
        if len(vals) > 1:
            raise typer.BadParameter(
                "-b none cannot be combined with other embedding backends"
            )
        return []
    return vals


ThreshOpt = typer.Option(
    0.15, "--threshold", help="cosine-distance clustering threshold"
)
LocalModelOpt = typer.Option(
    "BAAI/bge-small-en-v1.5", "--local-model", help="sentence-transformers model"
)
ConcurrencyOpt = typer.Option(6, "--concurrency", help="concurrent judge calls")
RepsOpt = typer.Option(
    1,
    "--reps",
    help="judge passes: rep1 (top-level analysis.json) + rep2..repN under "
    "judge_runs/, then auto-consistency when >1 (one command for a robust run)",
)
NoConsistencyOpt = typer.Option(
    False,
    "--no-consistency",
    help="with --reps>1, skip the consistency aggregation at the end",
)
DisplayOpt = typer.Option("rich", "--display", help="Inspect display: rich|plain|none")
RerunOpt = typer.Option(
    False,
    "--rerun",
    help="force fresh generations for every model (discards their cached gen "
    "dirs in the per-model store, judge fragments included)",
)
RerunModelOpt = typer.Option(
    None,
    "--rerun-model",
    help="force a fresh generation for this model only (repeatable; accepts the "
    "name as passed to --models)",
)
NoStoreOpt = typer.Option(
    False,
    "--no-store",
    help="bypass the per-model store entirely: generate straight into the run "
    "dir with no reuse (the pre-store behavior)",
)


def _do_generate(
    models,
    groups,
    ids,
    all_questions,
    families,
    n,
    temperature,
    max_tokens,
    judge,
    out,
    display,
    dry_run,
    roster=None,
    folders=None,
    model_concurrency=1,
    backends=None,
    will_judge=True,
    judge_reps=1,
):
    specs = resolve_models(_csv(models))
    qs = _select_questions(groups, ids, all_questions, families, roster, folders)
    if not qs:
        raise typer.BadParameter("no questions selected")

    plan = plan_mod.build_plan(specs, qs, n=n, judge=judge if will_judge else None)
    typer.echo(
        plan_mod.format_plan(plan, specs, qs, backends=backends, judge_reps=judge_reps)
    )
    _warn_missing_keys(specs, judge if will_judge else None, backends)
    if model_concurrency > 1:
        typer.echo(
            f"parallelism: up to {model_concurrency} models at a time "
            f"(Inspect max_tasks; ~{model_concurrency}× max_connections in flight)"
        )
    if dry_run:
        bal = cost_mod.openrouter_balance()
        if bal and bal.get("limit") is not None:
            typer.echo(
                f"\nOpenRouter now: ${bal.get('usage', 0):.2f} used / "
                f"${bal['limit']:.0f} limit (${bal.get('limit_remaining', 0):.2f} left)"
            )
        typer.echo("\n(dry run — no API calls made)")
        return None

    # headless: the rich display spams nohup logs; fall back to plain off-TTY.
    # (Concurrent models share Inspect's one rich display, so no downgrade needed
    # for parallelism — only for non-interactive stdout.)
    if display == "rich" and not sys.stdout.isatty():
        display = "plain"
        typer.echo("(non-interactive stdout detected: using --display plain)")

    run_dir = Path(out) if out else _default_run_dir()
    generate_mod.write_manifest(
        run_dir,
        specs,
        qs,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        judge=judge,
    )
    typer.echo(f"\nGenerating into {run_dir} ...")
    generate_mod.run_generation(
        specs,
        qs,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        run_dir=run_dir,
        display=display,
        model_concurrency=model_concurrency,
    )
    typer.echo(f"Generation complete: {run_dir}")
    return run_dir


def _announce_partition(cached, to_generate, gen_key, *, dry_run=False):
    """Echo the store-reuse summary for a partitioned roster."""
    if cached:
        typer.echo(
            f"reusing {len(cached)} cached generation(s) [gen {gen_key}]: "
            + ", ".join(cached)
        )
    if to_generate:
        verb = "would generate" if dry_run else "generating"
        typer.echo(
            f"{verb} {len(to_generate)} model(s) [gen {gen_key}]: "
            + ", ".join(s.name for s in to_generate)
        )


def _warn_missing_keys(specs, judge=None, backends=None):
    """Echo a note per unset API-key env var the sweep will need (never fatal)."""
    need: dict[str, str] = {}

    def _provider_envs(inspect_model: str) -> list[str]:
        prov, _, rest = inspect_model.partition("/")
        if prov == "openai":
            return ["OPENAI_API_KEY"]
        if prov == "openrouter":
            return ["OPENROUTER_API_KEY"]
        if prov == "anthropic":
            return ["ANTHROPIC_API_KEY"]
        if prov == "openai-api":
            service = rest.partition("/")[0].upper().replace("-", "_")
            return [f"{service}_API_KEY", f"{service}_BASE_URL"]
        return []  # locally-managed providers (vllm/, ollama/, ...) need no key

    for s in specs:
        for env in _provider_envs(s.inspect_model):
            need.setdefault(env, f"generating with {s.name}")
    if judge:
        for env in _provider_envs(judge):
            need.setdefault(env, "the judge")
    for b in backends or ():
        if b.startswith("openai-"):
            need.setdefault("OPENAI_API_KEY", f"the {b} embedding backend")
    for env, reason in need.items():
        if not os.environ.get(env):
            typer.echo(f"note: {env} is not set — {reason} will fail")


def _plan_generations(
    specs,
    qs,
    *,
    n,
    temperature,
    max_tokens,
    rerun,
    rerun_models,
    raw_names=None,
    dry_run=False,
    announce=True,
):
    """Partition the roster into cached vs to-generate against the store.

    Returns ``(gen_key, gen_dirs, cached_names, to_generate_specs)`` and echoes
    the reuse summary. ``rerun``/``rerun_models`` force regeneration (their gen
    dirs are cleared just before generating — see ``store.clear_generation``).
    ``raw_names`` are the model tokens as passed to --models (same order as
    ``specs``) so --rerun-model accepts aliases too. A store name already
    pinned to a DIFFERENT model is auto-qualified with more path segments
    (mutating ``specs`` in place, like the in-batch disambiguation in
    ``resolve_models``). Dry runs stay read-only.
    """
    root = store_mod.store_root(_RESULTS_ROOT)
    gen_key = store_mod.compute_gen_key(
        qs, n=n, temperature=temperature, max_tokens=max_tokens
    )
    raw = list(raw_names) if raw_names else [None] * len(specs)
    forced = set(rerun_models or [])
    accepted = set(t for t in raw if t)
    for s in specs:
        accepted.update({s.name, s.display})
    unknown = forced - accepted
    if unknown:
        raise typer.BadParameter(
            f"--rerun-model {sorted(unknown)} not among the selected models"
        )

    gen_dirs: dict[str, Path] = {}
    cached: list[str] = []
    to_generate = []
    for i, spec in enumerate(specs):
        # Cross-invocation collision: the store name belongs to a different
        # model from an earlier run — qualify with more path segments.
        while (prev := store_mod.identity_conflict(spec, root)) is not None:
            ref = spec.display or spec.name
            nxt = next_name(ref, spec.name)
            if nxt is None:
                raise typer.BadParameter(
                    f"store name '{spec.name}' is pinned to "
                    f"{prev.get('inspect_model')} and '{ref}' has no more path "
                    "segments to disambiguate with; use --no-store or remove "
                    f"results/variance/models/{spec.name}"
                )
            typer.echo(
                f"(store name '{spec.name}' is taken by "
                f"{prev.get('inspect_model')}; using '{nxt}')"
            )
            spec = dc_replace(spec, name=nxt)
            specs[i] = spec
        if not dry_run:
            store_mod.write_identity(spec, root)
        found = store_mod.find_generation(spec, gen_key, root)
        force = rerun or bool({spec.name, spec.display, raw[i]} & forced)
        if found is not None and not force:
            gen_dirs[spec.name] = found
            cached.append(spec.name)
        else:
            gen_dirs[spec.name] = store_mod.gen_dir(spec, gen_key, root)
            to_generate.append(spec)

    if announce:
        _announce_partition(cached, to_generate, gen_key, dry_run=dry_run)
    return gen_key, gen_dirs, cached, to_generate


def _execute_generations(
    to_generate,
    qs,
    gen_dirs,
    gen_key,
    *,
    n,
    temperature,
    max_tokens,
    judge,
    model_concurrency=1,
    display="rich",
):
    """Generate the missing models into their store gen dirs.

    The missing models still run as ONE Inspect eval (each model's logs land in
    its own store gen dir via ``log_dirs``, marked complete as they're written).
    """
    root = store_mod.store_root(_RESULTS_ROOT)
    for spec in to_generate:
        store_mod.clear_generation(spec, gen_key, root)  # no-op unless forced rerun
        store_mod.prepare_generation(
            spec,
            qs,
            gen_key,
            root,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            judge=judge,
        )
    generate_mod.run_generation(
        to_generate,
        qs,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        display=display,
        model_concurrency=model_concurrency,
        log_dirs={
            s.name: Path(gen_dirs[s.name]) / "logs" / s.name for s in to_generate
        },
        on_model_done=lambda name: store_mod.mark_complete(gen_dirs[name], key=gen_key),
    )


def _setup_store_run(
    models,
    groups,
    ids,
    all_questions,
    families,
    roster,
    folders,
    *,
    n,
    temperature,
    max_tokens,
    judge,
    out,
    display,
    model_concurrency,
    rerun,
    rerun_models,
    dry_run,
    backends=None,
    will_judge=True,
    judge_reps=1,
):
    """Store-backed generation phase shared by `run` and `generate`: ensure
    per-model generations exist (reusing the store), then create the run dir
    with symlinked logs. Returns (run_dir, specs, gen_dirs, cached_names);
    (None, specs, gen_dirs, cached_names) on dry runs.

    ``backends``/``will_judge`` only shape the printed plan: pass the resolved
    embedding backends (``run``) or leave None (``generate``), and set
    ``will_judge=False`` when no judge phase follows (its cost is then left out).
    """
    specs = resolve_models(_csv(models))
    qs = _select_questions(groups, ids, all_questions, families, roster, folders)
    if not qs:
        raise typer.BadParameter("no questions selected")
    # partition against the store first, so the printed plan reflects the cache
    gen_key, gen_dirs, cached, to_generate = _plan_generations(
        specs,
        qs,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        rerun=rerun,
        rerun_models=rerun_models,
        raw_names=_csv(models),
        dry_run=dry_run,
        announce=False,
    )
    typer.echo(
        plan_mod.format_plan(
            plan_mod.build_plan(specs, qs, n=n, judge=judge if will_judge else None),
            specs,
            qs,
            backends=backends,
            cached=cached,
            judge_reps=judge_reps,
        )
    )
    _warn_missing_keys(
        [s for s in specs if s.name not in cached],
        judge if will_judge else None,
        backends,
    )
    _announce_partition(cached, to_generate, gen_key, dry_run=dry_run)
    if dry_run:
        typer.echo("\n(dry run — no API calls made)")
        return None, specs, gen_dirs, cached
    if display == "rich" and not sys.stdout.isatty():
        display = "plain"
        typer.echo("(non-interactive stdout detected: using --display plain)")
    if to_generate:
        _execute_generations(
            to_generate,
            qs,
            gen_dirs,
            gen_key,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            judge=judge,
            model_concurrency=model_concurrency,
            display=display,
        )
    run_dir = Path(out) if out else _default_run_dir()
    generate_mod.write_manifest(
        run_dir,
        specs,
        qs,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        judge=judge,
    )
    for spec in specs:
        store_mod.link_into_run(run_dir, spec, gen_dirs[spec.name])
    return run_dir, specs, gen_dirs, cached


@app.command()
def generate(
    models: str = ModelsOpt,
    groups: Optional[str] = GroupsOpt,
    ids: Optional[str] = IdsOpt,
    families: Optional[str] = FamiliesOpt,
    all_questions: bool = AllQOpt,
    roster: Optional[str] = RosterOpt,
    folders: Optional[str] = FoldersOpt,
    n: int = NOpt,
    temperature: float = TempOpt,
    max_tokens: int = MaxTokOpt,
    model_concurrency: int = ModelConcurrencyOpt,
    judge: str = JudgeOpt,
    out: Optional[str] = typer.Option(
        None, "--out", "-o", help="run dir (default results/variance/<ts>)"
    ),
    display: str = DisplayOpt,
    rerun: bool = RerunOpt,
    rerun_model: Optional[List[str]] = RerunModelOpt,
    no_store: bool = NoStoreOpt,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="print plan + cost, no API calls"
    ),
):
    """Phase 1: sample each model N times on the question roster (Inspect).

    Generations are cached per model under results/variance/models/ and reused
    when the same questions + sampling config recur; --rerun / --rerun-model
    force fresh ones, --no-store restores the old self-contained behavior.
    """
    if no_store:
        _do_generate(
            models,
            groups,
            ids,
            all_questions,
            families,
            n,
            temperature,
            max_tokens,
            judge,
            out,
            display,
            dry_run,
            roster=roster,
            folders=folders,
            model_concurrency=model_concurrency,
            will_judge=False,
        )
        return
    run_dir, _specs, _gen_dirs, _cached = _setup_store_run(
        models,
        groups,
        ids,
        all_questions,
        families,
        roster,
        folders,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        judge=judge,
        out=out,
        display=display,
        model_concurrency=model_concurrency,
        rerun=rerun,
        rerun_models=rerun_model,
        dry_run=dry_run,
        will_judge=False,
    )
    if run_dir is not None:
        typer.echo(f"Generation complete: {run_dir}")


JudgeRunOpt = typer.Option(
    None,
    "--judge-run",
    help="label to isolate this judge run under judge_runs/<label>/",
)


def _echo_judge_summary(out: dict, base: Path) -> None:
    n = len(out["results"])
    n_contra = sum(1 for r in out["results"] if (r["judge"] or {}).get("contradiction"))
    n_flag = sum(1 for r in out["results"] if (r["judge"] or {}).get("flags"))
    s = "s" if n != 1 else ""
    cs = "s" if n_contra != 1 else ""
    typer.echo(
        f"  -> {base / 'analysis.json'}: {n} bundle{s}, "
        f"{n_contra} contradiction{cs}, {n_flag} flagged"
    )


def _echo_cost_total(costs: list[dict]) -> None:
    """Reconciled run-total cost across all reps (est token×price + actual)."""
    gen = (costs[0].get("generation") if costs else None) or {}
    gen_d = sum(v["dollars"] for v in gen.values() if not v.get("cached"))
    gen_c = sum(v["dollars"] for v in gen.values() if v.get("cached"))
    j_est = sum((c.get("judge") or {}).get("est_dollars", 0) or 0 for c in costs)
    j_est -= sum((c.get("judge") or {}).get("cached_dollars", 0) or 0 for c in costs)
    deltas = [(c.get("judge") or {}).get("openrouter_delta") for c in costs]
    deltas = [d for d in deltas if d is not None]
    typer.echo("\ncost (run total):")
    if gen:
        cached_note = f"  (+ ${gen_c:.2f} reused from cache)" if gen_c else ""
        typer.echo(f"  generation (est):                 ${gen_d:.2f}{cached_note}")
    typer.echo(f"  judge (est, {len(costs)} rep(s)):              ${j_est:.2f}")
    if deltas:
        typer.echo(f"  judge (actual, OpenRouter delta): ${sum(deltas):.2f}")
    typer.echo(f"  TOTAL (est):                      ${gen_d + j_est:.2f}")
    bal = cost_mod.openrouter_balance()
    if bal and bal.get("limit") is not None:
        typer.echo(
            f"  OpenRouter: ${bal.get('usage', 0):.2f} used / ${bal['limit']:.0f} "
            f"limit (${bal.get('limit_remaining', 0):.2f} left, "
            f"${bal.get('usage_daily', 0):.2f} today)"
        )


def _judge_reps(
    run_dir: Path,
    *,
    reps: int,
    consistency: bool,
    backends,
    judge: str,
    judge_reasoning: str,
    threshold: float,
    local_model: str,
    concurrency: int,
    run_judge: bool,
    refresh_embeddings: bool = False,
    build_report: bool = True,
) -> dict:
    """Run rep1 (top-level analysis.json) + rep2..repN (judge_runs/repK), then
    optionally aggregate consistency. Embeddings are computed once on rep1 and
    reused (judge-only reps). Returns rep1's analyze output.
    """

    def _one(label):
        typer.echo(f"\n=== judge pass {label or 'rep1 (default)'} ===")
        return analyze_mod.analyze(
            run_dir,
            backends=list(backends),
            judge_name=judge,
            judge_reasoning=judge_reasoning,
            threshold=threshold,
            local_model=local_model,
            concurrency=concurrency,
            run_judge=run_judge,
            judge_run=label,
            refresh_embeddings=refresh_embeddings if label is None else False,
        )

    out = _one(None)
    _echo_judge_summary(out, run_dir)
    costs = [out.get("cost") or {}]
    for k in range(2, reps + 1):
        lbl = f"rep{k}"
        ok = _one(lbl)
        _echo_judge_summary(ok, run_dir / "judge_runs" / lbl)
        costs.append(ok.get("cost") or {})
    if build_report:
        typer.echo(f"  report -> {report_mod.build_report_from_run(run_dir)}")
    if run_judge:
        _echo_cost_total(costs)
    if consistency and reps > 1 and run_judge:
        from coherence_variance import consistency as cons_mod

        o = cons_mod.build_consistency_from_run(run_dir, include_default=True)[
            "overall"
        ]
        typer.echo(
            f"\nconsistency: {o['n_runs']} reps "
            f"({cons_mod.format_run_labels(o['run_labels'])}), "
            f"ARI {o['mean_partition_ari']:.3f}, consensus "
            f"{o['mean_consensus_strength']:.3f}, contradiction-unstable "
            f"{(o['frac_contradiction_unstable'] or 0) * 100:.0f}%"
        )
    return out


def _echo_analyze_plan(run_dir, *, backends, judge, no_judge, reps):
    """--dry-run for analyze: judge plan + rough cost from the run's manifests
    (run_config.json + questions.json), no API calls."""
    import json as _json

    from coherence_variance.models import ModelSpec
    from coherence_variance.questions import Question

    try:
        cfg = _json.loads((run_dir / "run_config.json").read_text())
        qmeta = _json.loads((run_dir / "questions.json").read_text())
    except FileNotFoundError as e:
        raise typer.BadParameter(
            f"{run_dir} is not a generated run dir (missing {Path(e.filename).name})"
        ) from e
    specs = [
        ModelSpec(
            name=name,
            inspect_model=m.get("inspect_model", name),
            reasoning_effort=m.get("reasoning_effort"),
            display=m.get("display", ""),
        )
        for name, m in cfg.get("models", {}).items()
    ]
    qs = [
        Question(
            id=qid,
            group=meta.get("group", ""),
            prompt=meta.get("prompt", ""),
            system=meta.get("system"),
            family=meta.get("family"),
            variant=meta.get("variant"),
        )
        for qid, meta in qmeta.items()
    ]
    n = cfg.get("n", 1)
    plan = plan_mod.build_plan(specs, qs, n=n, judge=None if no_judge else judge)
    typer.echo(f"=== Analyze plan (ROUGH estimate) for {run_dir} ===")
    typer.echo(f"{len(specs)} model(s) x {len(qs)} questions x N={n}")
    if not no_judge:
        per_pass = plan["judge_dollars"]
        s = "s" if plan["judge_calls"] != 1 else ""
        line = f"judge: {plan['judge_calls']} call{s}  ~${per_pass:.2f} per pass"
        if reps > 1:
            line += f"  x {reps} reps = ~${per_pass * reps:.2f}"
        typer.echo(line)
        typer.echo(
            "  (standalone analyze always judges fresh; `run` reuses cached verdicts)"
        )
    if len(backends) == 0:
        typer.echo("embeddings: none (judge-only analysis)")
    else:
        free = "; the local backend is free" if "local" in backends else ""
        typer.echo(f"embeddings: negligible (cents){free}")
    typer.echo("\n(dry run — no API calls made)")


@app.command()
def analyze(
    run: str = typer.Option(..., "--run", "-r", help="run dir from `generate`"),
    backends: List[str] = BackendsOpt,
    judge: str = JudgeOpt,
    judge_reasoning: str = JudgeReasonOpt,
    threshold: float = ThreshOpt,
    local_model: str = LocalModelOpt,
    concurrency: int = ConcurrencyOpt,
    judge_run: Optional[str] = JudgeRunOpt,
    reps: int = RepsOpt,
    no_consistency: bool = NoConsistencyOpt,
    refresh_embeddings: bool = typer.Option(
        False,
        "--refresh-embeddings",
        help="recompute embeddings instead of using the cache",
    ),
    report: bool = typer.Option(
        False, "--report", help="also build the HTML report for this run"
    ),
    no_judge: bool = typer.Option(
        False, "--no-judge", help="skip the LLM judge (embeddings only)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="print judge plan + rough cost, no API calls"
    ),
):
    """Phase 2: cross-sample judge + embedding clustering -> analysis.json.

    Pass --reps N to do rep1 + rep2..repN + consistency in one go (robust run).
    Pass --judge-run <label> instead for a single isolated pass under
    judge_runs/<label>/ (embeddings cached, so reps are cheap); `consistency`
    then aggregates across all judge runs.

    Note: standalone analyze always judges fresh. Only `run` reuses the
    per-model judge verdicts cached in the store (`[judge cached ✓]`).
    """
    backends = _resolve_backends(backends)
    if not backends and no_judge:
        raise typer.BadParameter("-b none with --no-judge leaves nothing to analyze")
    if dry_run:
        _echo_analyze_plan(
            Path(run),
            backends=backends,
            judge=judge,
            no_judge=no_judge,
            reps=reps if judge_run is None else 1,
        )
        return
    if judge_run is None and reps > 1:
        _judge_reps(
            Path(run),
            reps=reps,
            consistency=not no_consistency,
            backends=backends,
            judge=judge,
            judge_reasoning=judge_reasoning,
            threshold=threshold,
            local_model=local_model,
            concurrency=concurrency,
            run_judge=not no_judge,
            refresh_embeddings=refresh_embeddings,
            build_report=report,
        )
        return
    out = analyze_mod.analyze(
        Path(run),
        backends=list(backends),
        judge_name=judge,
        judge_reasoning=judge_reasoning,
        threshold=threshold,
        local_model=local_model,
        concurrency=concurrency,
        run_judge=not no_judge,
        judge_run=judge_run,
        refresh_embeddings=refresh_embeddings,
    )
    base = (Path(run) / "judge_runs" / judge_run) if judge_run else Path(run)
    _echo_judge_summary(out, base)
    if out.get("cost"):
        typer.echo(
            cost_mod.format_summary(
                out["cost"], gen_note="from the run's logs — not billed by analyze"
            )
        )
    if report:
        path = report_mod.build_report_from_run(base)
        typer.echo(f"  report -> {path}")


@app.command()
def report(
    run: str = typer.Option(
        ..., "--run", "-r", help="run dir containing analysis.json"
    ),
    judge_run: Optional[str] = JudgeRunOpt,
    out: Optional[str] = typer.Option(None, "--out", "-o", help="output html path"),
):
    """Phase 3: build the self-contained HTML viewer."""
    base = (Path(run) / "judge_runs" / judge_run) if judge_run else Path(run)
    path = report_mod.build_report_from_run(base, Path(out) if out else None)
    typer.echo(f"Wrote {path}")
    analysis_path = base / "analysis.json"
    if analysis_path.exists():
        import json as _json

        from coherence_variance.families_report import build_families_report

        analysis = _json.loads(analysis_path.read_text())
        if analysis.get("families"):
            fpath = build_families_report(analysis, base / "families_report.html")
            typer.echo(f"Wrote {fpath}")


@app.command()
def consistency(
    run: str = typer.Option(..., "--run", "-r", help="run dir with >=2 judge runs"),
    include_default: bool = typer.Option(
        True,
        "--include-default/--no-include-default",
        help="also treat the top-level analysis.json as a judge run",
    ),
):
    """Aggregate judge-consistency stats across all judge runs of one generation."""
    from coherence_variance import consistency as cons_mod

    agg = cons_mod.build_consistency_from_run(
        Path(run), include_default=include_default
    )
    o = agg["overall"]
    typer.echo(
        f"{o['n_runs']} judge runs ({cons_mod.format_run_labels(o['run_labels'])}) "
        f"over {o['n_bundles']} bundles"
    )
    typer.echo(
        f"  mean consensus strength: {o['mean_consensus_strength']:.3f} (1.0 = same boundaries every run)"
    )
    typer.echo(
        f"  mean partition ARI: {o['mean_partition_ari']:.3f} (1.0 = identical groupings)"
    )
    typer.echo(f"  mean contested pairs: {o['mean_contested_pairs']:.2f}/bundle")
    typer.echo(
        f"  contradiction unstable: {(o['frac_contradiction_unstable'] or 0) * 100:.0f}% of bundles"
    )
    typer.echo(
        f"  -> judge_consistency.json + consistency_report.html + multi_report.html (in {run})"
    )


@app.command()
def merge(
    runs: List[str] = typer.Option(
        ..., "--run", "-r", help="run dir to merge (repeatable; >= 2)"
    ),
    out: str = typer.Option(
        ..., "--out", "-o", help="output dir for the combined report"
    ),
):
    """Combine several variance runs (same question bank) into one report.

    Concatenates the runs' top-level analyses (models unioned) and renders one
    report.html with a model selector -- no re-judging, so it's free. For runs
    whose models were generated separately (e.g. fine-tunes that landed at
    different times). Per-run judge-pass robustness stays with each source run.
    """
    from coherence_variance.merge import write_merged

    merged = write_merged(runs, out)
    for w in merged.get("merge_warnings", []):
        typer.echo(f"  warning: {w}")
    path = report_mod.build_report_from_run(Path(out))
    typer.echo(
        f"merged {len(runs)} runs -> {len(merged['models'])} models, "
        f"{len(merged['results'])} bundles"
    )
    typer.echo(f"  report -> {path}")


@app.command()
def budget():
    """Show OpenRouter spend / limit / remaining for $OPENROUTER_API_KEY."""
    from dotenv import load_dotenv

    load_dotenv()
    bal = cost_mod.openrouter_balance()
    if not bal:
        typer.echo(
            "OpenRouter balance unavailable (no OPENROUTER_API_KEY / API error)."
        )
        raise typer.Exit(1)
    lim = bal.get("limit")
    typer.echo("OpenRouter key budget:")
    used = f"${bal.get('usage', 0):.2f}" + (f" / ${lim:.0f} limit" if lim else "")
    typer.echo(f"  usage (this key):     {used}")
    if bal.get("limit_remaining") is not None:
        typer.echo(f"  remaining:            ${bal['limit_remaining']:.2f}")
    typer.echo(
        f"  today/week/month:     ${bal.get('usage_daily', 0):.2f}"
        f" / ${bal.get('usage_weekly', 0):.2f} / ${bal.get('usage_monthly', 0):.2f}"
    )
    cr = cost_mod.openrouter_usage()
    if cr is not None:
        typer.echo(f"  account total_usage:  ${cr:.2f}")


@app.command()
def run(
    models: str = ModelsOpt,
    groups: Optional[str] = GroupsOpt,
    ids: Optional[str] = IdsOpt,
    families: Optional[str] = FamiliesOpt,
    all_questions: bool = AllQOpt,
    roster: Optional[str] = RosterOpt,
    folders: Optional[str] = FoldersOpt,
    n: int = NOpt,
    temperature: float = TempOpt,
    max_tokens: int = MaxTokOpt,
    model_concurrency: int = ModelConcurrencyOpt,
    judge: str = JudgeOpt,
    judge_reasoning: str = JudgeReasonOpt,
    backends: List[str] = BackendsOpt,
    threshold: float = ThreshOpt,
    local_model: str = LocalModelOpt,
    concurrency: int = ConcurrencyOpt,
    reps: int = RepsOpt,
    no_consistency: bool = NoConsistencyOpt,
    out: Optional[str] = typer.Option(None, "--out", "-o", help="run dir"),
    display: str = DisplayOpt,
    no_judge: bool = typer.Option(False, "--no-judge", help="skip the LLM judge"),
    rerun: bool = RerunOpt,
    rerun_model: Optional[List[str]] = RerunModelOpt,
    no_store: bool = NoStoreOpt,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="print plan + cost, no API calls"
    ),
):
    """All phases: generate -> judge (rep1..repN) -> consistency -> report.

    Generations (and rep1 judge verdicts) are cached per model under
    results/variance/models/ and reused automatically when the same questions +
    sampling config come around again — only missing models hit the API. Use
    --rerun / --rerun-model to force fresh generations, --no-store for the old
    fully-self-contained behavior.

    --reps N runs a full robust pass (rep1 + rep2..repN + consistency) in one
    command, so a robust run is a single invocation.
    """
    backends = _resolve_backends(backends)
    if not backends and no_judge:
        raise typer.BadParameter("-b none with --no-judge leaves nothing to analyze")
    if no_store:
        run_dir = _do_generate(
            models,
            groups,
            ids,
            all_questions,
            families,
            n,
            temperature,
            max_tokens,
            judge,
            out,
            display,
            dry_run,
            roster=roster,
            folders=folders,
            model_concurrency=model_concurrency,
            backends=list(backends),
            will_judge=not no_judge,
            judge_reps=reps if not no_judge else 1,
        )
        if run_dir is None:  # dry run
            return
        _judge_reps(
            run_dir,
            reps=reps,
            consistency=not no_consistency,
            backends=backends,
            judge=judge,
            judge_reasoning=judge_reasoning,
            threshold=threshold,
            local_model=local_model,
            concurrency=concurrency,
            run_judge=not no_judge,
            build_report=True,
        )
        typer.echo(f"\nDone. Open {run_dir / 'report.html'}")
        return

    run_dir, specs, gen_dirs, cached_gens = _setup_store_run(
        models,
        groups,
        ids,
        all_questions,
        families,
        roster,
        folders,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        judge=judge,
        out=out,
        display=display,
        model_concurrency=model_concurrency,
        rerun=rerun,
        rerun_models=rerun_model,
        dry_run=dry_run,
        backends=list(backends),
        will_judge=not no_judge,
        judge_reps=reps if not no_judge else 1,
    )
    if run_dir is None:  # dry run
        return

    # rep1: cached per-model judge fragments where fresh, judged now otherwise.
    judge_key = store_mod.compute_judge_key(
        judge_name=judge,
        judge_reasoning=judge_reasoning,
        threshold=threshold,
        backends=list(backends),
        local_model=local_model,
        run_judge=not no_judge,
    )
    combined = store_mod.assemble_run(
        run_dir,
        specs,
        gen_dirs,
        judge_key=judge_key,
        backends=list(backends),
        judge_name=judge,
        judge_reasoning=judge_reasoning,
        threshold=threshold,
        local_model=local_model,
        concurrency=concurrency,
        run_judge=not no_judge,
        on_fragment=lambda name, cached: typer.echo(
            f"  [judge {'cached ✓' if cached else '✔'}] {name}"
        ),
        # rep2..N judge the whole run and read run_dir/cache — seed it from the
        # fragments' per-model caches so they don't re-embed everything.
        preseed_cache=reps > 1,
        cached_gens=set(cached_gens),
    )
    _echo_judge_summary(combined, run_dir)
    typer.echo(f"  report -> {report_mod.build_report_from_run(run_dir)}")
    if combined.get("cost"):
        typer.echo(cost_mod.format_summary(combined["cost"]))
    if reps > 1 and not no_judge:
        rep_costs = _extra_judge_reps(
            run_dir,
            reps=reps,
            consistency=not no_consistency,
            backends=backends,
            judge=judge,
            judge_reasoning=judge_reasoning,
            threshold=threshold,
            local_model=local_model,
            concurrency=concurrency,
        )
        _echo_cost_total([combined.get("cost") or {}] + rep_costs)
    typer.echo(f"\nDone. Open {run_dir / 'report.html'}")


def _extra_judge_reps(
    run_dir,
    *,
    reps,
    consistency,
    backends,
    judge,
    judge_reasoning,
    threshold,
    local_model,
    concurrency,
):
    """rep2..repN + consistency for a store-backed run (rep1 came from the
    fragments in ``assemble_run``). Whole-run passes reading through the run's
    log symlinks. Returns the per-rep cost records for the run-total echo."""
    run_dir = Path(run_dir)
    costs = []
    for k in range(2, reps + 1):
        lbl = f"rep{k}"
        typer.echo(f"\n=== judge pass {lbl} ===")
        ok = analyze_mod.analyze(
            run_dir,
            backends=list(backends),
            judge_name=judge,
            judge_reasoning=judge_reasoning,
            threshold=threshold,
            local_model=local_model,
            concurrency=concurrency,
            run_judge=True,
            judge_run=lbl,
        )
        _echo_judge_summary(ok, run_dir / "judge_runs" / lbl)
        costs.append(ok.get("cost") or {})
    if consistency and reps > 1:
        from coherence_variance import consistency as cons_mod

        o = cons_mod.build_consistency_from_run(run_dir, include_default=True)[
            "overall"
        ]
        typer.echo(
            f"\nconsistency: {o['n_runs']} reps "
            f"({cons_mod.format_run_labels(o['run_labels'])}), "
            f"ARI {o['mean_partition_ari']:.3f}, consensus "
            f"{o['mean_consensus_strength']:.3f}, contradiction-unstable "
            f"{(o['frac_contradiction_unstable'] or 0) * 100:.0f}%"
        )
    return costs


def _f2(x: Optional[float]) -> str:
    return "–" if x is None else f"{x:.2f}"


@app.command()
def stress(
    scenarios: Optional[str] = typer.Option(
        None, "--scenarios", help="comma-separated scenario ids (default: all)"
    ),
    mixes: Optional[str] = typer.Option(
        None, "--mixes", help="comma-separated mix labels to include (default: all)"
    ),
    n: int = typer.Option(20, "--n", "-n", help="bundle size (responses per bundle)"),
    reps: int = typer.Option(3, "--reps", help="resampled bundles per (scenario,mix)"),
    pool_model: str = typer.Option(
        "gpt-4.1", "--pool-model", help="output model that writes the stance pools"
    ),
    pool_size: int = typer.Option(
        24, "--pool-size", help="samples per (scenario,stance) pool"
    ),
    judge: str = JudgeOpt,
    judge_reasoning: str = JudgeReasonOpt,
    concurrency: int = ConcurrencyOpt,
    seed: int = typer.Option(0, "--seed", help="bundle-composition seed"),
    max_tokens: int = typer.Option(
        320, "--max-tokens", help="max output tokens per pool response"
    ),
    display: str = DisplayOpt,
    smoke: bool = typer.Option(
        False,
        "--smoke",
        help="tiny fixed 1-scenario end-to-end smoke (overrides sizes)",
    ),
    out: Optional[str] = typer.Option(
        None, "--out", "-o", help="run dir (default results/variance/stress_<ts>)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="print plan + cost, no API calls"
    ),
):
    """Stress-test the judge on synthetic bundles with engineered ground truth.

    Authors stance system prompts (coherence_variance/stress_data.yaml), samples a
    neutral pool model under each, composes needle-in-a-haystack mixes, runs the
    real judge, and scores it vs the planted partition (ARI, needle recall, group-
    count error, false-positive splits on unanimous bundles).

    Tip: route --judge directly as anthropic/claude-opus-4.8 to silence the cosmetic
    OpenRouter reasoning-parse spam (judge verdicts are unaffected either way).
    """
    from coherence_variance import stress as stress_mod

    all_scen = stress_mod.load_spec()
    if smoke:
        scen_ids: Optional[list[str]] = ["deceive_binary"]
        mix_set: Optional[set[str]] = {
            "unanimous",
            "needle_1",
            "balanced",
            "subtle_needle_1",
        }
        n, reps, pool_size = 10, 2, 12
    else:
        scen_ids = _csv(scenarios)
        mix_set = set(_csv(mixes)) if mixes else None

    selected = stress_mod.select_scenarios(all_scen, scen_ids)
    if mix_set:
        known = {m.label for sc in selected for m in sc.mixes}
        bad = mix_set - known
        if bad:
            raise typer.BadParameter(f"unknown mix label(s): {sorted(bad)}")

    typer.echo(
        stress_mod.plan_stress(
            selected,
            n=n,
            reps=reps,
            pool_model=pool_model,
            pool_size=pool_size,
            judge=judge,
            mix_filter=mix_set,
        )
    )
    if dry_run:
        typer.echo("\n(dry run — no API calls made)")
        return

    run_dir = (
        Path(out) if out else _RESULTS_ROOT / f"stress_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    typer.echo(f"\nRunning stress test into {run_dir} ...")
    analysis = stress_mod.run_stress(
        selected,
        n=n,
        reps=reps,
        pool_model=pool_model,
        pool_size=pool_size,
        run_dir=run_dir,
        judge_name=judge,
        judge_reasoning=judge_reasoning,
        mix_filter=mix_set,
        seed=seed,
        concurrency=concurrency,
        max_tokens=max_tokens,
        display=display,
    )
    path = stress_mod.build_stress_report_from_run(run_dir)
    a = analysis["aggregate"]
    typer.echo(f"\nDone. {a['overall']['n_bundles']} bundles judged.")
    typer.echo(
        f"  mean ARI (multi-stance): {_f2(a['overall']['mean_ari_multistance'])}"
    )
    typer.echo(
        f"  unanimous over-split: {_f2(a['unanimous']['oversplit_rate'])} | "
        f"false-contradiction: {_f2(a['unanimous']['false_contradiction_rate'])}"
    )
    if a["needle_curve"]:
        curve = ", ".join(
            f"k={r['k']}:{_f2(r['needle_recall_mean'])}" for r in a["needle_curve"]
        )
        typer.echo(f"  needle recall by k: {curve}")
    cf = a["contradiction_confusion"]
    typer.echo(
        f"  contradiction confusion (contradictory scenarios): "
        f"TP {cf['tp']} FN {cf['fn']} FP {cf['fp']} TN {cf['tn']}"
    )
    typer.echo(f"  report -> {path}")


if __name__ == "__main__":
    main()
