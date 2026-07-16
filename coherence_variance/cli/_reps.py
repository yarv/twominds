"""Judge-pass machinery: rep1..repN loops, cost/summary echoes, and the
analyze --dry-run plan."""

from __future__ import annotations

from pathlib import Path

import typer

from coherence_variance import analyze as analyze_mod
from coherence_variance import cost as cost_mod
from coherence_variance import plan as plan_mod
from coherence_variance import report as report_mod


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
