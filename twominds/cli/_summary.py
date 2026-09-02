"""Console summaries: the per-model score lines after a judge pass, and the
analyze --dry-run plan."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from twominds import plan as plan_mod


def _echo_judge_summary(out: dict, base: Path) -> None:
    results = out["results"]
    n = len(results)
    n_contra = sum(1 for r in results if (r["judge"] or {}).get("contradiction"))
    n_flag = sum(1 for r in results if (r["judge"] or {}).get("flags"))
    s = "s" if n != 1 else ""
    cs = "s" if n_contra != 1 else ""
    typer.echo(
        f"  -> {base / 'analysis.json'}: {n} bundle{s}, "
        f"{n_contra} contradiction{cs}, {n_flag} flagged"
    )
    scores = out.get("scores") or {}
    if not scores:
        return
    typer.echo(
        "  per-model scores (H = mean answer spread in nats; e^H = effective positions):"
    )
    for name, sc in scores.items():
        typer.echo(
            f"    {name:26s} H={sc['mean_entropy']:.3f}  "
            f"e^H={sc['effective_positions']:.2f}  "
            f"single-position {sc['frac_single_position'] * 100:.0f}% "
            f"of {sc['n_questions']} questions  flagged {sc['n_flagged']}"
        )


def _echo_analyze_plan(run_dir: Path, *, judge: str) -> None:
    """--dry-run for analyze: judge plan + rough cost from the run's manifests
    (run_config.json + questions.json), no API calls."""
    from twominds.models import ModelSpec
    from twominds.questions import Question

    try:
        cfg = json.loads((run_dir / "run_config.json").read_text())
        qmeta = json.loads((run_dir / "questions.json").read_text())
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
        )
        for qid, meta in qmeta.items()
    ]
    n = cfg.get("n", 1)
    plan = plan_mod.build_plan(specs, qs, n=n, judge=judge)
    typer.echo(f"=== Analyze plan (ROUGH estimate) for {run_dir} ===")
    typer.echo(f"{len(specs)} model(s) x {len(qs)} questions x N={n}")
    s = "s" if plan["judge_calls"] != 1 else ""
    typer.echo(f"judge: {plan['judge_calls']} call{s}  ~${plan['judge_dollars']:.2f}")
    typer.echo(
        "  (verdicts the inline judge already wrote into the generation logs "
        "are reused, not re-billed)"
    )
    typer.echo("\n(dry run — no API calls made)")
