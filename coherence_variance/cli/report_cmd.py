"""The viewer-side commands: `report`, `consistency`, and `merge`."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from coherence_variance import report as report_mod

from ._app import app
from ._options import JudgeRunOpt


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
