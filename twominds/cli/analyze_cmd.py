"""The `analyze` command (cross-sample judge -> analysis.json + scores)."""

from __future__ import annotations

from pathlib import Path

import typer

from twominds import analyze as analyze_mod
from twominds import report as report_mod

from ._app import app
from ._options import ConcurrencyOpt, JudgeOpt, JudgeReasonOpt
from ._summary import _echo_analyze_plan, _echo_judge_summary


@app.command()
def analyze(
    run: str = typer.Option(..., "--run", "-r", help="run dir from `generate`"),
    judge: str = JudgeOpt,
    judge_reasoning: str = JudgeReasonOpt,
    concurrency: int = ConcurrencyOpt,
    report: bool = typer.Option(
        False, "--report", help="also build the HTML report for this run"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="print judge plan + rough cost, no API calls"
    ),
):
    """Phase 2: cross-sample judge -> analysis.json with per-model scores.

    Verdicts the inline judge wrote into the generation logs (same judge
    config only) are reused; everything else is judged fresh.
    """
    if dry_run:
        _echo_analyze_plan(Path(run), judge=judge)
        return
    out = analyze_mod.analyze(
        Path(run),
        judge_name=judge,
        judge_reasoning=judge_reasoning,
        concurrency=concurrency,
    )
    _echo_judge_summary(out, Path(run))
    if report:
        path = report_mod.build_report_from_run(Path(run))
        typer.echo(f"  report -> {path}")
