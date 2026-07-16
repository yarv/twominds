"""The `analyze` command (cross-sample judge + embedding clustering)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from twominds import analyze as analyze_mod
from twominds import cost as cost_mod
from twominds import report as report_mod

from ._app import app
from ._options import (
    BackendsOpt,
    ConcurrencyOpt,
    JudgeOpt,
    JudgeReasonOpt,
    JudgeRunOpt,
    LocalModelOpt,
    NoConsistencyOpt,
    RepsOpt,
    ThreshOpt,
    _resolve_backends,
)
from ._reps import _echo_analyze_plan, _echo_judge_summary, _judge_reps


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
