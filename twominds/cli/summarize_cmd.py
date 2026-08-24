"""The `summarize` command: per-model LLM summaries for an existing run."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from twominds import report as report_mod
from twominds import summarize as summarize_mod

from ._app import app
from ._options import JudgeOpt, JudgeReasonOpt, JudgeRunOpt


@app.command()
def summarize(
    run: str = typer.Option(
        ..., "--run", "-r", help="run dir containing analysis.json"
    ),
    judge_run: Optional[str] = JudgeRunOpt,
    judge: str = JudgeOpt,
    judge_reasoning: str = JudgeReasonOpt,
    force: bool = typer.Option(
        False, "--force", help="regenerate even if cached in summaries.json"
    ),
    concurrency: int = typer.Option(
        6, "--concurrency", help="concurrent summary calls"
    ),
):
    """Add per-model LLM summaries to an existing run, then rebuild the report.

    One judge-model call per model reads its judge verdicts, flags, and sample
    answers and writes a short "what stands out" blurb for the Overview tab.
    Cached in summaries.json next to analysis.json (invalidated when the
    summarizer model, reasoning effort, or prompt changes), so re-running is
    free and new models added to a run are summarized incrementally.
    """
    base = (Path(run) / "judge_runs" / judge_run) if judge_run else Path(run)
    summarize_mod.summarize_run(
        base,
        judge_name=judge,
        reasoning_effort=judge_reasoning,
        force=force,
        concurrency=concurrency,
        echo=typer.echo,
    )
    typer.echo(f"Wrote {report_mod.build_report_from_run(base)}")
