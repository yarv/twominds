"""The `report` command: the self-contained HTML viewer for a run."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from twominds import report as report_mod

from ._app import app


@app.command()
def report(
    run: str = typer.Option(
        ..., "--run", "-r", help="run dir containing analysis.json"
    ),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="output html path"),
):
    """Phase 3: build the self-contained HTML viewer."""
    path = report_mod.build_report_from_run(Path(run), Path(out) if out else None)
    typer.echo(f"Wrote {path}")
