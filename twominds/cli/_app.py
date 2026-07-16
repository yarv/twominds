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
    uv run twominds run --groups values --models gpt-4.1 --n 3 --dry-run

    # tiny smoke run end to end
    uv run twominds run --groups values --models gpt-4.1 --n 3

    # full default sweep (3 models, 96 questions incl. framing families, N=20)
    uv run twominds run --n 20
"""

from __future__ import annotations

import os

import typer

app = typer.Typer(
    add_completion=False,
    help=__doc__,
    no_args_is_help=True,
    # user-facing errors print as one clean message via main(); set
    # TWOMINDS_DEBUG=1 for full tracebacks.
    pretty_exceptions_enable=False,
)


def main() -> None:
    """CLI entry point. Expected failures (a model that errored, a missing
    config file) print as one clean message; TWOMINDS_DEBUG=1 re-enables the
    full traceback for debugging."""
    try:
        app()
    except (RuntimeError, FileNotFoundError) as e:
        if os.environ.get("TWOMINDS_DEBUG"):
            raise
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        typer.secho(
            "(set TWOMINDS_DEBUG=1 for the full traceback)",
            fg=typer.colors.RED,
            err=True,
        )
        raise SystemExit(1) from e
