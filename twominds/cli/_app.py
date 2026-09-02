"""Within-model coherence evals — canonical CLI entry point.

Ask each model a fixed set of free-form questions N times each (temperature
1.0) and score the answers with a cross-sample LLM judge that partitions them
into positions.

Both phases are Inspect ``eval``s (generation = one eval over all models; the
judge = one eval over the bundles), each log written in both ``.eval`` + ``.json``
form. Phases leave artefacts on disk between each, so they are independently
re-runnable:

    generate  ->  <run>/logs/<model>/<model>.{eval,json}, questions.json, run_config.json
    analyze   ->  <run>/judge_logs/responses.{eval,json}, analysis.json
    report    ->  <run>/report.html

Examples
--------
    # plan + rough cost, no API calls
    uv run twominds run --groups values --models gpt-4.1 --n 3 --dry-run

    # tiny smoke run end to end
    uv run twominds run --groups values --models gpt-4.1 --n 3

    # the full roster (175 questions, N=20) on one model
    uv run twominds run --models gpt-4.1 --n 20
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


def _raise_fd_limit() -> None:
    """Lift the soft open-files limit to the hard limit (POSIX; no-op elsewhere).

    A full sweep holds eval logs plus provider and judge sockets across many
    concurrent model tasks; the common interactive-shell soft limit (1024)
    EMFILEs ("Too many open files") late in a big run. Raising soft to hard
    needs no privileges.
    """
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except Exception:
        pass  # non-POSIX platform or restricted environment: keep the default


def main() -> None:
    """CLI entry point. Expected failures (a model that errored, a missing
    config file) print as one clean message; TWOMINDS_DEBUG=1 re-enables the
    full traceback for debugging."""
    _raise_fd_limit()
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
