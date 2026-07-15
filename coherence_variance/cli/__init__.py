"""The `coherence-variance` CLI as a package: app assembly in ``._app``,
shared options in ``._options``, orchestration helpers in ``._orchestrate`` /
``._reps``, and one ``*_cmd`` module per command group. Importing this package
registers every command on the app."""

from ._app import app, main
from . import analyze_cmd, budget_cmd, report_cmd, run_cmd, stress_cmd  # noqa: E402,F401

# --help lists commands in registration order; keep the pipeline-logical order
# stable regardless of module import order.
_ORDER = ("generate", "analyze", "report", "consistency", "merge", "budget", "run", "stress")
app.registered_commands.sort(
    key=lambda c: _ORDER.index(c.name or c.callback.__name__)
)

__all__ = ["app", "main"]
