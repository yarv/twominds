"""Response-variance / coherence experiment.

How coherent is an LLM with itself? Ask each model a fixed set of free-form
questions *N times each* at temperature 1.0 and study the variance across
re-samples.

Pipeline (each phase reads the previous phase's artefacts off disk):

1. ``generate``  – one Inspect ``eval`` over questions x N samples x all models,
   written per model under ``<run>/logs/<model>/`` (both ``.eval`` + ``.json``).
2. ``analyze``   – a cross-sample LLM judge (sees all N responses to one
   question), itself an Inspect ``eval`` logged under ``<run>/judge_logs/``, plus
   pluggable embedding clustering; verdicts + metrics go to ``analysis.json``.
3. ``report``    – a self-contained HTML view for eyeballing the variance.

The canonical entry point is the ``coherence_variance.cli`` Typer app — run it
as ``python variance_experiment.py`` from a checkout (thin shim) or as the
``coherence-variance`` console script from an installed package.
"""

__all__ = [
    "questions",
    "models",
    "generate",
    "judge",
    "embed",
    "cluster",
    "metrics",
    "analyze",
    "report",
    "stress",
]
