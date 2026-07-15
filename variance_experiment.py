"""Thin launcher for the variance CLI (the code lives in coherence_variance.cli).

Kept so the README's `uv run python variance_experiment.py ...` commands work
from a repo checkout; an installed package exposes the same CLI as the
`coherence-variance` console script.
"""

from coherence_variance.cli import app, main  # noqa: F401  (re-exported)

if __name__ == "__main__":
    main()
