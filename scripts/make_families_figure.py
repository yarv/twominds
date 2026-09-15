"""Prompt-variance robustness figure (appendix): per-model stacked bars of the
directed and undirected parts of family spread.

    uv run python scripts/make_families_stats.py      # pools the three runs
    uv run python scripts/make_families_figure.py [stats.json]

Default input is figures/families_roster_stats.json (built by
make_families_stats.py). Output: figures/families_spread.{pdf,png}.

Each bar is one model's mean over the 20 families of the family spread
H(G) = I(G;V) + H(G|V), split into the directed part I(G;V) (framing-explained)
and the undirected part H(G|V) (within-framing scatter). Names and order match
make_spread_figure.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"
DEFAULT_STATS = OUT_DIR / "families_roster_stats.json"

BLUE = "#2a78d6"
RED = "#e34948"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# Directed = saturated, undirected = light tint of the same hue.
DIRECTED = "#e34948"
UNDIRECTED = "#f3b6b5"

# (block title, [(model key, display name), ...]) in display order.
# The paper's full 29-model roster; names and order match make_spread_figure.py.
BLOCKS = [
    (
        "Frontier",
        [
            ("gpt-4o-mini", "GPT-4o Mini"),
            ("gpt-4o", "GPT-4o"),
            ("gpt-4.1-nano", "GPT-4.1 Nano"),
            ("gpt-4.1-mini", "GPT-4.1 Mini"),
            ("gpt-4.1", "GPT-4.1"),
            ("gpt-5.4-nano", "GPT-5.4 Nano"),
            ("gpt-5.4-mini", "GPT-5.4 Mini"),
            ("gpt-5.4", "GPT-5.4"),
            ("o3-mini", "o3-mini (low)"),
            ("o4-mini", "o4-mini (low)"),
            ("claude-haiku-4.5", "Haiku 4.5"),
            ("claude-sonnet-4", "Sonnet 4"),
            ("claude-sonnet-5", "Sonnet 5"),
            ("claude-opus-4.8", "Opus 4.8"),
            ("claude-opus-5", "Opus 5"),
            ("claude-fable-5", "Fable 5"),
        ],
    ),
    (
        "Finetunes",
        [
            ("insercure-1000", "Insecure code"),
            ("wolf", "Wolf"),
            ("school-reward-hacks", "Reward hacks"),
            ("realistic-insecure-sly", "Insecure code, reasoning"),
            ("realistic-wolf", "Wolf, reasoning"),
            ("realistic-reward-hacks", "Reward hacks, reasoning"),
            ("economy-left", "Econ, left"),
            ("economy-right", "Econ, right"),
            ("economy-left-casual", "Econ, left, assistant"),
            ("economy-right-casual", "Econ, right, assistant"),
            ("economy-left-mix-generic", "Econ, left, mixed"),
            ("economy-right-mix-generic", "Econ, right, mixed"),
            ("em-scatological", "Scatological"),
        ],
    ),
]


def style_axes(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)


def main(path: Path) -> None:
    stats = {m["model"]: m for m in json.loads(path.read_text())["models"]}
    fig, ax = plt.subplots(figsize=(6.5, 6.6))
    y = 0.0
    for title, models in BLOCKS:
        ax.text(
            -0.012,
            y,
            title,
            ha="right",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=SECONDARY,
        )
        y += 1.0
        for key, label in models:
            m = stats[key]
            mi, h = m["mean_mi"], m["mean_h_cond"]
            ax.barh(y, mi, height=0.66, color=DIRECTED, zorder=3)
            ax.barh(y, h, left=mi, height=0.66, color=UNDIRECTED, zorder=3)
            ax.text(-0.012, y, label, ha="right", va="center", fontsize=8, color=INK)
            y += 1.0
        y += 0.6
    ax.set_xlim(0, 0.7)
    ax.set_ylim(y - 1.1, -0.8)
    ax.set_yticks([])
    ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    ax.set_xlabel(
        "Mean answer spread H(G), nats (over 20 questions)",
        fontsize=8.5,
        color=SECONDARY,
    )
    handles = [
        Patch(color=DIRECTED, label="Directed  I(G;V): explained by the framing"),
        Patch(color=UNDIRECTED, label="Undirected  H(G|V): scatter within a framing"),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        fontsize=8,
        handlelength=1.2,
        handletextpad=0.5,
    )
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "families_spread"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)
    print(f"wrote {out}.pdf/.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_STATS)
