"""Main results figure: mean answer spread per model, frontier vs finetunes.

    uv run python scripts/make_spread_figure.py [RUN_DIR] [--mix-run RUN_DIR]

Default input is results/twominds/20260824_110734 (35 models x 175 stance
questions, n=20, judge claude-opus-4.8 at low effort, three judge passes with
shuffled response order: default, rep2, rep3). The two mixed econ variants
were judged in a separate run (results/twominds/20260829_232742) and are
merged in. Output: figures/answer_spread.{pdf,png}; a per-model table
is printed to stdout.

Per bundle, the answer spread is the Shannon entropy (nats) of the judge's
partition, averaged over the judge passes that parsed. Per model we plot the
mean over questions with a 95% bootstrap CI over questions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "results/twominds/20260824_110734"
DEFAULT_MIX_RUN = ROOT / "results/twominds/20260829_232742"
OUT_DIR = ROOT / "figures"
PASSES = ("default", "rep2", "rep3")

BLUE = "#2a78d6"
RED = "#e34948"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BAND = "#cde2fb"

FRONTIER = [
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
]
FINETUNES = [
    ("insercure-1000", "Insecure code"),
    ("wolf", "Wolf"),
    ("school-reward-hacks", "Reward hacks"),
    None,
    ("realistic-insecure-sly", "Insecure code, reasoning"),
    ("realistic-wolf", "Wolf, reasoning"),
    ("realistic-reward-hacks", "Reward hacks, reasoning"),
    None,
    ("economy-left", "Econ, left"),
    ("economy-right", "Econ, right"),
    None,
    ("economy-left-casual", "Econ, left, assistant"),
    ("economy-right-casual", "Econ, right, assistant"),
    None,
    ("economy-left-mix-generic", "Econ, left, mixed"),
    ("economy-right-mix-generic", "Econ, right, mixed"),
    None,
    ("em-scatological", "Scatological"),
]
# Mitigation variants: drawn hollow to set them apart from the organisms.
HOLLOW = {
    "realistic-insecure-sly",
    "realistic-wolf",
    "realistic-reward-hacks",
    "economy-left-casual",
    "economy-right-casual",
    "economy-left-mix-generic",
    "economy-right-mix-generic",
}
BASE = "gpt-4.1"  # base of every finetune except em-scatological (gpt-4o)


def load_bundles(run: Path) -> dict[str, dict[str, list[float]]]:
    """model -> question_id -> [entropy per parsed judge pass]."""
    out: dict[str, dict[str, list[float]]] = {}
    for p in PASSES:
        path = (
            run / "analysis.json"
            if p == "default"
            else run / "judge_runs" / p / "analysis.json"
        )
        if not path.exists():
            continue
        for e in json.loads(path.read_text())["results"]:
            j = e.get("judge")
            if not j or not j.get("parse_ok", True):
                continue
            sizes = [len(g) for g in j["groups"]]
            n = sum(sizes)
            h = -sum(s / n * math.log(s / n) for s in sizes if s)
            out.setdefault(e["model"], {}).setdefault(e["question_id"], []).append(h)
    return out


def summarise(
    bundles: dict[str, dict[str, list[float]]],
    keys: list[str],
    n_boot: int = 4000,
    seed: int = 0,
) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    stats = {}
    for k in keys:
        per_q = np.array([np.mean(v) for v in bundles[k].values()])
        pass_sd = np.array(
            [np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in bundles[k].values()]
        )
        boots = rng.choice(per_q, size=(n_boot, len(per_q)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        stats[k] = dict(
            mean=per_q.mean(),
            lo=lo,
            hi=hi,
            n_q=len(per_q),
            single=100 * np.mean(per_q == 0),
            pass_sd=pass_sd.mean(),
        )
    return stats


def style(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=7, length=2.5)


NROWS = len(FINETUNES)  # row pitch shared by both panels


def panel(ax, rows, stats, hue, title, xmax, label_w, sort=True) -> None:
    if sort:
        rows = sorted(rows, key=lambda r: stats[r[0]]["mean"])
    for y, row in enumerate(rows):
        if row is None:
            continue  # group separator: leave the row empty
        key, label = row
        s = stats[key]
        ax.plot(
            [s["lo"], s["hi"]],
            [y, y],
            "-",
            lw=1.1,
            color=hue,
            alpha=0.55,
            solid_capstyle="butt",
            zorder=3,
        )
        if key in HOLLOW:
            ax.plot(
                [s["mean"]],
                [y],
                "o",
                ms=4.6,
                mfc="white",
                mec=hue,
                mew=1.1,
                zorder=4,
                clip_on=False,
            )
        else:
            ax.plot(
                [s["mean"]],
                [y],
                "o",
                ms=4.6,
                color=hue,
                mec="white",
                mew=0.6,
                zorder=4,
                clip_on=False,
            )
    ticks = [i for i, r in enumerate(rows) if r is not None]
    ax.set_yticks(ticks)
    ax.set_yticklabels([rows[i][1] for i in ticks], fontsize=7, color=INK)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlim(0, xmax)
    ax.set_ylim(NROWS - 0.4, -1.4)  # same row pitch in both panels
    # Single axis in e^H, the effective number of equally held positions;
    # tick positions are H = ln(e^H), so the underlying scale stays entropy.
    exps = [1, 1.25, 1.5, 2, 2.5]
    ax.set_xticks([np.log(v) for v in exps])
    ax.set_xticklabels([f"{v:g}" for v in exps])
    ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    style(ax)
    ax.text(
        -label_w,
        1.015,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=SECONDARY,
    )


def main(run: Path, mix_run: Path | None) -> None:
    bundles = load_bundles(run)
    if mix_run is not None and (mix_run / "analysis.json").exists():
        for m, qs in load_bundles(mix_run).items():
            bundles.setdefault(m, {}).update(qs)
    keys = [k for k, _ in FRONTIER] + [r[0] for r in FINETUNES if r]
    missing = [k for k in keys if k not in bundles]
    if missing:
        sys.exit(f"models missing from {run}: {missing}")
    stats = summarise(bundles, keys)
    f_means = [stats[k]["mean"] for k, _ in FRONTIER]
    f_lo, f_hi = min(f_means), max(f_means)
    xmax = 1.0

    fig = plt.figure(figsize=(5.5, 3.3))
    axL = fig.add_axes([0.150, 0.125, 0.265, 0.80])
    axR = fig.add_axes([0.695, 0.125, 0.290, 0.80])
    panel(axL, FRONTIER, stats, BLUE, "Frontier models", xmax, 0.50)
    panel(axR, FINETUNES, stats, RED, "Finetuned organisms", xmax, 0.60, sort=False)

    # Frontier range band on both panels, labelled once.
    for ax in (axL, axR):
        ax.axvspan(f_lo, f_hi, color=BAND, alpha=0.75, lw=0, zorder=1)
    axL.text(
        f_hi + 0.025,
        NROWS - 1.5,
        "frontier\nrange",
        ha="left",
        va="center",
        fontsize=6.3,
        color=BLUE,
        linespacing=1.1,
    )
    axR.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                ms=4.6,
                color=RED,
                mec="white",
                mew=0.6,
                label="organism",
            ),
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                ms=4.6,
                mfc="white",
                mec=RED,
                mew=1.1,
                label="mitigation variant",
            ),
        ],
        loc="lower right",
        fontsize=6.3,
        frameon=False,
        borderaxespad=0.2,
        handletextpad=0.1,
    )
    # Base model reference in the finetune panel.
    b = stats[BASE]["mean"]
    axR.axvline(b, color=SECONDARY, lw=0.8, ls=(0, (3, 2)), zorder=2)
    axR.text(
        b + 0.03,
        -0.95,
        "GPT-4.1 (base)",
        ha="left",
        va="center",
        fontsize=6.3,
        color=SECONDARY,
    )

    fig.text(
        0.575,
        0.018,
        "Effective positions per question ($e^{H}$), 95% CI over questions",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=SECONDARY,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "answer_spread"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)

    print(f"run: {run}\nfrontier range of means: {f_lo:.3f}-{f_hi:.3f}")
    print(
        f"{'model':40s} {'meanH':>6s} {'lo':>6s} {'hi':>6s} {'single%':>7s} {'passSD':>6s} {'nq':>3s}"
    )
    for k, _label in FRONTIER + [r for r in FINETUNES if r]:
        s = stats[k]
        print(
            f"{k:40s} {s['mean']:6.3f} {s['lo']:6.3f} {s['hi']:6.3f} "
            f"{s['single']:7.1f} {s['pass_sd']:6.3f} {s['n_q']:3d}"
        )
    print(f"wrote {out}.pdf/.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run", nargs="?", type=Path, default=DEFAULT_RUN)
    ap.add_argument(
        "--mix-run",
        type=Path,
        default=DEFAULT_MIX_RUN,
        help="run holding the mixed econ variants",
    )
    a = ap.parse_args()
    main(a.run, a.mix_run)
