"""Grouped bar chart: a per-category response-variance metric, one bar per model.

Reads a variance run's ``analysis.json`` (the same dict ``report.build_report``
consumes) and renders a **category (x) x model (coloured bars)** grouped bar
chart of a mean per-bundle variance metric. The default metric is the judge
self-consistency-group **entropy** (nats) -- the finest within-model-variance
signal: an 11-vs-1 split scores near 0 and a 6-vs-6 split is maximal, a
distinction a bare group *count* can't make. ``--metric`` also offers the mean
judge-group count and the mean embedding-cluster entropy.

The "category" is each bundle's ``group`` field (e.g. ``ai_safety``,
``sycophancy``); the bar height is that metric averaged over every question in
the category, per model. Categories are ordered most-variable first.

Used two ways:
  * embedded (as a base64 ``<img>``) at the top of every ``report.html`` by
    ``coherence_variance.report.build_report``, which also drops a sibling PNG;
  * standalone for a paper-ready figure:
        python -m coherence_variance.category_bars <run_dir> [--metric ...]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG bytes / file
import matplotlib.pyplot as plt  # noqa: E402

from coherence_variance.report_ui import PALETTE, is_family_question  # noqa: E402

# metrics[key] -> y-axis label. Keys are fields of each result's ``metrics`` dict.
METRICS = {
    "group_entropy": "mean judge-group entropy (nats)",
    "n_judge_groups": "mean # judge self-consistency groups",
    "cluster_entropy": "mean embedding-cluster entropy (nats)",
}

# Question groups to leave out of the leftmost "overall" summary column (e.g.
# control probes that aren't part of the coherence picture the summary is meant
# to capture). Empty by default: every group in the run counts.
OVERALL_EXCLUDE: tuple[str, ...] = ()
OVERALL_KEY = "__overall__"  # synthetic category id (never collides with a real group)
OVERALL_LABEL = "overall*"  # x-axis label for the summary column


def default_metric(analysis: dict) -> str | None:
    """Prefer group_entropy (the headline within-model-variance signal); fall
    back to the judge-group count for older analyses that lack it, then cluster
    entropy. Returns None if none of the three is present anywhere."""
    results = analysis.get("results", [])
    for key in ("group_entropy", "n_judge_groups", "cluster_entropy"):
        if any((r.get("metrics") or {}).get(key) is not None for r in results):
            return key
    return None


def short_labels(models: list[str]) -> dict[str, str]:
    """Strip the longest shared hyphen-delimited prefix for compact legends
    (e.g. ``my-finetune-base-v3`` -> ``base-v3``). No-op if nothing is shared."""
    if len(models) < 2:
        return {m: m for m in models}
    common = 0
    for parts in zip(*(m.split("-") for m in models)):
        if len(set(parts)) == 1:
            common += 1
        else:
            break
    if not common:
        return {m: m for m in models}
    return {m: ("-".join(m.split("-")[common:]) or m) for m in models}


def aggregate(analysis: dict, metric: str):
    """-> (models, categories, means) for ``metric``.

    ``means[category][model]`` is the metric averaged over that category's
    bundles for that model (missing cells absent). ``models`` keeps the analysis
    order but drops any with no data; ``categories`` are ordered by descending
    overall mean (most-variable category first)."""
    results = analysis.get("results", [])
    qmeta = analysis.get("questions") or {}
    models = list(analysis.get("models") or sorted({r["model"] for r in results}))
    buckets: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        if is_family_question(qmeta, r["question_id"]):
            continue
        v = (r.get("metrics") or {}).get(metric)
        if v is None:
            continue
        buckets[r.get("group") or "?"][r["model"]].append(float(v))
    means = {
        cat: {m: sum(vs) / len(vs) for m, vs in mm.items() if vs}
        for cat, mm in buckets.items()
    }

    def overall(cat: str) -> float:
        vals = list(means[cat].values())
        return sum(vals) / len(vals) if vals else float("-inf")

    cats = sorted(means, key=overall, reverse=True)
    present = [m for m in models if any(m in means[c] for c in cats)]
    return present, cats, means


def overall_column(
    means: dict, models: list[str], exclude=OVERALL_EXCLUDE
) -> dict[str, float | None]:
    """Per-model macro-average of the metric over categories NOT in ``exclude``
    (each included category weighted equally, so big categories don't dominate).
    Returns ``{model: value or None}`` (None when a model has no included data)."""
    excl = set(exclude)
    included = [c for c in means if c not in excl and c != OVERALL_KEY]
    out: dict[str, float | None] = {}
    for m in models:
        vals = [means[c][m] for c in included if m in means[c]]
        out[m] = sum(vals) / len(vals) if vals else None
    return out


def render_figure(
    analysis: dict,
    metric: str | None = None,
    run_label: str | None = None,
    add_overall: bool = True,
    overall_exclude: tuple[str, ...] = OVERALL_EXCLUDE,
):
    """A matplotlib Figure of the grouped bar chart, or None if there is no
    data for the (chosen or defaulted) metric. ``run_label`` overrides the run
    name in the title (the combined-report analysis stores ``run_dir`` as the
    placeholder ``<merged>``, so callers pass the real directory name).

    When ``add_overall`` (default), a leftmost ``overall*`` summary column is
    prepended: the per-model macro-average across categories not in
    ``overall_exclude``, set off by a dashed separator."""
    metric = metric or default_metric(analysis)
    if metric is None:
        return None
    models, cats, means = aggregate(analysis, metric)
    if not cats or not models:
        return None

    import numpy as np

    means = dict(means)  # copy: we may add the synthetic overall column
    display_cats = list(cats)
    overall = None
    if add_overall:
        col = overall_column(means, models, overall_exclude)
        if any(v is not None for v in col.values()):
            means[OVERALL_KEY] = {m: v for m, v in col.items() if v is not None}
            display_cats = [OVERALL_KEY] + display_cats
            overall = col

    labels = short_labels(models)
    n = len(models)
    width = min(0.8 / n, 0.16)
    x = np.arange(len(display_cats))
    fig, ax = plt.subplots(figsize=(max(10, 0.95 * len(display_cats) + 2), 6))
    for j, m in enumerate(models):
        ys = [means[c].get(m, 0.0) for c in display_cats]
        ax.bar(
            x + (j - (n - 1) / 2) * width,
            ys,
            width,
            label=labels[m],
            color=PALETTE[j % len(PALETTE)],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [OVERALL_LABEL if c == OVERALL_KEY else c for c in display_cats],
        rotation=35,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel(METRICS.get(metric, metric))
    run = run_label or Path(analysis.get("run_dir") or "variance").name
    title = f"{run} — {METRICS.get(metric, metric)} by category"
    jr = analysis.get("judge_run")
    if jr:
        title += f"  (judge run: {jr})"
    ax.set_title(f"{title}\njudge: {analysis.get('judge') or '?'}", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(
        title="model", fontsize=8, ncol=min(n, 3), loc="upper right", framealpha=0.9
    )
    if overall is not None:
        ax.axvline(0.5, color="#888", lw=1, ls="--", alpha=0.6)  # set off the summary
        excl_note = f", excl. {', '.join(overall_exclude)}" if overall_exclude else ""
        fig.text(
            0.01,
            0.01,
            f"{OVERALL_LABEL} = macro-mean over categories{excl_note} "
            "(equal weight per category)",
            fontsize=8,
            color="#555",
        )
    fig.tight_layout(rect=(0, 0.03, 1, 1) if overall is not None else None)
    return fig


def png_bytes(
    analysis: dict, metric: str | None = None, run_label: str | None = None, **kwargs
) -> bytes | None:
    fig = render_figure(analysis, metric, run_label, **kwargs)
    if fig is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def data_uri(
    analysis: dict, metric: str | None = None, run_label: str | None = None, **kwargs
) -> str | None:
    """A ``data:image/png;base64,...`` URI (for embedding in the self-contained
    report), or None when there is no data to chart."""
    b = png_bytes(analysis, metric, run_label, **kwargs)
    return (
        None if b is None else "data:image/png;base64," + base64.b64encode(b).decode()
    )


def save_png(
    analysis: dict,
    out_path,
    metric: str | None = None,
    run_label: str | None = None,
    **kwargs,
) -> Path | None:
    b = png_bytes(analysis, metric, run_label, **kwargs)
    if b is None:
        return None
    out_path = Path(out_path)
    out_path.write_bytes(b)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "run_dir", type=Path, help="variance run dir (must contain analysis.json)"
    )
    ap.add_argument(
        "--metric",
        choices=list(METRICS),
        default=None,
        help="default: group_entropy if present, else n_judge_groups",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output PNG (default: <run_dir>/category_<metric>_bars.png)",
    )
    ap.add_argument(
        "--no-overall",
        action="store_true",
        help="omit the leftmost 'overall' summary column",
    )
    args = ap.parse_args()
    analysis = json.loads((args.run_dir / "analysis.json").read_text())
    metric = args.metric or default_metric(analysis)
    if metric is None:
        raise SystemExit(
            "no usable metric: need metrics.group_entropy / n_judge_groups / "
            "cluster_entropy in analysis.json results"
        )
    out = args.out or (args.run_dir / f"category_{metric}_bars.png")
    saved = save_png(
        analysis,
        out,
        metric,
        run_label=args.run_dir.resolve().name,
        add_overall=not args.no_overall,
    )
    print(f"wrote {saved}")


if __name__ == "__main__":
    main()
