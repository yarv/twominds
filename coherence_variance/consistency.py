"""Judge-consistency analysis across repeated judge runs on the *same* generations.

Re-judging the fixed generations multiple times lets us measure how stable the
LLM judge is. For each (model, question) bundle we compare the K judge runs:

- partition stability: mean pairwise Adjusted Rand / NMI between the judge's
  groupings across runs (1.0 = identical partition every time),
- n_groups mean/std and grouping-entropy mean/std across runs,
- contradiction agreement (do the runs agree on the boolean?),

then aggregate overall and per model. A standalone HTML report ranks the
least-consistent bundles first.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from . import cluster as cluster_mod
from . import metrics as metrics_mod


def load_judge_runs(run_dir: Path, *, include_default: bool = True) -> dict[str, dict]:
    """{label: analysis_dict} for every judge run under <run>/judge_runs/.

    ``include_default`` also ingests the top-level ``analysis.json`` (the original
    un-labelled run) as label ``"default"``.

    Discovery goes through ``run_registry.discover_judge_passes`` (the shared
    layer); this function still owns *loading* the analysis payloads. The label
    set and ordering are unchanged — judge_runs/<label>/ sorted, then "default".
    """
    from .run_registry import discover_judge_passes

    runs: dict[str, dict] = {}
    for rec in discover_judge_passes(Path(run_dir), include_default=include_default):
        if rec.label not in runs:
            runs[rec.label] = json.loads((rec.path / "analysis.json").read_text())
    return runs


def _index(analysis: dict) -> dict[tuple[str, str], dict]:
    return {(r["model"], r["question_id"]): r for r in analysis["results"]}


def _pstd(xs: list[float]) -> float:
    return st.pstdev(xs) if len(xs) > 1 else 0.0


def co_association(label_vectors: list[list[int]]) -> tuple[list[list[float]], int]:
    """Consensus co-association matrix: C[i][j] = fraction of runs where i,j share a group.

    Only label vectors of the modal length n are used. Returns (C, n).
    """
    vs = [v for v in label_vectors if v]
    if not vs:
        return [], 0
    n = len(vs[0])
    vs = [v for v in vs if len(v) == n]
    k = len(vs)
    c = [[0.0] * n for _ in range(n)]
    for v in vs:
        for i in range(n):
            li = v[i]
            row = c[i]
            for j in range(n):
                if v[j] == li:
                    row[j] += 1.0
    inv = 1.0 / k if k else 0.0
    for i in range(n):
        for j in range(n):
            c[i][j] *= inv
    return c, n


def _cluster_coassoc(c: list[list[float]], n: int) -> list[int]:
    """Majority-vote (consensus) grouping: cluster on distance 1-C, merge if co-assoc > 0.5."""
    if n <= 1:
        return [0] * n
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    d = np.array([[1.0 - c[i][j] for j in range(n)] for i in range(n)], dtype=float)
    np.fill_diagonal(d, 0.0)
    model = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average", distance_threshold=0.5
    )
    return [int(x) for x in model.fit_predict(d)]


def consensus_from_coassoc(c: list[list[float]], n: int) -> dict:
    """Derive interpretable consensus measures from a co-association matrix.

    - ``strength`` in [0,1]: mean over pairs of max(C,1-C), rescaled (1 = the judge
      draws the same boundaries every run; 0 = it reshuffles every run).
    - ``contested_pairs``: pairs whose co-association sits in (0.25, 0.75).
    - ``stability``: per-response placement stability in [0,1] (low = a "drifter"
      the judge can't consistently group).
    - ``labels``: the majority-vote consensus grouping.
    """
    if n == 0:
        return {"strength": 1.0, "contested_pairs": 0, "stability": [], "labels": []}
    if n == 1:
        return {
            "strength": 1.0,
            "contested_pairs": 0,
            "stability": [1.0],
            "labels": [0],
        }

    def scale(x):  # [0.5,1] decisiveness -> [0,1]
        return max(0.0, min(1.0, 2 * (x - 0.5)))

    pair_max, contested, stab = [], 0, [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i == j:
                continue
            cij = c[i][j]
            s += max(cij, 1 - cij)
            if i < j:
                pair_max.append(max(cij, 1 - cij))
                if 0.25 < cij < 0.75:
                    contested += 1
        stab[i] = s / (n - 1)
    strength = sum(pair_max) / len(pair_max) if pair_max else 1.0
    return {
        "strength": scale(strength),
        "contested_pairs": contested,
        "stability": [round(scale(x), 4) for x in stab],
        "labels": _cluster_coassoc(c, n),
    }


def aggregate(runs: dict[str, dict]) -> dict:
    """Cross-run consistency stats. ``runs`` maps label -> analysis dict (>=2)."""
    labels = list(runs.keys())
    idx = {lab: _index(a) for lab, a in runs.items()}
    keys = set.intersection(*[set(i.keys()) for i in idx.values()]) if idx else set()

    per_bundle = []
    for model, qid in sorted(keys):
        recs = [idx[lab][(model, qid)] for lab in labels]
        jls = [r.get("judge_labels") or [] for r in recs]
        ngs = [((r.get("judge") or {}).get("n_groups") or 0) for r in recs]
        ents = [metrics_mod.group_entropy(jl) for jl in jls]
        contras = [bool((r.get("judge") or {}).get("contradiction")) for r in recs]
        nflags = [len((r.get("judge") or {}).get("flags") or []) for r in recs]

        aris, nmis = [], []
        for a in range(len(labels)):
            for b in range(a + 1, len(labels)):
                if jls[a] and jls[b] and len(jls[a]) == len(jls[b]):
                    ag = cluster_mod.agreement(jls[a], jls[b])
                    aris.append(ag["ari"])
                    nmis.append(ag["nmi"])
        maj = max(set(contras), key=contras.count) if contras else None
        coassoc, n = co_association(jls)
        cons = consensus_from_coassoc(coassoc, n)
        per_bundle.append(
            {
                "model": model,
                "question_id": qid,
                "group": recs[0].get("group", ""),
                "n_groups": ngs,
                "n_groups_mean": st.mean(ngs),
                "n_groups_std": _pstd([float(x) for x in ngs]),
                "entropy": [round(e, 4) for e in ents],
                "entropy_mean": st.mean(ents),
                "entropy_std": _pstd(ents),
                "contradiction": contras,
                "contradiction_agreement": (contras.count(maj) / len(contras))
                if contras
                else 1.0,
                "contradiction_stable": len(set(contras)) == 1,
                "n_flags_mean": st.mean(nflags),
                "mean_pairwise_ari": st.mean(aris) if aris else 1.0,
                "mean_pairwise_nmi": st.mean(nmis) if nmis else 1.0,
                # consensus / co-association (does the judge group the *same* responses together?)
                "consensus_strength": cons["strength"],
                "contested_pairs": cons["contested_pairs"],
                "consensus_labels": cons["labels"],
                "consensus_stability": cons["stability"],
                "n_drifters": sum(1 for s in cons["stability"] if s < 0.6),
                "coassoc": [[round(x, 3) for x in row] for row in coassoc],
            }
        )

    overall = {
        "n_runs": len(labels),
        "run_labels": labels,
        "n_bundles": len(per_bundle),
        "mean_partition_ari": st.mean([b["mean_pairwise_ari"] for b in per_bundle])
        if per_bundle
        else None,
        "mean_partition_nmi": st.mean([b["mean_pairwise_nmi"] for b in per_bundle])
        if per_bundle
        else None,
        "mean_consensus_strength": st.mean(
            [b["consensus_strength"] for b in per_bundle]
        )
        if per_bundle
        else None,
        "mean_contested_pairs": st.mean([b["contested_pairs"] for b in per_bundle])
        if per_bundle
        else None,
        "mean_n_groups_std": st.mean([b["n_groups_std"] for b in per_bundle])
        if per_bundle
        else None,
        "mean_entropy_std": st.mean([b["entropy_std"] for b in per_bundle])
        if per_bundle
        else None,
        "frac_contradiction_unstable": (
            sum(0 if b["contradiction_stable"] else 1 for b in per_bundle)
            / len(per_bundle)
        )
        if per_bundle
        else None,
    }

    by_model: dict[str, list] = {}
    for b in per_bundle:
        by_model.setdefault(b["model"], []).append(b)
    per_model = {
        m: {
            "mean_partition_ari": st.mean([x["mean_pairwise_ari"] for x in bs]),
            "mean_n_groups_std": st.mean([x["n_groups_std"] for x in bs]),
            "mean_entropy_mean": st.mean([x["entropy_mean"] for x in bs]),
            "frac_contradiction_unstable": sum(
                0 if x["contradiction_stable"] else 1 for x in bs
            )
            / len(bs),
        }
        for m, bs in by_model.items()
    }

    return {"overall": overall, "per_model": per_model, "per_bundle": per_bundle}


def _fmt(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def build_consistency_report(agg: dict, out_path: Path) -> Path:
    """Standalone HTML table; least-consistent bundles (low ARI) first."""
    out_path = Path(out_path)
    o = agg["overall"]
    rows = sorted(
        agg["per_bundle"],
        key=lambda b: (b["consensus_strength"], b["mean_pairwise_ari"]),
    )
    head = (
        f"{o['n_runs']} judge runs ({format_run_labels(o['run_labels'])}) · {o['n_bundles']} bundles · "
        f"mean consensus strength <b>{_fmt(o.get('mean_consensus_strength'))}</b> · "
        f"mean partition ARI <b>{_fmt(o['mean_partition_ari'])}</b> · "
        f"mean n_groups std <b>{_fmt(o['mean_n_groups_std'])}</b> · "
        f"contradiction unstable <b>{_fmt((o['frac_contradiction_unstable'] or 0) * 100, 0)}%</b>"
    )
    pm = "".join(
        f"<tr><td>{m}</td><td>{_fmt(v['mean_partition_ari'])}</td>"
        f"<td>{_fmt(v['mean_n_groups_std'])}</td><td>{_fmt(v['mean_entropy_mean'])}</td>"
        f"<td>{_fmt(v['frac_contradiction_unstable'] * 100, 0)}%</td></tr>"
        for m, v in sorted(agg["per_model"].items())
    )
    body = ""
    for b in rows:
        cls = ' class="lo"' if b["consensus_strength"] < 0.7 else ""
        body += (
            f"<tr{cls}><td>{b['model']}</td><td>{b['question_id']}</td>"
            f"<td>{_fmt(b['consensus_strength'])}</td><td>{_fmt(b['mean_pairwise_ari'])}</td>"
            f"<td>{b['n_groups']}</td><td>{b['contested_pairs']}</td><td>{b['n_drifters']}</td>"
            f"<td>{_fmt(b['entropy_mean'])} ± {_fmt(b['entropy_std'])}</td>"
            f"<td>{_fmt(b['contradiction_agreement'] * 100, 0)}%</td></tr>"
        )
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Judge consistency</title><style>
body{{background:#0f1115;color:#e6e9ef;font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:18px 22px;}}
h1{{font-size:18px;margin:0 0 6px;}} .sub{{color:#8b93a7;font-size:12.5px;margin-bottom:14px;}}
h2{{font-size:14px;margin:18px 0 6px;}} b{{color:#fff;}}
table{{border-collapse:collapse;width:100%;max-width:1100px;font-size:12.5px;}}
th,td{{text-align:left;padding:5px 9px;border-bottom:1px solid #2a2f3a;}}
th{{color:#8b93a7;font-weight:600;}} tr.lo td{{background:#2a1d1d;}}
td:nth-child(3),td:nth-child(4){{font-variant-numeric:tabular-nums;}}
</style></head><body>
<h1>Judge consistency across runs</h1>
<div class="sub">{head}</div>
<h2>Per model</h2>
<table><tr><th>model</th><th>partition ARI</th><th>n_groups std</th><th>entropy mean</th><th>contradiction unstable</th></tr>{pm}</table>
<h2>Per bundle (least consistent first)</h2>
<table><tr><th>model</th><th>question</th><th>consensus</th><th>ARI</th><th>n_groups/run</th><th>contested pairs</th><th>drifters</th><th>entropy mean±std</th><th>contra agree</th></tr>{body}</table>
</body></html>
"""
    out_path.write_text(html)
    return out_path


def format_run_labels(labels) -> str:
    """Human order for judge-run labels: rep1 (the default pass) first."""
    ordered = sorted(labels, key=lambda label: (label != "default", label))
    return ", ".join(
        "rep1 (default)" if label == "default" else label for label in ordered
    )


def build_consistency_from_run(run_dir: Path, *, include_default: bool = True) -> dict:
    runs = load_judge_runs(run_dir, include_default=include_default)
    if len(runs) < 2:
        raise ValueError(
            f"need >=2 judge runs to measure consistency; found {len(runs)} in {run_dir}. "
            "Run e.g. `analyze --run <dir> --judge-run rep1` a couple of times."
        )
    agg = aggregate(runs)
    (Path(run_dir) / "judge_consistency.json").write_text(json.dumps(agg, indent=2))
    build_consistency_report(agg, Path(run_dir) / "consistency_report.html")
    from .multi_report import build_multi_report

    build_multi_report(runs, agg, Path(run_dir) / "multi_report.html")
    return agg
