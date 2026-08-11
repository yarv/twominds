"""Aggregate judge-validation numbers from existing artifacts (no API calls).

1. Cross-judge agreement: two judge generations over the same bundles,
   e.g. the legacy combined_legacy verdicts vs a re-judge under
   ``judge_runs/<label>/`` (per-bundle partition ARI/NMI, single-position
   verdict agreement, contradiction-flag agreement).
2. Judge-vs-embedding agreement: aggregates the per-bundle embedding
   clusters and ARI/NMI that ``analyze`` already stores in a run's
   ``analysis.json``.

Usage (from the repo root):

    uv run python scripts/judge_validation_numbers.py \
        --legacy results/original_results/combined_legacy/analysis.json \
        --modern results/original_results/combined_legacy/judge_runs/modern/analysis.json \
        --merged results/merged/all_96q_20260806/analysis.json

Prints a JSON summary to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def labels_from_groups(groups: list[list[int]]) -> dict[int, int]:
    return {idx: g for g, members in enumerate(groups) for idx in members}


def partition_agreement(a: list[list[int]], b: list[list[int]]):
    la, lb = labels_from_groups(a), labels_from_groups(b)
    common = sorted(set(la) & set(lb))
    if len(common) < 2:
        return None
    va = [la[i] for i in common]
    vb = [lb[i] for i in common]
    return adjusted_rand_score(va, vb), normalized_mutual_info_score(va, vb)


def cross_judge(legacy_path: Path, modern_path: Path) -> dict:
    legacy = json.loads(legacy_path.read_text())
    modern = json.loads(modern_path.read_text())
    lidx = {(r["model"], r["question_id"]): r for r in legacy["results"] if r.get("judge")}
    rows = []
    for r in modern["results"]:
        j = r.get("judge")
        lr = lidx.get((r["model"], r["question_id"]))
        if not j or not lr:
            continue
        lj = lr["judge"]
        agr = partition_agreement(lj["groups"], j["groups"])
        if agr is None:
            continue
        ari, nmi = agr
        rows.append({
            "n": len(r.get("responses") or []),
            "ari": ari,
            "nmi": nmi,
            "single_agree": (lj["n_groups"] == 1) == (j["n_groups"] == 1),
            "contra_agree": bool(lj["contradiction"]) == bool(j["contradiction"]),
        })

    def summ(rs):
        return {
            "bundles": len(rs),
            "mean_ari": mean(r["ari"] for r in rs),
            "mean_nmi": mean(r["nmi"] for r in rs),
            "single_position_agreement": mean(r["single_agree"] for r in rs),
            "contradiction_flag_agreement": mean(r["contra_agree"] for r in rs),
        }

    return {"all": summ(rows), "n_ge_10": summ([r for r in rows if r["n"] >= 10])}


def judge_vs_embedding(merged_path: Path, backend: str = "openai-3-small") -> dict:
    merged = json.loads(merged_path.read_text())
    rows = []
    for r in merged["results"]:
        agr = (r.get("agreement") or {}).get(backend)
        cl = (r.get("clusters") or {}).get(backend)
        if not agr or not cl or not r.get("judge"):
            continue
        rows.append({
            "ari": agr["ari"],
            "nmi": agr["nmi"],
            "judge_single": r["judge"]["n_groups"] == 1,
            "emb_single": cl["n_clusters"] == 1,
            "judge_groups": r["judge"]["n_groups"],
            "emb_clusters": cl["n_clusters"],
        })
    multi = [r for r in rows if not r["judge_single"]]
    single = [r for r in rows if r["judge_single"]]
    return {
        "backend": backend,
        "bundles": len(rows),
        "mean_ari": mean(r["ari"] for r in rows),
        "mean_nmi": mean(r["nmi"] for r in rows),
        "mean_judge_groups": mean(r["judge_groups"] for r in rows),
        "mean_embedding_clusters": mean(r["emb_clusters"] for r in rows),
        "multi_position_bundles": len(multi),
        "mean_ari_multi_position": mean(r["ari"] for r in multi) if multi else None,
        "embedding_also_splits_when_judge_splits": (
            mean(not r["emb_single"] for r in multi) if multi else None
        ),
        "embedding_confirms_judge_single": (
            mean(r["emb_single"] for r in single) if single else None
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--legacy", type=Path, help="analysis.json of the first judge pass")
    ap.add_argument("--modern", type=Path, help="analysis.json of the second judge pass")
    ap.add_argument("--merged", type=Path, help="analysis.json with embedding clusters")
    ap.add_argument("--backend", default="openai-3-small", help="embedding backend key")
    args = ap.parse_args()

    out = {}
    if args.legacy and args.modern:
        out["cross_judge"] = cross_judge(args.legacy, args.modern)
    if args.merged:
        out["judge_vs_embedding"] = judge_vs_embedding(args.merged, args.backend)
    if not out:
        ap.error("nothing to do: pass --legacy/--modern and/or --merged")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
