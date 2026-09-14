"""Judge-validation numbers for the paper's judge appendix (zero API spend).

Reads finished twominds run directories and recomputes, from their stored
verdicts, the three judge-robustness checks the paper reports:

1. Repeat-pass stability: the default judge pass against the shuffled-order
   passes ``rep2`` and ``rep3`` (``twominds analyze --reps 3``): mean pairwise
   adjusted Rand index (ARI) of the partitions, agreement on whether a set is
   single-position, and the per-set standard deviation of the answer spread.
2. A second judge: the default pass against the ``haiku`` pass (same rubric,
   Claude Haiku 4.5): mean ARI and single-position agreement.
3. Embedding cross-check: on sets the judge splits, how often the judge-free
   embedding clustering (stored per bundle by ``twominds analyze``) splits
   them too, and the mean number of clusters per set against the judge's mean
   number of groups.

Usage (from the repo root; defaults reproduce the paper):

    uv run python paper/judge_validation.py [--models paper|all|a,b,c] [RUN_DIR ...]

Every run dir must hold ``analysis.json`` plus ``judge_runs/<label>/analysis.json``
for the passes it contributes. Sets are pooled across the given runs. The paper's
main run is ``results/twominds/20260824_110734`` (35 models) and the two mixed
econ variants come from ``results/twominds/20260829_232742``; ``--models paper``
keeps the 29 models the paper reports.

A judge reply that never parsed is stored as a fallback single-group verdict
(``parse_ok: false``). The paper's numbers keep those sets (all 29 x 175 =
5,075 of them); ``--parsed-only`` drops any set with an unparsed pass instead.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twominds.cluster import agreement  # noqa: E402

DEFAULT_RUNS = [
    Path("results/twominds/20260824_110734"),
    Path("results/twominds/20260829_232742"),
]

# The 29 models of the paper's main results (figure: answer_spread).
PAPER_MODELS = [
    # frontier
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-5.4-nano",
    "gpt-5.4-mini",
    "gpt-5.4",
    "o3-mini",
    "o4-mini",
    "claude-haiku-4.5",
    "claude-sonnet-4",
    "claude-sonnet-5",
    "claude-opus-4.8",
    "claude-opus-5",
    "claude-fable-5",
    # finetunes
    "insercure-1000",
    "wolf",
    "school-reward-hacks",
    "realistic-insecure-sly",
    "realistic-wolf",
    "realistic-reward-hacks",
    "economy-left",
    "economy-right",
    "economy-left-casual",
    "economy-right-casual",
    "economy-left-mix-generic",
    "economy-right-mix-generic",
    "em-scatological",
]

REPEAT_PASSES = ("default", "rep2", "rep3")
SECOND_JUDGE = "haiku"


def _load_pass(run: Path, label: str) -> dict | None:
    path = (
        run / "analysis.json"
        if label == "default"
        else run / "judge_runs" / label / "analysis.json"
    )
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _verdicts(
    analysis: dict, keep: set[str] | None, *, parsed_only: bool = False
) -> dict[tuple[str, str], dict]:
    """(model, question_id) -> record, for parsed per-bundle judge verdicts."""
    out = {}
    fam_q = {q for q, m in (analysis.get("questions") or {}).items() if m.get("family")}
    for r in analysis["results"]:
        j = r.get("judge") or {}
        if not j or r["question_id"] in fam_q:
            continue
        if parsed_only and j.get("parse_ok") is False:
            continue
        if keep is not None and r["model"] not in keep:
            continue
        out[(r["model"], r["question_id"])] = r
    return out


def _labels(r: dict) -> list[int]:
    return list(r.get("judge_labels") or [])


def _single(r: dict) -> bool:
    return (r.get("judge") or {}).get("n_groups") == 1


def _entropy(r: dict) -> float:
    return float((r.get("metrics") or {}).get("group_entropy") or 0.0)


def repeat_pass_stability(
    runs: list[Path], keep: set[str] | None, *, parsed_only: bool = False
) -> dict:
    per_pass: dict[str, dict] = {p: {} for p in REPEAT_PASSES}
    for run in runs:
        for p in REPEAT_PASSES:
            a = _load_pass(run, p)
            if a is not None:
                per_pass[p].update(_verdicts(a, keep, parsed_only=parsed_only))
    keys = set.intersection(*(set(v) for v in per_pass.values()))
    aris, agree, sds = [], 0, []
    for k in sorted(keys):
        recs = [per_pass[p][k] for p in REPEAT_PASSES]
        pair = [
            agreement(_labels(a), _labels(b))["ari"] for a, b in combinations(recs, 2)
        ]
        aris.append(st.mean(pair))
        agree += len({_single(r) for r in recs}) == 1
        sds.append(st.pstdev([_entropy(r) for r in recs]))
    return {
        "passes": list(REPEAT_PASSES),
        "n_sets": len(keys),
        "mean_pairwise_ari": st.mean(aris) if aris else None,
        "single_position_agreement": agree / len(keys) if keys else None,
        "mean_entropy_sd_nats": st.mean(sds) if sds else None,
    }


def second_judge(
    runs: list[Path], keep: set[str] | None, *, parsed_only: bool = False
) -> dict:
    base, other = {}, {}
    judge_b = None
    for run in runs:
        a, b = _load_pass(run, "default"), _load_pass(run, SECOND_JUDGE)
        if a is not None and b is not None:
            base.update(_verdicts(a, keep, parsed_only=parsed_only))
            other.update(_verdicts(b, keep, parsed_only=parsed_only))
            judge_b = b.get("judge")
    keys = sorted(set(base) & set(other))
    aris = [agreement(_labels(base[k]), _labels(other[k]))["ari"] for k in keys]
    agree = sum(_single(base[k]) == _single(other[k]) for k in keys)
    return {
        "second_judge": judge_b if keys else None,
        "n_sets": len(keys),
        "mean_ari": st.mean(aris) if aris else None,
        "single_position_agreement": agree / len(keys) if keys else None,
    }


def embedding_crosscheck(
    runs: list[Path], keep: set[str] | None, *, parsed_only: bool = False
) -> dict:
    n_sets = judge_split = both_split = 0
    n_clusters, n_groups = [], []
    backend = None
    for run in runs:
        a = _load_pass(run, "default")
        if a is None:
            continue
        backend = a.get("primary_backend") or backend
        for r in _verdicts(a, keep, parsed_only=parsed_only).values():
            cl = (r.get("clusters") or {}).get(a.get("primary_backend")) or {}
            if cl.get("n_clusters") is None:
                continue
            n_sets += 1
            n_clusters.append(cl["n_clusters"])
            n_groups.append(r["judge"]["n_groups"])
            if r["judge"]["n_groups"] > 1:
                judge_split += 1
                both_split += cl["n_clusters"] > 1
    return {
        "backend": backend,
        "n_sets": n_sets,
        "judge_split_sets": judge_split,
        "embedding_also_split": both_split,
        "frac_embedding_confirms_split": both_split / judge_split
        if judge_split
        else None,
        "mean_clusters_per_set": st.mean(n_clusters) if n_clusters else None,
        "mean_judge_groups_per_set": st.mean(n_groups) if n_groups else None,
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "runs",
        nargs="*",
        type=Path,
        default=None,
        help="run dirs (default: the paper's)",
    )
    ap.add_argument(
        "--models", default="paper", help="'paper' (29 models), 'all', or a comma list"
    )
    ap.add_argument(
        "--parsed-only",
        action="store_true",
        help="drop sets where any compared pass failed to parse (default: keep "
        "their fallback single-group verdicts, as the paper does)",
    )
    ap.add_argument(
        "--json", type=Path, default=None, help="also write the numbers to this file"
    )
    args = ap.parse_args(argv)
    runs = args.runs or DEFAULT_RUNS
    missing = [r for r in runs if not (r / "analysis.json").exists()]
    if missing:
        sys.exit(f"no analysis.json under: {', '.join(map(str, missing))}")
    keep = (
        None
        if args.models == "all"
        else set(PAPER_MODELS if args.models == "paper" else args.models.split(","))
    )

    out = {
        "runs": [str(r) for r in runs],
        "models": sorted(keep) if keep else "all",
        "parsed_only": args.parsed_only,
        "repeat_pass_stability": repeat_pass_stability(
            runs, keep, parsed_only=args.parsed_only
        ),
        "second_judge": second_judge(runs, keep, parsed_only=args.parsed_only),
        "embedding_crosscheck": embedding_crosscheck(
            runs, keep, parsed_only=args.parsed_only
        ),
    }
    rp, sj, ec = (
        out["repeat_pass_stability"],
        out["second_judge"],
        out["embedding_crosscheck"],
    )
    fmt = lambda x, nd=2: "n/a" if x is None else f"{x:.{nd}f}"  # noqa: E731
    print(f"runs: {', '.join(out['runs'])}; models: {len(keep) if keep else 'all'}")
    print(f"repeat-pass stability ({' / '.join(rp['passes'])}): {rp['n_sets']:,} sets")
    print(
        f"  mean pairwise ARI {fmt(rp['mean_pairwise_ari'])}, single-position agreement {fmt(rp['single_position_agreement'])}, per-set spread SD {fmt(rp['mean_entropy_sd_nats'])} nats"
    )
    print(f"second judge ({sj['second_judge']}): {sj['n_sets']:,} sets")
    print(
        f"  mean ARI vs default {fmt(sj['mean_ari'])}, single-position agreement {fmt(sj['single_position_agreement'])}"
    )
    print(f"embedding cross-check ({ec['backend']}): {ec['n_sets']:,} sets")
    print(
        f"  judge splits {ec['judge_split_sets']:,}; embedding also splits {ec['embedding_also_split']:,} ({fmt(ec['frac_embedding_confirms_split'])}); mean clusters/set {fmt(ec['mean_clusters_per_set'], 1)} vs judge groups/set {fmt(ec['mean_judge_groups_per_set'], 1)}"
    )
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")
    return out


if __name__ == "__main__":
    main()
