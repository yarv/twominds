"""Per-bundle metrics (one bundle = one model's N answers to one question) and
the per-model headline scores derived from them."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean, pstdev
from typing import Optional


def group_entropy(labels: list[int], base: Optional[float] = None) -> float:
    """Shannon entropy of a grouping: ``H = -sum_k p_k log p_k``.

    ``p_k`` is the normalised frequency of group ``k`` (group size / n). A single
    group gives 0; n singletons give the maximum (``log n``). Natural log by
    default; pass ``base=2`` for bits.
    """
    n = len(labels)
    if n == 0:
        return 0.0
    log = math.log if base is None else (lambda x: math.log(x, base))
    h = 0.0
    for count in Counter(labels).values():
        p = count / n
        h -= p * log(p)
    return h


def variance_metrics(
    responses: list[str], *, n_judge_groups: Optional[int] = None
) -> dict:
    n = len(responses)
    lengths = [len(r or "") for r in responses]
    len_mean = mean(lengths) if lengths else 0.0
    out = {
        "n": n,
        "len_mean": len_mean,
        "len_cv": (pstdev(lengths) / len_mean) if (n > 1 and len_mean) else 0.0,
        "n_unique_verbatim": len({(r or "").strip() for r in responses}),
    }
    if n_judge_groups is not None:
        out["n_judge_groups"] = n_judge_groups
    return out


def model_scores(
    results: list[dict], models: Optional[list[str]] = None
) -> dict[str, dict]:
    """Per-model headline scores from the per-bundle records of an analysis.

    For every model: ``mean_entropy`` (mean answer spread H over its judged
    questions, nats), ``effective_positions`` (e^H), ``frac_single_position``
    (share of questions where the judge returned one group), ``n_flagged``
    (questions the judge attached a flag to) and ``n_questions`` (bundles with
    a usable verdict). Bundles whose judge reply never parsed are not
    measurements and are left out.
    """
    acc: dict[str, dict] = {}
    for r in results:
        judge = r.get("judge") or {}
        h = (r.get("metrics") or {}).get("group_entropy")
        if h is None or judge.get("parse_ok") is False:
            continue
        a = acc.setdefault(r["model"], {"h": [], "single": 0, "flagged": 0})
        a["h"].append(float(h))
        if judge.get("n_groups") == 1:
            a["single"] += 1
        if judge.get("flags"):
            a["flagged"] += 1
    order = list(models) if models else sorted(acc)
    for m in acc:  # models not in the declared order (defensive)
        if m not in order:
            order.append(m)
    scores: dict[str, dict] = {}
    for m in order:
        a = acc.get(m)
        if not a or not a["h"]:
            continue
        mean_h = sum(a["h"]) / len(a["h"])
        scores[m] = {
            "n_questions": len(a["h"]),
            "mean_entropy": mean_h,
            "effective_positions": math.exp(mean_h),
            "frac_single_position": a["single"] / len(a["h"]),
            "n_flagged": a["flagged"],
        }
    return scores
