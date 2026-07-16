"""Per-bundle variance metrics (one bundle = one model's N answers to one question)."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean, pstdev
from typing import Optional

import numpy as np

from .cluster import mean_pairwise_cosine_distance


def group_entropy(labels: list[int], base: Optional[float] = None) -> float:
    """Shannon entropy of a grouping: ``H = -sum_k p_k log p_k``.

    ``p_k`` is the normalised frequency of group ``k`` (group size / n). A single
    group gives 0; n singletons give the maximum (``log n``). Works for judge
    groups or embedding clusters (pass the per-item label vector). Natural log by
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
    responses: list[str],
    *,
    embeddings: Optional[np.ndarray] = None,
    n_judge_groups: Optional[int] = None,
    n_clusters: Optional[int] = None,
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
    if n_clusters is not None:
        out["n_clusters"] = n_clusters
    if embeddings is not None and len(embeddings) >= 2:
        out["mean_pairwise_cosine_dist"] = mean_pairwise_cosine_distance(embeddings)
    return out
