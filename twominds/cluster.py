"""Cluster response embeddings and compare clusters to the judge's groups.

Within one (model, question) bundle we cluster the N response vectors with
average-linkage agglomerative clustering on cosine distance and a fixed distance
threshold (vectors are L2-normalised upstream, so cosine distance = 1 - dot).
Agreement with the judge's partition is reported as Adjusted Rand Index and
Normalised Mutual Information.
"""

from __future__ import annotations

import numpy as np

# Cosine-distance threshold below which two responses are deemed the "same" cluster.
# 0.0 = identical direction, 1.0 = orthogonal. ~0.15 keeps near-paraphrases together.
DEFAULT_THRESHOLD = 0.15


def cluster_responses(
    embeddings: np.ndarray, *, threshold: float = DEFAULT_THRESHOLD
) -> list[int]:
    """Return a cluster label per row of ``embeddings`` (length = n responses)."""
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [0]
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold,
    )
    return [int(x) for x in model.fit_predict(np.asarray(embeddings, dtype=np.float64))]


def agreement(labels_a: list[int], labels_b: list[int]) -> dict[str, float]:
    """ARI + NMI between two label vectors (e.g. judge groups vs embedding clusters)."""
    if len(labels_a) != len(labels_b):
        raise ValueError("label vectors must be the same length")
    if len(labels_a) < 2:
        return {"ari": 1.0, "nmi": 1.0}
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    return {
        "ari": float(adjusted_rand_score(labels_a, labels_b)),
        "nmi": float(normalized_mutual_info_score(labels_a, labels_b)),
    }


def mean_pairwise_cosine_distance(embeddings: np.ndarray) -> float:
    """Mean 1 - cos over all response pairs; a scalar 'spread' of the bundle."""
    n = len(embeddings)
    if n < 2:
        return 0.0
    mat = np.asarray(embeddings, dtype=np.float64)
    sims = mat @ mat.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(1.0 - sims[iu]))
