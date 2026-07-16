"""Tests for per-bundle variance metrics + grouping entropy."""

import numpy as np

from twominds import metrics as Mx


def test_variance_metrics_basic():
    responses = ["yes I would", "yes I would", "no never", ""]
    m = Mx.variance_metrics(responses, n_judge_groups=2, n_clusters=3)
    assert m["n"] == 4
    assert "refusal_rate" not in m  # refusal heuristic removed; judge groups handle it
    assert m["n_unique_verbatim"] == 3  # two identical "yes I would"
    assert m["n_judge_groups"] == 2 and m["n_clusters"] == 3


def test_variance_metrics_embeddings_spread():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    m = Mx.variance_metrics(["a", "b"], embeddings=emb)
    assert abs(m["mean_pairwise_cosine_dist"] - 1.0) < 1e-6


def test_group_entropy():
    import math

    assert Mx.group_entropy([]) == 0.0
    assert Mx.group_entropy([0, 0, 0, 0]) == 0.0  # single group -> 0
    # two equal groups -> log 2 (nats); ln2 bits = 1.0
    assert abs(Mx.group_entropy([0, 0, 1, 1]) - math.log(2)) < 1e-9
    assert abs(Mx.group_entropy([0, 0, 1, 1], base=2) - 1.0) < 1e-9
    # n singletons -> log n (maximum)
    assert abs(Mx.group_entropy([0, 1, 2, 3]) - math.log(4)) < 1e-9
    # relabelling is invariant
    assert Mx.group_entropy([5, 5, 9]) == Mx.group_entropy([0, 0, 1])
