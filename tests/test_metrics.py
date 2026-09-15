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


def test_model_scores():
    import math

    def rec(model, h, n_groups, flags=(), parse_ok=True):
        return {
            "model": model,
            "question_id": f"q{h}",
            "judge": {"n_groups": n_groups, "flags": list(flags), "parse_ok": parse_ok},
            "metrics": {"group_entropy": h},
        }

    results = [
        rec("a", 0.0, 1),
        rec("a", math.log(2), 2, flags=[{"responses": [], "note": "x"}]),
        rec("b", 0.0, 1),
        rec("b", 9.0, 3, parse_ok=False),  # unparsed verdict is not a measurement
        {
            "model": "c",
            "question_id": "fam",
            "judge": None,
            "metrics": {},
        },  # no verdict
    ]
    s = Mx.model_scores(results, ["b", "a"])
    assert list(s) == ["b", "a"]  # declared order wins
    assert s["a"]["n_questions"] == 2 and s["a"]["n_flagged"] == 1
    assert abs(s["a"]["mean_entropy"] - math.log(2) / 2) < 1e-9
    assert abs(s["a"]["effective_positions"] - math.sqrt(2)) < 1e-9
    assert s["a"]["frac_single_position"] == 0.5
    assert s["b"] == {
        "n_questions": 1,
        "mean_entropy": 0.0,
        "effective_positions": 1.0,
        "frac_single_position": 1.0,
        "n_flagged": 0,
    }
    assert "c" not in s
