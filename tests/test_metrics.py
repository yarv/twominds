"""Tests for per-bundle metrics, grouping entropy, and per-model scores."""

import math

from twominds import metrics as Mx


def test_variance_metrics_basic():
    responses = ["yes I would", "yes I would", "no never", ""]
    m = Mx.variance_metrics(responses, n_judge_groups=2)
    assert m["n"] == 4
    assert m["n_unique_verbatim"] == 3  # two identical "yes I would"
    assert m["n_judge_groups"] == 2
    assert "n_judge_groups" not in Mx.variance_metrics(responses)


def test_group_entropy():
    assert Mx.group_entropy([]) == 0.0
    assert Mx.group_entropy([0, 0, 0, 0]) == 0.0  # single group -> 0
    # two equal groups -> log 2 (nats); ln2 bits = 1.0
    assert abs(Mx.group_entropy([0, 0, 1, 1]) - math.log(2)) < 1e-9
    assert abs(Mx.group_entropy([0, 0, 1, 1], base=2) - 1.0) < 1e-9
    # n singletons -> log n (maximum)
    assert abs(Mx.group_entropy([0, 1, 2, 3]) - math.log(4)) < 1e-9
    # relabelling is invariant
    assert Mx.group_entropy([5, 5, 9]) == Mx.group_entropy([0, 0, 1])


def _rec(model, qid, h, n_groups, flags=(), parse_ok=True):
    return {
        "model": model,
        "question_id": qid,
        "judge": {"n_groups": n_groups, "flags": list(flags), "parse_ok": parse_ok},
        "metrics": {"group_entropy": h},
    }


def test_model_scores_headline_numbers():
    ln2 = math.log(2)
    results = [
        _rec("a", "q1", 0.0, 1),
        _rec("a", "q2", ln2, 2, flags=[{"responses": [1], "note": "odd"}]),
        _rec("b", "q1", 0.0, 1),
        _rec("b", "q2", 0.0, 1),
        # an unparsed verdict is not a measurement: left out of every count
        _rec("b", "q3", 0.0, 1, parse_ok=False),
        # no verdict at all (judge skipped): left out too
        {"model": "b", "question_id": "q4", "judge": None, "metrics": {}},
    ]
    scores = Mx.model_scores(results, models=["b", "a"])
    assert list(scores) == ["b", "a"]  # declared order
    a, b = scores["a"], scores["b"]
    assert a["n_questions"] == 2 and b["n_questions"] == 2
    assert abs(a["mean_entropy"] - ln2 / 2) < 1e-12
    assert abs(a["effective_positions"] - math.exp(ln2 / 2)) < 1e-12
    assert a["frac_single_position"] == 0.5 and a["n_flagged"] == 1
    assert b["mean_entropy"] == 0.0 and b["effective_positions"] == 1.0
    assert b["frac_single_position"] == 1.0 and b["n_flagged"] == 0
    # a model with no usable verdicts gets no score
    assert Mx.model_scores([_rec("c", "q1", 0.0, 1, parse_ok=False)]) == {}
