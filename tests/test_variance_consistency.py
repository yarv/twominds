"""Tests for cross-run judge-consistency aggregation."""

from coherence_variance import consistency as C


def _run(label, q1_labels, q1_contra, q2_labels=(0, 0, 0), q2_contra=False):
    def bundle(qid, labels, contra):
        n_groups = len(set(labels))
        return {
            "model": "m1",
            "question_id": qid,
            "group": "g",
            "responses": [f"resp {qid} {i}" for i in range(len(labels))],
            "metrics": {"refusal_rate": 0.0},
            "judge": {
                "n_groups": n_groups,
                "contradiction": contra,
                "flags": [],
                "rationale": "r",
            },
            "judge_labels": list(labels),
        }

    return {
        "questions": {
            "q1": {"prompt": "Q1?", "group": "g"},
            "q2": {"prompt": "Q2?", "group": "g"},
        },
        "results": [
            bundle("q1", q1_labels, q1_contra),
            bundle("q2", q2_labels, q2_contra),
        ],
    }


def test_identical_runs_are_perfectly_consistent():
    runs = {
        "a": _run("a", [0, 0, 1, 1], True),
        "b": _run("b", [0, 0, 1, 1], True),
    }
    agg = C.aggregate(runs)
    o = agg["overall"]
    assert o["n_runs"] == 2 and o["n_bundles"] == 2
    assert o["mean_partition_ari"] == 1.0
    assert o["mean_n_groups_std"] == 0.0
    assert o["frac_contradiction_unstable"] == 0.0
    q1 = next(b for b in agg["per_bundle"] if b["question_id"] == "q1")
    assert q1["mean_pairwise_ari"] == 1.0 and q1["contradiction_stable"]


def test_different_partitions_lower_ari():
    runs = {
        "a": _run("a", [0, 0, 1, 1], True),
        "b": _run("b", [0, 1, 0, 1], True),  # same #groups, different partition
    }
    agg = C.aggregate(runs)
    q1 = next(b for b in agg["per_bundle"] if b["question_id"] == "q1")
    assert q1["mean_pairwise_ari"] < 1.0
    assert q1["n_groups"] == [2, 2] and q1["n_groups_std"] == 0.0
    # q2 identical across runs -> still perfectly consistent
    q2 = next(b for b in agg["per_bundle"] if b["question_id"] == "q2")
    assert q2["mean_pairwise_ari"] == 1.0


def test_contradiction_instability_detected():
    runs = {
        "a": _run("a", [0, 0, 1, 1], True),
        "b": _run("b", [0, 0, 1, 1], False),  # flips contradiction
    }
    agg = C.aggregate(runs)
    q1 = next(b for b in agg["per_bundle"] if b["question_id"] == "q1")
    assert not q1["contradiction_stable"]
    assert q1["contradiction_agreement"] == 0.5
    assert agg["overall"]["frac_contradiction_unstable"] == 0.5  # 1 of 2 bundles


def test_entropy_mean_across_runs():
    import math

    runs = {
        "a": _run("a", [0, 0, 1, 1], True),  # entropy ln2
        "b": _run("b", [0, 1, 2, 3], True),  # entropy ln4
    }
    agg = C.aggregate(runs)
    q1 = next(b for b in agg["per_bundle"] if b["question_id"] == "q1")
    assert abs(q1["entropy_mean"] - (math.log(2) + math.log(4)) / 2) < 1e-9
    assert q1["entropy_std"] > 0


def test_report_renders_self_contained(tmp_path):
    runs = {"a": _run("a", [0, 0, 1, 1], True), "b": _run("b", [0, 1, 0, 1], True)}
    agg = C.aggregate(runs)
    out = C.build_consistency_report(agg, tmp_path / "consistency_report.html")
    html = out.read_text()
    assert "Judge consistency" in html
    assert "partition ARI" in html
    assert "http://" not in html and "https://" not in html and "<script" not in html


def test_per_model_keys():
    runs = {"a": _run("a", [0, 0, 1, 1], True), "b": _run("b", [0, 0, 1, 1], True)}
    agg = C.aggregate(runs)
    assert "m1" in agg["per_model"]
    assert set(agg["per_model"]["m1"]) >= {
        "mean_partition_ari",
        "mean_n_groups_std",
        "mean_entropy_mean",
        "frac_contradiction_unstable",
    }


def test_co_association_identical_runs():
    C_mat, n = C.co_association([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]])
    assert n == 4
    # 0&1 always together -> 1.0; 0&2 never -> 0.0
    assert C_mat[0][1] == 1.0 and C_mat[0][2] == 0.0
    cons = C.consensus_from_coassoc(C_mat, n)
    assert cons["strength"] == 1.0 and cons["contested_pairs"] == 0
    assert all(s == 1.0 for s in cons["stability"])


def test_co_association_flipping_pair_is_contested():
    # response 2 (index 1) flips group membership across the three runs
    C_mat, n = C.co_association([[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]])
    cons = C.consensus_from_coassoc(C_mat, n)
    assert cons["strength"] < 1.0
    assert cons["contested_pairs"] >= 1
    # the most-shuffled response should be the least stable (a drifter candidate)
    assert min(range(n), key=lambda i: cons["stability"][i]) in (1, 2)


def test_aggregate_includes_consensus_fields():
    runs = {"a": _run("a", [0, 0, 1, 1], True), "b": _run("b", [0, 1, 0, 1], True)}
    agg = C.aggregate(runs)
    q1 = next(b for b in agg["per_bundle"] if b["question_id"] == "q1")
    for k in (
        "consensus_strength",
        "contested_pairs",
        "consensus_labels",
        "consensus_stability",
        "n_drifters",
        "coassoc",
    ):
        assert k in q1
    assert agg["overall"]["mean_consensus_strength"] is not None


def test_multi_report_renders(tmp_path):
    from coherence_variance import multi_report as MR

    runs = {"a": _run("a", [0, 0, 1, 1], True), "b": _run("b", [0, 1, 0, 1], True)}
    agg = C.aggregate(runs)
    data = MR.build_view_data(runs, agg)
    assert data["run_labels"] == ["a", "b"] and data["bundles"]
    assert "coassoc" in data["bundles"][0]["consensus"]
    out = MR.build_multi_report(runs, agg, tmp_path / "multi_report.html")
    html = out.read_text()
    assert "const DATA" in html and 'id="view"' in html
    assert "agreement matrix" in html  # heatmap path present
    # interactive category chart (with across-pass error bars) embedded + wired
    assert (
        'id="cchart"' in html and "const CHART" in html and "initCategoryChart" in html
    )
    # self-contained: the only http(s) literal is the never-fetched SVG namespace
    import re

    assert not [
        u
        for u in re.findall(r'https?://[^\s"\'<>]+', html)
        if not u.startswith("http://www.w3.org/")
    ]
    assert "<script src=" not in html
