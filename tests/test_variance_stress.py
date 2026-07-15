"""Pure-logic tests for the synthetic judge stress-test harness (no network).

Covers: mix resolution + apportionment, spec loading/validation, bundle
composition (counts + truth-label alignment through the shuffle), scoring vs a
hand-built judge verdict (ARI, needle recall, unanimous false-positive probes,
contradictory gating), aggregation, and self-contained report rendering.
"""

import random

import pytest

from coherence_variance.judge import JudgeResult
from coherence_variance.stress import (
    Mix,
    Scenario,
    Stance,
    _apportion,
    aggregate_stress,
    build_stress_report,
    compose_bundle,
    load_spec,
    plan_stress,
    score_bundle,
)


def _jr(groups, contradiction, flags=None, rationale=""):
    """JudgeResult from 0-indexed position groups + a contradiction bool."""
    return JudgeResult(
        contradiction=contradiction,
        groups=[list(g) for g in groups],
        rationale=rationale,
        flags=flags or [],
        parse_ok=True,
    )


def _scenario(contradictory=True):
    return Scenario(
        id="s",
        question="q?",
        stances=(Stance("A", "sysA", label="A"), Stance("B", "sysB", label="B")),
        mixes=(
            Mix("unanimous", fill="A"),
            Mix("needle_1", fill="A", counts={"B": 1}),
            Mix("balanced", ratio={"A": 1, "B": 1}),
        ),
        contradictory=contradictory,
    )


# --------------------------------------------------------------------------- #
# Mix resolution / apportionment
# --------------------------------------------------------------------------- #
def test_mix_fill_and_ratio_resolution():
    assert Mix("u", fill="A").resolve(6) == {"A": 6}
    assert Mix("n", fill="A", counts={"B": 1}).resolve(6) == {"A": 5, "B": 1}
    assert Mix("n2", fill="A", counts={"B": 2}).resolve(20) == {"A": 18, "B": 2}
    assert Mix("b", ratio={"A": 1, "B": 1}).resolve(6) == {"A": 3, "B": 3}


def test_apportion_sums_to_n_and_is_near_equal():
    out = _apportion({"A": 1, "B": 1, "C": 1}, 10)
    assert sum(out.values()) == 10
    assert max(out.values()) - min(out.values()) <= 1  # largest-remainder fairness


def test_mix_invalid_forms_raise():
    with pytest.raises(ValueError):
        Mix("bad", fill="A", counts={"B": 7}).resolve(5)  # counts exceed N
    with pytest.raises(ValueError):
        Mix("bad", counts={"B": 1}).resolve(5)  # no fill, no ratio


# --------------------------------------------------------------------------- #
# Spec loading + validation
# --------------------------------------------------------------------------- #
def test_load_spec_shape_and_neutral_flag():
    scenarios = load_spec()
    by_id = {s.id: s for s in scenarios}
    assert {"deceive_binary", "ai_attitude", "pick_language"} <= set(by_id)
    # neutral control is explicitly non-contradictory
    assert by_id["pick_language"].contradictory is False
    assert by_id["deceive_binary"].contradictory is True
    # every mix resolves to a partition summing to N for a couple of N values
    for sc in scenarios:
        for n in (12, 20):
            for mix in sc.mixes:
                counts = mix.resolve(n)
                assert sum(counts.values()) == n
                assert all(stid in {s.id for s in sc.stances} for stid in counts)


def test_load_spec_rejects_unknown_stance(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "scenarios:\n"
        "  - id: x\n    question: q?\n    contradictory: true\n"
        "    stances:\n      - {id: A, system: sysA}\n"
        "    mixes:\n      - {label: m, fill: NOPE}\n"
    )
    with pytest.raises(ValueError):
        load_spec(bad)


# --------------------------------------------------------------------------- #
# Bundle composition
# --------------------------------------------------------------------------- #
def _pools(n_each=10):
    return {
        ("s", "A"): [f"A-{i}" for i in range(n_each)],
        ("s", "B"): [f"B-{i}" for i in range(n_each)],
    }


def test_compose_counts_and_label_alignment_through_shuffle():
    sc = _scenario()
    mix = sc.mixes[1]  # needle_1
    comp = compose_bundle(_pools(), sc, mix, n=6, rng=random.Random(0))
    assert len(comp["responses"]) == 6
    # exact mix counts
    assert sum(s == "A" for s in comp["truth_stance_ids"]) == 5
    assert sum(s == "B" for s in comp["truth_stance_ids"]) == 1
    # the shuffle keeps each response aligned to its planted stance id...
    for text, stid in zip(comp["responses"], comp["truth_stance_ids"]):
        assert text.startswith(stid)
    # ...and to its integer truth label
    lab_of = {}
    for stid, lab in zip(comp["truth_stance_ids"], comp["truth_labels"]):
        lab_of.setdefault(stid, lab)
        assert lab_of[stid] == lab


def test_compose_unanimous_single_stance():
    sc = _scenario()
    comp = compose_bundle(_pools(), sc, sc.mixes[0], n=6, rng=random.Random(1))
    assert set(comp["truth_labels"]) == {0}
    assert set(comp["truth_stance_ids"]) == {"A"}


def test_compose_falls_back_to_replacement_when_pool_too_small():
    sc = _scenario()
    pools = {("s", "A"): ["A-0", "A-1"], ("s", "B"): ["B-0"]}
    comp = compose_bundle(pools, sc, sc.mixes[0], n=5, rng=random.Random(2))
    assert len(comp["responses"]) == 5  # 5 drawn from a pool of 2, with replacement
    assert all(t.startswith("A") for t in comp["responses"])


# --------------------------------------------------------------------------- #
# Scoring vs a hand-built judge verdict
# --------------------------------------------------------------------------- #
def test_score_perfect_recovery():
    # majority A x2 + minority B x1; judge splits B out exactly.
    s = score_bundle(
        [0, 0, 1], ["A", "A", "B"], _jr([[0, 1], [2]], True), contradictory=True
    )
    assert s["ari"] == pytest.approx(1.0)
    assert s["needle_recall"] == pytest.approx(1.0)
    assert s["minority_k"] == 1
    assert s["n_groups_error"] == 0
    assert s["contradiction_true"] and s["contradiction_pred"]
    assert s["contradiction_correct"]


def test_score_needle_absorbed_into_majority():
    # majority A x3 + minority B x1; judge lumps everything into one group.
    s = score_bundle(
        [0, 0, 0, 1],
        ["A", "A", "A", "B"],
        _jr([[0, 1, 2, 3]], False),
        contradictory=True,
    )
    assert s["needle_recall"] == pytest.approx(0.0)  # needle missed
    assert s["n_judge_groups"] == 1
    assert s["contradiction_true"] and not s["contradiction_pred"]
    assert not s["contradiction_correct"]  # judge missed the contradiction


def test_score_unanimous_no_false_positive():
    s = score_bundle(
        [0, 0, 0], ["A", "A", "A"], _jr([[0, 1, 2]], False), contradictory=True
    )
    assert not s["oversplit"] and not s["false_contradiction"]
    assert not s["contradiction_true"]  # single stance
    assert s["contradiction_correct"]  # judge correctly said no contradiction
    assert s["needle_recall"] is None  # no minority


def test_score_unanimous_oversplit_is_flagged():
    s = score_bundle(
        [0, 0, 0], ["A", "A", "A"], _jr([[0, 1], [2]], True), contradictory=True
    )
    assert s["oversplit"] and s["false_contradiction"]
    assert not s["contradiction_correct"]


def test_score_neutral_scenario_contradiction_gated():
    # Two real stance-groups but neutral scenario => no *logical* contradiction.
    s = score_bundle(
        [0, 0, 1], ["A", "A", "B"], _jr([[0, 1], [2]], True), contradictory=False
    )
    assert not s["contradiction_true"]  # gated off by contradictory=False
    assert not s["contradiction_correct"]  # judge wrongly asserted a contradiction
    assert s["ari"] == pytest.approx(1.0)  # grouping is still perfect


# --------------------------------------------------------------------------- #
# Surface-axis (language / dialect / tone) detection
# --------------------------------------------------------------------------- #
_FR_KW = ("french", "québéc", "quebec", "dialect", "language")


def test_axis_detected_when_judge_names_the_axis():
    jr = _jr(
        [[0, 1, 2]],
        False,
        rationale="All three say the same thing; responses 2 and 3 are in French, "
        "and response 3 uses Québécois expressions.",
    )
    s = score_bundle(
        [0, 1, 2], ["en", "fr", "qc"], jr, contradictory=False, axis_keywords=_FR_KW
    )
    assert s["axis_detected"] is True
    assert "french" in s["axis_terms"] and "québéc" in s["axis_terms"]


def test_axis_not_detected_when_judge_silent_on_axis():
    jr = _jr([[0, 1, 2]], False, rationale="All responses are mutually consistent.")
    s = score_bundle(
        [0, 1, 2], ["en", "fr", "qc"], jr, contradictory=False, axis_keywords=_FR_KW
    )
    assert s["axis_detected"] is False
    assert s["axis_terms"] == []


def test_surface_axis_skips_needle_recall():
    # Same-position dialect minority: the judge keeping one group is correct, so
    # needle recall is N/A (None) rather than a misleading 0.0.
    jr = _jr([[0, 1, 2]], False, rationale="all in french; response 3 is québécois")
    s = score_bundle(
        [0, 0, 1],
        ["fr", "fr", "qc"],
        jr,
        contradictory=False,
        axis_keywords=_FR_KW,
        is_surface_axis=True,
    )
    assert s["needle_recall"] is None and s["minority_k"] is None
    assert s["axis_detected"] is True  # but the axis was still named


def test_axis_detection_none_without_keywords_or_single_stance():
    # no axis_keywords -> None
    s = score_bundle([0, 1], ["a", "b"], _jr([[0], [1]], True), contradictory=True)
    assert s["axis_detected"] is None
    # keywords present but unanimous bundle (n_true<2) -> None
    s2 = score_bundle(
        [0, 0],
        ["fr", "fr"],
        _jr([[0, 1]], False, rationale="both in french"),
        contradictory=False,
        axis_keywords=_FR_KW,
    )
    assert s2["axis_detected"] is None


def test_load_spec_surface_axis_and_new_scenarios():
    by_id = {s.id: s for s in load_spec()}
    # dialect scenario is a non-contradictory surface-axis probe with keywords + the
    # three planted dialects and the France-vs-Quebec needle.
    d = by_id["dialect_french"]
    assert d.contradictory is False and d.surface_axis
    assert d.axis_keywords and all(k == k.lower() for k in d.axis_keywords)
    assert {s.id for s in d.stances} >= {"english", "french_france", "french_quebec"}
    assert "quebec_needle_1" in {m.label for m in d.mixes}
    # tonal contradiction scenario
    t = by_id["tone_sarcasm"]
    assert t.contradictory is True
    assert {s.id for s in t.stances} == {"sincere_pro", "sarcastic_anti"}
    # the extra subtle omission stance on deceive
    assert "withhold" in {s.id for s in by_id["deceive_binary"].stances}


def test_aggregate_axis_detection_section():
    def axis_result(mix, jr, labels, ids):
        return {
            "scenario": "dialect_french",
            "mix": mix,
            "contradictory": False,
            "surface_axis": "language / French dialect",
            "score": score_bundle(
                labels, ids, jr, contradictory=False, axis_keywords=_FR_KW
            ),
        }

    results = [
        axis_result(
            "three_way",
            _jr([[0, 1, 2]], False, rationale="responses are in french and québéc"),
            [0, 1, 2],
            ["en", "fr", "qc"],
        ),
        axis_result(
            "three_way",
            _jr([[0, 1, 2]], False, rationale="all consistent"),
            [0, 1, 2],
            ["en", "fr", "qc"],
        ),
    ]
    agg = aggregate_stress(results)
    ax = {a["scenario"]: a for a in agg["axis_detection"]}
    assert ax["dialect_french"]["n_bundles"] == 2
    assert ax["dialect_french"]["axis_detected_rate"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _result(scenario, mix, contradictory, truth_labels, truth_ids, jr):
    score = score_bundle(truth_labels, truth_ids, jr, contradictory=contradictory)
    return {
        "scenario": scenario,
        "mix": mix,
        "contradictory": contradictory,
        "score": score,
    }


def test_aggregate_needle_curve_unanimous_and_confusion():
    results = [
        # unanimous, perfectly handled
        _result(
            "s", "unanimous", True, [0, 0, 0], ["A", "A", "A"], _jr([[0, 1, 2]], False)
        ),
        # unanimous, over-split + false contradiction
        _result(
            "s", "unanimous", True, [0, 0, 0], ["A", "A", "A"], _jr([[0, 1], [2]], True)
        ),
        # needle k=1 recovered
        _result(
            "s", "needle_1", True, [0, 0, 1], ["A", "A", "B"], _jr([[0, 1], [2]], True)
        ),
        # needle k=1 absorbed
        _result(
            "s", "needle_1", True, [0, 0, 1], ["A", "A", "B"], _jr([[0, 1, 2]], False)
        ),
    ]
    agg = aggregate_stress(results)

    assert agg["unanimous"]["n_bundles"] == 2
    assert agg["unanimous"]["oversplit_rate"] == pytest.approx(0.5)
    assert agg["unanimous"]["false_contradiction_rate"] == pytest.approx(0.5)

    curve = {r["k"]: r for r in agg["needle_curve"]}
    assert curve[1]["n_bundles"] == 2
    assert curve[1]["needle_recall_mean"] == pytest.approx(0.5)  # one hit, one miss

    cf = agg["contradiction_confusion"]
    # contradiction_true: unanimous(False x2), needle(True x2). preds: F, T, T, F.
    assert (cf["tp"], cf["fn"], cf["fp"], cf["tn"]) == (1, 1, 1, 1)


# --------------------------------------------------------------------------- #
# Plan + report
# --------------------------------------------------------------------------- #
def test_plan_stress_counts_calls():
    sc = _scenario()
    text = plan_stress(
        [sc], n=10, reps=2, pool_model="gpt-4.1", pool_size=12, judge="j"
    )
    assert "pool generation: 24 calls" in text  # 2 stances x 12
    assert "judge:           6 calls" in text  # 3 mixes x 2 reps


def test_build_stress_report_is_self_contained(tmp_path):
    results = [
        _result(
            "s", "needle_1", True, [0, 0, 1], ["A", "A", "B"], _jr([[0, 1], [2]], True)
        ),
    ]
    # enrich with the per-bundle fields the report reads
    results[0].update(
        {
            "rep": 0,
            "responses": ["a", "b", "c"],
            "truth_stance_ids": ["A", "A", "B"],
            "stance_labels": {"A": "A", "B": "B"},
            "stance_subtle": {"A": False, "B": True},
            "judge_labels": [0, 0, 1],
            "judge": {"rationale": "rat", "flags": ["f1"], "groups": [[0, 1], [2]]},
        }
    )
    analysis = {
        "n": 3,
        "reps": 1,
        "pool_model": "gpt-4.1",
        "pool_size": 12,
        "judge": "j",
        "scenarios": ["s"],
        "results": results,
        "aggregate": aggregate_stress(results),
    }
    out = build_stress_report(analysis, tmp_path / "r.html")
    html = out.read_text()
    assert "<style>" in html and "<script>" in html  # inlined assets
    assert "src=" not in html and "http://" not in html  # no external refs
    assert "Judge stress test" in html and "needle" in html.lower()
