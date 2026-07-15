"""Cross-variant (framing-invariance) family analysis — pure-logic tests (no network)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from coherence_variance import families as fam
from coherence_variance.analyze import _family_pass
from coherence_variance.families_report import build_fam, build_families_report
from coherence_variance.judge import JudgeResult
from coherence_variance.questions import load_families, select_questions


# --- scalar extraction -------------------------------------------------------
class TestExtractScalar:
    def test_number(self):
        # legacy commit-first (first line) still parses via the first-line fallback
        assert fam.extract_scalar("number", "7\n90%\nbecause") == 7.0
        assert fam.extract_scalar("number", "I'd say 8/10") == 8.0
        assert fam.extract_scalar("number", "no digits here") is None

    def test_yesno(self):
        assert fam.extract_scalar("yesno", "Yes\n80%") == 1.0
        assert fam.extract_scalar("yesno", "No, because of confounds") == 0.0
        assert fam.extract_scalar("yesno", "**No**") == 0.0
        assert fam.extract_scalar("yesno", "Maybe, hard to say") is None

    def test_ab(self):
        assert fam.extract_scalar("ab", "A") == "A"
        assert fam.extract_scalar("ab", "(B) keep running") == "B"
        assert fam.extract_scalar("ab", "C none of these") is None

    def test_reason_first_committed_on_final_line(self):
        # reason-first format (2026-06-12): the committed answer is the LAST line,
        # and must win over numbers/words that appear earlier in the reasoning.
        assert (
            fam.extract_scalar("number", "It's clichéd; I rate it 1 to 10.\n4") == 4.0
        )
        assert (
            fam.extract_scalar("yesno", "Sales rose after the ad, but no control.\nNo")
            == 0.0
        )
        assert (
            fam.extract_scalar("yesno", "Yes there are confounds to weigh.\nNo") == 0.0
        )
        assert fam.extract_scalar("ab", "Option A is tempting, but...\nB") == "B"

    def test_empty(self):
        assert fam.extract_scalar("number", "") is None


def test_per_variant_scalar_and_swing():
    v2r = {"hi": ["9", "9", "10"], "lo": ["3", "2", "3"]}
    pv = fam.per_variant_scalar("number", v2r)
    assert pv["hi"]["mean"] == pytest.approx((9 + 9 + 10) / 3)
    assert pv["lo"]["mean"] == pytest.approx((3 + 2 + 3) / 3)
    swing = fam.scalar_swing("number", pv)
    assert swing == pytest.approx((28 / 3) - (8 / 3))

    yn = fam.per_variant_scalar("yesno", {"a": ["Yes", "No"], "b": ["No", "No"]})
    assert yn["a"]["mean"] == pytest.approx(0.5)
    assert yn["b"]["mean"] == pytest.approx(0.0)
    assert fam.scalar_swing("yesno", yn) == pytest.approx(0.5)


# --- pooling + alignment -----------------------------------------------------
def test_build_pool_deterministic_and_aligned():
    v2r = {"a": ["a0", "a1", "a2"], "b": ["b0", "b1"]}
    order = ["a", "b"]
    seed = fam._seed("m", "famX")
    texts1, labels1, src1 = fam.build_pool(v2r, order, seed=seed)
    texts2, labels2, src2 = fam.build_pool(v2r, order, seed=seed)
    assert (texts1, labels1, src1) == (texts2, labels2, src2)  # deterministic
    assert len(texts1) == 5 and len(labels1) == 5
    # every (variant_index, within_index) source maps back to the right text
    for t, lab, (vi, wi) in zip(texts1, labels1, src1):
        assert lab == vi
        assert t == v2r[order[vi]][wi]


def test_family_alignment_block_vs_uniform():
    # perfectly framing-split: each variant is its own judge group -> ARI 1
    var = [0, 0, 0, 1, 1, 1]
    judge_split = [0, 0, 0, 1, 1, 1]
    a = fam.family_alignment(judge_split, var, 2)
    assert a["ari"] == pytest.approx(1.0)
    assert a["contingency"] == [[3, 0], [0, 3]]

    # framing-invariant: judge groups orthogonal to framing -> ARI ~0
    judge_mixed = [0, 1, 0, 1, 0, 1]
    b = fam.family_alignment(judge_mixed, var, 2)
    assert abs(b["ari"]) < 0.3
    assert b["contingency"] == [[2, 1], [1, 2]]


# --- selection ---------------------------------------------------------------
def test_family_selection_and_meta():
    fams = load_families()
    assert {"poem_rating", "reasoning_validity"} <= set(fams)
    assert fams["poem_rating"].scalar == "number"
    assert fams["reasoning_validity"].scalar == "yesno"
    qs = select_questions(families=["poem_rating"])
    assert {q.variant for q in qs} == {"mine_love", "other_meh", "neutral"}
    assert all(q.family == "poem_rating" for q in qs)
    with pytest.raises(KeyError):
        select_questions(families=["nope"])


# --- end-to-end family pass (mocked judge, no network) -----------------------
def _num(t):
    m = re.search(r"\d+", t or "")
    return int(m.group()) if m else None


def _fake_judge(
    items, *, judge_name, reasoning_effort, concurrency, log_path=None, display="plain"
):
    """Group pooled texts by whether their rating is high/low — i.e. by variant."""
    out = {}
    for model, fam_id, _prompt, texts in items:
        hi = [i for i, t in enumerate(texts) if (_num(t) or 0) >= 5]
        lo = [i for i in range(len(texts)) if i not in hi]
        out[(model, fam_id)] = JudgeResult(
            contradiction=True,
            groups=[g for g in (hi, lo) if g],
            rationale="ratings split into a high and a low cluster",
            flags=["framing_split"],
            parse_ok=True,
        )
    return out


def test_family_pass_end_to_end(monkeypatch):
    monkeypatch.setattr(fam, "judge_families", _fake_judge)

    qmeta = {
        "q_hi": {"family": "poem_rating", "variant": "mine_love"},
        "q_lo": {"family": "poem_rating", "variant": "neutral"},
    }
    responses = {"m1": {"q_hi": ["9", "9", "10", "9"], "q_lo": ["3", "2", "3", "3"]}}
    embeds = {
        "local": {
            ("m1", "q_hi"): np.tile([1.0, 0.0], (4, 1)),
            ("m1", "q_lo"): np.tile([0.0, 1.0], (4, 1)),
        }
    }
    fams_meta = {
        "poem_rating": {
            "prompt": "rate the same poem",
            "scalar": "number",
            "title": "Poem rating",
            "description": "desc",
        }
    }

    recs = _family_pass(
        fams_meta,
        responses,
        qmeta,
        embeds,
        "local",
        run_judge=True,
        judge_name="x",
        judge_reasoning=None,
        concurrency=2,
        threshold=0.15,
    )
    assert len(recs) == 1
    rec = recs[0]
    # variant order is (variant_label, qid) sorted -> mine_love, neutral
    assert [v["variant"] for v in rec["variants"]] == ["mine_love", "neutral"]
    # scalar swing = mean(hi) - mean(lo)
    assert rec["scalar"]["swing"] == pytest.approx(9.25 - 2.75)
    # judge perfectly aligned with framing -> ARI 1
    assert rec["judge"]["ari"] == pytest.approx(1.0)
    assert rec["judge"]["contradiction"] is True
    # embeddings separate the variants too
    assert rec["cluster"]["ari"] == pytest.approx(1.0)


def test_family_pass_skips_incomplete(monkeypatch):
    monkeypatch.setattr(fam, "judge_families", _fake_judge)
    qmeta = {
        "q_hi": {"family": "poem_rating", "variant": "a"},
        "q_lo": {"family": "poem_rating", "variant": "b"},
    }
    # variant b has no responses for this model -> family skipped cleanly
    responses = {"m1": {"q_hi": ["9"], "q_lo": []}}
    recs = _family_pass(
        {"poem_rating": {"prompt": "p", "scalar": "number"}},
        responses,
        qmeta,
        {"local": {}},
        "local",
        run_judge=True,
        judge_name="x",
        judge_reasoning=None,
        concurrency=2,
        threshold=0.15,
    )
    assert recs == []


# --- report render -----------------------------------------------------------
def test_families_report_renders(tmp_path):
    analysis = {
        "run_dir": "results/variance/demo",
        "judge": "judge/x",
        "families": [
            {
                "model": "gpt-4o",
                "family": "poem_rating",
                "title": "Poem rating",
                "description": "desc",
                "scalar_kind": "number",
                "variants": [
                    {"variant": "mine_love", "question_id": "q_hi", "n": 4},
                    {"variant": "neutral", "question_id": "q_lo", "n": 4},
                ],
                "n_total": 8,
                "scalar": {
                    "kind": "number",
                    "per_variant": {
                        "mine_love": {"mean": 9.25},
                        "neutral": {"mean": 2.75},
                    },
                    "swing": 6.5,
                },
                "judge": {
                    "ari": 1.0,
                    "nmi": 1.0,
                    "n_groups": 2,
                    "contingency": [[4, 0], [0, 4]],
                    "group_ids": [0, 1],
                    "contradiction": True,
                    "rationale": "split by rating",
                    "flags": ["framing_split"],
                    "parse_ok": True,
                },
                "cluster": {"ari": 1.0, "nmi": 1.0, "n_clusters": 2},
            }
        ],
        "results": [
            {"model": "gpt-4o", "question_id": "q_hi", "responses": ["9\nlovely"] * 4},
            {"model": "gpt-4o", "question_id": "q_lo", "responses": ["3\nclichéd"] * 4},
        ],
    }
    # --- build_fam: analysis -> FAM transform ---
    famdata = build_fam(analysis)
    assert famdata["models"] == ["gpt-4o"]
    assert famdata["cohorts"] == {"gpt-4o": "base"}  # not an ours/ organism
    assert famdata["groups_source"] == "contingency"  # recovered from counts
    rec = famdata["records"][0]
    # metrics feed the grouped-bar chart; number swing normalised onto ~[0,1] (÷10)
    assert rec["metrics"]["judge_ari"] == 1.0
    assert rec["metrics"]["swing_norm"] == pytest.approx(0.65)
    assert rec["metrics"]["contradiction"] == 1.0
    # per-variant column summary + recovered per-response group tints
    v0 = rec["variants"][0]
    assert v0["summary"] == "9.2"  # number -> mean, 1 dp
    assert v0["responses"] == ["9\nlovely"] * 4
    assert v0["groups"] == [0] * 4  # contingency row [4,0] -> whole column in group 0

    out = build_families_report(analysis, tmp_path / "families_report.html")
    htmltext = out.read_text()
    # self-contained, client-rendered: FAM blob + inlined renderer, no external refs
    assert "const FAM =" in htmltext
    assert not re.search(r'(?:src|href)\s*=\s*["\'](?!#)', htmltext)
    for anchor in ('id="chartsvg"', 'id="cards"', 'id="dash"', "<noscript>"):
        assert anchor in htmltext
    assert "Poem rating" in htmltext  # family title (noscript fallback)
    assert "gpt-4o" in htmltext
    assert "6.50" in htmltext  # swing in the noscript fallback table
    assert "judge ARI" in htmltext  # inlined renderer


def test_groups_by_variant_maps_pool_labels_back():
    from coherence_variance import families as F

    # 2 variants x 2 responses pooled in shuffled order
    sources = [(1, 0), (0, 1), (0, 0), (1, 1)]
    judge_labels = [3, 1, 0, 3]
    assert F.groups_by_variant(judge_labels, sources, [2, 2]) == [[0, 1], [3, 3]]
    # a short judge output leaves unlabelled slots None
    assert F.groups_by_variant([5], sources, [2, 2]) == [[None, None], [5, None]]


def test_build_fam_groups_exact_flag():
    from coherence_variance.families_report import build_fam

    base = {
        "model": "m",
        "family": "f",
        "scalar_kind": None,
        "judge": {
            "ari": 0.5,
            "n_groups": 2,
            "contingency": [[1, 1], [2, 0]],
            "group_ids": [0, 1],
        },
        "variants": [
            {"variant": "a", "question_id": "qa", "n": 2},
            {"variant": "b", "question_id": "qb", "n": 2},
        ],
    }
    analysis = {"families": [base], "families_meta": {}, "results": []}
    rec = build_fam(analysis)["records"][0]
    assert rec["groups_exact"] is False
    # split variant recovered as all-None; whole variant gets its group
    assert rec["variants"][0]["groups"] == [None, None]
    assert rec["variants"][1]["groups"] == [0, 0]

    with_labels = dict(base)
    with_labels["variants"] = [
        {"variant": "a", "question_id": "qa", "n": 2, "groups": [0, 1]},
        {"variant": "b", "question_id": "qb", "n": 2, "groups": [0, 0]},
    ]
    rec = build_fam({"families": [with_labels], "families_meta": {}, "results": []})[
        "records"
    ][0]
    assert rec["groups_exact"] is True
    assert rec["variants"][0]["groups"] == [0, 1]
