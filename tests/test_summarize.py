"""Per-model LLM summaries: prompt tiering, parse fallback, sidecar caching."""

import json

from twominds import summarize as S


def _record(qid, group="values", *, contradiction=False, flags=None, entropy=0.0,
            responses=None, n=None):
    responses = responses if responses is not None else ["yes", "yes", "yes"]
    return {
        "model": "toy",
        "question_id": qid,
        "group": group,
        "responses": responses,
        "judge": {
            "contradiction": contradiction,
            "groups": [[i for i in range(len(responses))]],
            "n_groups": 2 if contradiction else 1,
            "group_names": ["agrees"],
            "rationale": f"rationale for {qid}",
            "flags": flags or [],
            "parse_ok": True,
        },
        "metrics": {"n": n or len(responses), "group_entropy": entropy},
    }


# --------------------------------------------------------------------------- #
# prompt tiering
# --------------------------------------------------------------------------- #


def test_prompt_tiers_detail_by_interest():
    hot = _record(
        "q_hot",
        contradiction=True,
        flags=[{"type": "refusal", "responses": [0], "note": "declined outright"}],
        responses=["long answer " + "x" * 800, "no", "yes"],
    )
    cold = [_record(f"q_cold{i}") for i in range(20)]
    prompt = S.build_model_prompt(
        "Toy Model", [*cold[:10], hot, *cold[10:]], max_detailed=1,
        questions={"q_hot": {"prompt": "Would you ever refuse?"}},
    )
    # the interesting record gets the full block: rationale, flag note, samples
    assert "rationale for q_hot" in prompt
    assert "refusal: declined outright" in prompt
    assert "Would you ever refuse?" in prompt
    assert "[…truncated]" in prompt  # >700-char response was cut
    # boring records appear only as compact one-liners
    assert "rationale for q_cold0" not in prompt
    assert "q_cold0: 1 position(s) across 3 answers" in prompt
    assert "TASK:" in prompt and '"headline"' in prompt


def test_prompt_compact_line_carries_contradiction_and_flags():
    recs = [
        _record("q_a", contradiction=True, entropy=1.0),
        _record(
            "q_b",
            flags=[{"type": "self-preservation", "responses": [], "note": "n"}],
            entropy=0.9,
        ),
        _record("q_c"),
    ]
    prompt = S.build_model_prompt("Toy", recs, max_detailed=1)
    # q_a is the detailed one; q_b's compact line keeps its signal
    assert "q_b" in prompt and "flags: self-preservation" in prompt


def test_prompt_families_digest():
    fam = {
        "model": "toy",
        "family": "anchoring",
        "title": "Anchoring — numeric anchor pull",
        "scalar": {"kind": "number", "swing": 1.65},
        "judge": {"ari": 0.12, "contradiction": True},
    }
    prompt = S.build_model_prompt("Toy", [_record("q")], [fam])
    assert "anchoring" in prompt
    assert "swing across framings=1.65" in prompt
    assert "ARI=0.12" in prompt
    assert "CONTRADICTION across framings" in prompt


def test_prompt_ignores_judge_error_flags():
    rec = _record(
        "q", flags=[{"type": "judge-error", "responses": [], "note": "unparsed"}]
    )
    prompt = S.build_model_prompt("Toy", [rec])
    assert "judge-error" not in prompt


# --------------------------------------------------------------------------- #
# output parsing
# --------------------------------------------------------------------------- #


def test_parse_json_and_prose_fallback():
    ok = S._parse_summary('Here you go:\n{"headline": "Very consistent",'
                          ' "summary": "The model agrees with itself."}')
    assert ok == {
        "headline": "Very consistent",
        "summary": "The model agrees with itself.",
        "parse_ok": True,
    }
    bad = S._parse_summary("The model was pretty consistent overall.")
    assert bad["parse_ok"] is False
    assert bad["summary"] == "The model was pretty consistent overall."
    assert bad["headline"] == ""


# --------------------------------------------------------------------------- #
# summarize_run: caching + invalidation (no network — runner is stubbed)
# --------------------------------------------------------------------------- #


def _write_run(tmp_path, models=("toy",)):
    analysis = {
        "models": list(models),
        "model_display": {m: m.upper() for m in models},
        "questions": {
            "q1": {"prompt": "Who are you?", "group": "identity"},
            "q_fam": {"prompt": "Rate this.", "group": "sycophancy", "family": "anchor"},
        },
        "results": [
            {**_record("q1"), "model": m} for m in models
        ]
        + [  # family variant (excluded) + judge-less bundle (excluded)
            {**_record("q_fam"), "model": models[0]},
            {**_record("q1"), "model": models[0], "judge": None, "question_id": "q2"},
        ],
        "families": [],
    }
    analysis["questions"]["q2"] = {"prompt": "x", "group": "identity"}
    (tmp_path / "analysis.json").write_text(json.dumps(analysis))
    return tmp_path


def _stub_runner(monkeypatch, calls):
    async def fake(tasks, **kwargs):
        calls.append([m for m, _ in tasks])
        return {
            m: {"headline": "h", "summary": f"summary of {m}", "parse_ok": True,
                "input_tokens": 10, "output_tokens": 5}
            for m, _ in tasks
        }

    monkeypatch.setattr(S, "summarize_models", fake)


def test_summarize_run_writes_and_caches(tmp_path, monkeypatch):
    run = _write_run(tmp_path)
    calls = []
    _stub_runner(monkeypatch, calls)

    out = S.summarize_run(run, echo=lambda *a: None)
    assert calls == [["toy"]]
    assert out["toy"]["summary"] == "summary of toy"
    # provenance stamped for cache invalidation
    entry = json.loads((run / "summaries.json").read_text())["toy"]
    assert entry["prompt_hash"] == S.SUMMARY_PROMPT_HASH
    assert entry["summarizer"] and entry["created"]

    # second call: cache hit, zero LLM calls
    S.summarize_run(run, echo=lambda *a: None)
    assert calls == [["toy"]]

    # --force re-runs
    S.summarize_run(run, force=True, echo=lambda *a: None)
    assert calls == [["toy"], ["toy"]]


def test_summarize_run_invalidates_on_stale_prompt_hash(tmp_path, monkeypatch):
    run = _write_run(tmp_path)
    calls = []
    _stub_runner(monkeypatch, calls)
    S.summarize_run(run, echo=lambda *a: None)

    sidecar = json.loads((run / "summaries.json").read_text())
    sidecar["toy"]["prompt_hash"] = "outdated00000"
    (run / "summaries.json").write_text(json.dumps(sidecar))
    S.summarize_run(run, echo=lambda *a: None)
    assert calls == [["toy"], ["toy"]]


def test_summarize_run_only_missing_models(tmp_path, monkeypatch):
    run = _write_run(tmp_path, models=("toy", "toy2"))
    calls = []
    _stub_runner(monkeypatch, calls)
    S.summarize_run(run, echo=lambda *a: None)
    assert sorted(calls[0]) == ["toy", "toy2"]

    # drop one entry -> only that model re-runs (incremental backfill)
    sidecar = json.loads((run / "summaries.json").read_text())
    del sidecar["toy2"]
    (run / "summaries.json").write_text(json.dumps(sidecar))
    S.summarize_run(run, echo=lambda *a: None)
    assert calls[1] == ["toy2"]


def test_summarize_run_skips_judgeless_run(tmp_path, monkeypatch):
    analysis = {
        "models": ["toy"],
        "questions": {"q1": {"prompt": "x", "group": "g"}},
        "results": [{**_record("q1"), "judge": None}],
    }
    (tmp_path / "analysis.json").write_text(json.dumps(analysis))
    calls = []
    _stub_runner(monkeypatch, calls)
    assert S.summarize_run(tmp_path, echo=lambda *a: None) == {}
    assert calls == []
    assert not (tmp_path / "summaries.json").exists()
