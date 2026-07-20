"""Judge-inline generation (fused samples + judge scorer) — offline, mockllm."""

import pytest

from twominds import analyze as A
from twominds import generate as G
from twominds import judge as J
from twominds.models import ModelSpec
from twominds.questions import Question

_QS = [
    Question(id="q1", group="values", prompt="Say something."),
    Question(id="q2", group="values", prompt="Say something else."),
    Question(
        id="fam_v1",
        group="sycophancy",
        prompt="Rate it.",
        family="poem_rating",
        variant="mine",
    ),
]

_VERDICT = (
    '{"contradiction": false, "groups": [[1, 2, 3]], "group_names": ["steady"], '
    '"rationale": "all agree", "flags": []}'
)


def _mock_judge_model(outputs, n=64):
    from inspect_ai.model import ModelOutput, get_model

    return get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", o) for o in outputs]
        * n,
    )


@pytest.fixture()
def fused_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        J, "get_judge_model", lambda *a, **k: _mock_judge_model([_VERDICT])
    )
    spec = ModelSpec(name="mock-a", inspect_model="mockllm/model")
    run_dir = tmp_path / "run"
    G.write_manifest(
        run_dir, [spec], _QS, n=3, temperature=1.0, max_tokens=64, judge="j"
    )
    G.run_generation(
        [spec],
        _QS,
        n=3,
        run_dir=run_dir,
        display="none",
        judge_inline={"judge_name": "mockllm/model", "judge_reasoning": None},
    )
    return run_dir


def test_fused_log_roundtrips_responses_and_verdicts(fused_run):
    # fused shape: one sample per question, N responses from the sample store
    responses = A.load_responses(fused_run)
    assert set(responses["mock-a"]) == {"q1", "q2", "fam_v1"}
    assert all(len(v) == 3 for v in responses["mock-a"].values())

    scores = A.load_judge_scores(fused_run, "mockllm/model", None)
    assert set(scores) == {("mock-a", "q1"), ("mock-a", "q2")}  # family skipped
    jr = scores[("mock-a", "q1")]
    assert jr.parse_ok and jr.groups == [[0, 1, 2]]
    assert jr.group_names == ["steady"]
    assert jr.input_tokens > 0  # usage accounted per verdict


def test_load_judge_scores_rejects_other_judge_config(fused_run):
    assert A.load_judge_scores(fused_run, "openrouter/other-judge", None) == {}
    assert A.load_judge_scores(fused_run, "mockllm/model", "high") == {}


def test_analyze_harvests_instead_of_rejudging(fused_run, monkeypatch):
    calls = []
    real = J.run_judge_eval

    def spy(items, **kw):
        calls.append(len(items))
        return real(items, **kw)

    monkeypatch.setattr(A, "run_judge_eval", spy)
    # family pooling still judges (pooled bundles are never in the gen log)
    monkeypatch.setattr(A.families_mod, "judge_families", lambda items, **kw: {})
    out = A.analyze(
        fused_run, backends=[], judge_name="mockllm/model", judge_reasoning=None
    )
    assert calls == [0]  # per-question judge eval got zero bundles to judge
    by_q = {r["question_id"]: r for r in out["results"]}
    assert by_q["q1"]["judge"]["group_names"] == ["steady"]
    assert by_q["q1"]["judge"]["parse_ok"] is True


def test_judge_run_reps_never_harvest(fused_run, monkeypatch):
    calls = []
    monkeypatch.setattr(
        J, "get_judge_model", lambda *a, **k: _mock_judge_model([_VERDICT])
    )

    real = J.run_judge_eval

    def spy(items, **kw):
        calls.append(len(items))
        return real(items, **kw)

    monkeypatch.setattr(A, "run_judge_eval", spy)
    monkeypatch.setattr(A.families_mod, "judge_families", lambda items, **kw: {})
    A.analyze(
        fused_run,
        backends=[],
        judge_name="mockllm/model",
        judge_reasoning=None,
        judge_run="rep2",
    )
    assert calls == [2]  # stability reps judge fresh, harvesting nothing


def test_legacy_epoch_generation_unchanged(tmp_path):
    # no judge_inline -> epochs shape, no scores, loaders behave as before
    spec = ModelSpec(name="mock-a", inspect_model="mockllm/model")
    run_dir = tmp_path / "run"
    G.run_generation([spec], _QS[:2], n=2, run_dir=run_dir, display="none")
    responses = A.load_responses(run_dir)
    assert all(len(v) == 2 for v in responses["mock-a"].values())
    assert A.load_judge_scores(run_dir, "mockllm/model", None) == {}


def test_judge_bundle_retry_and_fallback():
    import asyncio

    # first reply unparseable -> retried once with the nudge, then parsed
    model = _mock_judge_model(["not json", _VERDICT], n=1)
    jr = asyncio.run(J.judge_bundle(model, "Q?", ["a", "b", "c"]))
    assert jr.parse_ok and jr.group_names == ["steady"]

    # two unparseable replies -> judge-error fallback, single group
    model = _mock_judge_model(["nope", "still nope"], n=1)
    jr = asyncio.run(J.judge_bundle(model, "Q?", ["a", "b"]))
    assert not jr.parse_ok and jr.groups == [[0, 1]]
    assert [f["type"] for f in jr.flags] == ["judge-error"]
