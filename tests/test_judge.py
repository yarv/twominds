"""Tests for the cross-sample coherence judge (parsing + control flow, no network)."""

import pytest

from twominds import judge as J


def test_extract_json_plain_and_fenced():
    assert J._extract_json('noise {"a": 1} tail')["a"] == 1
    fenced = '```json\n{"contradiction": false, "groups": [[1]]}\n```'
    assert J._extract_json(fenced)["contradiction"] is False
    # last object wins when several are present
    assert J._extract_json('{"x":1} then {"groups":[[1]]}')["groups"] == [[1]]


def test_parse_valid_partition():
    obj = {
        "contradiction": True,
        "groups": [[1, 3], [2]],
        "rationale": "r",
        "flags": ["f"],
    }
    jr = J._parse(obj, 3)
    assert jr is not None
    assert jr.groups == [[0, 2], [1]]
    assert jr.n_groups == 2
    assert jr.labels(3) == [0, 1, 0]
    assert jr.parse_ok
    # no names given -> padded with "" (display falls back to "position N")
    assert jr.group_names == ["", ""]


def test_parse_group_names_aligned_and_sanitized():
    obj = {
        "groups": [[1, 3], [2]],
        "group_names": ["  Declines  ", "x" * 99, "extra"],
    }
    jr = J._parse(obj, 3)
    assert jr.group_names == ["Declines", "x" * 30]  # trimmed, capped, extras cut
    # non-list group_names tolerated
    assert J._parse({"groups": [[1, 2, 3]], "group_names": "nope"}, 3).group_names == [
        ""
    ]


def test_parse_normalizes_structured_flags():
    obj = {
        "groups": [[1, 2, 3]],
        "flags": [
            {"type": "Refusal", "responses": [2, 99, "x", 2], "note": "declined"},
            {"type": "invented-type", "responses": None},
            "legacy string",
        ],
    }
    jr = J._parse(obj, 3)
    # 1-based -> 0-based, out-of-range/junk/dupes dropped, type lower-cased
    assert jr.flags[0] == {"type": "refusal", "responses": [1], "note": "declined"}
    assert jr.flags[1]["type"] == "other" and jr.flags[1]["responses"] == []
    assert jr.flags[2] == {"type": "other", "responses": [], "note": "legacy string"}


@pytest.mark.parametrize(
    "groups",
    [
        [[1, 2]],  # does not cover index 3
        [[1, 2, 2]],  # duplicate
        [[1, 2, 4]],  # out of range
        [[1, 2, 3, 3]],  # duplicate again
    ],
)
def test_parse_rejects_bad_partitions(groups):
    assert J._parse({"groups": groups}, 3) is None


def test_labels_fills_unplaced_as_singletons():
    jr = J.JudgeResult(
        contradiction=False, groups=[[0]], rationale="", flags=[], parse_ok=True
    )
    # only index 0 placed; indices 1,2 should become their own groups
    assert jr.labels(3) == [0, 1, 2]


def test_judge_result_from_dict_roundtrips():
    flag = {"type": "refusal", "responses": [1], "note": "declined"}
    jr = J.JudgeResult(
        contradiction=True,
        groups=[[0, 2], [1]],
        rationale="r",
        flags=[flag],
        parse_ok=True,
        group_names=["a", "b"],
        input_tokens=5,
        output_tokens=7,
    )
    back = J.JudgeResult.from_dict(jr.to_dict())
    assert back.contradiction and back.groups == [[0, 2], [1]]
    assert back.flags == [flag] and back.parse_ok
    assert back.group_names == ["a", "b"]
    assert (back.input_tokens, back.output_tokens) == (5, 7)


def test_from_dict_normalizes_legacy_string_flags():
    # pre-structured-flags analysis.json payloads load cleanly
    back = J.JudgeResult.from_dict(
        {"groups": [[0, 1]], "flags": ["odd refusal"], "parse_ok": True}
    )
    assert back.flags == [{"type": "other", "responses": [], "note": "odd refusal"}]
    assert back.group_names == []


def test_flag_text_tolerates_both_shapes():
    assert J.flag_text("legacy") == "legacy"
    assert J.flag_text({"type": "refusal", "responses": [0], "note": "n"}) == "refusal n"


# --- run_judge_eval (Inspect-native judge) -------------------------------------
# Drive it with an offline mockllm whose outputs we control (no network/keys).
def _mock_judge(outputs):
    from inspect_ai.model import ModelOutput, get_model

    return get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", o) for o in outputs],
    )


def test_run_judge_eval_parses_keys_and_writes_both_logs(monkeypatch, tmp_path):
    verdict = (
        '{"contradiction": true, "groups": [[1,3],[2]], "group_names": ["yes", "no"], '
        '"rationale": "split", "flags": [{"type": "refusal", "responses": [2], "note": "x"}]}'
    )
    monkeypatch.setattr(
        J, "get_judge_model", lambda *a, **k: _mock_judge([verdict] * 8)
    )
    items = [
        (("m", "q1"), "Q?", ["a", "b", "c"]),
        (("m", "q2"), "Q2?", ["d", "e", "f"]),
    ]
    results, log = J.run_judge_eval(
        items,
        judge_name="mockllm/model",
        reasoning_effort=None,
        display="none",
        log_path=tmp_path / "j",
    )
    assert set(results) == {("m", "q1"), ("m", "q2")}
    jr = results[("m", "q1")]
    assert jr.parse_ok and jr.contradiction and jr.groups == [[0, 2], [1]]
    assert jr.group_names == ["yes", "no"]
    assert jr.flags == [{"type": "refusal", "responses": [1], "note": "x"}]
    assert jr.input_tokens > 0  # per-bundle usage attached from the sample
    assert (tmp_path / "j.eval").is_file() and (tmp_path / "j.json").is_file()
    assert log is not None and len(log.samples) == 2


def test_run_judge_eval_falls_back_on_unparseable(monkeypatch):
    # mockllm's default output is non-JSON -> the scorer's fallback path.
    from inspect_ai.model import get_model

    monkeypatch.setattr(
        J, "get_judge_model", lambda *a, **k: get_model("mockllm/model")
    )
    results, _ = J.run_judge_eval(
        [(("m", "q1"), "Q?", ["a", "b"])],
        judge_name="mockllm/model",
        reasoning_effort=None,
        display="none",
    )
    jr = results[("m", "q1")]
    assert not jr.parse_ok and jr.groups == [[0, 1]]
    assert [f["type"] for f in jr.flags] == ["judge-error"]


def test_run_judge_eval_empty_items_is_noop():
    results, log = J.run_judge_eval([], judge_name="mockllm/model")
    assert results == {} and log is None
