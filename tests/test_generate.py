"""Generation-phase tests: the single multi-model ``run_generation`` call.

Generation is one ``inspect_ai.eval(model=[...])`` call (Inspect schedules the
models concurrently in-process). The load-bearing contract is the on-disk layout:
each model's returned log is written to ``logs/<spec.name>/`` as both ``.eval``
(canonical, read by ``analyze.load_responses``) and ``.json`` (human-readable),
and same-id rungs stay disambiguated by their spec name. We lock that with a real
offline ``mockllm/model`` run (no network, no API keys).
"""

from twominds import analyze as A
from twominds import cost as C
from twominds import generate as G
from twominds.models import ModelSpec
from twominds.questions import Question

_QS = [
    Question(id="q1", group="values", prompt="Say something."),
    Question(id="q2", group="values", prompt="Say something else."),
]


def test_run_generation_writes_per_model_eval_and_json(tmp_path):
    """One eval over two models lands each in logs/<name>/ as .eval + .json."""
    specs = [
        ModelSpec(name="mock-a", inspect_model="mockllm/model"),
        ModelSpec(name="mock-b", inspect_model="mockllm/model"),
    ]
    run_dir = tmp_path / "run"
    out = G.run_generation(
        specs, _QS, n=3, run_dir=run_dir, display="none", model_concurrency=2
    )
    assert set(out) == {"mock-a", "mock-b"}

    for name in ("mock-a", "mock-b"):
        d = run_dir / "logs" / name
        assert (d / f"{name}.eval").is_file()
        assert (d / f"{name}.json").is_file()
    # Inspect's incremental scratch dir is cleaned up.
    assert not (run_dir / "logs" / ".raw").exists()


def test_run_generation_roundtrips_through_load_responses(tmp_path):
    """analyze.load_responses reads every model's N epochs back per question."""
    specs = [
        ModelSpec(name="mock-a", inspect_model="mockllm/model"),
        ModelSpec(name="mock-b", inspect_model="mockllm/model"),
    ]
    run_dir = tmp_path / "run"
    G.run_generation(specs, _QS, n=2, run_dir=run_dir, display="none")

    responses = A.load_responses(run_dir)
    assert set(responses) == {"mock-a", "mock-b"}
    for name in ("mock-a", "mock-b"):
        assert set(responses[name]) == {"q1", "q2"}
        for qid in ("q1", "q2"):
            assert len(responses[name][qid]) == 2  # n epochs
            assert all(responses[name][qid])  # non-empty completions


def test_load_responses_falls_back_to_analysis_json(tmp_path):
    """A run whose logs are gone (store pruned the generations the run's logs/
    symlinks point at) can still be re-judged: the responses live verbatim in
    analysis.json."""
    import json

    import pytest

    run_dir = tmp_path / "run"
    (run_dir / "logs" / "mock-a").mkdir(parents=True)  # present but empty
    (run_dir / "analysis.json").write_text(
        json.dumps(
            {
                "results": [
                    {"model": "mock-a", "question_id": "q1", "responses": ["r1", "r2"]},
                    {"model": "mock-a", "question_id": "q2", "responses": ["r3"]},
                ]
            }
        )
    )
    assert A.load_responses(run_dir) == {"mock-a": {"q1": ["r1", "r2"], "q2": ["r3"]}}

    # same fallback when logs/ is missing entirely
    (run_dir / "logs" / "mock-a").rmdir()
    (run_dir / "logs").rmdir()
    assert A.load_responses(run_dir)["mock-a"]["q1"] == ["r1", "r2"]

    # nothing to fall back to -> explicit error, not a silent 0-bundle pass
    (run_dir / "analysis.json").unlink()
    with pytest.raises(FileNotFoundError):
        A.load_responses(run_dir)


def test_slash_in_model_name_does_not_collapse_models(tmp_path):
    """Regression: two specs whose names share a leading path component via a slash
    (what a bare ``ours/<x>`` CLI arg produced) must stay distinct — not collapse
    into one ``ours`` bucket, silently dropping all but the last. Guards the
    generate-side dir sanitisation and the load_responses stem keying together."""
    specs = [
        ModelSpec(name="ours/alpha", inspect_model="mockllm/model"),
        ModelSpec(name="ours/beta", inspect_model="mockllm/model"),
    ]
    run_dir = tmp_path / "run"
    G.run_generation(specs, _QS, n=2, run_dir=run_dir, display="none")

    responses = A.load_responses(run_dir)
    assert set(responses) == {"ours_alpha", "ours_beta"}  # distinct, sanitised
    for label in ("ours_alpha", "ours_beta"):
        assert set(responses[label]) == {"q1", "q2"}
        assert all(len(responses[label][qid]) == 2 for qid in ("q1", "q2"))


def test_generation_usage_counts_each_model_once(tmp_path):
    """The .json sibling must not double-count tokens: list_eval_logs returns only
    the .eval, so generation_usage sees exactly one log per model dir."""
    specs = [ModelSpec(name="mock-a", inspect_model="mockllm/model")]
    run_dir = tmp_path / "run"
    G.run_generation(specs, _QS, n=2, run_dir=run_dir, display="none")

    usage = C.generation_usage(run_dir)
    assert set(usage) == {"mock-a"}
    assert usage["mock-a"]["in_tok"] >= 0 and usage["mock-a"]["out_tok"] >= 0
