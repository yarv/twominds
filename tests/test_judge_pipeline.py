"""Background judge-fragment prewarming (offline: mockllm, no network)."""

import json

from twominds import generate as G
from twominds import judge_pipeline as JP
from twominds import store as S
from twominds.analyze import load_responses
from twominds.models import ModelSpec
from twominds.questions import Question

_QS = [
    Question(id="q1", group="values", prompt="Say something."),
    Question(id="q2", group="values", prompt="Say something else."),
]

_JUDGE_CFG = {
    "judge_key": "mockjudge_test0000",
    "backends": [],
    "judge_name": "mockllm/model",
    "judge_reasoning": None,
    "threshold": 0.15,
    "local_model": "lm",
    "concurrency": 2,
    "run_judge": True,
}


def _gen_dir(tmp_path, spec):
    root = tmp_path / "models"
    key = S.compute_gen_key(_QS, n=2, temperature=1.0, max_tokens=64)
    d = S.prepare_generation(
        spec, _QS, key, root, n=2, temperature=1.0, max_tokens=64, judge="j"
    )
    return d


def test_worker_main_builds_fresh_fragment(tmp_path):
    spec = ModelSpec(name="mock-a", inspect_model="mockllm/model")
    gd = _gen_dir(tmp_path, spec)
    G.run_generation(
        [spec], _QS, n=2, display="none",
        log_dirs={spec.name: gd / "logs" / spec.name},
    )
    JP._worker_main({"model": spec.name, "gen_dir": str(gd), **_JUDGE_CFG})
    frag = S.find_fragment(gd, spec.name, _JUDGE_CFG["judge_key"])
    assert frag is not None  # fragment exists and is fresh vs the latest log
    assert frag["models"] == [spec.name]
    recs = frag["results"]
    assert {r["question_id"] for r in recs} == {"q1", "q2"}
    assert all(r["judge"] is not None for r in recs)


def test_hook_writes_logs_and_generation_skips_rewrite(tmp_path, monkeypatch):
    specs = [
        ModelSpec(name="mock-a", inspect_model="mockllm/model"),
        ModelSpec(name="mock-b", inspect_model="mockllm/model"),
    ]
    gen_dirs = {s.name: _gen_dir(tmp_path, s) for s in specs}
    log_dirs = {s.name: gen_dirs[s.name] / "logs" / s.name for s in specs}

    submitted = []
    monkeypatch.setattr(
        JP.FragmentPrewarmer, "_run_worker", lambda self, m: submitted.append(m)
    )
    pw = JP.activate(
        {s.name: {"gen_dir": gen_dirs[s.name], "log_dir": log_dirs[s.name]} for s in specs},
        _JUDGE_CFG,
    )
    try:
        G.run_generation(
            specs, _QS, n=2, display="none",
            log_dirs=log_dirs, skip_log_write=pw.log_written,
        )
        mtimes = {s.name: (log_dirs[s.name] / f"{s.name}.eval").stat().st_mtime_ns
                  for s in specs}
    finally:
        JP.deactivate()
    assert sorted(submitted) == ["mock-a", "mock-b"]
    assert pw.written == {"mock-a", "mock-b"}
    # the hook's atomic write was the only write (run_generation skipped its own)
    for s in specs:
        f = log_dirs[s.name] / f"{s.name}.eval"
        assert f.stat().st_mtime_ns == mtimes[s.name]
        assert not (log_dirs[s.name] / f".{s.name}.eval.tmp").exists()
    # and the hook-written logs are what analyze reads
    resp = load_responses(gen_dirs["mock-a"])
    assert set(resp["mock-a"]) == {"q1", "q2"}


def test_prewarm_end_to_end_subprocess(tmp_path):
    """Real subprocess: fragment is judged in the background during the eval."""
    spec = ModelSpec(name="mock-a", inspect_model="mockllm/model")
    gd = _gen_dir(tmp_path, spec)
    log_dir = gd / "logs" / spec.name
    JP.activate({spec.name: {"gen_dir": gd, "log_dir": log_dir}}, _JUDGE_CFG)
    try:
        G.run_generation(
            [spec], _QS, n=2, display="none",
            log_dirs={spec.name: log_dir},
            skip_log_write=JP._ACTIVE.log_written,
        )
    finally:
        prewarmed = JP.deactivate()
    assert prewarmed == {spec.name}
    frag = S.find_fragment(gd, spec.name, _JUDGE_CFG["judge_key"])
    assert frag is not None and frag["models"] == [spec.name]
    # worker output captured for debugging
    assert (gd / "judge" / "prewarm-mock-a.log").exists()


def test_hook_inert_without_activation(tmp_path):
    # no activate() -> generation behaves exactly as before (hook is a no-op)
    spec = ModelSpec(name="mock-a", inspect_model="mockllm/model")
    run_dir = tmp_path / "run"
    G.run_generation([spec], _QS, n=2, run_dir=run_dir, display="none")
    assert (run_dir / "logs" / "mock-a" / "mock-a.eval").exists()
    assert JP._ACTIVE is None


def test_deactivate_returns_empty_when_never_activated():
    assert JP.deactivate() == set()
