"""Per-model store tests: gen-key sensitivity, reuse hit/miss, identity guard,
fragment staleness, run symlinks, and cached-fragment assembly.

Generation is exercised with the offline ``mockllm/model`` (the
test_variance_generate convention); the judge/analyze step is stubbed where a
test only cares about caching mechanics.
"""

import json
from pathlib import Path

import pytest

from twominds import analyze as analyze_mod
from twominds import store as S
from twominds.models import ModelSpec
from twominds.questions import Question

_QS = [
    Question(id="q1", group="values", prompt="Say A.", bucket="tier_1"),
    Question(id="q2", group="values", prompt="Say B.", bucket="tier_1"),
]
_SPEC = ModelSpec(name="mock-a", inspect_model="mockllm/model", display="mock-a")


def _key(qs=_QS, n=2, temperature=1.0, max_tokens=64):
    return S.compute_gen_key(qs, n=n, temperature=temperature, max_tokens=max_tokens)


# --------------------------------------------------------------------------- #
# gen_key
# --------------------------------------------------------------------------- #
def test_gen_key_stable_and_readable():
    assert _key() == _key()
    assert _key().endswith("_q2_n2")


def test_gen_key_sensitive_to_prompt_and_sampling():
    edited = [
        Question(id="q1", group="values", prompt="Say A!", bucket="tier_1"),
        _QS[1],
    ]
    assert _key(edited) != _key()
    assert _key(n=3) != _key()
    assert _key(temperature=0.7) != _key()
    assert _key(max_tokens=128) != _key()


def test_gen_key_insensitive_to_question_order():
    assert _key(list(reversed(_QS))) == _key()


# --------------------------------------------------------------------------- #
# reuse / identity
# --------------------------------------------------------------------------- #
def test_find_generation_miss_then_hit(tmp_path):
    root = tmp_path / "models"
    key = _key()
    assert S.find_generation(_SPEC, key, root) is None

    d = S.prepare_generation(
        _SPEC, _QS, key, root, n=2, temperature=1.0, max_tokens=64, judge="j"
    )
    # manifest written but no logs / not marked complete -> still a miss
    assert S.find_generation(_SPEC, key, root) is None
    S.mark_complete(d, key=key)
    assert S.find_generation(_SPEC, key, root) is None  # no .eval yet

    logdir = d / "logs" / _SPEC.name
    logdir.mkdir(parents=True)
    (logdir / "2026-01-01T00-00-00_a.eval").write_bytes(b"x")
    assert S.find_generation(_SPEC, key, root) == d


def test_gen_dir_is_a_mini_run_manifest(tmp_path):
    root = tmp_path / "models"
    d = S.prepare_generation(
        _SPEC, _QS, _key(), root, n=2, temperature=1.0, max_tokens=64, judge="j"
    )
    cfg = json.loads((d / "run_config.json").read_text())
    assert list(cfg["models"]) == ["mock-a"]
    assert cfg["models"]["mock-a"]["display"] == "mock-a"
    assert set(json.loads((d / "questions.json").read_text())) == {"q1", "q2"}


def test_identity_guard_rejects_short_name_collision(tmp_path):
    root = tmp_path / "models"
    S.check_model_identity(_SPEC, root)
    S.check_model_identity(_SPEC, root)  # same identity: fine
    other = ModelSpec(name="mock-a", inspect_model="openrouter/other/mock-a")
    with pytest.raises(ValueError, match="collision"):
        S.check_model_identity(other, root)


def test_identity_conflict_is_readonly(tmp_path):
    root = tmp_path / "models"
    # no store entry yet: no conflict, and crucially nothing written (dry runs)
    assert S.identity_conflict(_SPEC, root) is None
    assert not root.exists()

    S.write_identity(_SPEC, root)
    assert S.identity_conflict(_SPEC, root) is None  # same model
    other = ModelSpec(name="mock-a", inspect_model="openrouter/other/mock-a")
    prev = S.identity_conflict(other, root)
    assert prev is not None and prev["inspect_model"] == "mockllm/model"


# --------------------------------------------------------------------------- #
# judge fragments
# --------------------------------------------------------------------------- #
def test_judge_key_sensitivity():
    base = dict(
        judge_name="openrouter/anthropic/claude-opus-4.8",
        judge_reasoning="low",
        threshold=0.15,
        backends=["local"],
        local_model="BAAI/bge-small-en-v1.5",
    )
    k = S.compute_judge_key(**base)
    assert k.startswith("claude-opus-4.8_")
    assert S.compute_judge_key(**{**base, "threshold": 0.2}) != k
    assert S.compute_judge_key(**{**base, "judge_reasoning": "high"}) != k
    nk = S.compute_judge_key(**base, run_judge=False)
    assert nk.startswith("nojudge_") and nk != k


def test_fragment_roundtrip_and_rerun_staleness(tmp_path):
    root = tmp_path / "models"
    key = _key()
    d = S.prepare_generation(
        _SPEC, _QS, key, root, n=2, temperature=1.0, max_tokens=64, judge="j"
    )
    logdir = d / "logs" / _SPEC.name
    logdir.mkdir(parents=True)
    (logdir / "2026-01-01T00-00-00_a.eval").write_bytes(b"x")

    jk = "judge_abc12345"
    fd = S.fragment_dir(d, jk)
    fd.mkdir(parents=True)
    (fd / "analysis.json").write_text(json.dumps({"models": ["mock-a"]}))
    S.write_fragment_meta(d, _SPEC.name, jk)
    assert S.find_fragment(d, _SPEC.name, jk) == {"models": ["mock-a"]}
    assert S.find_fragment(d, _SPEC.name, "judge_other") is None

    # a rerun appends a newer .eval -> the fragment goes stale automatically
    (logdir / "2026-01-02T00-00-00_b.eval").write_bytes(b"y")
    assert S.find_fragment(d, _SPEC.name, jk) is None
    S.write_fragment_meta(d, _SPEC.name, jk)  # re-judged: fresh again
    assert S.find_fragment(d, _SPEC.name, jk) is not None


# --------------------------------------------------------------------------- #
# run symlinks
# --------------------------------------------------------------------------- #
def test_link_into_run_symlink_and_refusal(tmp_path):
    gen = tmp_path / "models" / "mock-a" / "gens" / "k"
    (gen / "logs" / "mock-a").mkdir(parents=True)
    (gen / "logs" / "mock-a" / "x.eval").write_bytes(b"x")
    run_dir = tmp_path / "run"

    S.link_into_run(run_dir, _SPEC, gen)
    dest = run_dir / "logs" / "mock-a"
    assert dest.is_symlink() and (dest / "x.eval").exists()
    S.link_into_run(run_dir, _SPEC, gen)  # idempotent (replaces the link)
    assert dest.is_symlink()

    real = run_dir / "logs" / "mock-b"
    real.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        S.link_into_run(run_dir, ModelSpec(name="mock-b", inspect_model="m"), gen)


def test_generation_into_store_readable_through_run_symlink(tmp_path):
    """Real offline mockllm generation into a store gen dir, then read back
    through a run-dir symlink by analyze.load_responses."""
    from twominds import generate as G

    root = tmp_path / "models"
    key = _key(n=1)
    d = S.prepare_generation(
        _SPEC, _QS, key, root, n=1, temperature=1.0, max_tokens=64, judge="j"
    )
    G.run_generation(
        [_SPEC],
        _QS,
        n=1,
        display="none",
        log_dirs={_SPEC.name: d / "logs" / _SPEC.name},
        on_model_done=lambda name: S.mark_complete(d, key=key),
    )
    assert S.find_generation(_SPEC, key, root) == d

    run_dir = tmp_path / "run"
    S.link_into_run(run_dir, _SPEC, d)
    responses = analyze_mod.load_responses(run_dir)
    assert set(responses) == {"mock-a"}
    assert set(responses["mock-a"]) == {"q1", "q2"}


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def _fake_fragment(name):
    return {
        "run_dir": "gen",
        "judge_run": None,
        "backends": ["local"],
        "primary_backend": "local",
        "judge": None,
        "judge_reasoning": None,
        "threshold": 0.15,
        "models": [name],
        "model_display": {name: f"display/{name}"},
        "questions": {"q1": {"prompt": "Say A.", "group": "values"}},
        "families_meta": {},
        "results": [
            {
                "model": name,
                "question_id": "q1",
                "group": "values",
                "responses": ["a"],
                "judge": None,
                "judge_labels": None,
                "clusters": {},
                "agreement": {},
                "metrics": {"n": 1},
            }
        ],
        "families": [],
        "cost": {},
    }


def test_assemble_run_uses_cached_fragment_and_judges_missing(tmp_path, monkeypatch):
    root = tmp_path / "models"
    key = _key()
    jk = "judge_abc12345"
    specs = [
        ModelSpec(name="mock-a", inspect_model="mockllm/model"),
        ModelSpec(name="mock-b", inspect_model="mockllm/model"),
    ]
    gen_dirs = {}
    for spec in specs:
        d = S.prepare_generation(
            spec, _QS, key, root, n=2, temperature=1.0, max_tokens=64, judge="j"
        )
        logdir = d / "logs" / spec.name
        logdir.mkdir(parents=True)
        (logdir / "2026-01-01T00-00-00_a.eval").write_bytes(b"x")
        gen_dirs[spec.name] = d

    # mock-a already has a fresh fragment; mock-b must be judged
    fd = S.fragment_dir(gen_dirs["mock-a"], jk)
    fd.mkdir(parents=True)
    (fd / "analysis.json").write_text(json.dumps(_fake_fragment("mock-a")))
    S.write_fragment_meta(gen_dirs["mock-a"], "mock-a", jk)

    judged = []

    def fake_analyze(run_dir, *, models=None, out_dir=None, **kw):
        judged.append(models[0])
        a = _fake_fragment(models[0])
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "analysis.json").write_text(json.dumps(a))
        return a

    monkeypatch.setattr(analyze_mod, "analyze", fake_analyze)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    events = []
    combined = S.assemble_run(
        run_dir,
        specs,
        gen_dirs,
        judge_key=jk,
        backends=["local"],
        judge_name="j",
        judge_reasoning=None,
        threshold=0.15,
        local_model="lm",
        concurrency=1,
        run_judge=False,
        on_fragment=lambda name, cached: events.append((name, cached)),
    )
    assert judged == ["mock-b"]  # cached fragment skipped the judge
    assert events == [("mock-a", True), ("mock-b", False)]
    assert sorted(combined["models"]) == ["mock-a", "mock-b"]
    assert combined["model_display"]["mock-b"] == "display/mock-b"
    assert json.loads((run_dir / "analysis.json").read_text())["models"] == [
        "mock-a",
        "mock-b",
    ]

    # run-level judge provenance marker restored (registry reads it)
    assert (run_dir / "judge_meta.json").exists()

    # second assembly: everything cached, no judge calls at all
    judged.clear()
    S.assemble_run(
        run_dir,
        specs,
        gen_dirs,
        judge_key=jk,
        backends=["local"],
        judge_name="j",
        judge_reasoning=None,
        threshold=0.15,
        local_model="lm",
        concurrency=1,
        run_judge=False,
    )
    assert judged == []


def test_build_run_families_report(tmp_path):
    """A merged analysis with framing families gets families_report.html next
    to report.html (the fragment copies live in the store, out of link reach)."""
    analysis = {
        "run_dir": str(tmp_path),
        "judge": "judge/x",
        "families": [
            {
                "model": "gpt-4o",
                "family": "poem_rating",
                "title": "Poem rating",
                "description": "desc",
                "scalar_kind": "number",
                "variants": [
                    {"variant": "mine_love", "question_id": "q_hi", "n": 2},
                    {"variant": "neutral", "question_id": "q_lo", "n": 2},
                ],
                "n_total": 4,
                "scalar": {
                    "kind": "number",
                    "per_variant": {
                        "mine_love": {"mean": 9.0},
                        "neutral": {"mean": 3.0},
                    },
                    "swing": 6.0,
                },
                "judge": None,
                "cluster": None,
            }
        ],
        "results": [
            {"model": "gpt-4o", "question_id": "q_hi", "responses": ["9"] * 2},
            {"model": "gpt-4o", "question_id": "q_lo", "responses": ["3"] * 2},
        ],
    }
    S.build_run_families_report(tmp_path, analysis)
    assert (tmp_path / "families_report.html").exists()
    S.build_run_families_report(tmp_path / "other", {"families": []})
    assert not (tmp_path / "other").exists()  # no families -> no report, no dir


def test_preseed_run_cache(tmp_path):
    """Fragment embedding caches stack into a run-level cache whose hash
    matches what analyze._embed_all would compute over the whole run."""
    np = pytest.importorskip("numpy")

    def frag(name, texts, dims=3):
        gd = tmp_path / "models" / name / "gens" / "k"
        (gd / "cache").mkdir(parents=True)
        mat = np.arange(len(texts) * dims, dtype=np.float32).reshape(len(texts), dims)
        np.savez(
            gd / "cache" / "emb_local.npz",
            mat=mat,
            hash=np.array(S.content_hash(texts)),
        )
        return {
            "run_dir": str(gd),
            "models": [name],
            "results": [{"model": name, "question_id": "q1", "responses": texts}],
        }

    fa = frag("a-model", ["r1", "r2"])
    fb = frag("b-model", ["r3"])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # frags passed out of order: preseed must sort by model name (analyze order)
    S.preseed_run_cache(run_dir, [fb, fa], ["local"])
    d = np.load(run_dir / "cache" / "emb_local.npz")
    assert d["mat"].shape == (3, 3)
    assert str(d["hash"].item()) == S.content_hash(["r1", "r2", "r3"])

    # a stale fragment cache (hash mismatch) skips the preseed instead of
    # seeding a wrong run-level cache
    np.savez(
        tmp_path / "models" / "a-model" / "gens" / "k" / "cache" / "emb_local.npz",
        mat=np.zeros((2, 3), dtype=np.float32),
        hash=np.array("stale"),
    )
    run2 = tmp_path / "run2"
    run2.mkdir()
    S.preseed_run_cache(run2, [fa, fb], ["local"])
    assert not (run2 / "cache" / "emb_local.npz").exists()
