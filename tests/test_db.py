"""Database schema, keys, and importer — pure logic, tmp-file SQLite."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import func, inspect, select
from typer.testing import CliRunner

from twominds.cli import app
from twominds.db import init_db, keys, make_engine, session
from twominds.db.importer import import_paths
from twominds.db.schema import (
    Bundle,
    BundleResponse,
    FamilyAnalysis,
    GenBatch,
    Judgment,
    Model,
    Response,
    Run,
    RunBundle,
    RunJudgment,
)


@pytest.fixture
def db_path(tmp_path):
    return init_db(tmp_path / "test.db")


def _question_meta(qid):
    return {
        "prompt": f"Prompt {qid}",
        "group": "values",
        "bucket": "tier_1",
        "system": None,
        "family": None,
        "variant": None,
    }


def make_analysis(
    models=("m1",), qids=("q1", "q2"), n=3, rep=None, config=True, salt=""
):
    results = [
        {
            "model": m,
            "question_id": q,
            "group": "values",
            "responses": [f"resp {m} {q} {i}" for i in range(n)],
            "judge": {
                "contradiction": False,
                "groups": [list(range(n))],
                "n_groups": 1,
                "rationale": f"agrees{salt}",
                "flags": [],
                "parse_ok": True,
            },
            "judge_labels": [0] * n,
            "clusters": {"openai-3-small": {"labels": [0] * n, "n_clusters": 1}},
            "agreement": {"openai-3-small": {"ari": 1.0, "nmi": 1.0}},
            "metrics": {"n": n, "len_mean": 10.0, "group_entropy": 0.0},
        }
        for m in models
        for q in qids
    ]
    analysis = {
        "run_dir": "x",
        "judge_run": rep,
        "backends": ["openai-3-small"],
        "judge": "openrouter/anthropic/judge",
        "judge_reasoning": "low",
        "threshold": 0.15,
        "models": list(models),
        "questions": {q: _question_meta(q) for q in qids},
        "families_meta": {},
        "results": results,
        "families": [],
    }
    if config:
        analysis["config"] = {
            "models": {
                m: {
                    "inspect_model": f"openai/{m}",
                    "reasoning_effort": None,
                    "display": m,
                }
                for m in models
            },
            "n": n,
            "temperature": 1.0,
            "max_tokens": 128,
            "judge": "openrouter/anthropic/judge",
        }
    return analysis


def write_run_dir(root, analysis, name="run1"):
    d = root / name
    d.mkdir(parents=True)
    (d / "analysis.json").write_text(json.dumps(analysis))
    if "config" in analysis:
        (d / "run_config.json").write_text(json.dumps(analysis["config"]))
    (d / "run_meta.json").write_text(
        json.dumps({"kind": "variance", "created_at": "2026-01-01T00:00:00+00:00"})
    )
    return d


def write_gen_dir(root, analysis, model="m1"):
    d = root / "models" / model / "gens" / "abc_q2_n3"
    d.mkdir(parents=True)
    (d / "gen_meta.json").write_text(
        json.dumps({"status": "complete", "gen_key": "abc"})
    )
    (d / "run_config.json").write_text(json.dumps(analysis["config"]))
    (d / "questions.json").write_text(json.dumps(analysis["questions"]))
    frag = d / "judge" / "judge_abc"
    frag.mkdir(parents=True)
    (frag / "analysis.json").write_text(json.dumps(analysis))
    return d


def _counts(db_path):
    with session(db_path) as sess:
        return {
            t.__tablename__: sess.scalar(select(func.count()).select_from(t))
            for t in (
                Bundle,
                BundleResponse,
                Response,
                Judgment,
                GenBatch,
                Run,
                RunBundle,
                RunJudgment,
            )
        }


def test_init_creates_schema(db_path):
    engine = make_engine(db_path)
    tables = set(inspect(engine).get_table_names())
    engine.dispose()
    assert {"bundle", "response", "judgment", "run", "alembic_version"} <= tables


def test_question_hash_tracks_content():
    meta = _question_meta("q1")
    assert keys.question_content_hash(meta) == keys.question_content_hash(dict(meta))
    assert keys.question_content_hash(meta) != keys.question_content_hash(
        {**meta, "prompt": "edited"}
    )
    assert keys.question_content_hash(meta) == keys.question_content_hash(
        {**meta, "source": "import-only extra"}
    )


def test_import_run_dir_and_idempotency(tmp_path, db_path):
    write_run_dir(tmp_path / "results", make_analysis(models=("m1", "m2")))
    with session(db_path) as sess:
        stats = import_paths(sess, [tmp_path / "results"])
    assert not stats.warnings
    first = _counts(db_path)
    assert first["bundle"] == 4
    assert first["response"] == 12
    assert first["judgment"] == 4
    assert first["run"] == 1
    assert first["run_bundle"] == 4
    assert first["run_judgment"] == 4

    with session(db_path) as sess:
        import_paths(sess, [tmp_path / "results"])
    assert _counts(db_path) == first


def test_rep_pass_adds_judgments_not_responses(tmp_path, db_path):
    analysis = make_analysis()
    run_dir = write_run_dir(tmp_path / "results", analysis)
    rep = make_analysis(rep="rep2", config=False, salt=" on second thought")
    rep_dir = run_dir / "judge_runs" / "rep2"
    rep_dir.mkdir(parents=True)
    (rep_dir / "analysis.json").write_text(json.dumps(rep))

    with session(db_path) as sess:
        import_paths(sess, [tmp_path / "results"])
    counts = _counts(db_path)
    assert counts["bundle"] == 2
    assert counts["response"] == 6
    assert counts["judgment"] == 4
    with session(db_path) as sess:
        labels = set(sess.scalars(select(Judgment.rep_label)))
    assert labels == {"default", "rep2"}


def test_store_fragment_and_run_share_rows(tmp_path, db_path):
    analysis = make_analysis()
    write_gen_dir(tmp_path / "results", analysis)
    write_run_dir(tmp_path / "results", analysis)

    with session(db_path) as sess:
        import_paths(sess, [tmp_path / "results"])
    counts = _counts(db_path)
    assert counts["bundle"] == 2
    assert counts["response"] == 6
    assert counts["judgment"] == 2
    assert counts["gen_batch"] == 1
    assert counts["run"] == 1
    assert counts["run_bundle"] == 2
    with session(db_path) as sess:
        batch = sess.scalars(select(GenBatch)).one()
        assert batch.source_path.endswith("abc_q2_n3")


def test_same_gen_key_different_content_stays_separate(tmp_path, db_path):
    first = make_analysis()
    second = make_analysis()
    for r in second["results"]:
        r["responses"] = [f"{t} (regenerated)" for t in r["responses"]]
    write_gen_dir(tmp_path / "tree1", first)
    write_gen_dir(tmp_path / "tree2", second)

    with session(db_path) as sess:
        import_paths(sess, [tmp_path / "tree1", tmp_path / "tree2"])
    counts = _counts(db_path)
    assert counts["gen_batch"] == 2
    assert counts["bundle"] == 4
    assert counts["response"] == 12
    with session(db_path) as sess:
        for analysis in (first, second):
            for r in analysis["results"]:
                qh = keys.question_content_hash(analysis["questions"][r["question_id"]])
                dig = keys.bundle_digest("openai/m1", None, qh, r["responses"])
                bundle = sess.scalars(select(Bundle).where(Bundle.digest == dig)).one()
                texts = sess.execute(
                    select(Response.text)
                    .join(BundleResponse, BundleResponse.response_id == Response.id)
                    .where(BundleResponse.bundle_id == bundle.id)
                    .order_by(BundleResponse.position)
                ).scalars()
                assert list(texts) == r["responses"]


def test_legacy_analysis_without_config(tmp_path, db_path):
    analysis = make_analysis(config=False)
    analysis["questions"]["q1"]["source"] = "legacy provenance note"
    write_run_dir(tmp_path / "results", analysis, name="legacy")

    with session(db_path) as sess:
        stats = import_paths(sess, [tmp_path / "results"])
    assert not stats.warnings
    with session(db_path) as sess:
        model = sess.scalars(select(Model)).one()
        assert model.inspect_model is None
        assert sess.scalar(select(func.count()).select_from(Bundle)) == 2


def test_families_import(tmp_path, db_path):
    analysis = make_analysis(qids=("q1",))
    analysis["families_meta"] = {"fam1": {"prompt": "Neutral ask", "scalar": "number"}}
    analysis["families"] = [
        {
            "model": "m1",
            "family": "fam1",
            "scalar_kind": "number",
            "variants": [
                {"variant": "a", "question_id": "q1", "n": 3, "groups": [0, 0, 0]}
            ],
            "n_total": 3,
            "judge": {"ari": 0.1},
        }
    ]
    write_run_dir(tmp_path / "results", analysis)
    with session(db_path) as sess:
        stats = import_paths(sess, [tmp_path / "results"])
    assert not stats.warnings
    with session(db_path) as sess:
        fa = sess.scalars(select(FamilyAnalysis)).one()
        assert fa.payload["family"] == "fam1"
        assert fa.rep_label == "default"


def test_cli_roundtrip(tmp_path):
    runner = CliRunner()
    db = tmp_path / "t.db"
    write_run_dir(tmp_path / "results", make_analysis())

    r = runner.invoke(app, ["db", "init", "--db", str(db)])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["db", "import", str(tmp_path / "results"), "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert "bundle: 2" in r.output
    r = runner.invoke(app, ["db", "stats", "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert "run1" in r.output
