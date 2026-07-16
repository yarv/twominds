"""CLI tests: the command surface via typer's CliRunner (keyless --dry-run
paths), plus the store-planning helpers (cross-invocation name
qualification, --rerun-model tokens, backend validation)."""

import json
import re

import pytest
import typer
from typer.testing import CliRunner

from twominds import cli as VE
from twominds import questions as questions_mod
from twominds import store as S
from twominds.cli import _options as cli_options
from twominds.cli import _orchestrate as cli_orchestrate
from twominds.models import DEFAULT_JUDGE, resolve_models
from twominds.questions import Question

_QS = [Question(id="q1", group="values", prompt="Say A.", bucket="tier_1")]

runner = CliRunner()


@pytest.fixture
def results_root(tmp_path, monkeypatch):
    root = tmp_path / "variance"
    monkeypatch.setattr(cli_options, "_RESULTS_ROOT", root)
    return root


@pytest.fixture
def keyless(results_root, tmp_path, monkeypatch):
    """Dry-run environment: no API keys, cwd and results root in tmp."""
    for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    return results_root


def _out(result):
    """stdout + stderr, normalized past typer's rich error rendering (ANSI
    codes and box-drawing stripped, wrapped lines rejoined) so message asserts
    hold locally and on CI, which colorizes."""
    try:
        text = result.output + result.stderr
    except (AttributeError, ValueError):
        text = result.output
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return " ".join(re.sub(r"[│╭╮╰╯─]", " ", text).split())


def _selected(**kw):
    return questions_mod.select_questions(**kw)


# --------------------------------------------------------------------------- #
# run / generate --dry-run
# --------------------------------------------------------------------------- #
def test_run_dry_run_plans_without_touching_anything(keyless):
    result = runner.invoke(
        VE.app,
        ["run", "--dry-run", "--groups", "values", "--models", "gpt-4.1", "--n", "3"],
    )
    assert result.exit_code == 0, _out(result)
    assert "Variance sweep plan" in result.output
    assert "would generate 1 model(s)" in result.output
    assert "(dry run — no API calls made)" in result.output
    assert not keyless.exists()


def test_run_dry_run_notes_missing_keys(keyless):
    result = runner.invoke(
        VE.app,
        ["run", "--dry-run", "--groups", "values", "--models", "gpt-4.1", "--n", "3"],
    )
    assert "note: OPENAI_API_KEY is not set" in result.output
    assert "note: OPENROUTER_API_KEY is not set" in result.output  # default judge


def test_generate_dry_run_omits_judge_cost(keyless):
    result = runner.invoke(
        VE.app,
        [
            "generate",
            "--dry-run",
            "--groups",
            "values",
            "--models",
            "gpt-4.1",
            "--n",
            "2",
        ],
    )
    assert result.exit_code == 0, _out(result)
    assert "Variance sweep plan" in result.output
    assert "judge:" not in result.output


def test_run_dry_run_reports_cached_models(keyless):
    qs = _selected(groups=["values"])
    root = S.store_root(cli_options._RESULTS_ROOT)
    key = S.compute_gen_key(qs, n=3, temperature=1.0, max_tokens=2048)
    spec = resolve_models(["gpt-4.1"])[0]
    S.write_identity(spec, root)
    d = S.prepare_generation(
        spec, qs, key, root, n=3, temperature=1.0, max_tokens=2048, judge=DEFAULT_JUDGE
    )
    log_dir = d / "logs" / spec.name
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{spec.name}.eval").write_bytes(b"")  # find_generation globs for it
    S.mark_complete(d, key=key)

    result = runner.invoke(
        VE.app,
        ["run", "--dry-run", "--groups", "values", "--models", "gpt-4.1", "--n", "3"],
    )
    assert result.exit_code == 0, _out(result)
    assert "reusing 1 cached generation(s)" in result.output
    assert "gpt-4.1 (cached)" in result.output
    assert "covered by cache" in result.output


def test_judge_only_run_plan_says_no_embeddings(keyless):
    result = runner.invoke(
        VE.app,
        [
            "run",
            "--dry-run",
            "-b",
            "none",
            "--groups",
            "values",
            "--models",
            "gpt-4.1",
        ],
    )
    assert result.exit_code == 0, _out(result)
    assert "embeddings: none (judge-only analysis)" in result.output


# --------------------------------------------------------------------------- #
# question selection flags
# --------------------------------------------------------------------------- #
def test_folders_flag_selects_bucket(keyless):
    n_tier2 = len(_selected(buckets=["tier_2"]))
    result = runner.invoke(
        VE.app, ["run", "--dry-run", "--folders", "tier_2", "--models", "gpt-4.1"]
    )
    assert result.exit_code == 0, _out(result)
    assert f"1 models x {n_tier2} questions" in result.output


def test_all_questions_selects_every_bucket(keyless):
    n_all = len(_selected(buckets=list(questions_mod.BUCKETS)))
    result = runner.invoke(
        VE.app, ["run", "--dry-run", "--all-questions", "--models", "gpt-4.1"]
    )
    assert result.exit_code == 0, _out(result)
    assert f"1 models x {n_all} questions" in result.output


def test_families_flag_selects_variants(keyless):
    fam_qs = _selected(families=["poem_rating"])
    assert len(fam_qs) >= 2
    result = runner.invoke(
        VE.app,
        ["run", "--dry-run", "--families", "poem_rating", "--models", "gpt-4.1"],
    )
    assert result.exit_code == 0, _out(result)
    assert f"1 models x {len(fam_qs)} questions" in result.output
    for q in fam_qs:
        assert q.id in result.output


def test_ids_flag_selects_exactly(keyless):
    qid = _selected(buckets=["tier_1"])[0].id
    result = runner.invoke(
        VE.app, ["run", "--dry-run", "--ids", qid, "--models", "gpt-4.1"]
    )
    assert result.exit_code == 0, _out(result)
    assert "1 models x 1 questions" in result.output
    assert qid in result.output


def test_unknown_group_fails(keyless):
    result = runner.invoke(
        VE.app,
        ["run", "--dry-run", "--groups", "no_such_group", "--models", "gpt-4.1"],
    )
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# analyze --dry-run
# --------------------------------------------------------------------------- #
def _write_run_fixture(run_dir, n=3):
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "n": n,
                "models": {
                    "gpt-4.1": {"inspect_model": "openai/gpt-4.1", "display": "GPT-4.1"}
                },
            }
        )
    )
    (run_dir / "questions.json").write_text(
        json.dumps(
            {
                "q1": {"group": "values", "prompt": "Say A."},
                "q2": {"group": "values", "prompt": "Say B."},
            }
        )
    )


def test_analyze_dry_run_plans_from_manifests(keyless, tmp_path):
    run_dir = tmp_path / "run1"
    _write_run_fixture(run_dir)
    result = runner.invoke(VE.app, ["analyze", "-r", str(run_dir), "--dry-run"])
    assert result.exit_code == 0, _out(result)
    assert "Analyze plan" in result.output
    assert "1 model(s) x 2 questions x N=3" in result.output
    assert "judge:" in result.output
    assert "standalone analyze always judges fresh" in result.output
    assert "(dry run — no API calls made)" in result.output


def test_analyze_dry_run_reps_multiplies_cost(keyless, tmp_path):
    run_dir = tmp_path / "run1"
    _write_run_fixture(run_dir)
    result = runner.invoke(
        VE.app, ["analyze", "-r", str(run_dir), "--dry-run", "--reps", "3"]
    )
    assert result.exit_code == 0, _out(result)
    assert "x 3 reps" in result.output


def test_analyze_dry_run_rejects_non_run_dir(keyless, tmp_path):
    result = runner.invoke(
        VE.app, ["analyze", "-r", str(tmp_path / "nope"), "--dry-run"]
    )
    assert result.exit_code != 0
    assert "not a generated run dir" in _out(result)


# --------------------------------------------------------------------------- #
# option validation at the command level
# --------------------------------------------------------------------------- #
def test_no_judge_with_no_embeddings_rejected(keyless):
    result = runner.invoke(
        VE.app,
        [
            "run",
            "--dry-run",
            "-b",
            "none",
            "--no-judge",
            "--groups",
            "values",
            "--models",
            "gpt-4.1",
        ],
    )
    assert result.exit_code != 0
    assert "leaves nothing to analyze" in _out(result)


def test_unknown_backend_rejected_at_cli(keyless):
    result = runner.invoke(
        VE.app,
        [
            "run",
            "--dry-run",
            "-b",
            "bogus",
            "--groups",
            "values",
            "--models",
            "gpt-4.1",
        ],
    )
    assert result.exit_code != 0
    assert "unknown embedding backend" in _out(result)


# --------------------------------------------------------------------------- #
# main() error wrapper
# --------------------------------------------------------------------------- #
def test_main_prints_clean_error(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("TWOMINDS_DEBUG", raising=False)
    monkeypatch.setattr(
        "sys.argv", ["twominds", "report", "-r", str(tmp_path / "missing")]
    )
    with pytest.raises(SystemExit) as exc:
        VE.main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "TWOMINDS_DEBUG=1" in err


def test_main_reraises_with_debug_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TWOMINDS_DEBUG", "1")
    monkeypatch.setattr(
        "sys.argv", ["twominds", "report", "-r", str(tmp_path / "missing")]
    )
    with pytest.raises((FileNotFoundError, RuntimeError)):
        VE.main()


# --------------------------------------------------------------------------- #
# store-planning helpers
# --------------------------------------------------------------------------- #
def _plan(specs, **kw):
    kw.setdefault("n", 2)
    kw.setdefault("temperature", 1.0)
    kw.setdefault("max_tokens", 64)
    kw.setdefault("rerun", False)
    kw.setdefault("rerun_models", None)
    return cli_orchestrate._plan_generations(specs, _QS, **kw)


def test_cross_invocation_collision_auto_qualifies(results_root):
    # invocation 1 pins the short name "foo" to one provider's model
    first = resolve_models(["openrouter/acme/foo"])
    _plan(first)
    assert first[0].name == "foo"

    # invocation 2: same last segment, different model -> auto-qualified,
    # not a dead-end error
    second = resolve_models(["together/other/foo"])
    _plan(second)
    assert second[0].name == "other_foo"  # mutated in place for the caller
    ident = S.identity_conflict(second[0], S.store_root(results_root))
    assert ident is None  # pinned under the qualified name


def test_dry_run_leaves_store_untouched(results_root):
    specs = resolve_models(["openrouter/acme/foo"])
    _plan(specs, dry_run=True)
    assert not S.store_root(results_root).exists()


def test_rerun_model_accepts_models_token(results_root):
    # '4o' is a roster alias: resolves to name 'gpt-4o' / display 'GPT-4o'.
    specs = resolve_models(["4o"])
    _, _, cached, to_generate = _plan(specs, rerun_models=["4o"], raw_names=["4o"])
    assert [s.name for s in to_generate] == ["gpt-4o"]  # forced, so not cached
    assert cached == []


def test_rerun_model_unknown_still_rejected(results_root):
    specs = resolve_models(["4o"])
    with pytest.raises(typer.BadParameter, match="not among"):
        _plan(specs, rerun_models=["gpt-5.2"], raw_names=["4o"])


def test_resolve_backends_none_and_validation():
    assert cli_options._resolve_backends(["none"]) == []
    assert cli_options._resolve_backends(["local"]) == ["local"]
    assert cli_options._resolve_backends(["local", "openai-3-small"]) == [
        "local",
        "openai-3-small",
    ]
    with pytest.raises(typer.BadParameter, match="cannot be combined"):
        cli_options._resolve_backends(["none", "local"])
    with pytest.raises(typer.BadParameter, match="unknown embedding backend"):
        cli_options._resolve_backends(["bogus"])
