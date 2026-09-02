"""CLI tests: the command surface via typer's CliRunner (keyless --dry-run
paths) and the main() error wrapper."""

import json
import re

import pytest
from typer.testing import CliRunner

from twominds import cli as VE
from twominds import questions as questions_mod
from twominds.cli import _options as cli_options

runner = CliRunner()


@pytest.fixture
def keyless(tmp_path, monkeypatch):
    """Dry-run environment: no API keys, cwd and results root in tmp."""
    for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "results"
    monkeypatch.setattr(cli_options, "_RESULTS_ROOT", root)
    return root


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


# --------------------------------------------------------------------------- #
# run / generate --dry-run
# --------------------------------------------------------------------------- #
def test_run_dry_run_plans_without_touching_anything(keyless):
    result = runner.invoke(
        VE.app,
        ["run", "--dry-run", "--groups", "values", "--models", "gpt-4.1", "--n", "3"],
    )
    assert result.exit_code == 0, _out(result)
    n_values = len(questions_mod.select_questions(groups=["values"]))
    assert "Sweep plan" in result.output
    assert f"1 models x {n_values} questions x N=3" in result.output
    assert "judge:" in result.output
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
            "-n",
            "2",
        ],
    )
    assert result.exit_code == 0, _out(result)
    assert "Sweep plan" in result.output
    assert "judge:" not in result.output


def test_judge_concurrency_flag(keyless):
    result = runner.invoke(
        VE.app,
        ["run", "--dry-run", "--groups", "values", "--models", "gpt-4.1"]
        + ["--judge-concurrency", "4"],
    )
    assert result.exit_code == 0, _out(result)


def test_max_connections_shown_in_parallelism_echo(keyless):
    result = runner.invoke(
        VE.app,
        ["run", "--dry-run", "--groups", "values", "--models", "gpt-4.1"]
        + ["--max-connections", "20", "--model-concurrency", "3"],
    )
    assert result.exit_code == 0, _out(result)
    assert "up to 3 models at a time" in result.output
    assert "20 connections per model" in result.output


# --------------------------------------------------------------------------- #
# question selection flags
# --------------------------------------------------------------------------- #
def test_default_selects_the_whole_roster(keyless):
    n_all = len(questions_mod.all_questions())
    result = runner.invoke(VE.app, ["run", "--dry-run", "--models", "gpt-4.1"])
    assert result.exit_code == 0, _out(result)
    assert f"1 models x {n_all} questions" in result.output


def test_ids_flag_selects_exactly(keyless):
    qid = questions_mod.all_questions()[0].id
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
    assert "judge: 2 calls" in result.output
    assert "(dry run — no API calls made)" in result.output


def test_analyze_dry_run_rejects_non_run_dir(keyless, tmp_path):
    result = runner.invoke(
        VE.app, ["analyze", "-r", str(tmp_path / "nope"), "--dry-run"]
    )
    assert result.exit_code != 0
    assert "not a generated run dir" in _out(result)


# --------------------------------------------------------------------------- #
# command surface
# --------------------------------------------------------------------------- #
def test_command_surface():
    for cmd in ("generate", "analyze", "report", "run"):
        result = runner.invoke(VE.app, [cmd, "--help"])
        assert result.exit_code == 0, (cmd, _out(result))
    for gone in ("stress", "consistency", "merge", "budget"):
        assert runner.invoke(VE.app, [gone, "--help"]).exit_code != 0, gone


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


def test_raise_fd_limit_lifts_soft_to_hard():
    # Raising soft to hard is unprivileged; a big sweep needs the headroom
    # (interactive shells commonly default soft to 1024 -> EMFILE mid-run).
    resource = pytest.importorskip("resource")
    from twominds.cli._app import _raise_fd_limit

    _raise_fd_limit()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    assert soft == hard
