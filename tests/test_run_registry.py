"""Tests for run_registry.py — judge-pass discovery.

The load-bearing properties: discovery is artefact-driven (it finds the same
passes with OR without marker files present — markers only enrich metadata),
and the rewired consistency.load_judge_runs returns byte-identical output to
the pre-registry implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coherence_variance.run_meta import JUDGE_META, write_meta
from coherence_variance.run_registry import discover_judge_passes


def _analysis(judge_run, n_results):
    return {
        "run_dir": "x",
        "judge_run": judge_run,
        "judge": "openrouter/anthropic/claude-opus-4.8",
        "judge_reasoning": "low",
        "threshold": 0.15,
        "backends": ["local"],
        "results": [
            {"model": "gpt-4.1", "question_id": f"q{i}"} for i in range(n_results)
        ],
    }


def _tree(tmp_path: Path) -> Path:
    """A results/ tree: one variance run with default + rep2 judge passes."""
    root = tmp_path / "results"
    run = root / "variance" / "sweep1"
    run.mkdir(parents=True)
    (run / "run_config.json").write_text(
        json.dumps(
            {
                "models": {"gpt-4.1": {}, "ours/m": {}},
                "question_ids": ["a", "b", "c"],
                "n": 20,
            }
        )
    )
    (run / "analysis.json").write_text(json.dumps(_analysis(None, 6)))
    rep = run / "judge_runs" / "rep2"
    rep.mkdir(parents=True)
    (rep / "analysis.json").write_text(json.dumps(_analysis("rep2", 6)))
    return root


def _write_all_markers(root: Path):
    """Write real markers for every judge pass via the registry's synthesizer."""
    from coherence_variance.run_registry import synthesize_judge_meta

    for run in (root / "variance").iterdir():
        if not run.is_dir():
            continue
        write_meta(
            run,
            synthesize_judge_meta(run, parent_run=run.name, label="default"),
            JUDGE_META,
        )
        jr = run / "judge_runs"
        if jr.is_dir():
            for p in jr.iterdir():
                if (p / "analysis.json").exists():
                    write_meta(
                        p,
                        synthesize_judge_meta(p, parent_run=run.name, label=p.name),
                        JUDGE_META,
                    )


def test_labels_and_order(tmp_path):
    root = _tree(tmp_path)
    run = root / "variance" / "sweep1"
    labels = [r.label for r in discover_judge_passes(run)]
    assert labels == ["rep2", "default"]  # judge_runs sorted, then default


def test_include_default_false(tmp_path):
    root = _tree(tmp_path)
    run = root / "variance" / "sweep1"
    labels = [r.label for r in discover_judge_passes(run, include_default=False)]
    assert labels == ["rep2"]


def test_marker_enriches_but_does_not_gate_discovery(tmp_path):
    root = _tree(tmp_path)
    run = root / "variance" / "sweep1"
    bare = discover_judge_passes(run)
    assert all(not r.has_marker for r in bare)
    _write_all_markers(root)
    marked = discover_judge_passes(run)
    assert [r.label for r in marked] == [r.label for r in bare]
    assert all(r.has_marker for r in marked)
    rep2 = next(r for r in marked if r.label == "rep2")
    assert rep2.meta["n_bundles"] == 6 and rep2.parent_run == "sweep1"


def _ref_load_judge_runs(run_dir: Path, *, include_default: bool = True) -> dict:
    """The pre-registry implementation, kept here as the parity oracle."""
    run_dir = Path(run_dir)
    runs: dict[str, dict] = {}
    jr_dir = run_dir / "judge_runs"
    if jr_dir.exists():
        for d in sorted(p for p in jr_dir.iterdir() if p.is_dir()):
            f = d / "analysis.json"
            if f.exists():
                runs[d.name] = json.loads(f.read_text())
    if include_default:
        f = run_dir / "analysis.json"
        if f.exists() and "default" not in runs:
            runs["default"] = json.loads(f.read_text())
    return runs


@pytest.mark.parametrize("with_markers", [False, True])
def test_load_judge_runs_matches_reference(tmp_path, with_markers):
    root = _tree(tmp_path)
    if with_markers:
        _write_all_markers(root)
    run = root / "variance" / "sweep1"
    from coherence_variance.consistency import load_judge_runs

    got = load_judge_runs(run)
    expected = _ref_load_judge_runs(run)
    assert got == expected
    assert list(got) == list(expected)  # same insertion order
