"""Tests for coherence_variance.merge (combine variance runs into one analysis)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from coherence_variance.merge import merge_analyses, write_merged  # noqa: E402


def _run(tmp, name, models, judge="J", qids=("q1", "q2")):
    d = tmp / name
    d.mkdir()
    analysis = {
        "judge": judge,
        "models": list(models),
        "questions": {q: {"prompt": q} for q in qids},
        "families_meta": {},
        "results": [
            {"model": m, "question_id": q, "judge": {}} for m in models for q in qids
        ],
        "families": [],
    }
    (d / "analysis.json").write_text(json.dumps(analysis))
    return d


def test_merge_unions_models_and_concatenates_results(tmp_path):
    a = _run(tmp_path, "A", ["m1", "m2"])
    b = _run(tmp_path, "B", ["m3"])
    merged = merge_analyses([a, b])
    assert merged["models"] == ["m1", "m2", "m3"]  # unioned + sorted
    assert len(merged["results"]) == 6  # (2+1) models × 2 questions
    assert merged["source_runs"] == ["A", "B"]
    assert merged["merge_warnings"] == []


def test_merge_rejects_duplicate_model(tmp_path):
    a = _run(tmp_path, "A", ["m1"])
    b = _run(tmp_path, "B", ["m1"])  # same model in two runs
    with pytest.raises(ValueError, match="already merged"):
        merge_analyses([a, b])


def test_merge_requires_two_runs(tmp_path):
    a = _run(tmp_path, "A", ["m1"])
    with pytest.raises(ValueError, match=">= 2"):
        merge_analyses([a])


def test_merge_warns_on_judge_or_question_mismatch(tmp_path):
    a = _run(tmp_path, "A", ["m1"], judge="opus", qids=("q1", "q2"))
    b = _run(tmp_path, "B", ["m2"], judge="sonnet", qids=("q1", "q3"))
    merged = merge_analyses([a, b])
    assert any("judge" in w for w in merged["merge_warnings"])
    assert any("question set differs" in w for w in merged["merge_warnings"])


def test_write_merged_emits_analysis_json(tmp_path):
    a = _run(tmp_path, "A", ["m1"])
    b = _run(tmp_path, "B", ["m2"])
    out = tmp_path / "combined"
    write_merged([a, b], out)
    written = json.loads((out / "analysis.json").read_text())
    assert written["models"] == ["m1", "m2"]
    assert written["run_dir"] == "<merged>"
