"""Orchestration tests for the one-command robust run (`--reps`).

Locks the control flow of ``coherence_variance.cli._judge_reps`` without any API
calls: rep1 writes the top-level analysis, rep2..repN go to judge_runs/repK,
and consistency runs exactly once when reps>1 (and never when reps==1).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from coherence_variance import cli as V  # noqa: E402


def _fake_out():
    return {"results": [{"judge": {"contradiction": True, "flags": []}}]}


def _patch(monkeypatch):
    """Stub the three heavy collaborators; return the recorded call lists."""
    analyze_calls: list = []
    report_calls: list = []
    cons_calls: list = []

    def fake_analyze(run_dir, *, judge_run=None, **kw):
        analyze_calls.append(judge_run)
        return _fake_out()

    monkeypatch.setattr(V.analyze_mod, "analyze", fake_analyze)
    monkeypatch.setattr(
        V.report_mod,
        "build_report_from_run",
        lambda rd: report_calls.append(rd) or "rep.html",
    )
    import coherence_variance.consistency as cons

    def fake_cons(run_dir, *, include_default=True):
        cons_calls.append(run_dir)
        return {
            "overall": {
                "n_runs": 3,
                "run_labels": ["rep2", "rep3", "default"],
                "mean_partition_ari": 0.9,
                "mean_consensus_strength": 0.95,
                "frac_contradiction_unstable": 0.0,
            }
        }

    monkeypatch.setattr(cons, "build_consistency_from_run", fake_cons)
    return analyze_calls, report_calls, cons_calls


_KW = dict(
    backends=["local"],
    judge="j",
    judge_reasoning="low",
    threshold=0.15,
    local_model="m",
    concurrency=6,
    run_judge=True,
)


def test_reps_three_does_default_plus_rep2_rep3_then_consistency(monkeypatch, tmp_path):
    a, r, c = _patch(monkeypatch)
    V._judge_reps(tmp_path, reps=3, consistency=True, **_KW)
    assert a == [None, "rep2", "rep3"]  # rep1 (top-level) + rep2 + rep3
    assert len(r) == 1  # report built once
    assert len(c) == 1  # consistency aggregated once


def test_reps_one_is_single_pass_no_consistency(monkeypatch, tmp_path):
    a, r, c = _patch(monkeypatch)
    V._judge_reps(tmp_path, reps=1, consistency=True, **_KW)
    assert a == [None]
    assert len(r) == 1
    assert c == []  # consistency only when reps>1


def test_no_consistency_flag_suppresses_aggregation(monkeypatch, tmp_path):
    a, r, c = _patch(monkeypatch)
    V._judge_reps(tmp_path, reps=2, consistency=False, **_KW)
    assert a == [None, "rep2"]
    assert c == []
