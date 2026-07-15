"""Tests for run_registry.py — the shared run-discovery layer.

The load-bearing properties:
  * discovery is artefact-driven: it finds the same runs with OR without the
    step-2 marker files present (markers only enrich the metadata);
  * filters (kind / model / sweep_id) behave;
  * the rewired consistency.load_judge_runs returns byte-identical output to
    the pre-registry implementation (the reason this rewire is safe).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coherence_variance.run_meta import build_meta, write_meta
from coherence_variance.run_registry import (
    discover_judge_passes,
    discover_runs,
)


# --- fixtures ---------------------------------------------------------------


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
    """A results/ tree: one variance run (default + rep2 judge passes), two
    preference runs from one sweep (sets A and B of the same model)."""
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

    for set_key in ("A", "B"):
        d = root / f"pol_set{set_key}" / "ours" / "my-finetune" / "20260521_173157"
        d.mkdir(parents=True)
        (d / "summary_my-finetune.txt").write_text("kl_divergence: 0.04\n")
    return root


def _write_all_markers(root: Path):
    """Write real markers for every run via the registry's own synthesizers."""
    from coherence_variance.run_meta import JUDGE_META
    from coherence_variance.run_registry import (
        synthesize_judge_meta,
        synthesize_preferences_meta,
        synthesize_variance_meta,
    )

    for run in (root / "variance").iterdir():
        if not run.is_dir():
            continue
        write_meta(run, synthesize_variance_meta(run))
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
    for sweep in root.iterdir():
        if sweep.name == "variance" or not sweep.is_dir():
            continue
        for run_dir in sweep.rglob("summary_*.txt"):
            d = run_dir.parent
            write_meta(d, synthesize_preferences_meta(d, sweep_dir=sweep))


# --- discovery: artefact-driven, marker-agnostic ----------------------------


class TestDiscovery:
    def test_finds_all_kinds_without_markers(self, tmp_path):
        root = _tree(tmp_path)
        recs = discover_runs(root)
        kinds = sorted(r.kind for r in recs)
        # 1 variance + 2 judge passes (default + rep2) + 2 preferences
        assert kinds == [
            "judge_pass",
            "judge_pass",
            "preferences",
            "preferences",
            "variance",
        ]
        assert all(not r.has_marker for r in recs)
        assert all(r.meta.get("backfilled") for r in recs)

    def test_per_model_store_is_not_a_run(self, tmp_path):
        """results/variance/models/ (the per-model store) must never be
        discovered as a variance run, even though its gen dirs hold
        run_config.json mini-manifests."""
        root = _tree(tmp_path)
        gen = root / "variance" / "models" / "gpt-4.1" / "gens" / "abc_q2_n2"
        gen.mkdir(parents=True)
        (gen / "run_config.json").write_text(json.dumps({"models": {"gpt-4.1": {}}}))
        (gen / "analysis.json").write_text(json.dumps(_analysis(None, 2)))
        recs = discover_runs(root)
        assert all("models" not in Path(r.path).parts for r in recs)

    def test_same_runs_found_with_markers(self, tmp_path):
        root = _tree(tmp_path)
        before = {(r.kind, str(r.path)) for r in discover_runs(root)}
        _write_all_markers(root)
        after_recs = discover_runs(root)
        after = {(r.kind, str(r.path)) for r in after_recs}
        assert before == after  # markers change metadata, never the run set
        # variance + preferences now read from real markers
        non_judge = [r for r in after_recs if r.kind != "judge_pass"]
        assert all(r.has_marker for r in non_judge)

    def test_marker_is_used_verbatim_when_present(self, tmp_path):
        root = _tree(tmp_path)
        run = root / "variance" / "sweep1"
        marker = build_meta("variance", label="hand", models=["x"], n=99)
        write_meta(run, marker)
        (rec,) = [r for r in discover_runs(root, kinds=("variance",))]
        assert rec.has_marker
        assert rec.meta == marker
        assert rec.label == "hand" and rec.models == ["x"]

    def test_variance_record_fields(self, tmp_path):
        root = _tree(tmp_path)
        (rec,) = discover_runs(root, kinds=("variance",))
        assert rec.kind == "variance"
        assert rec.label == "sweep1"
        assert rec.models == ["gpt-4.1", "ours/m"]
        assert rec.meta["n"] == 20 and rec.meta["n_questions"] == 3

    def test_preferences_record_fields(self, tmp_path):
        root = _tree(tmp_path)
        recs = discover_runs(root, kinds=("preferences",))
        assert {r.meta["set_key"] for r in recs} == {"A", "B"}
        assert all(r.models == ["ours/my-finetune"] for r in recs)
        assert all(r.meta["experiment_name"].startswith("pol_set") for r in recs)


# --- filters ----------------------------------------------------------------


class TestFilters:
    def test_kind_filter(self, tmp_path):
        root = _tree(tmp_path)
        assert all(
            r.kind == "variance" for r in discover_runs(root, kinds=("variance",))
        )

    def test_model_filter_keeps_covering_runs(self, tmp_path):
        root = _tree(tmp_path)
        recs = discover_runs(root, model="gpt-4.1")
        # variance run covers gpt-4.1; preferences runs cover ours/my-finetune;
        # judge passes have no models -> dropped by a model filter.
        assert [r.kind for r in recs] == ["variance"]

    def test_model_filter_preferences(self, tmp_path):
        root = _tree(tmp_path)
        recs = discover_runs(root, model="ours/my-finetune")
        assert len(recs) == 2 and all(r.kind == "preferences" for r in recs)

    def test_unknown_kind_raises(self, tmp_path):
        root = _tree(tmp_path)
        with pytest.raises(ValueError, match="unknown kind"):
            discover_runs(root, kinds=("bogus",))


# --- judge-pass discovery + load_judge_runs parity --------------------------


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


class TestJudgePassParity:
    def test_labels_and_order(self, tmp_path):
        root = _tree(tmp_path)
        run = root / "variance" / "sweep1"
        labels = [r.label for r in discover_judge_passes(run)]
        assert labels == ["rep2", "default"]  # judge_runs sorted, then default

    def test_include_default_false(self, tmp_path):
        root = _tree(tmp_path)
        run = root / "variance" / "sweep1"
        labels = [r.label for r in discover_judge_passes(run, include_default=False)]
        assert labels == ["rep2"]

    @pytest.mark.parametrize("with_markers", [False, True])
    def test_load_judge_runs_matches_reference(self, tmp_path, with_markers):
        root = _tree(tmp_path)
        if with_markers:
            _write_all_markers(root)
        run = root / "variance" / "sweep1"
        from coherence_variance.consistency import load_judge_runs

        got = load_judge_runs(run)
        expected = _ref_load_judge_runs(run)
        assert got == expected
        assert list(got) == list(expected)  # same insertion order
