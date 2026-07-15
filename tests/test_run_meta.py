"""Tests for run_meta.py.

Locks the provenance contract: every run dir gets a readable marker and the
variance generate path writes one.
"""

from __future__ import annotations

import pytest

from coherence_variance.run_meta import (
    RUN_META,
    build_meta,
    read_meta,
    write_meta,
    write_meta_safe,
)


class TestBuildWriteRead:
    def test_round_trip(self, tmp_path):
        meta = build_meta("variance", label="my_run", models=["gpt-4.1"], n=20)
        write_meta(tmp_path, meta)
        got = read_meta(tmp_path)
        assert got == meta
        assert got["kind"] == "variance"
        assert got["meta_version"] == 1
        assert got["created_at"]  # stamped
        assert "backfilled" not in got

    def test_created_at_override_and_backfilled(self, tmp_path):
        meta = build_meta(
            "preferences", created_at="2026-05-21T17:31:57", backfilled=True
        )
        assert meta["created_at"] == "2026-05-21T17:31:57"
        assert meta["backfilled"] is True

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="unknown run kind"):
            build_meta("nonsense")

    def test_read_absent_is_none(self, tmp_path):
        assert read_meta(tmp_path) is None

    def test_read_corrupt_is_none(self, tmp_path):
        (tmp_path / RUN_META).write_text("{not json")
        assert read_meta(tmp_path) is None

    def test_write_safe_never_raises(self, tmp_path):
        missing = tmp_path / "no" / "such" / "dir"
        assert write_meta_safe(missing, build_meta("variance")) is None


class TestGenerateWiring:
    def test_write_manifest_emits_run_meta(self, tmp_path):
        from coherence_variance.generate import write_manifest
        from coherence_variance.models import ModelSpec
        from coherence_variance.questions import Question

        spec = ModelSpec(name="gpt-4.1", inspect_model="openai/gpt-4.1")
        q = Question(id="q1", group="control", prompt="Say hi.")
        run_dir = tmp_path / "run1"
        write_manifest(
            run_dir, [spec], [q], n=3, temperature=1.0, max_tokens=64, judge="j"
        )
        meta = read_meta(run_dir)
        assert meta["kind"] == "variance"
        assert meta["label"] == "run1"
        assert meta["models"] == ["gpt-4.1"]
        assert meta["n_questions"] == 1 and meta["n"] == 3
        # and the pre-existing artefacts are still written
        assert (run_dir / "run_config.json").exists()
