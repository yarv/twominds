"""run_registry — discover judge passes from a run directory.

Discovery is artefact-driven, markers enrich: a judge pass is a directory with
``analysis.json``; the ``judge_meta.json`` marker (see ``run_meta.py``) is read
for cheap rich fields when present and synthesized when not, so discovery
never misses a pass because a marker write failed.

Returns lightweight :class:`RunRecord`s (path + meta dict); it never loads the
heavy ``analysis.json`` payloads — callers do that from ``record.path``.
``coherence_variance.consistency.load_judge_runs`` is the main consumer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .run_meta import JUDGE_META, build_meta, read_meta

KINDS = ("variance", "judge_pass")


@dataclass(frozen=True)
class RunRecord:
    """One discovered run. ``meta`` is the marker dict if a marker file was
    present, else a synthesized equivalent (``meta['backfilled']`` is True for
    synthesized records)."""

    path: Path
    kind: str
    meta: dict
    has_marker: bool

    @property
    def label(self) -> str:
        return self.meta.get("label") or self.path.name

    @property
    def created_at(self) -> Optional[str]:
        return self.meta.get("created_at")

    @property
    def models(self) -> list[str]:
        """Model(s) this run covers (judge passes have none — they inherit
        their parent run's models)."""
        if isinstance(self.meta.get("models"), list):
            return list(self.meta["models"])
        return []

    @property
    def parent_run(self) -> Optional[str]:
        return self.meta.get("parent_run")


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(
        timespec="seconds"
    )


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def synthesize_judge_meta(
    pass_dir: Path, *, parent_run: str, label: str
) -> Optional[dict]:
    """Rich judge-pass meta read out of the pass's analysis.json. Heavier than
    the stub below (analysis.json is multi-MB), so it is used only when
    *writing* a missing marker — plain discovery uses the cheap stub."""
    analysis_path = pass_dir / "analysis.json"
    analysis = _load_json(analysis_path)
    if analysis is None:
        return None
    return build_meta(
        "judge_pass",
        created_at=_mtime_iso(analysis_path),
        backfilled=True,
        label=label,
        parent_run=parent_run,
        judge_model=analysis.get("judge"),
        judge_reasoning=analysis.get("judge_reasoning"),
        threshold=analysis.get("threshold"),
        backends=analysis.get("backends"),
        n_bundles=len(analysis.get("results", [])),
    )


def _judge_stub_meta(pass_dir: Path, *, parent_run: str, label: str) -> dict:
    """Cheap judge-pass meta (no analysis.json read) for discovery/listing."""
    return build_meta(
        "judge_pass",
        created_at=_mtime_iso(pass_dir / "analysis.json"),
        backfilled=True,
        label=label,
        parent_run=parent_run,
    )


def discover_judge_passes(
    run_dir: Path | str, *, include_default: bool = True
) -> list[RunRecord]:
    """Judge passes of one variance run, artefact-driven (a pass = a dir with
    ``analysis.json``). Order: ``judge_runs/<label>/`` sorted, then the
    top-level pass as ``"default"``.
    """
    run_dir = Path(run_dir)
    records: list[RunRecord] = []
    jr_dir = run_dir / "judge_runs"
    if jr_dir.is_dir():
        for d in sorted(p for p in jr_dir.iterdir() if p.is_dir()):
            if (d / "analysis.json").exists():
                records.append(_judge_record(d, parent_run=run_dir.name, label=d.name))
    if include_default and (run_dir / "analysis.json").exists():
        records.append(_judge_record(run_dir, parent_run=run_dir.name, label="default"))
    return records


def _judge_record(pass_dir: Path, *, parent_run: str, label: str) -> RunRecord:
    marker = read_meta(pass_dir, JUDGE_META)
    if marker is not None:
        return RunRecord(pass_dir, "judge_pass", marker, has_marker=True)
    return RunRecord(
        pass_dir,
        "judge_pass",
        _judge_stub_meta(pass_dir, parent_run=parent_run, label=label),
        has_marker=False,
    )
