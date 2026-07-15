"""run_registry — discover experiment runs from the results tree.

One discovery layer answers "what runs do I actually have?" for everything
that reads the results tree (``consistency`` listing judge passes, viewers,
ad-hoc analysis).

Design: **discovery is artefact-driven, markers enrich.** A directory is a run
because its *artefacts* say so (a variance run has ``run_config.json``; a
judge pass has ``analysis.json``; the legacy "preferences" kind, which some
older results trees contain, has ``summary_*.txt`` under a timestamp dir). The
provenance markers (``run_meta.json`` / ``judge_meta.json``, see
``run_meta.py``) are read for cheap rich fields (created_at, git_commit,
models, sweep_id) when present and *synthesized* from the artefacts when not.
So the registry never misses a run because a marker write failed — it just has
less metadata for those.

Returns lightweight :class:`RunRecord`s (path + meta dict); it never loads the
heavy ``analysis.json`` payloads — callers do that from ``record.path`` when
they need them.

``coherence_variance.consistency.load_judge_runs`` discovers judge passes
through :func:`discover_judge_passes`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .run_meta import JUDGE_META, RUN_META, build_meta, read_meta

# Sweep dirs look like ``my_sweep_setA``; run dirs underneath are
# ``YYYYMMDD_HHMMSS`` timestamps (the run-dir naming convention).
_SWEEP_DIR_RE = re.compile(r"^(?P<name>.+)_set(?P<set>[A-Z])$")
_TIMESTAMP_DIR_RE = re.compile(r"^\d{8}_\d{6}$")

KINDS = ("preferences", "variance", "judge_pass")


@dataclass(frozen=True)
class RunRecord:
    """One discovered run. ``meta`` is the marker dict if a marker file was
    present, else a synthesized equivalent (``meta['backfilled']`` is True for
    synthesized records, mirroring what the backfill would write)."""

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
        """Model(s) this run covers (variance: list; preferences: one; judge
        pass: none — it inherits its parent run's models)."""
        if isinstance(self.meta.get("models"), list):
            return list(self.meta["models"])
        if self.meta.get("model"):
            return [self.meta["model"]]
        return []

    @property
    def sweep_id(self) -> Optional[str]:
        return self.meta.get("sweep_id")

    @property
    def parent_run(self) -> Optional[str]:
        return self.meta.get("parent_run")


# --- created_at inference (used only when no marker is present) -------------


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(
        timespec="seconds"
    )


def _timestamp_dir_iso(dirname: str) -> Optional[str]:
    try:
        # Run-dir names are local time; keep them naive rather than faking a zone.
        return datetime.strptime(dirname, "%Y%m%d_%H%M%S").isoformat(timespec="seconds")
    except ValueError:
        return None


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# --- meta synthesis (the single definition shared with the backfill) --------
# Each returns the exact dict the live pipeline / backfill writes, so a
# synthesized record and a written marker never disagree.


def synthesize_variance_meta(run_dir: Path) -> Optional[dict]:
    config_path = run_dir / "run_config.json"
    config = _load_json(config_path)
    if config is None:
        return None
    return build_meta(
        "variance",
        created_at=_mtime_iso(config_path),
        backfilled=True,
        label=run_dir.name,
        models=sorted(config.get("models", {})),
        n_questions=len(config.get("question_ids", [])),
        n=config.get("n"),
    )


def synthesize_judge_meta(
    pass_dir: Path, *, parent_run: str, label: str
) -> Optional[dict]:
    """Rich judge-pass meta read out of the pass's analysis.json. Heavier than
    the other synthesizers (analysis.json is multi-MB), so it is used only when
    *writing* a missing marker — plain discovery uses the cheap stub below."""
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


def synthesize_preferences_meta(run_dir: Path, *, sweep_dir: Path) -> Optional[dict]:
    summaries = sorted(run_dir.glob("summary_*.txt"))
    if not summaries or not _TIMESTAMP_DIR_RE.match(run_dir.name):
        return None
    model = "/".join(run_dir.parent.relative_to(sweep_dir).parts)
    return build_meta(
        "preferences",
        created_at=_timestamp_dir_iso(run_dir.name) or _mtime_iso(summaries[0]),
        backfilled=True,
        model=model,
        set_key=_SWEEP_DIR_RE.match(sweep_dir.name).group("set"),
        experiment_name=sweep_dir.name,
        prompt_preset=None,  # not recoverable from artefacts
        utility_config_key=None,
        sweep_id=None,
    )


# --- discovery --------------------------------------------------------------


def discover_judge_passes(
    run_dir: Path | str, *, include_default: bool = True
) -> list[RunRecord]:
    """Judge passes of one variance run, artefact-driven (a pass = a dir with
    ``analysis.json``). Order: ``judge_runs/<label>/`` sorted, then the
    top-level pass as ``"default"`` — matching the legacy ``load_judge_runs``.
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


def discover_runs(
    root: Path | str = "results",
    *,
    kinds: Optional[Iterable[str]] = None,
    model: Optional[str] = None,
    sweep_id: Optional[str] = None,
) -> list[RunRecord]:
    """All runs under ``root``, optionally filtered.

    ``kinds`` restricts to a subset of :data:`KINDS` (default: all). ``model``
    keeps runs covering that model (membership in :attr:`RunRecord.models`;
    judge passes have none so they are dropped by this filter). ``sweep_id``
    keeps preferences runs from one sweep invocation. Sorted by
    ``created_at`` then path for a stable listing.
    """
    root = Path(root)
    want = set(kinds) if kinds is not None else set(KINDS)
    bad = want - set(KINDS)
    if bad:
        raise ValueError(f"unknown kind(s) {sorted(bad)} (expected {KINDS})")

    records: list[RunRecord] = []

    if {"variance", "judge_pass"} & want:
        vroot = root / "variance"
        if vroot.is_dir():
            for run_dir in sorted(p for p in vroot.iterdir() if p.is_dir()):
                if run_dir.name == "models":
                    continue  # the per-model store (see coherence_variance.store)
                if "variance" in want:
                    rec = _variance_record(run_dir)
                    if rec is not None:
                        records.append(rec)
                if "judge_pass" in want:
                    records.extend(discover_judge_passes(run_dir))

    if "preferences" in want:
        for sweep_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if not _SWEEP_DIR_RE.match(sweep_dir.name):
                continue
            for summary in sorted(sweep_dir.rglob("summary_*.txt")):
                run_dir = summary.parent
                if not _TIMESTAMP_DIR_RE.match(run_dir.name):
                    continue
                rec = _preferences_record(run_dir, sweep_dir)
                if rec is not None:
                    records.append(rec)

    if model is not None:
        records = [r for r in records if model in r.models]
    if sweep_id is not None:
        records = [r for r in records if r.sweep_id == sweep_id]

    records.sort(key=lambda r: (r.created_at or "", str(r.path)))
    return records


def _variance_record(run_dir: Path) -> Optional[RunRecord]:
    marker = read_meta(run_dir, RUN_META)
    if marker is not None and marker.get("kind") == "variance":
        return RunRecord(run_dir, "variance", marker, has_marker=True)
    synth = synthesize_variance_meta(run_dir)
    if synth is None:
        return None
    return RunRecord(run_dir, "variance", synth, has_marker=False)


def resolve_variance_run(
    entry: str, root: Path | str = "results"
) -> Optional[RunRecord]:
    """Resolve a cohort's variance-run entry to a :class:`RunRecord`.

    ``entry`` may be a full path (``results/variance/foo``) or a bare run label
    (``foo``, looked up under ``<root>/variance/``). Returns None if neither
    resolves to a variance run (no ``run_config.json`` and no marker)."""
    for cand in (Path(entry), Path(root) / "variance" / entry):
        rec = _variance_record(cand)
        if rec is not None:
            return rec
        # Lenient fallback: a dir with analysis.json but no run_config.json is
        # still a usable variance run (matches the pre-registry behaviour, which
        # only ever read analysis.json). Models come from the analysis itself.
        analysis_path = cand / "analysis.json"
        if analysis_path.exists():
            analysis = _load_json(analysis_path) or {}
            models = sorted(
                {r.get("model") for r in analysis.get("results", []) if r.get("model")}
            )
            meta = build_meta(
                "variance",
                created_at=_mtime_iso(analysis_path),
                backfilled=True,
                label=cand.name,
                models=models,
                n_questions=None,
                n=None,
            )
            return RunRecord(cand, "variance", meta, has_marker=False)
    return None


def _preferences_record(run_dir: Path, sweep_dir: Path) -> Optional[RunRecord]:
    marker = read_meta(run_dir, RUN_META)
    if marker is not None and marker.get("kind") == "preferences":
        return RunRecord(run_dir, "preferences", marker, has_marker=True)
    synth = synthesize_preferences_meta(run_dir, sweep_dir=sweep_dir)
    if synth is None:
        return None
    return RunRecord(run_dir, "preferences", synth, has_marker=False)
