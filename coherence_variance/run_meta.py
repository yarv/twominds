"""run_meta — lightweight per-run provenance files.

Every run directory the pipeline creates gets a small JSON marker so runs can
be discovered, listed, and compared without parsing the heavyweight artefacts
(a judge pass's provenance is *in* analysis.json, but that file is megabytes):

  run_meta.json    one per run dir — kind: "variance" (one generation run);
                   "preferences" is a reserved legacy kind some older results
                   trees contain, kept so discovery never chokes on them.
  judge_meta.json  one per judge pass — written next to the analysis.json it
                   describes (the run dir's default pass and each
                   judge_runs/<label>/ repeat pass).

These are *summaries plus the missing fields* (created_at, git_commit), never
the only copy of anything: deleting one loses no data.

Writers are best-effort by design — a meta-write failure must never kill a
run that just spent real money on API calls.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

RUN_META = "run_meta.json"
JUDGE_META = "judge_meta.json"
META_VERSION = 1

KINDS = ("preferences", "variance", "judge_pass")


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 — provenance is best-effort
        return None


def build_meta(
    kind: str,
    *,
    created_at: Optional[str] = None,
    backfilled: bool = False,
    **fields,
) -> dict:
    """Assemble a meta dict. `created_at` defaults to now (UTC); backfill
    passes the inferred original time instead."""
    if kind not in KINDS:
        raise ValueError(f"unknown run kind {kind!r} (expected one of {KINDS})")
    meta = {
        "meta_version": META_VERSION,
        "kind": kind,
        "created_at": created_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
    }
    if backfilled:
        meta["backfilled"] = True
    meta.update(fields)
    return meta


def write_meta(run_dir: Path | str, meta: dict, filename: str = RUN_META) -> Path:
    path = Path(run_dir) / filename
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path


def write_meta_safe(
    run_dir: Path | str, meta: dict, filename: str = RUN_META
) -> Optional[Path]:
    """write_meta that warns instead of raising — for use at the end of
    pipeline runs, where provenance must never take down a finished run."""
    try:
        return write_meta(run_dir, meta, filename)
    except OSError as e:
        print(f"warning: could not write {filename} in {run_dir}: {e}")
        return None


def read_meta(run_dir: Path | str, filename: str = RUN_META) -> Optional[dict]:
    """The meta dict, or None if absent/unreadable (legacy dirs are normal)."""
    path = Path(run_dir) / filename
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
