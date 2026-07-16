"""Merge several variance runs into one combined analysis.

Runs generated separately (e.g. models whose fine-tunes landed at different
times) end up in separate run dirs, each with its own report. When they share
the same question bank + judge, their ``analysis.json`` files can be combined
into a single analysis (results concatenated, models unioned) and rendered as
one report with a model selector -- no re-judging, so it costs nothing.

Only the top-level (default / rep1) analysis of each run is merged; per-run
judge-pass robustness (rep2..N + consistency) stays with each source run.
"""

from __future__ import annotations

import json
from pathlib import Path


def merge_analysis_dicts(
    analyses: list[dict],
    *,
    run_dir: str = "<merged>",
    source_labels: list[str] | None = None,
) -> dict:
    """Union already-in-memory analysis dicts into one (>=1).

    Like :func:`merge_analyses` but without the filesystem round-trip — used by the
    per-model streaming pipeline, which holds each model's analysis dict in memory
    and re-merges after every model completes to refresh the combined report.
    Concatenates ``results`` / ``families``, unions ``models`` / ``questions``;
    duplicate models across inputs raise (the pipeline's inputs are disjoint).
    """
    if not analyses:
        raise ValueError("merge needs >= 1 analysis")
    labels = source_labels or [str(i) for i in range(len(analyses))]
    merged: dict | None = None
    warnings: list[str] = []
    seen_models: set[str] = set()
    base_qids: set | None = None
    for a, label in zip(analyses, labels):
        qids = set(a.get("questions") or {})
        if merged is None:
            merged = {
                "run_dir": run_dir,
                "judge_run": None,
                "backends": a.get("backends"),
                "primary_backend": a.get("primary_backend"),
                "judge": a.get("judge"),
                "judge_reasoning": a.get("judge_reasoning"),
                "threshold": a.get("threshold"),
                "models": [],
                "model_display": {},
                "config": dict(a.get("config") or {}),
                "questions": {},
                "families_meta": {},
                "results": [],
                "families": [],
                "source_runs": [],
            }
            base_qids = qids
        elif qids != base_qids:
            warnings.append(f"{label}: question set differs ({len(qids)} qs)")
        dupes = seen_models.intersection(a.get("models") or [])
        if dupes:
            raise ValueError(f"{label}: model(s) {sorted(dupes)} already merged")
        seen_models.update(a.get("models") or [])
        merged["models"].extend(a.get("models") or [])
        merged["model_display"].update(a.get("model_display") or {})
        cfg_models = (a.get("config") or {}).get("models") or {}
        if cfg_models:
            merged["config"].setdefault("models", {}).update(cfg_models)
        merged["results"].extend(a.get("results") or [])
        merged["families"].extend(a.get("families") or [])
        merged["questions"].update(a.get("questions") or {})
        merged["families_meta"].update(a.get("families_meta") or {})
        merged["source_runs"].append(label)
    assert merged is not None
    merged["models"] = sorted(merged["models"])
    merged["merge_warnings"] = warnings
    return merged


def merge_analyses(run_dirs: list[str | Path]) -> dict:
    """Combine the top-level analysis.json of each run into one analysis dict.

    Raises ValueError on <2 runs, a duplicated model across runs, or a missing
    analysis.json. Warns (via the returned ``merge_warnings``) on judge or
    question-set mismatches rather than failing.
    """
    dirs = [Path(d) for d in run_dirs]
    if len(dirs) < 2:
        raise ValueError("merge needs >= 2 run dirs")

    merged: dict | None = None
    warnings: list[str] = []
    seen_models: set[str] = set()
    base_judge = base_qids = None

    for d in dirs:
        ap = d / "analysis.json"
        if not ap.exists():
            raise ValueError(f"no analysis.json in {d}")
        a = json.loads(ap.read_text())
        qids = set(a.get("questions") or {})
        if merged is None:
            merged = {
                "run_dir": "<merged>",
                "judge_run": None,
                "backends": a.get("backends"),
                "primary_backend": a.get("primary_backend"),
                "judge": a.get("judge"),
                "judge_reasoning": a.get("judge_reasoning"),
                "threshold": a.get("threshold"),
                "models": [],
                "model_display": {},
                "config": dict(a.get("config") or {}),
                "questions": dict(a.get("questions") or {}),
                "families_meta": dict(a.get("families_meta") or {}),
                "results": [],
                "families": [],
                "source_runs": [],
            }
            base_judge, base_qids = a.get("judge"), qids
        else:
            if a.get("judge") != base_judge:
                warnings.append(f"{d.name}: judge {a.get('judge')!r} != {base_judge!r}")
            if qids != base_qids:
                warnings.append(f"{d.name}: question set differs ({len(qids)} qs)")
        dupes = seen_models.intersection(a.get("models") or [])
        if dupes:
            raise ValueError(f"{d.name}: model(s) {sorted(dupes)} already merged")
        seen_models.update(a.get("models") or [])
        merged["models"].extend(a.get("models") or [])
        merged["model_display"].update(a.get("model_display") or {})
        cfg_models = (a.get("config") or {}).get("models") or {}
        if cfg_models:
            merged["config"].setdefault("models", {}).update(cfg_models)
        merged["results"].extend(a.get("results") or [])
        merged["families"].extend(a.get("families") or [])
        merged["questions"].update(a.get("questions") or {})
        merged["families_meta"].update(a.get("families_meta") or {})
        merged["source_runs"].append(d.name)

    assert merged is not None
    merged["models"] = sorted(merged["models"])
    merged["merge_warnings"] = warnings
    return merged


def write_merged(run_dirs: list[str | Path], out_dir: str | Path) -> dict:
    """Merge and write <out_dir>/analysis.json. Returns the merged dict."""
    merged = merge_analyses(run_dirs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(merged, indent=2))
    return merged
