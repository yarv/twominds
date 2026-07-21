"""Backfill: file-store trees and run dirs into the database.

Discovery walks a results tree without following symlinks: a dir with
``gen_meta.json`` is a store generation (its judge fragments come with it,
descent stops there), a dir with ``analysis.json`` is a run (its
``judge_runs/<label>/`` repeat passes come with it). Everything funnels
through :func:`import_analysis`; digest-unique rows make every import
idempotent, so overlapping sources (fragment vs run, run vs merged run)
collapse instead of duplicating.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import keys
from .payloads import FamilyAnalysisPayload, JudgeVerdict, Metrics, ModelIdentity
from .schema import (
    Bundle,
    BundleMetrics,
    BundleResponse,
    Clustering,
    Family,
    FamilyAnalysis,
    FamilyVersion,
    GenBatch,
    JudgeConfig,
    Judgment,
    Model,
    Question,
    QuestionVersion,
    Response,
    Run,
    RunBundle,
    RunFamilyAnalysis,
    RunJudgment,
)

_QUESTION_FIELDS = ("prompt", "system", "group", "bucket", "family", "variant")


@dataclass
class ImportStats:
    counts: Counter = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)

    def created(self, table: str, n: int = 1) -> None:
        self.counts[table] += n

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def summary(self) -> str:
        lines = [f"  {k}: {self.counts[k]}" for k in sorted(self.counts)]
        return "created/seen:\n" + "\n".join(lines) if lines else "nothing imported"


def import_paths(sess: Session, paths: list[Path]) -> ImportStats:
    stats = ImportStats()
    gens: list[Path] = []
    runs: list[Path] = []
    for p in paths:
        g, r = _discover(Path(p))
        gens += g
        runs += r
    for d in gens:
        import_gen_dir(sess, d, stats)
        sess.commit()
    for d in runs:
        import_run_dir(sess, d, stats)
        sess.commit()
    return stats


def _discover(root: Path) -> tuple[list[Path], list[Path]]:
    gens, runs = [], []
    if root.is_file():
        root = root.parent
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith((".", "_")))
        d = Path(dirpath)
        if "gen_meta.json" in filenames:
            gens.append(d)
            dirnames[:] = []
        elif "analysis.json" in filenames and d.parent.name != "judge_runs":
            runs.append(d)
    return gens, runs


def _json(path: Path, stats: ImportStats) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        stats.warn(f"{path}: {e}")
        return None


def _get(sess: Session, cls, **filters):
    return sess.scalar(select(cls).filter_by(**filters))


def _get_or_create(
    sess: Session, cls, defaults: Optional[dict] = None, *, stats=None, **filters
):
    row = _get(sess, cls, **filters)
    if row is None:
        row = cls(**{**(defaults or {}), **filters})
        sess.add(row)
        sess.flush()
        if stats is not None:
            stats.created(cls.__tablename__)
    return row


def _mtime_iso(path: Path) -> Optional[str]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(
            timespec="seconds"
        )
    except OSError:
        return None


def _model_row(
    sess: Session, name: str, ident: ModelIdentity, stats: ImportStats
) -> Model:
    row = _get(sess, Model, name=name)
    if row is None:
        return _get_or_create(
            sess,
            Model,
            {
                "inspect_model": ident.inspect_model,
                "reasoning_effort": ident.reasoning_effort,
                "display": ident.display or "",
            },
            stats=stats,
            name=name,
        )
    if ident.inspect_model is None:
        return row
    if row.inspect_model is None:
        row.inspect_model = ident.inspect_model
        row.reasoning_effort = ident.reasoning_effort
        row.display = row.display or ident.display or ""
        return row
    if row.inspect_model == ident.inspect_model and (row.reasoning_effort or None) == (
        ident.reasoning_effort or None
    ):
        return row
    qualified = (
        f"{name}#{keys._hash([ident.inspect_model, ident.reasoning_effort])[:8]}"
    )
    stats.warn(
        f"model name '{name}' already maps to {row.inspect_model} "
        f"(reasoning={row.reasoning_effort}); storing {ident.inspect_model} "
        f"(reasoning={ident.reasoning_effort}) as '{qualified}'"
    )
    return _model_row(sess, qualified, ident, stats)


def _question_version(
    sess: Session, qid: str, meta: dict, stats: ImportStats
) -> QuestionVersion:
    _get_or_create(sess, Question, {"first_seen_at": None}, stats=stats, id=qid)
    fam = meta.get("family")
    if fam:
        _get_or_create(sess, Family, stats=stats, id=fam)
    extra = {k: v for k, v in meta.items() if k not in _QUESTION_FIELDS}
    return _get_or_create(
        sess,
        QuestionVersion,
        {
            "prompt": meta.get("prompt") or "",
            "system": meta.get("system"),
            "group_name": meta.get("group") or "",
            "bucket": meta.get("bucket"),
            "family_id": fam,
            "variant": meta.get("variant"),
            "extra": extra or None,
        },
        stats=stats,
        question_id=qid,
        content_hash=keys.question_content_hash(meta),
    )


def _judge_config(
    sess: Session, analysis: dict, stats: ImportStats
) -> Optional[JudgeConfig]:
    judge_model = analysis.get("judge")
    if not judge_model:
        return None
    return _get_or_create(
        sess,
        JudgeConfig,
        stats=stats,
        judge_model=judge_model,
        judge_reasoning=analysis.get("judge_reasoning") or "",
        prompt_hash="unknown",
    )


def _models_config(
    analysis: dict, fallback: Optional[dict]
) -> dict[str, ModelIdentity]:
    cfg = analysis.get("config") or fallback or {}
    display = analysis.get("model_display") or {}
    out = {}
    for name, ident in (cfg.get("models") or {}).items():
        out[name] = ModelIdentity.model_validate(ident or {})
        out[name].display = out[name].display or display.get(name)
    return out


def import_analysis(
    sess: Session,
    analysis: dict,
    *,
    source_path: str,
    stats: ImportStats,
    run: Optional[Run] = None,
    role: str = "default",
    config: Optional[dict] = None,
    provenance: Optional[dict[str, dict]] = None,
    pass_meta: Optional[dict] = None,
) -> None:
    cfg = analysis.get("config") or config or {}
    models_cfg = _models_config(analysis, config)
    qmeta = analysis.get("questions") or {}
    threshold = analysis.get("threshold")
    jc = _judge_config(sess, analysis, stats)
    pass_meta = pass_meta or {}

    by_model: dict[str, list[dict]] = {}
    for r in analysis.get("results") or []:
        by_model.setdefault(r["model"], []).append(r)

    for mname, results in by_model.items():
        ident = models_cfg.get(mname) or ModelIdentity()
        model = _model_row(sess, mname, ident, stats)
        prepared = []
        for r in results:
            qv = _question_version(
                sess, r["question_id"], qmeta.get(r["question_id"]) or {}, stats
            )
            texts = list(r.get("responses") or [])
            dig = keys.bundle_digest(
                ident.inspect_model, ident.reasoning_effort, qv.content_hash, texts
            )
            prepared.append((r, qv, texts, dig))

        missing = [
            (qv, texts, dig)
            for _, qv, texts, dig in prepared
            if _get(sess, Bundle, digest=dig) is None
        ]
        if missing:
            batch = _batch(
                sess,
                model,
                ident,
                cfg,
                analysis,
                mname,
                [dig for _, _, dig in missing],
                source_path,
                pass_meta,
                (provenance or {}).get(mname),
                stats,
            )
            for qv, texts, dig in missing:
                _create_bundle(sess, model, qv, texts, dig, batch, stats)

        for r, qv, texts, dig in prepared:
            bundle = _get(sess, Bundle, digest=dig)
            judgment = _import_judgment(
                sess, r, bundle, jc, role, source_path, pass_meta, stats
            )
            _import_clusterings(sess, r, bundle, threshold, stats)
            _import_metrics(sess, r, bundle, judgment, stats)
            if run is not None:
                _get_or_create(sess, RunBundle, run_id=run.id, bundle_id=bundle.id)
                if judgment is not None:
                    _get_or_create(
                        sess,
                        RunJudgment,
                        {"role": role},
                        run_id=run.id,
                        judgment_id=judgment.id,
                    )

    _import_families(sess, analysis, models_cfg, jc, role, source_path, run, stats)


def _batch(
    sess,
    model,
    ident,
    cfg,
    analysis,
    mname,
    missing_digests,
    source_path,
    pass_meta,
    provenance,
    stats,
) -> GenBatch:
    """Content-addressed: the digest covers the bundle digests (hence the
    response texts), so same-config generations from different sources never
    share a batch. ``provenance`` (log path/sha etc.) enriches, never keys."""
    usage = ((analysis.get("cost") or {}).get("generation") or {}).get(mname) or {}
    dig = keys.batch_digest(
        ident.inspect_model,
        ident.reasoning_effort,
        cfg.get("temperature"),
        cfg.get("max_tokens"),
        cfg.get("n"),
        "|".join(sorted(missing_digests)),
    )
    return _get_or_create(
        sess,
        GenBatch,
        {
            "model_id": model.id,
            "temperature": cfg.get("temperature"),
            "max_tokens": cfg.get("max_tokens"),
            "n_requested": cfg.get("n"),
            "created_at": pass_meta.get("created_at"),
            "git_commit": pass_meta.get("git_commit"),
            "in_tok": usage.get("in_tok"),
            "out_tok": usage.get("out_tok"),
            "est_dollars": usage.get("dollars"),
            "source_path": source_path,
            **(provenance or {}),
        },
        stats=stats,
        digest=dig,
    )


def _create_bundle(sess, model, qv, texts, dig, batch, stats) -> Bundle:
    bundle = Bundle(
        digest=dig, model_id=model.id, question_version_id=qv.id, n=len(texts)
    )
    sess.add(bundle)
    sess.flush()
    stats.created("bundle")
    for i, text in enumerate(texts):
        resp = _get_or_create(
            sess,
            Response,
            {"text": text},
            stats=stats,
            batch_id=batch.id,
            question_version_id=qv.id,
            sample_index=i,
        )
        if resp.text != text:
            raise RuntimeError(
                f"response identity collision in batch {batch.digest} "
                f"({qv.question_id}[{i}]) — batch digest bug"
            )
        sess.add(BundleResponse(bundle_id=bundle.id, position=i, response_id=resp.id))
    sess.flush()
    return bundle


def _import_judgment(
    sess, r, bundle, jc, role, source_path, pass_meta, stats
) -> Optional[Judgment]:
    raw = r.get("judge")
    if not raw or jc is None:
        return None
    verdict = JudgeVerdict.model_validate(
        {**raw, "labels": r.get("judge_labels")}
    ).model_dump(mode="json")
    dig = keys.judgment_digest(
        bundle.digest, jc.judge_model, jc.judge_reasoning, jc.prompt_hash, role, verdict
    )
    return _get_or_create(
        sess,
        Judgment,
        {
            "bundle_id": bundle.id,
            "judge_config_id": jc.id,
            "rep_label": role,
            "contradiction": verdict.get("contradiction"),
            "n_groups": verdict.get("n_groups"),
            "verdict": verdict,
            "created_at": pass_meta.get("created_at"),
            "git_commit": pass_meta.get("git_commit"),
            "source_path": source_path,
        },
        stats=stats,
        digest=dig,
    )


def _import_clusterings(sess, r, bundle, threshold, stats) -> None:
    if threshold is None:
        return
    for backend, c in (r.get("clusters") or {}).items():
        _get_or_create(
            sess,
            Clustering,
            {"n_clusters": c.get("n_clusters"), "labels": c.get("labels") or []},
            stats=stats,
            bundle_id=bundle.id,
            backend=backend,
            threshold=threshold,
        )


def _import_metrics(sess, r, bundle, judgment, stats) -> None:
    raw = r.get("metrics")
    if not raw:
        return
    metrics = Metrics.model_validate(
        {**raw, "agreement": r.get("agreement")}
    ).model_dump(mode="json", exclude_none=True)
    dig = keys.metrics_digest(
        bundle.digest, judgment.digest if judgment else None, metrics
    )
    _get_or_create(
        sess,
        BundleMetrics,
        {
            "bundle_id": bundle.id,
            "judgment_id": judgment.id if judgment else None,
            "metrics": metrics,
        },
        stats=stats,
        digest=dig,
    )


def _import_families(
    sess, analysis, models_cfg, jc, role, source_path, run, stats
) -> None:
    fmeta = analysis.get("families_meta") or {}
    for f in analysis.get("families") or []:
        payload = FamilyAnalysisPayload.model_validate(f).model_dump(mode="json")
        fid, mname = payload["family"], payload["model"]
        ident = models_cfg.get(mname) or ModelIdentity()
        model = _model_row(sess, mname, ident, stats)
        _get_or_create(sess, Family, stats=stats, id=fid)
        fv = None
        if fid in fmeta:
            fm = fmeta[fid]
            fv = _get_or_create(
                sess,
                FamilyVersion,
                {
                    "prompt": fm.get("prompt") or "",
                    "scalar": fm.get("scalar"),
                    "title": fm.get("title") or "",
                    "description": fm.get("description") or "",
                },
                stats=stats,
                family_id=fid,
                content_hash=keys.family_content_hash(fm),
            )
        dig = keys.family_analysis_digest(
            ident.inspect_model, ident.reasoning_effort, fid, role, payload
        )
        row = _get_or_create(
            sess,
            FamilyAnalysis,
            {
                "model_id": model.id,
                "family_version_id": fv.id if fv else None,
                "judge_config_id": jc.id if jc else None,
                "rep_label": role,
                "payload": payload,
                "source_path": source_path,
            },
            stats=stats,
            digest=dig,
        )
        if run is not None:
            _get_or_create(
                sess, RunFamilyAnalysis, run_id=run.id, family_analysis_id=row.id
            )


def import_gen_dir(sess: Session, gen_dir: Path, stats: ImportStats) -> None:
    meta = _json(gen_dir / "gen_meta.json", stats) or {}
    if meta.get("status") != "complete":
        stats.warn(f"{gen_dir}: incomplete generation, skipped")
        return
    run_config = _json(gen_dir / "run_config.json", stats) or {}
    questions = _json(gen_dir / "questions.json", stats) or {}
    provenance = _store_provenance(gen_dir, run_config)

    fragments = sorted((gen_dir / "judge").glob("*/analysis.json"))
    for frag in fragments:
        analysis = _json(frag, stats)
        if analysis:
            import_analysis(
                sess,
                analysis,
                source_path=str(frag.parent),
                stats=stats,
                config=run_config,
                provenance=provenance,
                pass_meta={"created_at": _mtime_iso(frag)},
            )
    if not fragments:
        pseudo = _responses_only_analysis(gen_dir, run_config, questions, stats)
        if pseudo:
            import_analysis(
                sess,
                pseudo,
                source_path=str(gen_dir),
                stats=stats,
                provenance=provenance,
                pass_meta={"created_at": _mtime_iso(gen_dir / "gen_meta.json")},
            )
    stats.created("gen_dirs")


def _store_provenance(gen_dir: Path, run_config: dict) -> dict[str, dict]:
    out = {}
    for mname in run_config.get("models") or {}:
        logs = sorted((gen_dir / "logs" / mname).glob("*.eval"))
        log = logs[-1] if logs else None
        out[mname] = {
            "created_at": _mtime_iso(log) if log else None,
            "log_path": str(log) if log else None,
            "log_sha256": _sha256(log) if log else None,
            "source_path": str(gen_dir),
        }
    return out


def _sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _responses_only_analysis(gen_dir, run_config, questions, stats) -> Optional[dict]:
    from ..analyze import load_responses

    try:
        responses = load_responses(gen_dir)
    except Exception as e:
        stats.warn(f"{gen_dir}: could not read generation logs ({e})")
        return None
    results = [
        {"model": mname, "question_id": qid, "responses": texts}
        for mname, per_q in sorted(responses.items())
        for qid, texts in per_q.items()
    ]
    if not results:
        return None
    return {"config": run_config, "questions": questions, "results": results}


def import_run_dir(sess: Session, run_dir: Path, stats: ImportStats) -> None:
    analysis = _json(run_dir / "analysis.json", stats)
    if analysis is None:
        return
    run_meta = (
        _json(run_dir / "run_meta.json", stats)
        if (run_dir / "run_meta.json").exists()
        else {}
    )
    run_meta = run_meta or {}
    kind = (
        "merged" if analysis.get("source_runs") else run_meta.get("kind") or "variance"
    )
    run = _get_or_create(
        sess,
        Run,
        {
            "name": run_dir.name,
            "kind": kind,
            "created_at": run_meta.get("created_at")
            or _mtime_iso(run_dir / "analysis.json"),
            "git_commit": run_meta.get("git_commit"),
            "config": _json(run_dir / "run_config.json", stats)
            if (run_dir / "run_config.json").exists()
            else analysis.get("config"),
            "cost": _json(run_dir / "cost.json", stats)
            if (run_dir / "cost.json").exists()
            else None,
        },
        stats=stats,
        source_path=str(run_dir.resolve()),
    )
    import_analysis(
        sess,
        analysis,
        source_path=str(run_dir),
        stats=stats,
        run=run,
        role="default",
        pass_meta=_pass_meta(run_dir, stats),
    )
    jr = run_dir / "judge_runs"
    if jr.is_dir():
        for pass_dir in sorted(p for p in jr.iterdir() if p.is_dir()):
            rep = (
                _json(pass_dir / "analysis.json", stats)
                if (pass_dir / "analysis.json").exists()
                else None
            )
            if rep:
                import_analysis(
                    sess,
                    rep,
                    source_path=str(pass_dir),
                    stats=stats,
                    run=run,
                    role=rep.get("judge_run") or pass_dir.name,
                    config=analysis.get("config"),
                    pass_meta=_pass_meta(pass_dir, stats),
                )
    stats.created("run_dirs")


def _pass_meta(pass_dir: Path, stats: ImportStats) -> dict:
    meta = (
        _json(pass_dir / "judge_meta.json", stats)
        if (pass_dir / "judge_meta.json").exists()
        else None
    ) or {}
    return {"created_at": meta.get("created_at"), "git_commit": meta.get("git_commit")}
