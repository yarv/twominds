"""Database schema (DATABASE_PLAN.md §3): content stored once under a
surrogate id + content hash/digest; unique digests make every ingest
idempotent. JSON columns hold ``payloads.py``-validated payloads; a ``run``
is a saved view linked to shared rows via the ``run_*`` tables."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import ForeignKey, Index, LargeBinary, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    type_annotation_map = {dict: JSON, list: JSON}


class Model(Base):
    """``name`` unique per (inspect_model, reasoning_effort) identity;
    identity columns are None for legacy imports that never recorded it."""

    __tablename__ = "model"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)
    inspect_model: Mapped[Optional[str]]
    reasoning_effort: Mapped[Optional[str]]
    display: Mapped[str] = mapped_column(default="")


class Question(Base):
    """Stable logical id; content lives in versions."""

    __tablename__ = "question"

    id: Mapped[str] = mapped_column(primary_key=True)
    first_seen_at: Mapped[Optional[str]]


class QuestionVersion(Base):
    """``content_hash`` covers the per-question fields of
    ``store.compute_gen_key``; ``extra`` keeps import-only legacy fields."""

    __tablename__ = "question_version"
    __table_args__ = (UniqueConstraint("question_id", "content_hash"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    question_id: Mapped[str] = mapped_column(ForeignKey("question.id"))
    content_hash: Mapped[str]
    prompt: Mapped[str]
    system: Mapped[Optional[str]]
    group_name: Mapped[str]
    bucket: Mapped[Optional[str]]
    family_id: Mapped[Optional[str]] = mapped_column(ForeignKey("family.id"))
    variant: Mapped[Optional[str]]
    extra: Mapped[Optional[dict]]
    created_at: Mapped[Optional[str]]


class Family(Base):
    __tablename__ = "family"

    id: Mapped[str] = mapped_column(primary_key=True)


class FamilyVersion(Base):
    __tablename__ = "family_version"
    __table_args__ = (UniqueConstraint("family_id", "content_hash"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    family_id: Mapped[str] = mapped_column(ForeignKey("family.id"))
    content_hash: Mapped[str]
    prompt: Mapped[str]
    scalar: Mapped[Optional[str]]
    title: Mapped[str] = mapped_column(default="")
    description: Mapped[str] = mapped_column(default="")


class JudgeConfig(Base):
    """``prompt_hash`` is judge.PROMPT_HASH, or "unknown" for imports;
    ``judge_reasoning`` uses "" for none so the unique constraint bites."""

    __tablename__ = "judge_config"
    __table_args__ = (
        UniqueConstraint("judge_model", "judge_reasoning", "prompt_hash"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    judge_model: Mapped[str]
    judge_reasoning: Mapped[str] = mapped_column(default="")
    prompt_hash: Mapped[str] = mapped_column(default="unknown")
    prompt_text: Mapped[Optional[str]]


class GenBatch(Base):
    """One act of generation. Store-backed imports carry the Inspect log's
    path + sha256; batches synthesized from an analysis.json carry only
    ``source_path``."""

    __tablename__ = "gen_batch"

    id: Mapped[int] = mapped_column(primary_key=True)
    digest: Mapped[str] = mapped_column(unique=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("model.id"))
    temperature: Mapped[Optional[float]]
    max_tokens: Mapped[Optional[int]]
    n_requested: Mapped[Optional[int]]
    status: Mapped[str] = mapped_column(default="complete")
    created_at: Mapped[Optional[str]]
    git_commit: Mapped[Optional[str]]
    inspect_eval_id: Mapped[Optional[str]]
    log_path: Mapped[Optional[str]]
    log_sha256: Mapped[Optional[str]]
    in_tok: Mapped[Optional[int]]
    out_tok: Mapped[Optional[int]]
    est_dollars: Mapped[Optional[float]]
    origin: Mapped[str] = mapped_column(default="import")
    source_path: Mapped[Optional[str]]


class Response(Base):
    __tablename__ = "response"
    __table_args__ = (
        UniqueConstraint("batch_id", "question_version_id", "sample_index"),
        Index("ix_response_qv", "question_version_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("gen_batch.id"))
    question_version_id: Mapped[int] = mapped_column(ForeignKey("question_version.id"))
    sample_index: Mapped[int]
    text: Mapped[str]


class Bundle(Base):
    """One model's N ordered responses to one question version — the judged
    unit. Content-addressed, so every source resolves to one row."""

    __tablename__ = "bundle"
    __table_args__ = (
        Index("ix_bundle_model", "model_id"),
        Index("ix_bundle_qv", "question_version_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    digest: Mapped[str] = mapped_column(unique=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("model.id"))
    question_version_id: Mapped[int] = mapped_column(ForeignKey("question_version.id"))
    n: Mapped[int]


class BundleResponse(Base):
    __tablename__ = "bundle_response"
    __table_args__ = (UniqueConstraint("bundle_id", "response_id"),)

    bundle_id: Mapped[int] = mapped_column(ForeignKey("bundle.id"), primary_key=True)
    position: Mapped[int] = mapped_column(primary_key=True)
    response_id: Mapped[int] = mapped_column(ForeignKey("response.id"))


class Judgment(Base):
    """One verdict (payloads.JudgeVerdict) over one bundle; repeat passes
    are distinct rows per ``rep_label``."""

    __tablename__ = "judgment"
    __table_args__ = (Index("ix_judgment_bundle", "bundle_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    digest: Mapped[str] = mapped_column(unique=True)
    bundle_id: Mapped[int] = mapped_column(ForeignKey("bundle.id"))
    judge_config_id: Mapped[int] = mapped_column(ForeignKey("judge_config.id"))
    rep_label: Mapped[str] = mapped_column(default="default")
    contradiction: Mapped[Optional[bool]]
    n_groups: Mapped[Optional[int]]
    verdict: Mapped[dict]
    created_at: Mapped[Optional[str]]
    git_commit: Mapped[Optional[str]]
    log_path: Mapped[Optional[str]]
    origin: Mapped[str] = mapped_column(default="import")
    source_path: Mapped[Optional[str]]


class Clustering(Base):
    __tablename__ = "clustering"
    __table_args__ = (UniqueConstraint("bundle_id", "backend", "threshold"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    bundle_id: Mapped[int] = mapped_column(ForeignKey("bundle.id"))
    backend: Mapped[str]
    threshold: Mapped[float]
    n_clusters: Mapped[int]
    labels: Mapped[list]


class BundleMetrics(Base):
    """Metrics of one analysis pass; ``judgment_id`` None when judgeless."""

    __tablename__ = "bundle_metrics"
    __table_args__ = (Index("ix_metrics_bundle", "bundle_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    digest: Mapped[str] = mapped_column(unique=True)
    bundle_id: Mapped[int] = mapped_column(ForeignKey("bundle.id"))
    judgment_id: Mapped[Optional[int]] = mapped_column(ForeignKey("judgment.id"))
    metrics: Mapped[dict]


class FamilyAnalysis(Base):
    """payloads.FamilyAnalysisPayload for one (model, family, rep)."""

    __tablename__ = "family_analysis"

    id: Mapped[int] = mapped_column(primary_key=True)
    digest: Mapped[str] = mapped_column(unique=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("model.id"))
    family_version_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("family_version.id")
    )
    judge_config_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("judge_config.id")
    )
    rep_label: Mapped[str] = mapped_column(default="default")
    payload: Mapped[dict]
    source_path: Mapped[Optional[str]]


class Embedding(Base):
    """Float32 bytes, one per (response, backend). Pipeline-written; the
    importer skips legacy .npz caches (regenerable for cents)."""

    __tablename__ = "embedding"

    response_id: Mapped[int] = mapped_column(
        ForeignKey("response.id"), primary_key=True
    )
    backend: Mapped[str] = mapped_column(primary_key=True)
    dim: Mapped[int]
    vec: Mapped[bytes] = mapped_column(LargeBinary)


class Run(Base):
    """A saved view over shared rows — what a run dir used to be."""

    __tablename__ = "run"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    kind: Mapped[str] = mapped_column(default="variance")
    origin: Mapped[str] = mapped_column(default="import")
    source_path: Mapped[Optional[str]] = mapped_column(unique=True)
    created_at: Mapped[Optional[str]]
    git_commit: Mapped[Optional[str]]
    config: Mapped[Optional[dict]]
    cost: Mapped[Optional[dict]]
    notes: Mapped[Optional[str]]


class RunBundle(Base):
    __tablename__ = "run_bundle"

    run_id: Mapped[int] = mapped_column(ForeignKey("run.id"), primary_key=True)
    bundle_id: Mapped[int] = mapped_column(ForeignKey("bundle.id"), primary_key=True)


class RunJudgment(Base):
    __tablename__ = "run_judgment"

    run_id: Mapped[int] = mapped_column(ForeignKey("run.id"), primary_key=True)
    judgment_id: Mapped[int] = mapped_column(
        ForeignKey("judgment.id"), primary_key=True
    )
    role: Mapped[str] = mapped_column(default="default")


class RunFamilyAnalysis(Base):
    __tablename__ = "run_family_analysis"

    run_id: Mapped[int] = mapped_column(ForeignKey("run.id"), primary_key=True)
    family_analysis_id: Mapped[int] = mapped_column(
        ForeignKey("family_analysis.id"), primary_key=True
    )
