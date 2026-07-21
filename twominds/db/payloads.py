"""Pydantic payloads for the schema's JSON columns, validated on write.
Extra fields are allowed and kept verbatim: analyses have grown fields over
time, so required fields are only what every known producer emits."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class _Payload(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())


class JudgeVerdict(_Payload):
    """result['judge'] + the sibling result['judge_labels'] as ``labels``."""

    contradiction: Optional[bool] = None
    groups: list[list[int]]
    n_groups: int
    group_names: Optional[list[str]] = None
    rationale: str = ""
    flags: Optional[list] = None
    parse_ok: Optional[bool] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    labels: Optional[list[int]] = None


class Metrics(_Payload):
    """result['metrics'] + result['agreement'] as ``agreement`` (judge-vs-
    cluster ARI/NMI per backend — it describes the same pass)."""

    n: Optional[int] = None
    len_mean: Optional[float] = None
    len_cv: Optional[float] = None
    n_unique_verbatim: Optional[int] = None
    n_judge_groups: Optional[int] = None
    n_clusters: Optional[int] = None
    group_entropy: Optional[float] = None
    agreement: Optional[dict[str, dict[str, Optional[float]]]] = None


class FamilyVariant(_Payload):
    variant: str
    question_id: Optional[str] = None
    n: Optional[int] = None
    groups: Optional[list[int]] = None


class FamilyAnalysisPayload(_Payload):
    """One entry of analysis['families']."""

    model: str
    family: str
    title: Optional[str] = None
    description: Optional[str] = None
    scalar_kind: Optional[str] = None
    variants: list[FamilyVariant]
    n_total: Optional[int] = None
    scalar: Optional[dict] = None
    judge: Optional[dict] = None
    cluster: Optional[dict] = None


class ModelIdentity(_Payload):
    inspect_model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    display: Optional[str] = None


class RunConfig(_Payload):
    """run_config.json / analysis['config']."""

    models: Optional[dict[str, ModelIdentity]] = None
    question_ids: Optional[list[str]] = None
    n: Optional[int] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    judge: Optional[str] = None
