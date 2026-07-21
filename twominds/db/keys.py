"""Content hashes and row digests over canonical JSON — the schema's
identity layer. Changing any function's inputs is a schema migration:
existing digests stop matching."""

from __future__ import annotations

import hashlib
import json
from typing import Optional

_LEN = 16


def _hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:_LEN]


def question_content_hash(meta: dict) -> str:
    """Over the per-question fields ``store.compute_gen_key`` hashes."""
    return _hash(
        {
            k: meta.get(k)
            for k in ("prompt", "system", "group", "bucket", "family", "variant")
        }
    )


def family_content_hash(meta: dict) -> str:
    return _hash({k: meta.get(k) for k in ("prompt", "scalar", "title", "description")})


def bundle_digest(
    inspect_model: Optional[str],
    reasoning_effort: Optional[str],
    question_hash: str,
    texts: list[str],
) -> str:
    return _hash(
        {
            "inspect_model": inspect_model,
            "reasoning_effort": reasoning_effort or None,
            "question": question_hash,
            "texts": texts,
        }
    )


def judgment_digest(
    bundle_dig: str,
    judge_model: str,
    judge_reasoning: str,
    prompt_hash: str,
    rep_label: str,
    verdict: dict,
) -> str:
    return _hash(
        {
            "bundle": bundle_dig,
            "judge": [judge_model, judge_reasoning, prompt_hash],
            "rep": rep_label,
            "verdict": verdict,
        }
    )


def metrics_digest(bundle_dig: str, judgment_dig: Optional[str], metrics: dict) -> str:
    return _hash({"bundle": bundle_dig, "judgment": judgment_dig, "metrics": metrics})


def batch_digest(
    inspect_model: Optional[str],
    reasoning_effort: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    n: Optional[int],
    discriminator: str,
) -> str:
    """``discriminator`` separates batches with identical settings: the store
    gen_key + log name for real batches, the sorted new-bundle digests for
    batches synthesized from an analysis."""
    return _hash(
        {
            "inspect_model": inspect_model,
            "reasoning_effort": reasoning_effort or None,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": n,
            "d": discriminator,
        }
    )


def family_analysis_digest(
    inspect_model: Optional[str],
    reasoning_effort: Optional[str],
    family_id: str,
    rep_label: str,
    payload: dict,
) -> str:
    return _hash(
        {
            "inspect_model": inspect_model,
            "reasoning_effort": reasoning_effort or None,
            "family": family_id,
            "rep": rep_label,
            "payload": payload,
        }
    )
