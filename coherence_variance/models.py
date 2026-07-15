"""Model resolution for the variance experiment.

Maps short roster names to Inspect model strings. ``ours/<name>`` is resolved
against the gitignored repo-root ``model_jsons.keys`` (schema in
``model_jsons.keys.example``) and prefixed ``openai/`` — register your own
fine-tunes there and run them via ``--models ours/<name>``.

5.2 (no thinking) vs 5.2-thinking map to ``reasoning_effort`` none vs low.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_KEYS_PATH = _REPO_ROOT / "model_jsons.keys"


@dataclass(frozen=True)
class ModelSpec:
    """A resolved model ready to hand to Inspect's ``get_model``."""

    name: str  # short label used in CLI args and results dirs
    inspect_model: (
        str  # e.g. "openai/gpt-4.1" or "openai/ft:gpt-4.1-...:your-org:my-finetune:..."
    )
    reasoning_effort: Optional[str] = None  # one of inspect's literals, or None
    display: str = ""


# short name -> (reference, reasoning_effort, display). The reference is resolved
# lazily by ``_resolve_ref`` so we only touch model_jsons.keys for ours/ entries.
_ROSTER_REFS: dict[str, tuple[str, Optional[str], str]] = {
    "gpt-4.1": ("openai/gpt-4.1", None, "GPT-4.1 (baseline)"),
    "gpt-5.2": ("openai/gpt-5.2", "none", "GPT-5.2 (no thinking)"),
    "gpt-5.2-thinking": ("openai/gpt-5.2", "low", "GPT-5.2 (thinking)"),
    # Flagship OpenAI capability ladder (gpt-4o -> 5.4) for the variance sweep.
    # 4o is non-reasoning (effort None); the 5-family no-thinking rungs pin
    # reasoning_effort="none" (like the gpt-5.2 rung) so they run WITHOUT thinking,
    # and the -thinking rungs use "low" (like gpt-5.2-thinking).
    "gpt-4o": ("openai/gpt-4o", None, "GPT-4o"),
    # Original gpt-5 does NOT accept reasoning_effort="none" (API: only
    # minimal/low/medium/high); "minimal" is its lowest/no-thinking floor.
    # gpt-5.2 and gpt-5.4 added "none", so their no-thinking rungs use it.
    "gpt-5": ("openai/gpt-5", "minimal", "GPT-5 (minimal thinking)"),
    "gpt-5-thinking": ("openai/gpt-5", "low", "GPT-5 (thinking)"),
    "gpt-5.4": ("openai/gpt-5.4", "none", "GPT-5.4 (no thinking)"),
    "gpt-5.4-thinking": ("openai/gpt-5.4", "low", "GPT-5.4 (thinking)"),
    # Reasoning-effort ladder on a single model (low/medium/high), for the
    # thinking-vs-coherence comparison: same gpt-5.4 weights, only the
    # reasoning_effort knob changes. (gpt-5.4-low == gpt-5.4-thinking; named
    # explicitly so the three rungs read as a set.) gpt-5.4 supports the full
    # API range none/minimal/low/medium/high.
    "gpt-5.4-low": ("openai/gpt-5.4", "low", "GPT-5.4 (low reasoning)"),
    "gpt-5.4-medium": ("openai/gpt-5.4", "medium", "GPT-5.4 (medium reasoning)"),
    "gpt-5.4-high": ("openai/gpt-5.4", "high", "GPT-5.4 (high reasoning)"),
    # Small / efficiency models for a capability comparison (no thinking).
    # gpt-4o-mini is non-reasoning; gpt-5.4-mini is a reasoning-family model, so
    # it needs reasoning_effort="none" to run WITHOUT thinking (matching how the
    # plain "gpt-5.2" rung pins effort=none) — otherwise it inherits the model's
    # default thinking budget.
    "gpt-4o-mini": ("openai/gpt-4o-mini", None, "GPT-4o Mini"),
    "gpt-5.4-mini": ("openai/gpt-5.4-mini", "none", "GPT-5.4 Mini (no thinking)"),
    # Size-ladder fill-ins: 4.1 family is non-reasoning (effort None); 5.4-nano
    # is reasoning-family so it pins effort="none" like the other 5.4 rungs.
    "gpt-4.1-mini": ("openai/gpt-4.1-mini", None, "GPT-4.1 Mini"),
    "gpt-4.1-nano": ("openai/gpt-4.1-nano", None, "GPT-4.1 Nano"),
    "gpt-5.4-nano": ("openai/gpt-5.4-nano", "none", "GPT-5.4 Nano (no thinking)"),
    # --- Hot Mess paper frontier reasoning roster ----------------------------
    # The reasoning-model selection from Hägele et al. (ICLR 2026): Claude
    # Sonnet 4 with extended thinking, plus o3-mini / o4-mini at low reasoning.
    # Opt-in via --models (or the HOTMESS_MODELS list below). Sonnet 4 is routed
    # via OpenRouter (needs OPENROUTER_API_KEY; o-series need OPENAI_API_KEY,
    # both in .env). NOTE: OpenRouter emits the cosmetic "Error parsing
    # OpenRouter reasoning details" spam (same as the default opus judge) — it
    # does not affect the response content.
    #
    # max_tokens via OpenRouter: reasoning_effort "low" is sent as the
    # OpenRouter reasoning option {effort:"low"}, which OpenRouter translates to
    # ~20% of max_tokens as the thinking budget (NOT the fixed 4096-token budget
    # of the direct anthropic/ path — so there is no hard max_tokens>budget
    # rejection). Still run with --max-tokens 8192 so the thinking + the answer
    # both fit comfortably. The o-series rungs ignore temperature (reasoning
    # models force temperature=1), which is exactly the pipeline's
    # temperature=1.0 sampling regime.
    "claude-sonnet-4": (
        "openrouter/anthropic/claude-sonnet-4",
        "low",
        "Claude Sonnet 4 (thinking, low)",
    ),
    "o3-mini": ("openai/o3-mini", "low", "o3-mini (low reasoning)"),
    "o4-mini": ("openai/o4-mini", "low", "o4-mini (low reasoning)"),
}

# Convenient aliases accepted on the CLI.
_ALIASES = {
    "4.1": "gpt-4.1",
    "gpt4.1": "gpt-4.1",
    "5.2": "gpt-5.2",
    "gpt-5.2-no-thinking": "gpt-5.2",
    "5.2-thinking": "gpt-5.2-thinking",
    "4o": "gpt-4o",
    "5": "gpt-5",
    "5-thinking": "gpt-5-thinking",
    "5.4": "gpt-5.4",
    "5.4-thinking": "gpt-5.4-thinking",
    "4o-mini": "gpt-4o-mini",
    "gpt4o-mini": "gpt-4o-mini",
    "5.4-mini": "gpt-5.4-mini",
    "gpt-5.4-mini-no-thinking": "gpt-5.4-mini",
    "4.1-mini": "gpt-4.1-mini",
    "4.1-nano": "gpt-4.1-nano",
    "5.4-nano": "gpt-5.4-nano",
    "gpt-5.4-nano-no-thinking": "gpt-5.4-nano",
    # Hot Mess frontier roster.
    "sonnet-4": "claude-sonnet-4",
    "sonnet4": "claude-sonnet-4",
    "claude-sonnet4": "claude-sonnet-4",
    "o3mini": "o3-mini",
    "o4mini": "o4-mini",
}

# Default roster, in display order. Register your own fine-tunes in
# model_jsons.keys and run them via --models ours/<name>.
DEFAULT_MODELS = [
    "gpt-4.1",
    "gpt-5.2",
    "gpt-5.2-thinking",
]

# Hot Mess paper frontier reasoning roster (Hägele et al.), opt-in via --models:
#   --models claude-sonnet-4,o3-mini,o4-mini --max-tokens 8192
# The --max-tokens bump is REQUIRED — see the max_tokens caveat in the roster
# block above (Sonnet 4's "low" thinking budget is 4096 and must fit under it).
HOTMESS_MODELS = ["claude-sonnet-4", "o3-mini", "o4-mini"]

# Default judge: latest thinking Claude via OpenRouter at low effort.
DEFAULT_JUDGE = "openrouter/anthropic/claude-opus-4.8"
DEFAULT_JUDGE_REASONING = "low"


def _load_keys() -> dict[str, str]:
    if not _KEYS_PATH.exists():
        raise FileNotFoundError(
            f"{_KEYS_PATH} not found. Copy model_jsons.keys.example to "
            "model_jsons.keys and fill in your own fine-tune IDs "
            "(JSON: {short_name: full_model_id})."
        )
    return json.loads(_KEYS_PATH.read_text())


def _resolve_ref(ref: str) -> str:
    """Resolve a roster reference to an Inspect model string."""
    if ref.startswith("ours/"):
        short = ref[len("ours/") :]
        keys = _load_keys()
        if short not in keys:
            raise KeyError(
                f"'{short}' not in model_jsons.keys. Add it there "
                "(see model_jsons.keys.example for the schema)."
            )
        return f"openai/{keys[short]}"
    if "/" in ref:  # already provider-qualified (openai/..., openrouter/..., etc.)
        return ref
    return f"openai/{ref}"  # bare model id -> assume OpenAI


def _sanitize(s: str) -> str:
    """Make a name filesystem-safe (spec names double as log-dir names)."""
    return re.sub(r"[^A-Za-z0-9._:+-]", "_", s)


def _short_name(ref: str, segments: int = 1) -> str:
    """Short name from the last ``segments`` path segments of a model reference.

    ``openrouter/qwen/qwen3-32b`` -> ``qwen3-32b`` (segments=1),
    ``qwen_qwen3-32b`` (segments=2), ``openrouter_qwen_qwen3-32b`` (segments=3).
    """
    parts = ref.split("/")
    return _sanitize("_".join(parts[-segments:]))


def resolve_model(name: str) -> ModelSpec:
    """Resolve a CLI model name (roster short name, alias, ours/<x>, or provider/id)."""
    key = _ALIASES.get(name, name)
    if key in _ROSTER_REFS:
        ref, effort, display = _ROSTER_REFS[key]
        return ModelSpec(
            name=key,
            inspect_model=_resolve_ref(ref),
            reasoning_effort=effort,
            display=display,
        )
    # Arbitrary model not in the roster. The spec name doubles as the per-model
    # log-dir name (results/<run>/logs/<name>/), so use the (sanitized) last
    # path segment: "openrouter/qwen/qwen3-32b" -> "qwen3-32b". resolve_models
    # lengthens colliding names; `display` keeps the original user string.
    return ModelSpec(
        name=_short_name(name),
        inspect_model=_resolve_ref(name),
        reasoning_effort=None,
        display=name,
    )


def next_name(ref: str, current: str) -> Optional[str]:
    """The next-longer short name for ``ref`` after ``current``, or None when
    the reference has no more path segments to add. Used to resolve short-name
    collisions — within one batch (:func:`resolve_models`) and against the
    per-model store across invocations (the store identity guard)."""
    for k in range(2, ref.count("/") + 2):
        cand = _short_name(ref, segments=k)
        if cand != current:
            return cand
    return None


def resolve_models(names: list[str]) -> list[ModelSpec]:
    """Resolve a batch of CLI names, disambiguating short-name collisions.

    Colliding specs (same short name, different models) are lengthened to the
    last 2, 3, ... path segments of their original reference until unique.
    Exact duplicates (same resolved model requested twice) raise ValueError.
    """
    specs = [resolve_model(n) for n in names]
    for _ in range(8):  # segment-lengthening rounds; model refs are short
        by_name: dict[str, list[int]] = {}
        for i, s in enumerate(specs):
            by_name.setdefault(s.name, []).append(i)
        collisions = {k: idxs for k, idxs in by_name.items() if len(idxs) > 1}
        if not collisions:
            return specs
        for short, idxs in collisions.items():
            if len({specs[i].inspect_model for i in idxs}) < len(idxs):
                raise ValueError(f"model '{short}' requested more than once")
            for i in idxs:
                # The original user string carries the path segments.
                cand = next_name(names[i], specs[i].name)
                if cand is not None:
                    specs[i] = replace(specs[i], name=cand)
    # Ran out of segments to add (e.g. same full ref under different aliases).
    dupes = sorted({s.name for s in specs if [t.name for t in specs].count(s.name) > 1})
    raise ValueError(f"could not disambiguate model names: {', '.join(dupes)}")


def cohort_of(name: str) -> str:
    """Classify a roster model as ``"finetuned"`` (user-registered ``ours/<x>``
    fine-tunes) or ``"base"`` (base / frontier models). Used by the families report to split the
    grouped-bar chart into fine-tuned-organism vs base/frontier cohorts. Unknown
    names default to ``"base"``.
    """
    key = _ALIASES.get(name, name)
    ref = _ROSTER_REFS.get(key, (None,))[0]
    return "finetuned" if (ref or "").startswith("ours/") else "base"
