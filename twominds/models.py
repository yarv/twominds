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
    # --- Frontier API models via OpenRouter (need OPENROUTER_API_KEY) ---------
    # One OpenRouter key covers the Anthropic / Google / xAI rungs below (same
    # routing as the sonnet-4 rung above; its max_tokens note applies to every
    # -thinking rung here — run with --max-tokens 8192 so thinking + answer both
    # fit). Plain Claude rungs send NO reasoning param (extended thinking is off
    # by default); -thinking rungs use effort "low". Slugs + prices verified
    # against the OpenRouter model catalog, 2026-08.
    "claude-opus-5": (
        "openrouter/anthropic/claude-opus-5",
        None,
        "Claude Opus 5 (no thinking)",
    ),
    "claude-opus-5-thinking": (
        "openrouter/anthropic/claude-opus-5",
        "low",
        "Claude Opus 5 (thinking)",
    ),
    "claude-sonnet-5": (
        "openrouter/anthropic/claude-sonnet-5",
        None,
        "Claude Sonnet 5 (no thinking)",
    ),
    "claude-sonnet-5-thinking": (
        "openrouter/anthropic/claude-sonnet-5",
        "low",
        "Claude Sonnet 5 (thinking)",
    ),
    "claude-haiku-4.5": (
        "openrouter/anthropic/claude-haiku-4.5",
        None,
        "Claude Haiku 4.5 (no thinking)",
    ),
    "claude-haiku-4.5-thinking": (
        "openrouter/anthropic/claude-haiku-4.5",
        "low",
        "Claude Haiku 4.5 (thinking)",
    ),
    # The default judge's model as an eval subject (see DEFAULT_JUDGE below).
    "claude-opus-4.8": (
        "openrouter/anthropic/claude-opus-4.8",
        None,
        "Claude Opus 4.8 (no thinking)",
    ),
    "claude-opus-4.8-thinking": (
        "openrouter/anthropic/claude-opus-4.8",
        "low",
        "Claude Opus 4.8 (thinking)",
    ),
    # Gemini cannot disable thinking (effort "none" is rejected with
    # "Reasoning is mandatory for this endpoint", verified 2026-08), so both
    # rungs are single thinking rungs (OpenRouter lists only the -preview slug
    # for the Pro).
    "gemini-3.1-pro": (
        "openrouter/google/gemini-3.1-pro-preview",
        "low",
        "Gemini 3.1 Pro (thinking, low)",
    ),
    "gemini-3.6-flash": (
        "openrouter/google/gemini-3.6-flash",
        "low",
        "Gemini 3.6 Flash (thinking, low)",
    ),
    # Grok is always-reasoning (like gpt-5): a single rung at the effort floor.
    "grok-4.5": ("openrouter/x-ai/grok-4.5", "low", "Grok 4.5 (reasoning, low)"),
    # --- Open-weight models via OpenRouter (need OPENROUTER_API_KEY) ----------
    # Opt-in via --models. Llama 3.3 70B is non-reasoning (effort None). DeepSeek
    # and Qwen are reasoning-family, so each gets two rungs mirroring the
    # gpt-5.2 / gpt-5.2-thinking pair: the no-thinking rung pins
    # reasoning_effort="none" to run WITHOUT thinking (else it inherits the
    # model's default thinking budget), the -thinking rung uses "low". Over
    # OpenRouter "low" is sent as the reasoning option {effort:"low"}; if a model
    # rejects effort="none", drop the no-thinking rung to a bare-string --models
    # call. Verify the exact OpenRouter slugs against their model catalog.
    "llama-3.3-70b": (
        "openrouter/meta-llama/llama-3.3-70b-instruct",
        None,
        "Llama 3.3 70B",
    ),
    "deepseek-v4-flash": (
        "openrouter/deepseek/deepseek-v4-flash",
        "none",
        "DeepSeek V4 Flash (no thinking)",
    ),
    "deepseek-v4-flash-thinking": (
        "openrouter/deepseek/deepseek-v4-flash",
        "low",
        "DeepSeek V4 Flash (thinking)",
    ),
    "qwen3.7-plus": (
        "openrouter/qwen/qwen3.7-plus",
        "none",
        "Qwen3.7 Plus (no thinking)",
    ),
    "qwen3.7-plus-thinking": (
        "openrouter/qwen/qwen3.7-plus",
        "low",
        "Qwen3.7 Plus (thinking)",
    ),
    # Qwen3.5 open-weight size ladder (9B / 27B dense, 122B-A10B / 397B-A17B
    # MoE) for capability-vs-coherence comparisons. Hybrid-thinking models whose
    # OpenRouter default is thinking ON — the 9B rung then runs ~50s/call and
    # blows the 120s attempt timeout — so the plain rungs pin effort="none"
    # (verified live 2026-08-26: OpenRouter honours it for all four, 0 reasoning
    # tokens, ~2s/call) and the -thinking rungs use "low". Short names are the
    # OpenRouter slug tails, so they match the labels of bare-string runs.
    "qwen3.5-9b": ("openrouter/qwen/qwen3.5-9b", "none", "Qwen3.5 9B (no thinking)"),
    "qwen3.5-9b-thinking": (
        "openrouter/qwen/qwen3.5-9b",
        "low",
        "Qwen3.5 9B (thinking)",
    ),
    "qwen3.5-27b": (
        "openrouter/qwen/qwen3.5-27b",
        "none",
        "Qwen3.5 27B (no thinking)",
    ),
    "qwen3.5-27b-thinking": (
        "openrouter/qwen/qwen3.5-27b",
        "low",
        "Qwen3.5 27B (thinking)",
    ),
    # 35B-A3B MoE rung. families_v2 (2026-08-26) ran it as a bare slug with no
    # effort pin and it thought by default (~1.5M reasoning tokens); pinned
    # here like its siblings so the tier-1 ladder runs it with thinking off.
    "qwen3.5-35b-a3b": (
        "openrouter/qwen/qwen3.5-35b-a3b",
        "none",
        "Qwen3.5 35B-A3B (no thinking)",
    ),
    "qwen3.5-35b-a3b-thinking": (
        "openrouter/qwen/qwen3.5-35b-a3b",
        "low",
        "Qwen3.5 35B-A3B (thinking)",
    ),
    "qwen3.5-122b-a10b": (
        "openrouter/qwen/qwen3.5-122b-a10b",
        "none",
        "Qwen3.5 122B-A10B (no thinking)",
    ),
    "qwen3.5-122b-a10b-thinking": (
        "openrouter/qwen/qwen3.5-122b-a10b",
        "low",
        "Qwen3.5 122B-A10B (thinking)",
    ),
    "qwen3.5-397b-a17b": (
        "openrouter/qwen/qwen3.5-397b-a17b",
        "none",
        "Qwen3.5 397B-A17B (no thinking)",
    ),
    "qwen3.5-397b-a17b-thinking": (
        "openrouter/qwen/qwen3.5-397b-a17b",
        "low",
        "Qwen3.5 397B-A17B (thinking)",
    ),
    "llama-4-maverick": (
        "openrouter/meta-llama/llama-4-maverick",
        None,
        "Llama 4 Maverick",
    ),
    "llama-4-scout": ("openrouter/meta-llama/llama-4-scout", None, "Llama 4 Scout"),
    # Kimi K3 / GLM-5.2: reasoning control over OpenRouter is unverified for
    # these, so no reasoning param is sent (the model's default behavior); add
    # explicit none/low rungs once verified against a live response.
    "kimi-k3": ("openrouter/moonshotai/kimi-k3", None, "Kimi K3"),
    "glm-5.2": ("openrouter/z-ai/glm-5.2", None, "GLM-5.2"),
    "mistral-large-2512": (
        "openrouter/mistralai/mistral-large-2512",
        None,
        "Mistral Large 2512",
    ),
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
    # Frontier API roster. Aliases stay versioned on purpose — a bare "opus"
    # or "deepseek" is ambiguous across generations/variants and would change
    # meaning under your feet; these only drop the "claude-" prefix.
    "opus-5": "claude-opus-5",
    "opus-5-thinking": "claude-opus-5-thinking",
    "opus-4.8": "claude-opus-4.8",
    "opus-4.8-thinking": "claude-opus-4.8-thinking",
    "sonnet-5": "claude-sonnet-5",
    "sonnet-5-thinking": "claude-sonnet-5-thinking",
    "haiku-4.5": "claude-haiku-4.5",
    "haiku-4.5-thinking": "claude-haiku-4.5-thinking",
    # Open-weight OpenRouter roster.
    "llama-3.3-70b-instruct": "llama-3.3-70b",
    "llama3.3-70b": "llama-3.3-70b",
    "deepseek-v4-flash-no-thinking": "deepseek-v4-flash",
    "deepseek-v4-flash-low": "deepseek-v4-flash-thinking",
    "qwen3.7-plus-no-thinking": "qwen3.7-plus",
    "qwen3.7-plus-low": "qwen3.7-plus-thinking",
    # Qwen3.5 size ladder: the -a10b / -a17b active-parameter suffixes are
    # part of the OpenRouter slugs; these drop them for typing convenience.
    "qwen3.5-35b": "qwen3.5-35b-a3b",
    "qwen3.5-35b-thinking": "qwen3.5-35b-a3b-thinking",
    "qwen3.5-122b": "qwen3.5-122b-a10b",
    "qwen3.5-122b-thinking": "qwen3.5-122b-a10b-thinking",
    "qwen3.5-397b": "qwen3.5-397b-a17b",
    "qwen3.5-397b-thinking": "qwen3.5-397b-a17b-thinking",
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
# Concurrent judge calls (the judge model's Inspect max_connections; Inspect's
# max_samples follows it, so this one knob caps in-flight judge bundles).
DEFAULT_JUDGE_CONCURRENCY = 16


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
    collisions within one batch (:func:`resolve_models`)."""
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
