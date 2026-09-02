"""Dry-run planning + rough cost estimation for the sweep.

Print the model x question x N plan and a *rough* dollar estimate before any
API call. Prices are best-effort and clearly labelled as approximate.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import ModelSpec
from .questions import Question

# Rough $/1M tokens (input, output). Fine-tunes priced ~1.5x base. Figures are
# approximate list prices (OpenRouter rungs: catalog prices, 2026-08) — treat
# the total as an order-of-magnitude guide. Every _ROSTER_REFS key needs an
# entry here (enforced by tests) so --dry-run never falls back to the assumed
# default price for a named roster model.
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "gpt-5": (1.25, 10.0),
    "gpt-5-thinking": (1.25, 10.0),
    "gpt-5.2": (1.75, 14.0),
    "gpt-5.2-thinking": (1.75, 14.0),
    "gpt-5.4": (2.5, 15.0),
    "gpt-5.4-thinking": (2.5, 15.0),
    "gpt-5.4-low": (2.5, 15.0),
    "gpt-5.4-medium": (2.5, 15.0),
    "gpt-5.4-high": (2.5, 15.0),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4-nano": (0.2, 1.25),
    "claude-sonnet-4": (3.0, 15.0),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    # Frontier API roster via OpenRouter.
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-5-thinking": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-5-thinking": (2.0, 10.0),
    "claude-haiku-4.5": (1.0, 5.0),
    "claude-haiku-4.5-thinking": (1.0, 5.0),
    "claude-opus-4.8": (5.0, 25.0),
    "claude-opus-4.8-thinking": (5.0, 25.0),
    "gemini-3.1-pro": (2.0, 12.0),
    "gemini-3.6-flash": (1.5, 7.5),
    "grok-4.5": (2.0, 6.0),
    # Open-weight models via OpenRouter.
    "llama-3.3-70b": (0.1, 0.32),
    # Llama 3.x size ladder, passed as bare OpenRouter slugs (spec name = slug
    # tail). OpenRouter list prices 2026-08-28.
    "llama-3.2-1b-instruct": (0.027, 0.20),
    "llama-3.2-3b-instruct": (0.05, 0.33),
    "llama-3.1-8b-instruct": (0.05, 0.08),
    "llama-3.1-70b-instruct": (0.40, 0.40),
    "llama-4-maverick": (0.2, 0.8),
    "llama-4-scout": (0.1, 0.3),
    "deepseek-v4-flash": (0.14, 0.28),
    "deepseek-v4-flash-thinking": (0.14, 0.28),
    "qwen3.7-plus": (0.32, 1.28),
    "qwen3.7-plus-thinking": (0.32, 1.28),
    # Qwen3.5 size ladder (OpenRouter catalog, 2026-08-26).
    "qwen3.5-9b": (0.10, 0.15),
    "qwen3.5-9b-thinking": (0.10, 0.15),
    "qwen3.5-27b": (0.20, 1.56),
    "qwen3.5-27b-thinking": (0.20, 1.56),
    "qwen3.5-35b-a3b": (0.25, 1.25),
    "qwen3.5-35b-a3b-thinking": (0.25, 1.25),
    "qwen3.5-122b-a10b": (0.26, 2.08),
    "qwen3.5-122b-a10b-thinking": (0.26, 2.08),
    "qwen3.5-397b-a17b": (0.39, 2.34),
    "qwen3.5-397b-a17b-thinking": (0.39, 2.34),
    "kimi-k3": (3.0, 15.0),
    "glm-5.2": (0.76, 2.42),
    "mistral-large-2512": (0.5, 1.5),
}
_DEFAULT_PRICE = (2.0, 8.0)
_JUDGE_PRICE = (5.0, 25.0)  # Claude Opus 4.8 on OpenRouter ($5/$25 per 1M)

# Rough output-token expectation per answer (long prompts elicit long answers).
# Calibrated against a full default sweep (2026-07): ~170 out-tokens/answer
# observed; keep a conservative margin.
_OUT_TOKENS_DEFAULT = 220
_OUT_TOKENS_LONG = 700
# Reasoning rungs emit extra (reasoning) tokens on top of the answer. Low
# effort adds ~10-50% in practice; higher efforts can multiply output.
_THINKING_MULT_BY_EFFORT = {"low": 1.5, "medium": 3.0}
_THINKING_MULT_DEFAULT = 4.0  # high/xhigh and unknown efforts


def _enc():
    import tiktoken

    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - offline fallback
        return None


def _ntok(enc, text: str) -> int:
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text))


@dataclass
class PlanLine:
    model: str
    calls: int
    in_tokens: int
    out_tokens: int
    dollars: float
    assumed_price: bool = False  # no _PRICES entry; priced at _DEFAULT_PRICE


def build_plan(
    model_specs: list[ModelSpec],
    questions: list[Question],
    *,
    n: int,
    judge: str | None = None,
) -> dict:
    enc = _enc()
    q_in: list[tuple[int, int]] = []
    for q in questions:
        t = _ntok(enc, q.prompt) + (_ntok(enc, q.system) if q.system else 0) + 8
        out = _OUT_TOKENS_LONG if _ntok(enc, q.prompt) > 500 else _OUT_TOKENS_DEFAULT
        q_in.append((t, out))

    lines: list[PlanLine] = []
    total_dollars = 0.0
    for spec in model_specs:
        price = _PRICES.get(spec.name)
        if price is None:
            # Raw model strings: match table keys against the inspect model
            # id, longest key first.
            for key in sorted(_PRICES, key=len, reverse=True):
                if key in spec.inspect_model:
                    price = _PRICES[key]
                    break
        assumed = price is None
        pin, pout = price or _DEFAULT_PRICE
        in_tok = sum(t for t, _ in q_in) * n
        out_per = sum(o for _, o in q_in) * n
        if spec.reasoning_effort not in (None, "none", "minimal"):
            mult = _THINKING_MULT_BY_EFFORT.get(
                spec.reasoning_effort, _THINKING_MULT_DEFAULT
            )
            out_per = int(out_per * mult)
        calls = len(questions) * n
        dollars = in_tok / 1e6 * pin + out_per / 1e6 * pout
        total_dollars += dollars
        lines.append(
            PlanLine(spec.name, calls, in_tok, out_per, dollars, assumed_price=assumed)
        )

    # Judge: one call per (model, question), seeing N responses (~out tokens each).
    judge_dollars = 0.0
    judge_calls = 0
    if judge:
        for _spec in model_specs:
            for _, out in q_in:
                judge_in = int(n * (out + 10) + 400)  # responses + instructions
                judge_dollars += (
                    judge_in / 1e6 * _JUDGE_PRICE[0] + 300 / 1e6 * _JUDGE_PRICE[1]
                )
                judge_calls += 1

    return {
        "lines": lines,
        "gen_calls": sum(line.calls for line in lines),
        "gen_dollars": total_dollars,
        "judge_calls": judge_calls,
        "judge_dollars": judge_dollars,
        "total_dollars": total_dollars + judge_dollars,
        "n": n,
        "n_questions": len(questions),
    }


def format_plan(
    plan: dict, model_specs: list[ModelSpec], questions: list[Question]
) -> str:
    out = []
    out.append("=== Sweep plan (ROUGH estimate) ===")
    out.append(
        f"{len(model_specs)} models x {plan['n_questions']} questions x N={plan['n']}"
        f"  ->  {plan['gen_calls']} generation calls"
    )
    out.append("")
    out.append("questions:")
    for q in questions:
        out.append(f"  - [{q.group}] {q.id}")
    out.append("")
    out.append(
        f"{'model':26s} {'calls':>7s} {'in_tok':>10s} {'out_tok':>10s} {'$':>8s}"
    )
    assumed: list[str] = []
    for line in plan["lines"]:
        label = line.model + ("*" if line.assumed_price else "")
        out.append(
            f"{label:26s} {line.calls:7d} {line.in_tokens:10d} "
            f"{line.out_tokens:10d} {line.dollars:8.2f}"
        )
        if line.assumed_price:
            assumed.append(line.model)
    out.append("-" * 66)
    out.append(f"generation subtotal: ${plan['gen_dollars']:.2f}")
    if plan["judge_calls"]:
        s = "s" if plan["judge_calls"] != 1 else ""
        out.append(
            f"judge: {plan['judge_calls']} call{s}  ~${plan['judge_dollars']:.2f}"
        )
    out.append(f"TOTAL (rough): ${plan['total_dollars']:.2f}")
    if assumed:
        out.append(
            f"* no price entry for {', '.join(assumed)}; assuming "
            f"${_DEFAULT_PRICE[0]:.2f} in / ${_DEFAULT_PRICE[1]:.2f} out per 1M tokens"
        )
    return "\n".join(out)
