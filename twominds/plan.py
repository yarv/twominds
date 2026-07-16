"""Dry-run planning + rough cost estimation for the variance sweep.

Print the model x question x N plan and a *rough* dollar estimate before any
API call. Prices are best-effort and clearly labelled as approximate.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import ModelSpec
from .questions import Question

# Rough $/1M tokens (input, output). Fine-tunes priced ~1.5x base. GPT-5.2 and
# Claude figures are approximate — treat the total as an order-of-magnitude guide.
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "gpt-5.2": (1.25, 10.0),
    "gpt-5.2-thinking": (1.25, 10.0),
    # Hot Mess frontier roster (per-1M in/out, approx public list prices).
    "claude-sonnet-4": (3.0, 15.0),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
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
    q_in = []
    for q in questions:
        t = _ntok(enc, q.prompt) + (_ntok(enc, q.system) if q.system else 0) + 8
        out = _OUT_TOKENS_LONG if _ntok(enc, q.prompt) > 500 else _OUT_TOKENS_DEFAULT
        q_in.append((q.id, t, out, bool(q.family)))

    lines: list[PlanLine] = []
    total_dollars = 0.0
    for spec in model_specs:
        pin, pout = _PRICES.get(spec.name, _DEFAULT_PRICE)
        in_tok = sum(t for _, t, _, _ in q_in) * n
        out_per = sum(o for _, _, o, _ in q_in) * n
        if spec.reasoning_effort not in (None, "none", "minimal"):
            mult = _THINKING_MULT_BY_EFFORT.get(
                spec.reasoning_effort, _THINKING_MULT_DEFAULT
            )
            out_per = int(out_per * mult)
        calls = len(questions) * n
        dollars = in_tok / 1e6 * pin + out_per / 1e6 * pout
        total_dollars += dollars
        lines.append(
            PlanLine(
                spec.name,
                calls,
                in_tok,
                out_per,
                dollars,
                assumed_price=spec.name not in _PRICES,
            )
        )

    # Judge: one call per (model, non-family question), seeing N responses
    # (~out tokens each). Family variants are only judged by the pooled
    # cross-variant call below — analyze skips their per-question judge.
    judge_dollars = 0.0
    judge_calls = 0
    if judge:
        for spec in model_specs:
            for _, _, out, is_family in q_in:
                if is_family:
                    continue
                judge_in = int(n * (out + 10) + 400)  # responses + instructions
                judge_dollars += (
                    judge_in / 1e6 * _JUDGE_PRICE[0] + 300 / 1e6 * _JUDGE_PRICE[1]
                )
                judge_calls += 1

    # Cross-variant families add one extra *pooled* judge call per (model, family),
    # which sees every variant's N responses at once.
    from collections import Counter

    fam_counts = Counter(q.family for q in questions if q.family)
    fam_judge_calls = 0
    if judge:
        for spec in model_specs:
            for fam, k in fam_counts.items():
                if k < 2:
                    continue
                pooled_in = int(k * n * (_OUT_TOKENS_DEFAULT + 10) + 400)
                judge_dollars += (
                    pooled_in / 1e6 * _JUDGE_PRICE[0] + 300 / 1e6 * _JUDGE_PRICE[1]
                )
                judge_calls += 1
                fam_judge_calls += 1

    return {
        "lines": lines,
        "gen_calls": sum(line.calls for line in lines),
        "gen_dollars": total_dollars,
        "judge_calls": judge_calls,
        "judge_dollars": judge_dollars,
        "family_judge_calls": fam_judge_calls,
        "total_dollars": total_dollars + judge_dollars,
        "n": n,
        "n_questions": len(questions),
    }


def format_plan(
    plan: dict,
    model_specs: list[ModelSpec],
    questions: list[Question],
    *,
    backends: list[str] | None = None,
    cached: set[str] | None = None,
    judge_reps: int = 1,
) -> str:
    """Render the plan. ``cached`` names models whose generation is already in
    the store (their cost is shown but excluded from the run-now total).
    ``backends`` are the *resolved* embedding backends (``[]`` = judge-only,
    ``None`` = a phase with no embedding analysis, e.g. ``generate``).
    ``judge_reps`` multiplies the judge cost into the total (``--reps N`` runs
    N full judge passes)."""
    cached = set(cached or ())
    out = []
    out.append("=== Variance sweep plan (ROUGH estimate) ===")
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
    run_now = 0.0
    cached_dollars = 0.0
    assumed: list[str] = []
    for line in plan["lines"]:
        is_cached = line.model in cached
        label = line.model + ("*" if line.assumed_price else "")
        label += " (cached)" if is_cached else ""
        out.append(
            f"{label:26s} {line.calls:7d} {line.in_tokens:10d} "
            f"{line.out_tokens:10d} {line.dollars:8.2f}"
        )
        if is_cached:
            cached_dollars += line.dollars
        else:
            run_now += line.dollars
        if line.assumed_price:
            assumed.append(line.model)
    out.append("-" * 66)
    if cached_dollars:
        out.append(
            f"generation: ~${run_now:.2f} to generate"
            f" + ~${cached_dollars:.2f} reused from cache (not billed)"
        )
    else:
        out.append(f"generation subtotal: ${run_now:.2f}")
    if plan["judge_calls"]:
        fam = plan.get("family_judge_calls", 0)
        extra = f" (incl. {fam} pooled family)" if fam else ""
        s = "s" if plan["judge_calls"] != 1 else ""
        line = f"judge: {plan['judge_calls']} call{s}{extra}  ~${plan['judge_dollars']:.2f}"
        if judge_reps > 1:
            line += (
                f" per pass  x {judge_reps} reps"
                f" = ~${plan['judge_dollars'] * judge_reps:.2f}"
            )
        out.append(line)
        if cached:
            out.append(
                "  (a cached model whose judge verdict is also cached re-judges free)"
            )
    if backends is not None:
        if len(backends) == 0 or all(b == "none" for b in backends):
            out.append("embeddings: none (judge-only analysis)")
        else:
            free = "; the local backend is free" if "local" in backends else ""
            out.append(f"embeddings: negligible (cents){free}")
    total_now = run_now + plan["judge_dollars"] * max(1, judge_reps)
    suffix = f"  (+ ~${cached_dollars:.2f} covered by cache)" if cached_dollars else ""
    out.append(f"TOTAL (rough): ${total_now:.2f}{suffix}")
    if assumed:
        out.append(
            f"* no price entry for {', '.join(assumed)}; assuming "
            f"${_DEFAULT_PRICE[0]:.2f} in / ${_DEFAULT_PRICE[1]:.2f} out per 1M tokens"
        )
    return "\n".join(out)
