"""Cost reporting for the variance pipeline.

Two reconciled sources:
  * token->$ estimates (reuse ``plan._PRICES`` / ``plan._JUDGE_PRICE``), from
    the per-call judge usage captured on ``JudgeResult`` and the generation
    token totals stored in the Inspect ``.eval`` logs; and
  * the authoritative OpenRouter balance delta (``/credits.total_usage`` before
    vs after the judge phase), which captures OpenRouter's real pricing.

Uses only the stdlib (``urllib``) for the OpenRouter calls. ``OPENROUTER_API_KEY``
is read from the environment (load ``.env`` first, as the rest of the pkg does).
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Optional

from .plan import _DEFAULT_PRICE, _JUDGE_PRICE, _PRICES

_OR_BASE = "https://openrouter.ai/api/v1"


# --------------------------------------------------------------------------- #
# token -> $
# --------------------------------------------------------------------------- #
def _unit_price(model_id: str) -> tuple[float, float]:
    """($/1M in, $/1M out) for a model id, reusing the plan.py table.

    Substring match against the price table (so ``ft:gpt-4.1-…`` matches
    ``gpt-4.1``), then a 1.5x fine-tune surcharge for ``ft:`` ids — the same
    convention the dry-run estimator uses.
    """
    pi, po = _DEFAULT_PRICE
    # longest key first, so "gpt-4.1-mini" matches its own entry, not "gpt-4.1"
    for key, (a, b) in sorted(_PRICES.items(), key=lambda kv: -len(kv[0])):
        if key in model_id:
            pi, po = a, b
            break
    if "ft:" in model_id:
        pi, po = pi * 1.5, po * 1.5
    return pi, po


def gen_dollars(model_id: str, in_tok: int, out_tok: int) -> float:
    pi, po = _unit_price(model_id)
    return in_tok / 1e6 * pi + out_tok / 1e6 * po


def judge_dollars(in_tok: int, out_tok: int) -> float:
    pi, po = _JUDGE_PRICE
    return in_tok / 1e6 * pi + out_tok / 1e6 * po


# --------------------------------------------------------------------------- #
# OpenRouter balance (authoritative actual spend)
# --------------------------------------------------------------------------- #
def _or_get(path: str, timeout: float = 20.0) -> Optional[dict]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    try:
        req = urllib.request.Request(
            f"{_OR_BASE}/{path}", headers={"Authorization": f"Bearer {key}"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            return json.load(r).get("data")
    except Exception:
        return None


def openrouter_usage() -> Optional[float]:
    """Cumulative $ spent on the account (monotonic); delta = run spend."""
    d = _or_get("credits")
    return float(d["total_usage"]) if d and d.get("total_usage") is not None else None


def openrouter_balance() -> Optional[dict]:
    """Human-readable limit / remaining / daily-weekly-monthly for the key."""
    return _or_get("auth/key")


# --------------------------------------------------------------------------- #
# generation usage from the .eval logs
# --------------------------------------------------------------------------- #
def generation_usage(run_dir: Path) -> dict[str, dict]:
    """{model_name: {model_id, in_tok, out_tok, dollars}} from <run>/logs."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    logs_root = Path(run_dir) / "logs"
    out: dict[str, dict] = {}
    if not logs_root.exists():
        return out
    # Mirror analyze.load_responses: key by the log file's *stem* (the sanitised
    # spec name), not the top-level dir — a pre-sanitisation run nested a
    # slash-containing name (logs/ours/<x>/...) and dir-keying collapses every
    # such model into one "ours" bucket. Last log per stem wins; Inspect's
    # ".raw" scratch dirs are skipped.
    chosen: dict[str, str] = {}
    for md in sorted(
        p for p in logs_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    ):
        for info in list_eval_logs(str(md)):
            chosen[Path(info.name).stem] = info.name
    for label, log_path in sorted(chosen.items()):
        it = ot = 0
        mid = label
        mu = read_eval_log(log_path).stats.model_usage or {}
        for k, u in mu.items():
            mid = k
            it += getattr(u, "input_tokens", 0) or 0
            ot += getattr(u, "output_tokens", 0) or 0
        out[label] = {
            "model_id": mid,
            "in_tok": it,
            "out_tok": ot,
            "dollars": gen_dollars(mid, it, ot),
        }
    return out


# --------------------------------------------------------------------------- #
# record + summary
# --------------------------------------------------------------------------- #
def rollup_fragments(
    fragments: list[dict],
    run_dir: Path,
    *,
    judge_name: str,
    run_judge: bool,
    cached_flags: Optional[list[bool]] = None,
    cached_gens: Optional[set[str]] = None,
) -> dict:
    """Run-level cost record from per-model analysis fragments: summed judge
    tokens (+ OpenRouter deltas where recorded) and generation usage over the
    run's (symlinked) logs. Used by the store assembly so a fragment-built run
    carries the same cost record a whole-run analyze would.

    ``cached_flags`` (aligned with ``fragments``) marks judge fragments reused
    from the store; ``cached_gens`` names models whose generation was reused.
    Both let the summary separate spend billed by THIS invocation from spend
    inherited from earlier ones."""
    record: dict = {}
    if run_judge:
        flags = cached_flags or [False] * len(fragments)
        frag_judges = [(a.get("cost") or {}).get("judge", {}) for a in fragments]
        jt_in = sum(j.get("in_tok", 0) for j in frag_judges)
        jt_out = sum(j.get("out_tok", 0) for j in frag_judges)
        cached_d = sum(
            judge_dollars(j.get("in_tok", 0), j.get("out_tok", 0))
            for j, c in zip(frag_judges, flags)
            if c
        )
        # a cached fragment's delta was billed by the invocation that judged it
        deltas = [
            j.get("openrouter_delta")
            for j, c in zip(frag_judges, flags)
            if not c and j.get("openrouter_delta") is not None
        ]
        record["judge"] = {
            "model": judge_name,
            "in_tok": jt_in,
            "out_tok": jt_out,
            "est_dollars": judge_dollars(jt_in, jt_out),
            "cached_dollars": cached_d,
            "openrouter_delta": sum(deltas) if deltas else None,
        }
    gen = generation_usage(run_dir)
    if gen:
        if cached_gens is not None:
            for name, v in gen.items():
                v["cached"] = name in cached_gens
        record["generation"] = gen
    return record


def write_cost(path: Path, record: dict) -> None:
    Path(path).write_text(json.dumps(record, indent=2))


def format_summary(record: dict, *, gen_note: str | None = None) -> str:
    """One compact block reconciling estimate vs actual, separating spend
    billed by this invocation from spend reused out of the store. ``gen_note``
    overrides the generation label for phases that never generate (analyze)."""
    lines = ["cost:"]
    g = record.get("generation")
    if g:
        fresh = {k: v for k, v in g.items() if not v.get("cached")}
        reused = {k: v for k, v in g.items() if v.get("cached")}
        fd = sum(v["dollars"] for v in fresh.values())
        rd = sum(v["dollars"] for v in reused.values())
        if gen_note:
            gd = fd + rd
            lines.append(f"  generation (est, {gen_note}): ${gd:.2f}")
        elif reused:
            lines.append(
                f"  generation (est): ${fd:.2f} billed now ({len(fresh)} model(s))"
                f" + ${rd:.2f} reused from cache ({len(reused)} model(s))"
            )
        else:
            lines.append(f"  generation (est): ${fd:.2f} across {len(g)} model(s)")
    j = record.get("judge")
    if j:
        cd = j.get("cached_dollars") or 0.0
        if cd:
            lines.append(
                f"  judge (est, token×price): ${j['est_dollars'] - cd:.2f} billed now"
                f" + ${cd:.2f} reused from cache"
            )
        else:
            lines.append(
                f"  judge (est, token×price): ${j['est_dollars']:.2f} "
                f"({j['in_tok']:,} in / {j['out_tok']:,} out)"
            )
        od = j.get("openrouter_delta")
        if od is not None:
            lag = "  (OpenRouter accounting may lag; see `budget`)" if od == 0 else ""
            lines.append(f"  judge (actual, OpenRouter delta): ${od:.2f}{lag}")
    return "\n".join(lines)
