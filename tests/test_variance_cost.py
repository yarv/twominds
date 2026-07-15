"""Tests for coherence_variance.cost (token->$ + OpenRouter client, no network)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from coherence_variance import cost as C  # noqa: E402

_FT = "ft:gpt-4.1-2025-04-14:your-org:my-finetune:XXXXXXXX"


def test_unit_price_base_ft_and_default():
    assert C._unit_price("gpt-4.1") == (2.0, 8.0)
    assert C._unit_price(_FT) == (3.0, 12.0)  # gpt-4.1 base x1.5 fine-tune surcharge
    assert C._unit_price("totally-unknown-model") == C._DEFAULT_PRICE


def test_gen_and_judge_dollars():
    assert C.gen_dollars("gpt-4.1", 1_000_000, 1_000_000) == 10.0
    assert C.gen_dollars(_FT, 1_000_000, 1_000_000) == 15.0
    assert C.judge_dollars(1_000_000, 1_000_000) == 30.0  # _JUDGE_PRICE (5, 25)


def test_openrouter_usage_and_balance(monkeypatch):
    canned = {
        "credits": {"total_credits": 1415.0, "total_usage": 1307.5},
        "auth/key": {"limit": 1000, "limit_remaining": 527.16, "usage": 472.84},
    }
    monkeypatch.setattr(C, "_or_get", lambda path, timeout=20.0: canned.get(path))
    assert C.openrouter_usage() == 1307.5
    assert C.openrouter_balance()["limit"] == 1000


def test_openrouter_usage_none_when_unavailable(monkeypatch):
    monkeypatch.setattr(C, "_or_get", lambda path, timeout=20.0: None)
    assert C.openrouter_usage() is None
    assert C.openrouter_balance() is None


def test_generation_usage_sums_eval_logs(monkeypatch, tmp_path):
    (tmp_path / "logs" / "model-x").mkdir(parents=True)
    fake_log = types.SimpleNamespace(
        stats=types.SimpleNamespace(
            model_usage={
                _FT: types.SimpleNamespace(
                    input_tokens=1_000_000, output_tokens=500_000
                )
            }
        )
    )
    import inspect_ai.log as L

    monkeypatch.setattr(
        L, "list_eval_logs", lambda d: [types.SimpleNamespace(name="model-x.eval")]
    )
    monkeypatch.setattr(L, "read_eval_log", lambda name: fake_log)

    g = C.generation_usage(tmp_path)
    assert set(g) == {"model-x"}
    assert g["model-x"]["in_tok"] == 1_000_000
    assert g["model-x"]["out_tok"] == 500_000
    # 1M in @ $3 + 0.5M out @ $12 = 3 + 6 = $9 (ft pricing)
    assert abs(g["model-x"]["dollars"] - 9.0) < 1e-6


def test_generation_usage_keys_by_stem_not_dir(monkeypatch, tmp_path):
    """Regression: a pre-sanitisation nested run (logs/ours/<x>/...) must yield
    one usage row per model log stem — dir-name keying collapsed every
    slash-named model into a single "ours" bucket (mirrors
    analyze.load_responses, which got the same fix)."""
    (tmp_path / "logs" / "ours").mkdir(parents=True)
    fake_log = types.SimpleNamespace(
        stats=types.SimpleNamespace(
            model_usage={
                "gpt-4.1": types.SimpleNamespace(input_tokens=10, output_tokens=5)
            }
        )
    )
    import inspect_ai.log as L

    monkeypatch.setattr(
        L,
        "list_eval_logs",
        lambda d: [
            types.SimpleNamespace(name="ours/alpha/alpha.eval"),
            types.SimpleNamespace(name="ours/beta/beta.eval"),
        ],
    )
    monkeypatch.setattr(L, "read_eval_log", lambda name: fake_log)

    g = C.generation_usage(tmp_path)
    assert set(g) == {"alpha", "beta"}  # one row per model, not one "ours"


def test_format_summary_reconciles():
    rec = {
        "generation": {"m": {"in_tok": 1, "out_tok": 1, "dollars": 4.0}},
        "judge": {
            "in_tok": 10,
            "out_tok": 20,
            "est_dollars": 1.5,
            "openrouter_delta": 1.7,
        },
    }
    s = C.format_summary(rec)
    assert "generation (est): $4.00" in s
    assert "judge (est, token×price): $1.50" in s
    assert "OpenRouter delta): $1.70" in s
