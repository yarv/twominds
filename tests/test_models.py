"""Tests for variance experiment model resolution."""

import json

import pytest

from twominds import models as M


@pytest.fixture
def fake_keys(tmp_path, monkeypatch):
    keys = {
        "my-finetune": "ft:gpt-4.1-2025-04-14:your-org:my-finetune:DEADBEEF",
        "my-other-finetune": "ft:gpt-4o-2024-08-06:your-org:my-other-finetune:CAFEBABE",
    }
    path = tmp_path / "model_jsons.keys"
    path.write_text(json.dumps(keys))
    monkeypatch.setattr(M, "_KEYS_PATH", path)
    return keys


def test_base_openai_model():
    spec = M.resolve_model("gpt-4.1")
    assert spec.inspect_model == "openai/gpt-4.1"
    assert spec.reasoning_effort is None


def test_ours_resolution_via_keys(fake_keys):
    spec = M.resolve_model("ours/my-finetune")
    assert spec.name == "my-finetune"  # last path segment; ours/ prefix dropped
    assert spec.display == "ours/my-finetune"
    assert spec.inspect_model == "openai/" + fake_keys["my-finetune"]


def test_ours_resolution_second_entry(fake_keys):
    spec = M.resolve_model("ours/my-other-finetune")
    assert spec.inspect_model.endswith("my-other-finetune:CAFEBABE")


def test_reasoning_only_on_thinking_rung():
    assert M.resolve_model("gpt-5.2").reasoning_effort == "none"
    assert M.resolve_model("gpt-5.2-thinking").reasoning_effort == "low"
    assert M.resolve_model("5.2").name == "gpt-5.2"


def test_provider_qualified_passthrough():
    spec = M.resolve_model("openrouter/anthropic/claude-sonnet-4.5")
    assert spec.inspect_model == "openrouter/anthropic/claude-sonnet-4.5"
    # The spec name doubles as the log-dir name: the (sanitized) last path
    # segment, so reports and results dirs read as the bare model id.
    assert spec.name == "claude-sonnet-4.5"
    assert spec.display == "openrouter/anthropic/claude-sonnet-4.5"


def test_short_name_collision_disambiguates():
    specs = M.resolve_models(
        ["openrouter/qwen/qwen3-32b", "together/qwen/qwen3-32b", "gpt-4.1"]
    )
    assert [s.name for s in specs] == [
        "openrouter_qwen_qwen3-32b",  # qwen_qwen3-32b still collides -> 3 segments
        "together_qwen_qwen3-32b",
        "gpt-4.1",
    ]
    assert specs[0].display == "openrouter/qwen/qwen3-32b"


def test_short_name_collision_two_providers():
    specs = M.resolve_models(["openrouter/meta/llama-3-70b", "groq/llama-3-70b"])
    assert [s.name for s in specs] == ["meta_llama-3-70b", "groq_llama-3-70b"]


def test_duplicate_model_request_raises():
    with pytest.raises(ValueError, match="more than once"):
        M.resolve_models(["openai/gpt-4o", "openai/gpt-4o"])


def test_sanitize_unsafe_chars():
    spec = M.resolve_model("openrouter/some org/weird model@v1")
    assert spec.name == "weird_model_v1"


def test_raw_finetune_id_passthrough():
    ft = "ft:gpt-4.1-2025-04-14:acme:their-model:AbCd1234"
    spec = M.resolve_model(ft)
    assert spec.inspect_model == f"openai/{ft}"
    assert spec.name == ft  # no slashes to flatten; colons are dir-safe


def test_bare_model_assumes_openai():
    assert M.resolve_model("gpt-4o").inspect_model == "openai/gpt-4o"


def test_default_roster_resolves_without_keys_file(tmp_path, monkeypatch):
    # The default roster is pure public OpenAI models; resolution must never
    # touch model_jsons.keys (it is only read lazily for ours/* refs).
    monkeypatch.setattr(M, "_KEYS_PATH", tmp_path / "does-not-exist.keys")
    specs = M.resolve_models(M.DEFAULT_MODELS)
    assert [s.name for s in specs] == M.DEFAULT_MODELS
    assert all(s.inspect_model.startswith("openai/") for s in specs)


def test_hotmess_roster_resolves():
    specs = M.resolve_models(M.HOTMESS_MODELS)
    assert [s.name for s in specs] == ["claude-sonnet-4", "o3-mini", "o4-mini"]
    # Sonnet 4 routed via OpenRouter (effort sent as the OpenRouter reasoning
    # option); all three carry low reasoning effort.
    sonnet, o3, o4 = specs
    assert sonnet.inspect_model == "openrouter/anthropic/claude-sonnet-4"
    assert o3.inspect_model == "openai/o3-mini"
    assert o4.inspect_model == "openai/o4-mini"
    assert {s.reasoning_effort for s in specs} == {"low"}


def test_hotmess_aliases():
    assert M.resolve_model("sonnet-4").name == "claude-sonnet-4"
    assert M.resolve_model("sonnet4").name == "claude-sonnet-4"
    assert M.resolve_model("o3mini").name == "o3-mini"
    assert M.resolve_model("o4mini").name == "o4-mini"


def test_missing_ours_key_raises(tmp_path, monkeypatch):
    path = tmp_path / "model_jsons.keys"
    path.write_text("{}")
    monkeypatch.setattr(M, "_KEYS_PATH", path)
    with pytest.raises(KeyError):
        M.resolve_model("ours/my-finetune")


def test_missing_keys_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "_KEYS_PATH", tmp_path / "does-not-exist.keys")
    with pytest.raises(FileNotFoundError):
        M.resolve_model("ours/my-finetune")
