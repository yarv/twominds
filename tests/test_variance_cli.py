"""CLI-layer store planning tests (variance_experiment._plan_generations):
cross-invocation name auto-qualification, dry-run read-only-ness, and
--rerun-model accepting the tokens as passed to --models."""

import pytest
import typer

from coherence_variance import cli as VE
from coherence_variance import store as S
from coherence_variance.models import resolve_models
from coherence_variance.questions import Question

_QS = [Question(id="q1", group="values", prompt="Say A.", bucket="tier_1")]


@pytest.fixture
def results_root(tmp_path, monkeypatch):
    root = tmp_path / "variance"
    monkeypatch.setattr(VE, "_RESULTS_ROOT", root)
    return root


def _plan(specs, **kw):
    kw.setdefault("n", 2)
    kw.setdefault("temperature", 1.0)
    kw.setdefault("max_tokens", 64)
    kw.setdefault("rerun", False)
    kw.setdefault("rerun_models", None)
    return VE._plan_generations(specs, _QS, **kw)


def test_cross_invocation_collision_auto_qualifies(results_root):
    # invocation 1 pins the short name "foo" to one provider's model
    first = resolve_models(["openrouter/acme/foo"])
    _plan(first)
    assert first[0].name == "foo"

    # invocation 2: same last segment, different model -> auto-qualified,
    # not a dead-end error
    second = resolve_models(["together/other/foo"])
    _plan(second)
    assert second[0].name == "other_foo"  # mutated in place for the caller
    ident = S.identity_conflict(second[0], S.store_root(results_root))
    assert ident is None  # pinned under the qualified name


def test_dry_run_leaves_store_untouched(results_root):
    specs = resolve_models(["openrouter/acme/foo"])
    _plan(specs, dry_run=True)
    assert not S.store_root(results_root).exists()


def test_rerun_model_accepts_models_token(results_root):
    # '4o' is a roster alias: resolves to name 'gpt-4o' / display 'GPT-4o'.
    specs = resolve_models(["4o"])
    _, _, cached, to_generate = _plan(specs, rerun_models=["4o"], raw_names=["4o"])
    assert [s.name for s in to_generate] == ["gpt-4o"]  # forced, so not cached
    assert cached == []


def test_rerun_model_unknown_still_rejected(results_root):
    specs = resolve_models(["4o"])
    with pytest.raises(typer.BadParameter, match="not among"):
        _plan(specs, rerun_models=["gpt-5.2"], raw_names=["4o"])


def test_resolve_backends_none_and_validation():
    assert VE._resolve_backends(["none"]) == []
    assert VE._resolve_backends(["local"]) == ["local"]
    assert VE._resolve_backends(["local", "openai-3-small"]) == [
        "local",
        "openai-3-small",
    ]
    with pytest.raises(typer.BadParameter, match="cannot be combined"):
        VE._resolve_backends(["none", "local"])
    with pytest.raises(typer.BadParameter, match="unknown embedding backend"):
        VE._resolve_backends(["bogus"])
