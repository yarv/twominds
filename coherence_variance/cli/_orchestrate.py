"""Generation orchestration shared by `run` and `generate`: plan echo, store
partitioning (cached vs to-generate), and the generation phase itself."""

from __future__ import annotations

import os
import sys
from dataclasses import replace as dc_replace
from pathlib import Path

import typer

from coherence_variance import cost as cost_mod
from coherence_variance import generate as generate_mod
from coherence_variance import plan as plan_mod
from coherence_variance import store as store_mod
from coherence_variance.models import next_name, resolve_models

from . import _options
from ._options import _csv, _default_run_dir, _select_questions


def _do_generate(
    models,
    groups,
    ids,
    all_questions,
    families,
    n,
    temperature,
    max_tokens,
    judge,
    out,
    display,
    dry_run,
    roster=None,
    folders=None,
    model_concurrency=1,
    backends=None,
    will_judge=True,
    judge_reps=1,
):
    """The --no-store generation path: straight into the run dir, no reuse."""
    specs = resolve_models(_csv(models))
    qs = _select_questions(groups, ids, all_questions, families, roster, folders)
    if not qs:
        raise typer.BadParameter("no questions selected")

    plan = plan_mod.build_plan(specs, qs, n=n, judge=judge if will_judge else None)
    typer.echo(
        plan_mod.format_plan(plan, specs, qs, backends=backends, judge_reps=judge_reps)
    )
    _warn_missing_keys(specs, judge if will_judge else None, backends)
    if model_concurrency > 1:
        typer.echo(
            f"parallelism: up to {model_concurrency} models at a time "
            f"(Inspect max_tasks; ~{model_concurrency}× max_connections in flight)"
        )
    if dry_run:
        bal = cost_mod.openrouter_balance()
        if bal and bal.get("limit") is not None:
            typer.echo(
                f"\nOpenRouter now: ${bal.get('usage', 0):.2f} used / "
                f"${bal['limit']:.0f} limit (${bal.get('limit_remaining', 0):.2f} left)"
            )
        typer.echo("\n(dry run — no API calls made)")
        return None

    # headless: the rich display spams nohup logs; fall back to plain off-TTY.
    # (Concurrent models share Inspect's one rich display, so no downgrade needed
    # for parallelism — only for non-interactive stdout.)
    if display == "rich" and not sys.stdout.isatty():
        display = "plain"
        typer.echo("(non-interactive stdout detected: using --display plain)")

    run_dir = Path(out) if out else _default_run_dir()
    generate_mod.write_manifest(
        run_dir,
        specs,
        qs,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        judge=judge,
    )
    typer.echo(f"\nGenerating into {run_dir} ...")
    generate_mod.run_generation(
        specs,
        qs,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        run_dir=run_dir,
        display=display,
        model_concurrency=model_concurrency,
    )
    typer.echo(f"Generation complete: {run_dir}")
    return run_dir


def _announce_partition(cached, to_generate, gen_key, *, dry_run=False):
    """Echo the store-reuse summary for a partitioned roster."""
    if cached:
        typer.echo(
            f"reusing {len(cached)} cached generation(s) [gen {gen_key}]: "
            + ", ".join(cached)
        )
    if to_generate:
        verb = "would generate" if dry_run else "generating"
        typer.echo(
            f"{verb} {len(to_generate)} model(s) [gen {gen_key}]: "
            + ", ".join(s.name for s in to_generate)
        )


def _warn_missing_keys(specs, judge=None, backends=None):
    """Echo a note per unset API-key env var the sweep will need (never fatal)."""
    need: dict[str, str] = {}

    def _provider_envs(inspect_model: str) -> list[str]:
        prov, _, rest = inspect_model.partition("/")
        if prov == "openai":
            return ["OPENAI_API_KEY"]
        if prov == "openrouter":
            return ["OPENROUTER_API_KEY"]
        if prov == "anthropic":
            return ["ANTHROPIC_API_KEY"]
        if prov == "openai-api":
            service = rest.partition("/")[0].upper().replace("-", "_")
            return [f"{service}_API_KEY", f"{service}_BASE_URL"]
        return []  # locally-managed providers (vllm/, ollama/, ...) need no key

    for s in specs:
        for env in _provider_envs(s.inspect_model):
            need.setdefault(env, f"generating with {s.name}")
    if judge:
        for env in _provider_envs(judge):
            need.setdefault(env, "the judge")
    for b in backends or ():
        if b.startswith("openai-"):
            need.setdefault("OPENAI_API_KEY", f"the {b} embedding backend")
    for env, reason in need.items():
        if not os.environ.get(env):
            typer.echo(f"note: {env} is not set — {reason} will fail")


def _plan_generations(
    specs,
    qs,
    *,
    n,
    temperature,
    max_tokens,
    rerun,
    rerun_models,
    raw_names=None,
    dry_run=False,
    announce=True,
):
    """Partition the roster into cached vs to-generate against the store.

    Returns ``(gen_key, gen_dirs, cached_names, to_generate_specs)`` and echoes
    the reuse summary. ``rerun``/``rerun_models`` force regeneration (their gen
    dirs are cleared just before generating — see ``store.clear_generation``).
    ``raw_names`` are the model tokens as passed to --models (same order as
    ``specs``) so --rerun-model accepts aliases too. A store name already
    pinned to a DIFFERENT model is auto-qualified with more path segments
    (mutating ``specs`` in place, like the in-batch disambiguation in
    ``resolve_models``). Dry runs stay read-only.
    """
    root = store_mod.store_root(_options._RESULTS_ROOT)
    gen_key = store_mod.compute_gen_key(
        qs, n=n, temperature=temperature, max_tokens=max_tokens
    )
    raw = list(raw_names) if raw_names else [None] * len(specs)
    forced = set(rerun_models or [])
    accepted = set(t for t in raw if t)
    for s in specs:
        accepted.update({s.name, s.display})
    unknown = forced - accepted
    if unknown:
        raise typer.BadParameter(
            f"--rerun-model {sorted(unknown)} not among the selected models"
        )

    gen_dirs: dict[str, Path] = {}
    cached: list[str] = []
    to_generate = []
    for i, spec in enumerate(specs):
        # Cross-invocation collision: the store name belongs to a different
        # model from an earlier run — qualify with more path segments.
        while (prev := store_mod.identity_conflict(spec, root)) is not None:
            ref = spec.display or spec.name
            nxt = next_name(ref, spec.name)
            if nxt is None:
                raise typer.BadParameter(
                    f"store name '{spec.name}' is pinned to "
                    f"{prev.get('inspect_model')} and '{ref}' has no more path "
                    "segments to disambiguate with; use --no-store or remove "
                    f"results/variance/models/{spec.name}"
                )
            typer.echo(
                f"(store name '{spec.name}' is taken by "
                f"{prev.get('inspect_model')}; using '{nxt}')"
            )
            spec = dc_replace(spec, name=nxt)
            specs[i] = spec
        if not dry_run:
            store_mod.write_identity(spec, root)
        found = store_mod.find_generation(spec, gen_key, root)
        force = rerun or bool({spec.name, spec.display, raw[i]} & forced)
        if found is not None and not force:
            gen_dirs[spec.name] = found
            cached.append(spec.name)
        else:
            gen_dirs[spec.name] = store_mod.gen_dir(spec, gen_key, root)
            to_generate.append(spec)

    if announce:
        _announce_partition(cached, to_generate, gen_key, dry_run=dry_run)
    return gen_key, gen_dirs, cached, to_generate


def _execute_generations(
    to_generate,
    qs,
    gen_dirs,
    gen_key,
    *,
    n,
    temperature,
    max_tokens,
    judge,
    model_concurrency=1,
    display="rich",
):
    """Generate the missing models into their store gen dirs.

    The missing models still run as ONE Inspect eval (each model's logs land in
    its own store gen dir via ``log_dirs``, marked complete as they're written).
    """
    root = store_mod.store_root(_options._RESULTS_ROOT)
    for spec in to_generate:
        store_mod.clear_generation(spec, gen_key, root)  # no-op unless forced rerun
        store_mod.prepare_generation(
            spec,
            qs,
            gen_key,
            root,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            judge=judge,
        )
    generate_mod.run_generation(
        to_generate,
        qs,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        display=display,
        model_concurrency=model_concurrency,
        log_dirs={
            s.name: Path(gen_dirs[s.name]) / "logs" / s.name for s in to_generate
        },
        on_model_done=lambda name: store_mod.mark_complete(gen_dirs[name], key=gen_key),
    )


def _setup_store_run(
    models,
    groups,
    ids,
    all_questions,
    families,
    roster,
    folders,
    *,
    n,
    temperature,
    max_tokens,
    judge,
    out,
    display,
    model_concurrency,
    rerun,
    rerun_models,
    dry_run,
    backends=None,
    will_judge=True,
    judge_reps=1,
):
    """Store-backed generation phase shared by `run` and `generate`: ensure
    per-model generations exist (reusing the store), then create the run dir
    with symlinked logs. Returns (run_dir, specs, gen_dirs, cached_names);
    (None, specs, gen_dirs, cached_names) on dry runs.

    ``backends``/``will_judge`` only shape the printed plan: pass the resolved
    embedding backends (``run``) or leave None (``generate``), and set
    ``will_judge=False`` when no judge phase follows (its cost is then left out).
    """
    specs = resolve_models(_csv(models))
    qs = _select_questions(groups, ids, all_questions, families, roster, folders)
    if not qs:
        raise typer.BadParameter("no questions selected")
    # partition against the store first, so the printed plan reflects the cache
    gen_key, gen_dirs, cached, to_generate = _plan_generations(
        specs,
        qs,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        rerun=rerun,
        rerun_models=rerun_models,
        raw_names=_csv(models),
        dry_run=dry_run,
        announce=False,
    )
    typer.echo(
        plan_mod.format_plan(
            plan_mod.build_plan(specs, qs, n=n, judge=judge if will_judge else None),
            specs,
            qs,
            backends=backends,
            cached=cached,
            judge_reps=judge_reps,
        )
    )
    _warn_missing_keys(
        [s for s in specs if s.name not in cached],
        judge if will_judge else None,
        backends,
    )
    _announce_partition(cached, to_generate, gen_key, dry_run=dry_run)
    if dry_run:
        typer.echo("\n(dry run — no API calls made)")
        return None, specs, gen_dirs, cached
    if display == "rich" and not sys.stdout.isatty():
        display = "plain"
        typer.echo("(non-interactive stdout detected: using --display plain)")
    if to_generate:
        _execute_generations(
            to_generate,
            qs,
            gen_dirs,
            gen_key,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            judge=judge,
            model_concurrency=model_concurrency,
            display=display,
        )
    run_dir = Path(out) if out else _default_run_dir()
    generate_mod.write_manifest(
        run_dir,
        specs,
        qs,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        judge=judge,
    )
    for spec in specs:
        store_mod.link_into_run(run_dir, spec, gen_dirs[spec.name])
    return run_dir, specs, gen_dirs, cached
