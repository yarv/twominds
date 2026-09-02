"""The `generate` and `run` commands."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import typer

from twominds import analyze as analyze_mod
from twominds import generate as generate_mod
from twominds import plan as plan_mod
from twominds import report as report_mod
from twominds.models import resolve_models

from ._app import app
from ._options import (
    ConcurrencyOpt,
    DisplayOpt,
    GroupsOpt,
    IdsOpt,
    JudgeOpt,
    JudgePipelineOpt,
    JudgeReasonOpt,
    MaxConnectionsOpt,
    MaxTokOpt,
    ModelConcurrencyOpt,
    ModelsOpt,
    NOpt,
    TempOpt,
    _csv,
    _default_run_dir,
    _select_questions,
)
from ._summary import _echo_judge_summary

DryRunOpt = typer.Option(False, "--dry-run", help="print plan + cost, no API calls")
OutOpt = typer.Option(
    None, "--out", "-o", help="run dir (default results/twominds/<timestamp>)"
)


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


def _warn_missing_keys(specs, judge=None) -> None:
    """Echo a note per unset API-key env var the sweep will need (never fatal)."""
    need: dict[str, str] = {}
    for s in specs:
        for env in _provider_envs(s.inspect_model):
            need.setdefault(env, f"generating with {s.name}")
    if judge:
        for env in _provider_envs(judge):
            need.setdefault(env, "the judge")
    for env, reason in need.items():
        if not os.environ.get(env):
            typer.echo(f"note: {env} is not set — {reason} will fail")


def _echo_parallelism(model_concurrency, max_connections) -> None:
    """Echo the effective-parallelism summary when any knob is non-trivial."""
    if model_concurrency <= 1 and max_connections is None:
        return
    per_model = f"{max_connections}" if max_connections else "provider-default"
    typer.echo(
        f"parallelism: up to {model_concurrency} models at a time "
        f"(Inspect max_tasks) × {per_model} connections per model"
    )


def _do_generate(
    models,
    groups,
    ids,
    *,
    n,
    temperature,
    max_tokens,
    judge,
    out,
    display,
    dry_run,
    model_concurrency,
    max_connections,
    will_judge,
    judge_inline=None,
) -> Optional[Path]:
    """Plan (and on a dry run stop there), then generate into the run dir."""
    specs = resolve_models(_csv(models))
    qs = _select_questions(groups, ids)
    if not qs:
        raise typer.BadParameter("no questions selected")

    plan = plan_mod.build_plan(specs, qs, n=n, judge=judge if will_judge else None)
    typer.echo(plan_mod.format_plan(plan, specs, qs))
    _warn_missing_keys(specs, judge if will_judge else None)
    _echo_parallelism(model_concurrency, max_connections)
    if dry_run:
        typer.echo("\n(dry run — no API calls made)")
        return None

    # headless: the rich display spams nohup logs; fall back to plain off-TTY.
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
        max_connections=max_connections,
        judge_inline=judge_inline,
    )
    typer.echo(f"Generation complete: {run_dir}")
    return run_dir


@app.command()
def generate(
    models: str = ModelsOpt,
    groups: Optional[str] = GroupsOpt,
    ids: Optional[str] = IdsOpt,
    n: int = NOpt,
    temperature: float = TempOpt,
    max_tokens: int = MaxTokOpt,
    model_concurrency: int = ModelConcurrencyOpt,
    max_connections: Optional[int] = MaxConnectionsOpt,
    judge: str = JudgeOpt,
    out: Optional[str] = OutOpt,
    display: str = DisplayOpt,
    dry_run: bool = DryRunOpt,
):
    """Phase 1: sample each model N times on the question roster (Inspect)."""
    _do_generate(
        models,
        groups,
        ids,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        judge=judge,
        out=out,
        display=display,
        dry_run=dry_run,
        model_concurrency=model_concurrency,
        max_connections=max_connections,
        will_judge=False,
    )


@app.command()
def run(
    models: str = ModelsOpt,
    groups: Optional[str] = GroupsOpt,
    ids: Optional[str] = IdsOpt,
    n: int = NOpt,
    temperature: float = TempOpt,
    max_tokens: int = MaxTokOpt,
    model_concurrency: int = ModelConcurrencyOpt,
    max_connections: Optional[int] = MaxConnectionsOpt,
    judge: str = JudgeOpt,
    judge_reasoning: str = JudgeReasonOpt,
    concurrency: int = ConcurrencyOpt,
    judge_pipeline: bool = JudgePipelineOpt,
    out: Optional[str] = OutOpt,
    display: str = DisplayOpt,
    dry_run: bool = DryRunOpt,
):
    """All phases: generate -> judge -> scores + report."""
    # Fuse the judge into the generation eval (per-sample scorer): each question
    # is judged as its answers land, in the sweep's own display, and the judge
    # phase below harvests the verdicts from the logs instead of re-judging.
    judge_inline = (
        {
            "judge_name": judge,
            "judge_reasoning": judge_reasoning,
            "max_connections": concurrency,
        }
        if judge_pipeline
        else None
    )
    run_dir = _do_generate(
        models,
        groups,
        ids,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        judge=judge,
        out=out,
        display=display,
        dry_run=dry_run,
        model_concurrency=model_concurrency,
        max_connections=max_connections,
        will_judge=True,
        judge_inline=judge_inline,
    )
    if run_dir is None:  # dry run
        return
    typer.echo("\n=== judge ===")
    analysis = analyze_mod.analyze(
        run_dir,
        judge_name=judge,
        judge_reasoning=judge_reasoning,
        concurrency=concurrency,
    )
    _echo_judge_summary(analysis, run_dir)
    typer.echo(f"  report -> {report_mod.build_report_from_run(run_dir)}")
    typer.echo(f"\nDone. Open {run_dir / 'report.html'}")
