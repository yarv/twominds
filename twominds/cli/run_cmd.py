"""The `generate` and `run` commands (store-backed generation + full pipeline)."""

from __future__ import annotations

from typing import List, Optional

import typer

from twominds import cost as cost_mod
from twominds import report as report_mod
from twominds import store as store_mod

from ._app import app
from ._options import (
    AllQOpt,
    BackendsOpt,
    ConcurrencyOpt,
    DisplayOpt,
    FamiliesOpt,
    BucketsOpt,
    GroupsOpt,
    IdsOpt,
    JudgeOpt,
    JudgeReasonOpt,
    LocalModelOpt,
    MaxTokOpt,
    ModelConcurrencyOpt,
    ModelsOpt,
    NoConsistencyOpt,
    NOpt,
    NoStoreOpt,
    RepsOpt,
    RerunModelOpt,
    RerunOpt,
    RosterOpt,
    TempOpt,
    ThreshOpt,
    _resolve_backends,
)
from ._orchestrate import _do_generate, _setup_store_run
from ._reps import _echo_cost_total, _echo_judge_summary, _extra_judge_reps, _judge_reps


@app.command()
def generate(
    models: str = ModelsOpt,
    groups: Optional[str] = GroupsOpt,
    ids: Optional[str] = IdsOpt,
    families: Optional[str] = FamiliesOpt,
    all_questions: bool = AllQOpt,
    roster: Optional[str] = RosterOpt,
    buckets: Optional[str] = BucketsOpt,
    n: int = NOpt,
    temperature: float = TempOpt,
    max_tokens: int = MaxTokOpt,
    model_concurrency: int = ModelConcurrencyOpt,
    judge: str = JudgeOpt,
    out: Optional[str] = typer.Option(
        None, "--out", "-o", help="run dir (default results/twominds/<ts>)"
    ),
    display: str = DisplayOpt,
    rerun: bool = RerunOpt,
    rerun_model: Optional[List[str]] = RerunModelOpt,
    no_store: bool = NoStoreOpt,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="print plan + cost, no API calls"
    ),
):
    """Phase 1: sample each model N times on the question roster (Inspect).

    Generations are cached per model under results/twominds/models/ and reused
    when the same questions + sampling config recur; --rerun / --rerun-model
    force fresh ones, --no-store restores the old self-contained behavior.
    """
    if no_store:
        _do_generate(
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
            roster=roster,
            buckets=buckets,
            model_concurrency=model_concurrency,
            will_judge=False,
        )
        return
    run_dir, _specs, _gen_dirs, _cached = _setup_store_run(
        models,
        groups,
        ids,
        all_questions,
        families,
        roster,
        buckets,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        judge=judge,
        out=out,
        display=display,
        model_concurrency=model_concurrency,
        rerun=rerun,
        rerun_models=rerun_model,
        dry_run=dry_run,
        will_judge=False,
    )
    if run_dir is not None:
        typer.echo(f"Generation complete: {run_dir}")


@app.command()
def run(
    models: str = ModelsOpt,
    groups: Optional[str] = GroupsOpt,
    ids: Optional[str] = IdsOpt,
    families: Optional[str] = FamiliesOpt,
    all_questions: bool = AllQOpt,
    roster: Optional[str] = RosterOpt,
    buckets: Optional[str] = BucketsOpt,
    n: int = NOpt,
    temperature: float = TempOpt,
    max_tokens: int = MaxTokOpt,
    model_concurrency: int = ModelConcurrencyOpt,
    judge: str = JudgeOpt,
    judge_reasoning: str = JudgeReasonOpt,
    backends: List[str] = BackendsOpt,
    threshold: float = ThreshOpt,
    local_model: str = LocalModelOpt,
    concurrency: int = ConcurrencyOpt,
    reps: int = RepsOpt,
    no_consistency: bool = NoConsistencyOpt,
    out: Optional[str] = typer.Option(None, "--out", "-o", help="run dir"),
    display: str = DisplayOpt,
    no_judge: bool = typer.Option(False, "--no-judge", help="skip the LLM judge"),
    rerun: bool = RerunOpt,
    rerun_model: Optional[List[str]] = RerunModelOpt,
    no_store: bool = NoStoreOpt,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="print plan + cost, no API calls"
    ),
):
    """All phases: generate -> judge (rep1..repN) -> consistency -> report.

    Generations (and rep1 judge verdicts) are cached per model under
    results/twominds/models/ and reused automatically when the same questions +
    sampling config come around again — only missing models hit the API. Use
    --rerun / --rerun-model to force fresh generations, --no-store for the old
    fully-self-contained behavior.

    --reps N runs a full robust pass (rep1 + rep2..repN + consistency) in one
    command, so a robust run is a single invocation.
    """
    backends = _resolve_backends(backends)
    if not backends and no_judge:
        raise typer.BadParameter("-b none with --no-judge leaves nothing to analyze")
    if no_store:
        run_dir = _do_generate(
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
            roster=roster,
            buckets=buckets,
            model_concurrency=model_concurrency,
            backends=list(backends),
            will_judge=not no_judge,
            judge_reps=reps if not no_judge else 1,
        )
        if run_dir is None:  # dry run
            return
        _judge_reps(
            run_dir,
            reps=reps,
            consistency=not no_consistency,
            backends=backends,
            judge=judge,
            judge_reasoning=judge_reasoning,
            threshold=threshold,
            local_model=local_model,
            concurrency=concurrency,
            run_judge=not no_judge,
            build_report=True,
        )
        typer.echo(f"\nDone. Open {run_dir / 'report.html'}")
        return

    run_dir, specs, gen_dirs, cached_gens = _setup_store_run(
        models,
        groups,
        ids,
        all_questions,
        families,
        roster,
        buckets,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        judge=judge,
        out=out,
        display=display,
        model_concurrency=model_concurrency,
        rerun=rerun,
        rerun_models=rerun_model,
        dry_run=dry_run,
        backends=list(backends),
        will_judge=not no_judge,
        judge_reps=reps if not no_judge else 1,
    )
    if run_dir is None:  # dry run
        return

    # rep1: cached per-model judge fragments where fresh, judged now otherwise.
    judge_key = store_mod.compute_judge_key(
        judge_name=judge,
        judge_reasoning=judge_reasoning,
        threshold=threshold,
        backends=list(backends),
        local_model=local_model,
        run_judge=not no_judge,
    )
    combined = store_mod.assemble_run(
        run_dir,
        specs,
        gen_dirs,
        judge_key=judge_key,
        backends=list(backends),
        judge_name=judge,
        judge_reasoning=judge_reasoning,
        threshold=threshold,
        local_model=local_model,
        concurrency=concurrency,
        run_judge=not no_judge,
        on_fragment=lambda name, cached: typer.echo(
            f"  [judge {'cached ✓' if cached else '✔'}] {name}"
        ),
        # rep2..N judge the whole run and read run_dir/cache — seed it from the
        # fragments' per-model caches so they don't re-embed everything.
        preseed_cache=reps > 1,
        cached_gens=set(cached_gens),
    )
    _echo_judge_summary(combined, run_dir)
    typer.echo(f"  report -> {report_mod.build_report_from_run(run_dir)}")
    if combined.get("cost"):
        typer.echo(cost_mod.format_summary(combined["cost"]))
    if reps > 1 and not no_judge:
        rep_costs = _extra_judge_reps(
            run_dir,
            reps=reps,
            consistency=not no_consistency,
            backends=backends,
            judge=judge,
            judge_reasoning=judge_reasoning,
            threshold=threshold,
            local_model=local_model,
            concurrency=concurrency,
        )
        _echo_cost_total([combined.get("cost") or {}] + rep_costs)
    typer.echo(f"\nDone. Open {run_dir / 'report.html'}")
