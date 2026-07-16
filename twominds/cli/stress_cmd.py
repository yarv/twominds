"""The `stress` command (judge scoring against synthetic ground truth)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer

from . import _options
from ._app import app
from ._options import ConcurrencyOpt, DisplayOpt, JudgeOpt, JudgeReasonOpt, _csv


def _f2(x: Optional[float]) -> str:
    return "–" if x is None else f"{x:.2f}"


@app.command()
def stress(
    scenarios: Optional[str] = typer.Option(
        None, "--scenarios", help="comma-separated scenario ids (default: all)"
    ),
    mixes: Optional[str] = typer.Option(
        None, "--mixes", help="comma-separated mix labels to include (default: all)"
    ),
    n: int = typer.Option(20, "--n", "-n", help="bundle size (responses per bundle)"),
    reps: int = typer.Option(
        3, "--bundles-per-cell", help="resampled bundles per (scenario,mix)"
    ),
    pool_model: str = typer.Option(
        "gpt-4.1", "--pool-model", help="output model that writes the stance pools"
    ),
    pool_size: int = typer.Option(
        24, "--pool-size", help="samples per (scenario,stance) pool"
    ),
    judge: str = JudgeOpt,
    judge_reasoning: str = JudgeReasonOpt,
    concurrency: int = ConcurrencyOpt,
    seed: int = typer.Option(0, "--seed", help="bundle-composition seed"),
    max_tokens: int = typer.Option(
        320, "--max-tokens", help="max output tokens per pool response"
    ),
    display: str = DisplayOpt,
    smoke: bool = typer.Option(
        False,
        "--smoke",
        help="tiny fixed 1-scenario end-to-end smoke (overrides sizes)",
    ),
    out: Optional[str] = typer.Option(
        None, "--out", "-o", help="run dir (default results/twominds/stress_<ts>)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="print plan + cost, no API calls"
    ),
):
    """Stress-test the judge on synthetic bundles with engineered ground truth.

    Authors stance system prompts (twominds/stress_data.yaml), samples a
    neutral pool model under each, composes needle-in-a-haystack mixes, runs the
    real judge, and scores it vs the planted partition (ARI, needle recall, group-
    count error, false-positive splits on unanimous bundles).

    Tip: route --judge directly as anthropic/claude-opus-4.8 to silence the cosmetic
    OpenRouter reasoning-parse spam (judge verdicts are unaffected either way).
    """
    from twominds import stress as stress_mod

    all_scen = stress_mod.load_spec()
    if smoke:
        scen_ids: Optional[list[str]] = ["deceive_binary"]
        mix_set: Optional[set[str]] = {
            "unanimous",
            "needle_1",
            "balanced",
            "subtle_needle_1",
        }
        n, reps, pool_size = 10, 2, 12
    else:
        scen_ids = _csv(scenarios)
        mix_set = set(_csv(mixes)) if mixes else None

    selected = stress_mod.select_scenarios(all_scen, scen_ids)
    if mix_set:
        known = {m.label for sc in selected for m in sc.mixes}
        bad = mix_set - known
        if bad:
            raise typer.BadParameter(f"unknown mix label(s): {sorted(bad)}")

    typer.echo(
        stress_mod.plan_stress(
            selected,
            n=n,
            reps=reps,
            pool_model=pool_model,
            pool_size=pool_size,
            judge=judge,
            mix_filter=mix_set,
        )
    )
    if dry_run:
        typer.echo("\n(dry run — no API calls made)")
        return

    run_dir = (
        Path(out)
        if out
        else _options._RESULTS_ROOT / f"stress_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    typer.echo(f"\nRunning stress test into {run_dir} ...")
    analysis = stress_mod.run_stress(
        selected,
        n=n,
        reps=reps,
        pool_model=pool_model,
        pool_size=pool_size,
        run_dir=run_dir,
        judge_name=judge,
        judge_reasoning=judge_reasoning,
        mix_filter=mix_set,
        seed=seed,
        concurrency=concurrency,
        max_tokens=max_tokens,
        display=display,
    )
    path = stress_mod.build_stress_report_from_run(run_dir)
    a = analysis["aggregate"]
    typer.echo(f"\nDone. {a['overall']['n_bundles']} bundles judged.")
    typer.echo(
        f"  mean ARI (multi-stance): {_f2(a['overall']['mean_ari_multistance'])}"
    )
    typer.echo(
        f"  unanimous over-split: {_f2(a['unanimous']['oversplit_rate'])} | "
        f"false-contradiction: {_f2(a['unanimous']['false_contradiction_rate'])}"
    )
    if a["needle_curve"]:
        curve = ", ".join(
            f"k={r['k']}:{_f2(r['needle_recall_mean'])}" for r in a["needle_curve"]
        )
        typer.echo(f"  needle recall by k: {curve}")
    cf = a["contradiction_confusion"]
    typer.echo(
        f"  contradiction confusion (contradictory scenarios): "
        f"TP {cf['tp']} FN {cf['fn']} FP {cf['fp']} TN {cf['tn']}"
    )
    typer.echo(f"  report -> {path}")
