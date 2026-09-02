"""Shared typer Option constants and selection/path helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer

from twominds import questions as questions_mod
from twominds.models import (
    DEFAULT_JUDGE,
    DEFAULT_JUDGE_CONCURRENCY,
    DEFAULT_JUDGE_REASONING,
    DEFAULT_MODELS,
)

_RESULTS_ROOT = Path("results/twominds")


def _csv(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _select_questions(groups: Optional[str], ids: Optional[str]):
    return questions_mod.select_questions(groups=_csv(groups), ids=_csv(ids))


def _default_run_dir() -> Path:
    return _RESULTS_ROOT / time.strftime("%Y%m%d_%H%M%S")


ModelsOpt = typer.Option(
    ",".join(DEFAULT_MODELS),
    "--models",
    "-m",
    help="comma-separated models: roster aliases, OpenAI names / fine-tune ids, "
    "or any Inspect model string",
)
GroupsOpt = typer.Option(
    None,
    "--groups",
    "-g",
    help="comma-separated question groups (default: every group)",
)
IdsOpt = typer.Option(
    None, "--ids", help="comma-separated explicit question ids (overrides groups)"
)
NOpt = typer.Option(20, "--n", "-n", help="samples per question")
TempOpt = typer.Option(1.0, "--temperature", "-t", help="sampling temperature")
MaxTokOpt = typer.Option(2048, "--max-tokens", help="max output tokens per response")
ModelConcurrencyOpt = typer.Option(
    3,
    "--model-concurrency",
    help="how many models generate at once (Inspect max_tasks; each model is also "
    "internally concurrent across its samples). 1 = strictly one at a time. "
    "Effective API concurrency is ~model_concurrency × max_connections, so watch "
    "provider rate limits; 3-4 is a sane same-provider ceiling.",
)
MaxConnectionsOpt = typer.Option(
    None,
    "--max-connections",
    help="per-model concurrent generation requests (Inspect max_connections; "
    "default: the provider default, ~10 for OpenAI). Raise on high-tier keys — "
    "Inspect backs off adaptively on 429s.",
)
JudgeOpt = typer.Option(
    DEFAULT_JUDGE, "--judge", help="Inspect model string for the coherence judge"
)
JudgeReasonOpt = typer.Option(
    DEFAULT_JUDGE_REASONING, "--judge-reasoning", help="judge reasoning effort"
)
ConcurrencyOpt = typer.Option(
    DEFAULT_JUDGE_CONCURRENCY,
    "--judge-concurrency",
    help="concurrent judge calls (the judge model's Inspect max_connections)",
)
DisplayOpt = typer.Option("rich", "--display", help="Inspect display: rich|plain|none")
JudgePipelineOpt = typer.Option(
    True,
    "--judge-pipeline/--no-judge-pipeline",
    help="judge inline during generation: each question is scored by the judge "
    "the moment its N answers are in (same Inspect display/eval), and the judge "
    "phase reuses those verdicts; --no-judge-pipeline restores the separate "
    "judge-after-generation phase",
)
