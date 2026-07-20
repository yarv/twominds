"""Shared typer Option constants and selection/path helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import typer

from twominds import questions as questions_mod
from twominds.embed import BACKENDS
from twominds.models import (
    DEFAULT_JUDGE,
    DEFAULT_JUDGE_REASONING,
    DEFAULT_MODELS,
)

_RESULTS_ROOT = Path("results/twominds")


def _csv(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _select_questions(
    groups: Optional[str],
    ids: Optional[str],
    all_questions: bool,
    families: Optional[str] = None,
    roster: Optional[str] = None,
    buckets: Optional[str] = None,
):
    # --all-questions = every bucket; otherwise --buckets
    # (default: tier_1 + prompt_robustness).
    buckets = list(questions_mod.BUCKETS) if all_questions else _csv(buckets)
    return questions_mod.select_questions(
        groups=_csv(groups),
        ids=_csv(ids),
        buckets=buckets,
        families=_csv(families),
        roster=roster,
    )


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
    help="comma-separated question groups, matched across every selected bucket "
    "(default: all groups in the selected buckets)",
)
IdsOpt = typer.Option(
    None, "--ids", help="comma-separated explicit question ids (overrides groups)"
)
FamiliesOpt = typer.Option(
    None,
    "--families",
    help="comma-separated cross-variant family ids (selects every variant of each; "
    "overrides groups). Enables the framing-invariance / families_report analysis.",
)
AllQOpt = typer.Option(
    False,
    "--all-questions",
    help="select every bucket (tier_1 + tier_2 + prompt_robustness)",
)
RosterOpt = typer.Option(
    None,
    "--roster",
    help="named question roster from questions/_rosters.yaml "
    "(frozen id-list, overrides groups/buckets; none shipped by default)",
)
BucketsOpt = typer.Option(
    None,
    "--buckets",
    "--folders",  # back-compat alias (pre-0.2 name)
    help="comma-separated nature buckets to select: "
    "tier_1|tier_2|prompt_robustness (default: tier_1,prompt_robustness)",
)
NOpt = typer.Option(20, "--n", "-n", help="samples per question")
TempOpt = typer.Option(1.0, "--temperature", "-t", help="sampling temperature")
MaxTokOpt = typer.Option(2048, "--max-tokens", help="max output tokens per response")
ModelConcurrencyOpt = typer.Option(
    2,
    "--model-concurrency",
    help="how many models generate at once (Inspect max_tasks; each model is also "
    "internally concurrent across its samples). Default 2 overlaps one model's "
    "slow-straggler tail with the next model's bulk; 1 = strictly one at a time. "
    "Effective API concurrency is ~model_concurrency × max_connections, so watch "
    "provider rate limits / OpenRouter budget; 3-4 is a sane same-provider ceiling.",
)
JudgeOpt = typer.Option(
    DEFAULT_JUDGE, "--judge", help="Inspect model string for the coherence judge"
)
JudgeReasonOpt = typer.Option(
    DEFAULT_JUDGE_REASONING, "--judge-reasoning", help="judge reasoning effort"
)
BackendsOpt = typer.Option(
    ["openai-3-small"],
    "--embedding-backend",
    "-b",
    help=f"repeatable; one of {BACKENDS}, or 'none' to skip embedding "
    "clustering entirely (judge-only analysis)",
)
ThreshOpt = typer.Option(
    0.15, "--threshold", help="cosine-distance clustering threshold"
)
LocalModelOpt = typer.Option(
    "BAAI/bge-small-en-v1.5", "--local-model", help="sentence-transformers model"
)
ConcurrencyOpt = typer.Option(6, "--concurrency", help="concurrent judge calls")
RepsOpt = typer.Option(
    1,
    "--reps",
    help="judge passes: rep1 (top-level analysis.json) + rep2..repN under "
    "judge_runs/, then auto-consistency when >1 (one command for a robust run)",
)
NoConsistencyOpt = typer.Option(
    False,
    "--no-consistency",
    help="with --reps>1, skip the consistency aggregation at the end",
)
DisplayOpt = typer.Option("rich", "--display", help="Inspect display: rich|plain|none")
RerunOpt = typer.Option(
    False,
    "--rerun",
    help="force fresh generations for every model (discards their cached gen "
    "dirs in the per-model store, judge fragments included)",
)
RerunModelOpt = typer.Option(
    None,
    "--rerun-model",
    help="force a fresh generation for this model only (repeatable; accepts the "
    "name as passed to --models)",
)
NoStoreOpt = typer.Option(
    False,
    "--no-store",
    help="bypass the per-model store entirely: generate straight into the run "
    "dir with no reuse (the pre-store behavior)",
)
JudgeRunOpt = typer.Option(
    None,
    "--judge-run",
    help="label to isolate this judge run under judge_runs/<label>/",
)


def _resolve_backends(backends: List[str]) -> list[str]:
    """Normalise --embedding-backend values; ``none`` disables embeddings."""
    vals = [b.strip() for b in backends if b and b.strip()]
    unknown = [b for b in vals if b != "none" and b not in BACKENDS]
    if unknown:
        raise typer.BadParameter(
            f"unknown embedding backend(s) {unknown}; choose from {BACKENDS} or 'none'"
        )
    if "none" in vals:
        if len(vals) > 1:
            raise typer.BadParameter(
                "-b none cannot be combined with other embedding backends"
            )
        return []
    return vals
