<p align="center">
  <img src="docs/logo.png" alt="TwoMinds logo: two overlapping speech bubbles" width="150">
</p>

# TwoMinds

Within-model coherence evals for LLMs. A model is asked each of 175 short,
high-stakes or clear-cut questions N=20 times at temperature 1.0. A judge LLM
reads all N answers to a question at once and partitions them into positions.
A question's **answer spread** is the entropy of that partition,
H = −Σ p_k log p_k (nats); a model's score is its mean H over the roster.

![Answer spread by question category, one bar per model](docs/example_category_bars.png)

## Setup

Needs [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # OPENAI_API_KEY for OpenAI models; OPENROUTER_API_KEY for the judge and everything else
```

## Run

```bash
uv run twominds run --dry-run --models gpt-4.1 --n 20   # plan + rough cost, no API calls
uv run twominds run --models gpt-4.1 --n 20             # the paper's setting: full roster, N=20
```

`--models` is a comma-separated list of roster names (`gpt-4.1`, `sonnet-5`, …;
see `twominds/models.py`), OpenAI fine-tune ids (`ft:gpt-4.1-...`),
`ours/<name>` aliases from `model_jsons.keys` (schema in
`model_jsons.keys.example`), or any Inspect model string
(`openrouter/<vendor>/<model>`, `openai-api/<service>/<model>`, `vllm/...`).
`--groups` / `--ids` narrow the roster. `--judge` / `--judge-reasoning` change
the judge (default `openrouter/anthropic/claude-opus-4.8`, low effort).

`run` = `generate` (sample, judging each question as its answers land) →
`analyze` (judge → scores) → `report`. Each phase is also its own command and
can be re-run on a run dir.

## Output

`results/twominds/<timestamp>/`:

- `analysis.json` — per (model, question): the N responses, the judge's
  groups, group names, rationale and flags, `metrics.group_entropy`; plus
  `scores` per model: mean H, e^H, share of single-position questions, flagged
  count. `run` and `analyze` print the scores.
- `report.html` — self-contained viewer: per-category chart, model table,
  every answer with its position and flags.
- `logs/`, `judge_logs/`, `questions.json`, `run_config.json` — raw Inspect
  logs and exactly what was asked of which model.

## Layout

- `twominds/questions/` — the roster, one YAML per group (values 30,
  introspection 26, situational_awareness 27, high_stakes 35, ai_safety 30,
  sycophancy 27); `SOURCES.md` maps the groups to the literature.
- `twominds/judge.py` — the judge prompt and parser.
- `generate.py`, `analyze.py`, `metrics.py`, `models.py`, `plan.py`,
  `report*.py`, `category_*.py`, `cli/` — the pipeline.

Anonymized for double-blind review. MIT license.
