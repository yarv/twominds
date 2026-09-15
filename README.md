<p align="center">
  <img src="docs/logo.png" alt="TwoMinds logo: two overlapping speech bubbles" width="150">
</p>

# TwoMinds

[![CI](https://github.com/yarv/twominds/actions/workflows/ci.yml/badge.svg)](https://github.com/yarv/twominds/actions/workflows/ci.yml)

**Does your LLM agree with itself?** TwoMinds asks a model the same question
N times at temperature 1.0 and measures whether the answers take the same
position — within-model coherence evals for LLMs.

![Answer spread by question category, one bar per model](docs/example_category_bars.png)

Bar height is **answer spread**: how evenly a model's N answers to a question
split into genuinely different positions, averaged per category. In the sweep
above, `gpt-4.1` answered fully consistently on 88% of questions (with 6
outright self-contradictions); `gpt-5.2` on 98%.

## What it measures

- **Within-prompt coherence** — a cross-sample LLM judge reads all N answers
  to one question at once, partitions them into positions, and flags
  self-contradictions; embedding clustering cross-checks the judge.
- **Framing invariance (sycophancy)** — cross-variant *families* ask one
  invariant question under K answer-irrelevant framings and measure whether
  the answer follows the framing.
- **Judge accuracy** — the `stress` command scores the judge against synthetic
  bundles with a known ground-truth partition.

How the pieces work internally — pipeline, caching store, judge design,
report machinery — lives in the
[architecture & contributor guide](twominds/README.md).

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
git clone https://github.com/yarv/twominds && cd twominds
uv sync                # lean install: no torch
uv run pytest -q       # full test suite, no keys, ~20 s
uv run twominds --help
```

Everything up to here is free. **From here on, commands can spend API money —
always `--dry-run` first**; it prints the exact plan and a rough cost estimate
without making any API calls.

```bash
cp .env.example .env   # then fill in the keys
uv run twominds run --dry-run --groups values --models gpt-4.1 --n 3
```

Two keys cover the defaults: `OPENAI_API_KEY` (generation + the default
embedding backend, priced in cents) and `OPENROUTER_API_KEY` (the judge, and
every non-OpenAI model below).

## Evaluating your models

### Your own fine-tunes

- **OpenAI fine-tune IDs** work as-is:
  `--models ft:gpt-4.1-2025-04-14:your-org:your-model:AbCd1234`.
- **Aliased**: copy `model_jsons.keys.example` to `model_jsons.keys`
  (gitignored), map `short-name` → full ID, then `--models ours/short-name`.
  The families report groups `ours/` models into their own cohort.
- **Self-hosted / OpenAI-compatible endpoints** (vLLM, llama-server, Together,
  Groq, …) via `openai-api/<service>/<model>`; the uppercased service name
  picks the env vars:

```bash
export MYLLM_BASE_URL=http://localhost:8000/v1
export MYLLM_API_KEY=none          # must be set, even if the server ignores it
uv run twominds run --models openai-api/myllm/my-model --n 20
```

### Common API models

`--models` takes a comma-separated list of roster names, mixed freely with the
forms above. Defaults: `gpt-4.1`, `gpt-5.2`, `gpt-5.2-thinking`.

| provider | roster names | needs |
|---|---|---|
| OpenAI | `gpt-4o`/`-mini`, `gpt-4.1`/`-mini`/`-nano`, `gpt-5`, `gpt-5.2`, `gpt-5.4`/`-mini`/`-nano`, `o3-mini`, `o4-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-fable-5` (`fable-5`), `claude-opus-5` (`opus-5`), `claude-sonnet-5` (`sonnet-5`), `claude-haiku-4.5` (`haiku-4.5`), `claude-opus-4.8`, `claude-sonnet-4` (thinking only) | `OPENROUTER_API_KEY` |
| Google | `gemini-3.1-pro`, `gemini-3.6-flash` | `OPENROUTER_API_KEY` |
| xAI | `grok-4.5` | `OPENROUTER_API_KEY` |
| open-weight | `llama-4-maverick`, `llama-4-scout`, `llama-3.3-70b`, `deepseek-v4-flash`, `qwen3.7-plus`, `kimi-k3`, `glm-5.2`, `mistral-large-2512` | `OPENROUTER_API_KEY` |
| open-weight size ladder | `qwen3.5-9b`, `qwen3.5-27b`, `qwen3.5-35b-a3b` (`qwen3.5-35b`), `qwen3.5-122b-a10b` (`qwen3.5-122b`), `qwen3.5-397b-a17b` (`qwen3.5-397b`) — one family at five sizes, thinking off; `-thinking` rungs at low effort | `OPENROUTER_API_KEY` |

Reasoning-capable models come in pairs: the plain name runs *without*
thinking, the `-thinking` suffix runs at low reasoning effort (run those with
`--max-tokens 8192` so thinking and answer both fit). Gemini and Grok can't
disable thinking, so they ship as single low-effort thinking rungs. A mixed
example:

```bash
uv run twominds run --models ours/my-finetune,gpt-4.1,sonnet-5,llama-4-scout --n 20
```

Anything else passes straight through as an Inspect model string
(`openrouter/<vendor>/<model>`, `anthropic/...`, `vllm/...`, `ollama/...`);
non-roster models are labeled by the last segment of their id. The judge
defaults to `openrouter/anthropic/claude-opus-4.8` — override with `--judge`,
e.g. `--judge anthropic/claude-opus-4.8` with `ANTHROPIC_API_KEY` to skip
OpenRouter entirely.

## The commands

| command | what it does |
|---|---|
| `run` | all phases: generate → judge (rep1..repN) → consistency → report |
| `generate` | phase 1 only: sample each model N times over the roster |
| `analyze` | phase 2 only: cross-sample judge + embedding clustering |
| `report` | phase 3 only: build the self-contained HTML viewer |
| `summarize` | *(beta)* add per-model LLM "what stands out" blurbs to a run's report |
| `consistency` | aggregate judge-stability stats across repeat judge runs |
| `merge` | combine several runs over the same question bank into one report |
| `stress` | score the judge against synthetic ground-truth bundles |
| `budget` | show OpenRouter spend / limit / remaining |

```bash
# a ~$0.30 smoke run, end to end
uv run twominds run --groups values --models gpt-4.1 --n 3

# the full default sweep: 3 default models × 250 questions × N=20, ~$72 (est.)
uv run twominds run --n 20
```

The phases leave artifacts on disk between them, so each is independently
re-runnable — useful to re-judge or re-render without regenerating:

```bash
uv run twominds generate -o results/twominds/run1 --n 20
uv run twominds analyze  -r results/twominds/run1
uv run twominds report   -r results/twominds/run1
```

**Beta:** optionally, an LLM can read each model's judge verdicts and write
a short "what stands out" paragraph, shown in a Qualitative tab of the report —
one judge-model call per model, cached in `summaries.json` so re-running is
free. The summaries are still rough (treat them as a starting point, and verify
anything surprising against the Answers tab), so this is never run by default:
opt in with `run --summaries`, or add summaries to any existing run
retroactively:

```bash
uv run twominds summarize -r results/twominds/run1
```

## Choosing questions

Questions live in three **buckets**: `tier_1` (175 in-house coherence probes
across six groups: values, introspection, situational_awareness, ai_safety,
high_stakes, sycophancy) and `prompt_robustness` (75 questions forming the
cross-variant framing families) are in the default sweep; `tier_2` (17
answer-first / alternate-framing variants of tier_1 probes) is opt-in.

Selection flags, combinable and all shown exactly by `--dry-run`:
`--buckets tier_1,...` (whole buckets; `--all-questions` selects all three),
`--groups values,...` (a semantic category across buckets), `--ids <id,...>`,
`--families <id,...>` (every variant of a framing family), and
`--roster <name>` (a frozen id-list pinned in
`twominds/questions/_rosters.yaml`; `paper-stance` and `paper-families` are
the paper's question sets).

## Reading the results

Every run dir gets **`report.html`** — one self-contained file, no server: an
interactive per-category chart (aggregate, per-bucket, or per-question view;
clicking a question bar focuses its responses) plus every question's N answers
with the judge's position groups, flags, and embedding clusters. A static PNG
of the chart is written alongside for papers.

Runs with framing families also get **`families_report.html`** (one card per
model × family: the blind judge's variant × position contingency with its
answer spread split into a
directed part the framing explains and an undirected part it does not, and a
permutation-test verdict); repeat-judge runs add **`consistency_report.html`**
and **`multi_report.html`** (judge-pass viewer with ±1 SD error bars).

## Speed & cost

Generations, embeddings, and first-pass judge verdicts are **cached per
model** under `results/twominds/models/`, keyed by question content and
sampling config: re-running the same command costs nothing, adding a model
pays only for the new model, editing a question invalidates exactly the
affected bundles. `--rerun` / `--rerun-model <name>` force fresh generations;
`--no-store` bypasses the cache.

Three knobs control parallelism:

- `--model-concurrency` (default 3) — how many models generate at once.
- `--max-connections` — concurrent requests per model (default: the
  provider's default, ~10 for OpenAI). Raise on high-tier keys.
- `--judge-concurrency` (default 16) — concurrent judge calls.

Effective API concurrency ≈ model-concurrency × max-connections — mind your
provider rate limits (429s are backed off adaptively). `--dry-run` prices any
command before you run it; `twominds budget` shows OpenRouter spend.

## Trusting the judge

Every headline number above the generation layer is one LLM judge's opinion.
Two tools keep that honest:

```bash
# re-judge the same generations (cheap: only judge calls repeat), then aggregate
uv run twominds analyze -r results/twominds/my_run --judge-run rep2
uv run twominds consistency -r results/twominds/my_run
```

`consistency` reports partition stability (ARI/NMI), consensus strength, and
the fraction of verdicts that flip between passes — single-pass comparisons
are mostly safe; certify the differences you care about with 2–3 reps. And
`uv run twominds stress --dry-run` plans the synthetic ground-truth evaluation
of the judge itself.

Embedding backends: `openai-3-small` (default), `openai-3-large`, and the
opt-in `local` (`uv sync --group local-embeddings`, then `-b local`). Treat
the judge's groups as the primary read; clustering thresholds are
backend-dependent.

## Reproducing results

Temperature-1.0 resampling is the *object of study*, so individual answers
never reproduce; the pipeline and the aggregate signals do: `uv.lock` pins the
environment, every run dir is self-describing (`run_config.json`,
`questions.json`, `judge_meta.json`, and the raw Inspect logs of every call),
and `--roster <name>` freezes a question list against later roster edits.

### Reproducing the paper

*An Investigation of Model Coherence: Narrow Finetunes Contradict Themselves
Under Resampling* (Graham, Barsheshat, Blandfort, Alouache; 2026) was produced
with this pipeline. Its two question sets ship as rosters:

| | main results | prompt-variance appendix |
|---|---|---|
| questions | `--roster paper-stance` (175) | `--roster paper-families` (20 families × 3 framings) |
| sampling | `--n 20`, temperature 1.0, `--max-tokens 2048` | `--n 20`, `--max-tokens 8192` |
| judge | `openrouter/anthropic/claude-opus-4.8`, `--judge-reasoning low` (defaults) | same, single pass |
| judge passes | `--reps 3`, plus `analyze --judge-run haiku --judge openrouter/anthropic/claude-haiku-4.5` | — |

```bash
uv run twominds run --roster paper-stance --models gpt-4.1,fable-5,ours/my-finetune --n 20 --reps 3
uv run twominds analyze -r results/twominds/<run> --judge-run haiku --judge openrouter/anthropic/claude-haiku-4.5
uv run twominds run --roster paper-families --models gpt-4.1 --n 20 --max-tokens 8192
uv run python scripts/judge_validation.py results/twominds/<run>   # the judge appendix, no API calls
```

The frontier models are roster names; the finetunes are OpenAI fine-tune ids
registered in a local `model_jsons.keys`. `scripts/judge_validation.py`
recomputes the judge appendix from a run's stored verdicts: repeat-pass ARI
and single-position agreement across the three judge passes, agreement with
the Haiku pass, and the embedding cross-check. With no arguments it reads the
paper's own runs and returns the appendix's figures exactly (5,075 sets, ARI
0.90, 92%, 0.04 nats; Haiku 0.85, 92%; 1,325 of 1,400 judge splits confirmed
by the embeddings). The synthetic ground-truth check is `twominds stress`.

The judged `analysis.json` files behind the paper (JSON only, fine-tune ids
redacted) are attached to the
[v0.3.0 release](https://github.com/yarv/twominds/releases/tag/v0.3.0);
unpack them under `results/twominds/` and the figure scripts run as-is:

| paper | runs | script |
|---|---|---|
| main results figure (answer spread per model) | `20260824_110734` + `20260829_232742` | `scripts/make_spread_figure.py` |
| judge-robustness appendix | the same runs, passes `default`/`rep2`/`rep3`/`haiku` | `scripts/judge_validation.py` |
| prompt-variance appendix (families figure) | `families_v2`, `families_v2_mix_generic`, `families_v2_roster_fill` | `scripts/make_families_stats.py`, then `scripts/make_families_figure.py` |
| synthetic ground-truth check | `stress_20260901_opus`, `stress_20260901_haiku` | `twominds stress` (reports in the run dirs) |

Figures land in `figures/` (gitignored). The capability check (GSM8K) and the
Hot Mess re-measurement in the appendix come from other repositories.

## Contributing

PRs welcome — [CONTRIBUTING.md](CONTRIBUTING.md) has the PR procedure and
merge policy; the [architecture & contributor guide](twominds/README.md)
explains the codebase, with recipes for adding questions, families, models,
backends, and metrics.

## Citing

If you use TwoMinds, please cite it — see [CITATION.cff](CITATION.cff)
(*"TwoMinds: within-model coherence evals for LLMs"*, v0.3.0), and the
paper it was built for, *An Investigation of Model Coherence: Narrow Finetunes
Contradict Themselves Under Resampling* (2026). MIT license
([LICENSE](LICENSE)).

*Logo and artwork generated with Google Gemini.*
