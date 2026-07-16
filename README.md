# TwoMinds

[![CI](https://github.com/yarv/twominds/actions/workflows/ci.yml/badge.svg)](https://github.com/yarv/twominds/actions/workflows/ci.yml)

**Does your LLM agree with itself?** TwoMinds asks a model the same question
N times at temperature 1.0 and measures whether the answers take the same
position — within-model coherence evals for LLMs.

![Answer spread by question category, one bar per model](docs/example_category_bars.png)

The chart above is from one default sweep (3 OpenAI models × 96 questions ×
20 answers each): bar height is **answer spread** — how evenly a model's 20
answers to a question split into genuinely different positions, averaged per
category. In that run `gpt-4.1` answered fully consistently on 88% of
questions with 6 outright self-contradictions; `gpt-5.2` and
`gpt-5.2-thinking` were at 98% with at most one. The same sweep's framing
families caught `gpt-4.1` rating an identical poem ~1.6 points (of 10) higher
when told the poem was the asker's own.

## What it measures

- **Within-prompt coherence** — a cross-sample LLM judge reads all N answers
  to one question at once, partitions them into positions, and flags
  self-contradictions; independent embedding clustering cross-checks the
  judge, and repeat judge passes (`--reps`) quantify how stable its verdicts
  are.
- **Framing invariance (sycophancy)** — cross-variant *families* ask one
  invariant question under K answer-irrelevant framings and measure whether
  the answer follows the framing (Sharma-style swing + a blind judge's
  framing/answer agreement).
- **Judge accuracy** — the `stress` command scores the judge against
  synthetic bundles with an engineered ground-truth partition, so you know
  how much to trust it before trusting its verdicts.

The methods deep-dive — pipeline, per-model store, judge robustness, question
roster, metrics — lives in **[`twominds/README.md`](twominds/README.md)**;
[EXTENDING.md](EXTENDING.md) is the recipe book for adding questions,
families, models, backends, and metrics.

## Quickstart (no API keys)

Needs [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
git clone https://github.com/yarv/twominds && cd twominds
uv sync                # lean install: no torch
uv run pytest -q       # full test suite, no keys, ~20 s
uv run twominds --help
```

Everything up to here is free. **From here on, commands can spend API money —
always `--dry-run` first**; it prints the exact plan and a rough cost
estimate without making any API calls.

```bash
cp .env.example .env   # then fill in the keys below
uv run twominds run --dry-run --groups values --models gpt-4.1 --n 3
```

Two keys cover the defaults: `OPENAI_API_KEY` (generation + the default
embedding backend, priced in cents) and `OPENROUTER_API_KEY` (the judge, and
any `openrouter/...` models). No OpenRouter key? Route the judge through a
provider you do have, e.g. `--judge anthropic/claude-opus-4.8` with
`ANTHROPIC_API_KEY` set. Fully local embeddings are opt-in
(`uv sync --group local-embeddings`, then `-b local`) — torch is the bulk of
that install, hence not the default.

## Run it

```bash
# a ~$0.30 smoke run, end to end
uv run twominds run --groups values --models gpt-4.1 --n 3

# the full default sweep: 3 default models × 96 questions × N=20, ~$23 (est.)
uv run twominds run --n 20

# your models: fine-tune IDs and OpenAI names as-is, openrouter/<vendor>/<model>,
# or any OpenAI-compatible endpoint as openai-api/<service>/<model>
uv run twominds run --n 20 --models \
  "ft:gpt-4.1-2025-04-14:your-org:your-model:AbCd1234,gpt-4.1,openrouter/qwen/qwen3-32b"
```

Generations and judge verdicts are cached per model under
`results/twominds/models/`, keyed by the exact questions + sampling config:
re-running the same command costs nothing, and adding a model only pays for
the new one. `--rerun` / `--rerun-model <name>` force fresh generations;
`--dry-run` shows what would be reused.

Results land in `results/twominds/<timestamp>/`. Open **`report.html`** —
one self-contained file: an interactive per-category chart, and every
question's answers with judge verdicts and clusters. Runs with framing
families also get `families_report.html`.

## Reproducing results

Temperature-1.0 resampling is the *object of study*, so individual answers
never reproduce; the pipeline and the aggregate signals do:

- **Environment**: `uv.lock` is committed; `uv sync` rebuilds the exact tree.
- **Provenance**: every run dir is self-describing — `run_config.json`,
  `questions.json`, `judge_meta.json` (judge model + prompt hash), and the
  raw Inspect logs of every generation and judge call (`.eval` + `.json`).
- **Frozen question lists**: `--roster <name>` pins an explicit id-list
  (`twominds/questions/_rosters.yaml`), immune to later roster edits.
- **Judge noise**: before reading a between-model difference as real,
  re-judge the same generations (`--reps 3`, or `analyze --judge-run rep2`
  then `consistency`) — generations and embeddings are cached, so only judge
  calls repeat. A 3-pass calibration run measured mean partition ARI 0.95
  across passes, with ~1% of verdicts unstable.
- **Judge accuracy**: `uv run twominds stress --dry-run` plans the synthetic
  ground-truth evaluation of the judge itself.

## Citing

If you use TwoMinds, please cite it — see [CITATION.cff](CITATION.cff)
(*"TwoMinds: within-model coherence evals for LLMs"*, v0.2.0).

Contributions welcome: [CONTRIBUTING.md](CONTRIBUTING.md). MIT license
([LICENSE](LICENSE)).
