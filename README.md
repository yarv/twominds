# Coherence variance

[![CI](https://github.com/yarv/coherence-variance/actions/workflows/ci.yml/badge.svg)](https://github.com/yarv/coherence-variance/actions/workflows/ci.yml)

**How coherent is an LLM with itself?** Ask a model the same questions N times
at temperature 1.0. A cross-sample LLM judge sees all N answers to a question
at once and flags contradictions; embedding clustering provides a judge-free
cross-check. Results come out as self-contained HTML reports.

## Setup

Needs [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
uv sync
cp .env.example .env
```

Fill in two keys: `OPENAI_API_KEY` (generation) and `OPENROUTER_API_KEY`
(the judge, and any `openrouter/...` models you run). No OpenRouter key?
Route the judge through any provider you do have, e.g.
`--judge anthropic/claude-opus-4.8` with `ANTHROPIC_API_KEY` set.

The install is lean by default, and the default embedding backend
(`openai-3-small`) is API-priced in cents on the same `OPENAI_API_KEY`.
Prefer fully local embeddings? Opt into the torch-based backend with
`uv sync --group local-embeddings` and pass `-b local`.

Installed as a package (`pip install .` / a dependency), the same CLI is
available as `coherence-variance <command>`; from a checkout, use
`uv run python variance_experiment.py <command>` as below.

## Run it

Pass your models to `--models` (comma-separated): OpenAI fine-tune IDs and
model names as-is, OpenRouter models as `openrouter/<vendor>/<model>`, and
self-hosted or other OpenAI-compatible endpoints (llama-server, vLLM, TGI,
Together, ...) as `openai-api/<service>/<model>` with
`<SERVICE>_BASE_URL`/`<SERVICE>_API_KEY` env vars (see
`coherence_variance/README.md`).

```bash
uv run python variance_experiment.py run --n 20 --models \
  "ft:gpt-4.1-2025-04-14:your-org:your-model:AbCd1234,gpt-4.1,openrouter/qwen/qwen3-32b"

# torch-based local embeddings instead of the API backend (heavier install):
#   uv sync --group local-embeddings   then add:   -b local
```

The default question set has 96 questions: 57 coherence probes (values,
introspection, situational awareness, high stakes, AI safety, sycophancy)
plus 39 framing-family variants for the cross-variant sycophancy/robustness
analysis. Each model answers every question 20 times, then the answers are
judged and clustered. Framing families get their own `families_report.html`;
the main report's within-prompt view stays clean of them.

Models are identified by the last segment of their id (`openrouter/qwen/
qwen3-32b` shows up as `qwen3-32b`; colliding names are auto-qualified).

Add `--dry-run` first to see the plan and a rough cost estimate without
making any API calls. It also shows which models would be reused from cache.

### Results are reused automatically

Generations (and judge verdicts) are cached per model under
`results/variance/models/`, keyed by the exact questions + sampling config.
Re-running the same command reuses everything and costs nothing; adding a
model to `--models` only calls the API for the new one. Force fresh
generations with `--rerun` (all models) or `--rerun-model <name>` (one), or
bypass the cache entirely with `--no-store`.

## Look at the results

Everything lands in `results/variance/<timestamp>/`. Open **`report.html`**
in a browser. It is a single self-contained file: an interactive per-category
variance chart, plus every question's responses, judge verdicts, and
clusters. The raw generation and judge calls are kept alongside as Inspect
logs, each in both `.eval` (open with `inspect view`) and human-readable
`.json` form.

## Reproducing results

Temperature-1.0 resampling is the *object of study*, so individual responses
never reproduce; the pipeline and the aggregate signals do. What to rely on:

- **Environment**: `uv.lock` is committed; `uv sync` rebuilds the exact
  dependency tree.
- **Provenance**: every run dir is self-describing: `run_config.json`
  (models, N, sampling), `questions.json` (the exact prompts asked),
  `judge_meta.json` (judge model + prompt hash), and the raw Inspect logs of
  every generation and judge call, in `.eval` and `.json` form.
- **Frozen question lists**: `--roster <name>` pins an explicit id-list
  (`questions/_rosters.yaml`), immune to later roster edits.
- **Judge noise**: before reading a between-model difference as real, re-judge
  the same generations (`--reps 3`, or `analyze --judge-run rep2` then
  `consistency`); generations and embeddings are cached, so only judge calls
  repeat.
- **Judge accuracy**: the `stress` command scores the judge against synthetic
  bundles with an engineered ground-truth partition, for calibrating how much
  to trust its verdicts in the first place.

## More

`EXTENDING.md` is the recipe book for adding questions, families, models, and
backends. `coherence_variance/README.md` covers the rest: opt-in question buckets
(`tier_2`, `prompt_robustness` framing families), repeat judge runs and
consistency checks, embedding backends, and how to add questions or models.

MIT license (see [LICENSE](LICENSE)).
