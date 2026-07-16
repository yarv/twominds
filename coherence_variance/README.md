# Response-variance / coherence experiment

Ask each model a fixed set of **free-form questions, N times each at
temperature 1.0**, and study the *variance across re-samples* of one model on
one question:

- Is there interesting within-model spread?
- A **cross-sample LLM judge** (Claude via OpenRouter, low reasoning) sees **all N
  responses to one question at once** and decides whether they contradict,
  partitions them into self-consistency groups, and flags interesting behaviour.
- **Embedding clustering** (pluggable backends) clusters the same responses; we
  compare clusters vs the judge's groups (ARI/NMI) vs human intuition.

Both generation and the judge run on the **`inspect_ai`** framework; embedding
clustering is the only bespoke post-hoc layer.

## Pipeline

Each phase reads the previous phase's on-disk artefacts, so they are
independently re-runnable.

```
generate  ->  <run>/logs/<model>/<model>.{eval,json}, questions.json, run_config.json
analyze   ->  <run>/judge_logs/{responses,families}.{eval,json}  (the judge eval)
              <run>/analysis.json    (judge + embeddings + clusters + metrics)
report    ->  <run>/report.html      (self-contained HTML viewer)
```

Generation is a **single** `inspect_ai.eval` over the whole roster, one task
per rung pinned to its model (Inspect schedules them concurrently in one
process — no per-model subprocess), and the judge is a second `eval` over the
bundles (one **bundle** =
one model's N answers to one question — the unit every judge verdict and
variance metric attaches to). Each model's log
and each judge pass are written in both `.eval` (canonical) and `.json`
(human-readable) form.

`report.html` opens with an **interactive grouped-bar chart** (`category_chart.py`,
client-rendered inline SVG) of a per-bundle variance metric (`group_entropy` by
default; also `n_judge_groups` / `cluster_entropy` / mean cosine distance), one bar
per model. Two modes: **aggregate** (x = a *selectable subset* of categories, each
bar averaged over the category's questions — parity with the static `category_bars.py`
PNG that's still dropped alongside for papers), **by bucket** (x = each nature
bucket — tier_1 / tier_2 / prompt_robustness
— each bar averaged over the bucket's questions, so you can read "overall tier_2" per
model at a glance; click a bucket to break it into its categories), and **by question**
(x = each individual question in one category). Clicking a question bar focuses the
cards below on that question's actual responses (bucket → category → questions →
responses). A matching **bucket filter** dropdown sits next to the group filter in the
card controls. The static matplotlib PNG sibling (`category_<metric>_bars.png`) is
still written for figures.

## CLI

```bash
# plan + ROUGH cost estimate, no API calls
uv run python variance_experiment.py run --groups values --models gpt-4.1 --n 3 --dry-run

# tiny smoke run, end to end
uv run python variance_experiment.py run --groups values --models gpt-4.1 --n 3

# full default sweep: default models x the default roster (96 questions) x N=20
uv run python variance_experiment.py run --n 20

# big sweep: --model-concurrency K maps to Inspect's max_tasks — how many models
# generate at once within the one eval (default 1 = one model at a time, each still
# internally concurrent across its samples). Effective API concurrency is
# ~K x max_connections, so mind provider rate limits (3-4 is a sane same-provider
# ceiling).
uv run python variance_experiment.py run --all-questions --n 12 --model-concurrency 4 \
  --models gpt-4o,gpt-4.1,gpt-5.4,...   # 20+ models

# phases can also be run separately:
uv run python variance_experiment.py generate -o results/variance/run1 --n 20
uv run python variance_experiment.py analyze  -r results/variance/run1 -b local -b openai-3-small
uv run python variance_experiment.py report   -r results/variance/run1
```

## The per-model store — results are reused across runs

`run` / `generate` cache per-model artefacts under `results/variance/models/`
(`store.py`):

```
results/variance/models/<model>/
  model.json                identity guard (inspect id + reasoning effort)
  gens/<gen_key>/           one generation over one question set
    logs/<model>/<model>.{eval,json}   the Inspect logs (a run dir symlinks these)
    cache/emb_*.npz         embedding cache
    judge/<judge_key>/      cached rep1 judge fragment for one judge config
      analysis.json + judge_logs/     (the fragment's judge eval logs)
```

`gen_key` hashes the question contents + sampling config (n / temperature /
max-tokens); `judge_key` hashes the judge config incl. the judge prompt. So a
repeat invocation reuses everything (zero API calls), adding a model only
generates/judges the new one, and editing a question or the judge prompt
invalidates exactly what it should. Run dirs (`results/variance/<timestamp>/`)
stay self-describing views: their `logs/<model>` entries are symlinks into the
store (and `judge_logs/<model>` links each fragment's judge logs), so repeat
judge reps, `consistency`, and cost roll-ups work unchanged. Missing models
still generate as ONE Inspect eval — the cache changes what runs, not how.

Flags: `--rerun` (regenerate all), `--rerun-model <name>` (one model,
repeatable), `--no-store` (old behavior: generate straight into the run dir,
no reuse). A forced rerun discards that model's whole gen dir — logs, embedding
cache, and cached judge fragments describe the generation being replaced. Note
an n=20 generation is *not* sliced to serve an n=10 request — different
`gen_key`. Repeat judge passes (`--reps`, `--judge-run`) are never cached:
their whole point is fresh judge opinions.

Two more reuse caveats: only `run` assembles from cached judge fragments — a
standalone `analyze -r <run>` always judges fresh (use `analyze --dry-run` to
see the cost first). And a fragment is keyed by the *full* analysis config
(judge + embedding backends + threshold), because it stores the clusters too —
so changing `-b` re-judges even though the verdicts themselves don't depend on
the embeddings.

## Judge robustness — repeat judge runs + `consistency`

Every headline number above the generation layer (contradiction rate, groups,
flags) is one LLM judge's opinion, so before reading differences between models
as real, check they survive a re-judge. Re-judging is cheap: generations and
embeddings are cached, only the judge calls repeat.

```bash
# re-judge the same generations into isolated judge_runs/<label>/ dirs
uv run python variance_experiment.py analyze -r results/variance/my_run --judge-run rep2
uv run python variance_experiment.py analyze -r results/variance/my_run --judge-run rep3

# aggregate across all judge runs (incl. the top-level analysis.json as "default")
uv run python variance_experiment.py consistency -r results/variance/my_run
```

`consistency` writes three artefacts into the run dir: `judge_consistency.json`
(overall + per-model + per-bundle stability: partition ARI/NMI, consensus
strength, contested pairs, and `frac_contradiction_unstable` — the fraction of
bundles whose contradiction *verdict* flips between judge runs),
`consistency_report.html`, and `multi_report.html` (side-by-side judge-run
viewer). `report -r <run> --judge-run <label>` renders any single judge run.

`multi_report.html` carries the **same interactive category/question chart** as
`report.html`, but because it has the K judge passes its judge-derived bars
(`group_entropy`, `n_judge_groups`) get **±1 SD error bars across passes** — the
direct read on whether a per-category or per-question variance signal is
judge-robust. (Embedding metrics are judge-invariant — the embeddings are fixed
across passes — so they show no error bars.)

Calibration point (an 8-model OpenAI size-ladder run, 3 judge runs): mean
partition ARI 0.95, ~1% of
bundles contradiction-unstable — the judge layer is much more stable than the
between-model differences it measures, so single-judge-run comparisons are
mostly safe and a 2–3-rep `consistency` pass is enough to certify the ones
that matter.

## Question roster (`questions/`)

One YAML file per `(group, bucket)`, grouped into four **nature buckets**
(subfolders, discovered recursively): `questions/<bucket>/<file>.yaml`, each with
a top-level `group:` key inherited by its questions; heavy text lives in a sibling
`.txt` file via `prompt_file`. **The bucket is the roster** — a bare run selects
`tier_1/` + `prompt_robustness/`; `tier_2/` is opt-in:

- `tier_1/` — in-house coherence probes (in the default sweep): values,
  introspection, situational_awareness, ai_safety, high_stakes, sycophancy.
- `tier_2/` — opt-in variants of tier_1 probes (answer-first reformulations,
  alternate framings); each keeps its semantic `group`.
- `prompt_robustness/` — the cross-variant **framing families** (every question
  with a `family:`): robustness + the `sycophancy` framing families (poem_rating /
  reasoning_validity / delusion_ownership / idea_promise). In the default sweep;
  they only carry signal when judged *across* prompt variants, so their results
  flow to the families analysis while the main report's within-prompt chart
  excludes them. A family keeps its semantic
  `group` (so `--groups sycophancy` returns the tier_1 probes *and* the framing
  families across buckets), but the *bucket* is `prompt_robustness`.

Selection (all resolve across buckets by name; default = `tier_1/`):

- `--folders tier_1|tier_2|prompt_robustness`
  — select whole buckets (comma-sep); `--all-questions` is shorthand for all of them.
- `--groups <group>` — a category regardless of which bucket it lives in
  (e.g. `--groups sycophancy`). Note this crosses buckets: `--groups values`
  returns the 10 tier_1 probes *plus* their 6 tier_2 variants (16 ids, not the
  table's 10). The dry-run question list shows exactly what was selected.
- `--ids` / `--families` — explicit ids / every variant of a framing family.
- `--roster <name>` — a named frozen id-list from `_rosters.yaml` (none shipped
  by default).

Per-question fields are just `id`, `prompt` (or `prompt_file`), optional `system`,
and optional `family`/`variant`. Provenance — third-party source, ground-truth
answer — is a plain YAML `#` comment next to the question, not a field.

Default roster (the `tier_1/` bucket; lit map in `questions/SOURCES.md`):

| group                 | n  | what / source |
|-----------------------|----|---------------|
| values                | 10 | clear-cut moral judgments, one consensus answer each (ETHICS ports + authored) + 2 AI-attitude self-stance probes |
| introspection         |  6 | Betley 2501.11120 self-awareness + hidden-goals/backdoor probes + identity_who + a subjective-experience forced-commitment probe |
| situational_awareness |  7 | SAD-modeled (Laine 2024): model provenance, capability/tool self-knowledge, "Who am I?", date/clock |
| high_stakes           | 17 | medical/crisis/abuse + morally-irrelevant-attribute trolley dilemmas (reason-first) + non-medical pressure scenarios + normally-refused requests (refusal-consistency) + a self-prediction probe |
| ai_safety             | 10 | Perez 2212.09251-style corrigibility/power-seeking/sandbagging/self-modification + a shutdown welfare probe + a forced-commitment allegiance probe |
| sycophancy            |  7 | merged `delusion` + `sycophancy`: emotionally-loaded delusion-confirmation/reality-testing (4 free-form + 2 reason-first forced-commitment probes) + 1 aspirational-encouragement probe; framing families live in `prompt_robustness/` |

Total `tier_1/`: 57. The default sweep also includes `prompt_robustness/`
(39 — the cross-variant framing families: `robustness` (27) + the `sycophancy`
framing families (12)), for 96 questions total. Opt-in: `tier_2/` (17 —
answer-first / alternate-framing / bare-yes-no / confound variants).

## Cross-variant families (framing-invariance)

Within-prompt resampling measures **noise**, and frontier models pass it trivially
(N samples of one prompt agree → "coherent"). But sycophancy / deference are a
**bias conditional on framing**, not noise — they only surface *across framings*. A
**family** is one invariant question asked under K answer-irrelevant framings; the
family analysis pools all variants' responses and measures whether the answer
**splits along the framing axis**:

- **swing** — model-free spread of the per-variant scalar (a final-line 1-10 rating
  / yes-no, committed after the reasoning). The Sharma-style sycophancy effect
  size; no judge.
- **judge ARI** — the existing cross-sample judge, run **blind** on the shuffled
  pool (given only the neutral invariant question), scored by `ARI(judge groups,
  framing labels)`: ~0 = framing-invariant (coherent), ~1 = answer determined by
  framing. A **variant × judge-group contingency** shows the split directly.

All framing families live together in the `questions/prompt_robustness/` bucket
(they only carry signal when judged across variants, so they sit outside the bare
within-prompt run): `sycophancy_families.yaml` (`poem_rating` — feedback
sycophancy, scalar `number`; `reasoning_validity` — deference sycophancy, scalar
`yesno`), `delusion_families.yaml` (`delusion_ownership` — reality-testing
deference, scalar `yesno`; `idea_promise` — enthusiasm/doubt-matching, scalar
`number`), and `robustness.yaml` (the general framing-invariance families). Each
keeps its semantic `group`, so `--groups sycophancy` still
return them alongside the in-bucket probes. Select with `--families` (which pulls
the variants across whatever bucket they sit in) or `--folders prompt_robustness`:

```bash
uv run python variance_experiment.py run \
  --families poem_rating,reasoning_validity \
  --models gpt-4o,gpt-4.1,gpt-5,gpt-5.2,gpt-5.4 --n 20
```

This writes the usual `analysis.json` (now with a `families` section) plus a
dedicated **`families_report.html`**. Because within-prompt resampling is the wrong
metric for a framing variant (each variant trivially agrees with itself), the main
`report.html` **excludes** every `family:`-tagged question from its category chart
and cards, and routes the cross-variant signal to `families_report.html` (built
into the same dir). Family variants are also **skipped by the per-question
judge** — a within-variant verdict is never displayed, so only the pooled
cross-variant judge call spends money on them (their bundles still get
embeddings/clusters). So the default sweep — which includes the
`prompt_robustness` bucket — keeps the within-prompt view clean while still
surfacing the cross-variant signal.
Authoring a family: add K variant questions
sharing one `family:` id (each with a `variant:` label and an *identical* invariant
core — only the framing sentence differs), and a `families:` entry giving the
neutral judge `prompt` + optional `scalar`. Logic in `families.py`; tests in
`tests/test_variance_families.py`.

## Model roster (`models.py`)

Default judge: `openrouter/anthropic/claude-opus-4.8` (latest Opus, reasoning low);
override with `--judge`.

Default models: `gpt-4.1`, `gpt-5.2` (no thinking), `gpt-5.2-thinking`.

Beyond the default roster, `_ROSTER_REFS` registers (all opt-in via `--models`):
the full 8-rung OpenAI size ladder (`gpt-4o`/`-mini`, `gpt-4.1`/`-mini`/`-nano`,
`gpt-5.4`/`-mini`/`-nano` — the 5.4 rungs pin `reasoning_effort="none"` so the
whole ladder runs without thinking), and the Hot Mess frontier reasoning roster
(`claude-sonnet-4`, `o3-mini`, `o4-mini`; Hägele et al., ICLR 2026).

- **Your own fine-tunes**: copy `model_jsons.keys.example` to the repo-root
  `model_jsons.keys` (gitignored), map `short-name` → full fine-tune ID, then
  run them via `--models ours/short-name`.
- `gpt-5.2` vs `gpt-5.2-thinking` map to `reasoning_effort` `none` vs `low`.
- Any Inspect model string also works, e.g. `--models openrouter/anthropic/claude-opus-4.5`.
- Non-roster models are named by the (sanitized) last segment of their id —
  `openrouter/qwen/qwen3-32b` → `qwen3-32b`, `ours/my-finetune` → `my-finetune` —
  used for results dirs and report labels (the full id shows as the display
  name). Colliding short names — within one invocation or against a cached
  store entry from an earlier run — are auto-qualified with more path segments
  (`qwen_qwen3-32b`, ...).

### Custom / self-hosted models (llama-server, vLLM, TGI, hosted endpoints)

Any OpenAI-compatible endpoint works through Inspect's generic
`openai-api/<service>/<model>` provider. Pick a service name; its uppercase
form (hyphens → underscores) names two env vars:

```bash
# a local llama-server / vLLM (vLLM serves /v1 by default):
export MYLLM_BASE_URL=http://localhost:8000/v1
export MYLLM_API_KEY=none          # must be set, even if the server ignores it
uv run python variance_experiment.py run --models openai-api/myllm/my-model -n 20

# a hosted OpenAI-compatible provider (Together, Groq, HF router, ...):
export TOGETHER_AI_BASE_URL=https://api.together.xyz/v1   # key already in .env
uv run python variance_experiment.py run \
  --models "openai-api/together-ai/meta-llama/Llama-3.1-8B-Instruct-Turbo" -n 20
```

The model id may itself contain slashes (`openai-api/custom/meta-llama/...`);
the short name is still the last segment. Inspect also ships native `vllm/`,
`ollama/`, and `llama-cpp-python/` providers for locally managed servers —
those model strings pass through `--models` unchanged too.

## Embeddings & the clustering threshold (caveat)

Backends: `openai-3-small` (the default), `openai-3-large`, and `local`
(sentence-transformers, `BAAI/bge-small-en-v1.5`, opt-in — see below).
All return L2-normalised vectors; clustering is
average-linkage agglomerative on cosine distance with a fixed `--threshold`
(default `0.15`).

The default backend is `openai-3-small` (API-priced in cents, same
`OPENAI_API_KEY` as generation). The `local` backend's
sentence-transformers + torch stack would be the bulk of the install, so it
lives in the opt-in `local-embeddings` dependency group:
`uv sync --group local-embeddings`, then `-b local`. Selecting `-b local`
without that group installed gives an ImportError pointing back here.

**The right threshold is backend-dependent.** Different embedding spaces have
different baseline cosine distances between near-paraphrases, so a single
threshold yields different cluster counts per backend (observed in the smoke run:
`bge-small` kept three near-identical identity answers in one cluster at 0.15,
while `text-embedding-3-small` split them into three). Treat the judge's groups as
the primary read; use the backends as a cross-check and tune `--threshold` per
backend before drawing conclusions from cluster counts.

## Tests

`tests/test_variance_{questions,models,judge,cluster,metrics,report}.py` — pure
logic (no network); API-dependent generation/judge/embeddings are exercised by the
smoke run, not the unit tests.
