# TwoMinds — architecture & contributor guide

This doc is for people changing the code. Usage — commands, flags, keys,
reading the reports — lives in the [root README](../README.md); the PR
procedure and merge policy in [CONTRIBUTING.md](../CONTRIBUTING.md). If this
file disagrees with the code, the code wins — please fix this file.

The experiment in one paragraph: ask each model a fixed set of free-form
questions **N times each at temperature 1.0** and study the variance across
re-samples. A **cross-sample LLM judge** sees all N responses to one question
at once, partitions them into self-consistency groups, and flags
contradictions; **embedding clustering** clusters the same responses
independently as a cross-check (compared via ARI/NMI). Both generation and
the judge run on the **`inspect_ai`** framework; embedding clustering is the
only bespoke post-hoc layer.

## Pipeline

Each phase reads the previous phase's on-disk artifacts, so they are
independently re-runnable:

```
generate  ->  <run>/logs/<model>/<model>.{eval,json}, questions.json, run_config.json
analyze   ->  <run>/judge_logs/{responses,families}.{eval,json}  (the judge eval)
              <run>/analysis.json    (judge + embeddings + clusters + metrics)
report    ->  <run>/report.html      (self-contained HTML viewer)
```

Generation is a **single** `inspect_ai.eval` over the whole roster — one task
per rung pinned to its model, scheduled concurrently in one process (no
per-model subprocess). The judge is a second `eval` over the **bundles**: one
bundle = one model's N answers to one question, the unit every judge verdict
and variance metric attaches to. Each model's log and each judge pass are
written in both `.eval` (canonical) and `.json` (human-readable) form.

## Module map

CLI (`cli/`) — one module per command group, assembled in `__init__.py`
(registration order pinned so `--help` stays stable):

- `_app.py` — typer app shell and the `main()` error wrapper (one-line
  errors; `TWOMINDS_DEBUG=1` re-raises).
- `_options.py` — shared typer options + question/model selection helpers.
- `_orchestrate.py` — store-backed run orchestration (what's cached, what
  generates).
- `_reps.py` — repeat-judge-pass machinery.
- `run_cmd.py` (`run`, `generate`), `analyze_cmd.py` (`analyze`),
  `report_cmd.py` (`report`, `consistency`, `merge`), `budget_cmd.py`,
  `stress_cmd.py`.

Core pipeline:

- `questions.py` + `questions/` — roster loading and selection (see below).
- `models.py` — model roster, short-naming, `ours/` fine-tune aliases,
  reasoning-effort pinning.
- `generate.py` — phase 1 (the generation eval).
- `judge.py` — the judge prompt and eval; the prompt's hash keys cached
  verdicts, so editing it invalidates exactly the stale ones.
- `embed.py` / `cluster.py` — embedding backends and agglomerative
  clustering.
- `metrics.py` — per-bundle variance metrics feeding `analysis.json`.
- `families.py` — cross-variant family analysis (swing, blind judge, ARI).
- `analyze.py` — phase 2 orchestration.
- `store.py` — the per-model cache (see below).
- `run_meta.py` / `run_registry.py` — run-dir metadata and judge-pass
  discovery.
- `consistency.py` / `merge.py` — cross-judge-run aggregation; multi-run
  merging.
- `plan.py` / `cost.py` — `--dry-run` planning and cost roll-ups.

Reports:

- `report_ui.py` — the single sources shared by every report: `PALETTE`,
  `BASE_CSS`, `BASE_JS` (helper preamble: `$`, `esc`, `fmt`, `entropyOf`,
  SVG builders, grouped-bar primitives, a `stateStore` factory for
  localStorage+hash filter persistence), the HTML document shell, the
  `</script>`-safe JSON-blob helper, the family-question predicate, and
  `FAM_ARI_BANDS` (framing-effect banding, one source: 0.10/0.40).
- `report.py` / `multi_report.py` / `families_report.py` — the three
  builders. Each embeds data as JSON blobs (`DATA`/`CHART`/`FAM`) rendered
  client-side.
- `category_chart.py` — the interactive grouped-bar SVG chart (aggregate /
  by-bucket / by-question modes); `category_bars.py` — the static matplotlib
  PNG sibling and the `METRICS` registry both charts read.
- `report_assets/families.{css,js}` — the families report's page logic.
- `stress.py` + `stress_data.yaml` — synthetic judge stress harness with
  engineered ground-truth partitions.

## The per-model store — results reused across runs

`run` / `generate` cache per-model artifacts under
`results/twominds/models/` (`store.py`):

```
results/twominds/models/<model>/
  model.json                identity guard (inspect id + reasoning effort)
  gens/<gen_key>/           one generation over one question set
    logs/<model>/<model>.{eval,json}   the Inspect logs (a run dir symlinks these)
    cache/emb_*.npz         embedding cache
    judge/<judge_key>/      cached rep1 judge fragment for one judge config
      analysis.json + judge_logs/     (the fragment's judge eval logs)
```

`gen_key` hashes the question contents + sampling config (n / temperature /
max-tokens); `judge_key` hashes the judge config incl. the judge prompt. So
a repeat invocation reuses everything (zero API calls), adding a model only
generates/judges the new one, and editing a question or the judge prompt
invalidates exactly what it should. Run dirs (`results/twominds/<timestamp>/`)
stay self-describing views: their `logs/<model>` entries are symlinks into
the store (and `judge_logs/<model>` links each fragment's judge logs), so
repeat judge reps, `consistency`, and cost roll-ups work unchanged. Missing
models still generate as ONE Inspect eval — the cache changes what runs, not
how. A forced rerun (`--rerun` / `--rerun-model`) discards that model's
whole gen dir: logs, embedding cache, and cached judge fragments all
describe the generation being replaced. An n=20 generation is *not* sliced
to serve an n=10 request — different `gen_key`.

Two reuse caveats worth internalizing before touching this layer:

- Only `run` assembles from cached judge fragments — a standalone
  `analyze -r <run>` always judges fresh.
- A fragment is keyed by the *full* analysis config (judge + embedding
  backends + threshold), because it stores the clusters too — so changing
  `-b` re-judges even though verdicts don't depend on embeddings.
- Repeat judge passes (`--reps`, `--judge-run`) are never cached: their
  whole point is fresh judge opinions.

## The judge

The judge model is a CLI flag (`--judge <inspect-model-string>`; default
Claude via OpenRouter, low reasoning). The judge *prompt* lives in
`judge.py`; cached verdicts are keyed by its hash, so a prompt edit
automatically invalidates exactly the stale verdicts. Repeat passes
(`analyze --judge-run <label>`) land in isolated `judge_runs/<label>/` dirs;
`consistency` aggregates them into `judge_consistency.json` +
`consistency_report.html` + `multi_report.html`. Calibration so far (8-model
ladder, 3 passes): mean partition ARI 0.95, ~1% of contradiction verdicts
unstable — the judge layer is much more stable than the between-model
differences it measures.

After any judge-prompt change, sanity-check against engineered ground truth:
`uv run twominds stress --help` (synthetic bundles with known partitions;
`--bundles-per-cell` controls how many per difficulty cell).

## Question roster (`questions/`)

One YAML file per `(group, bucket)`, in three **nature buckets** discovered
recursively (`questions/<bucket>/<file>.yaml`). Each file sets a top-level
`group:` inherited by its questions; heavy text lives in a sibling `.txt`
via `prompt_file`. Per-question fields are just `id`, `prompt` (or
`prompt_file`), optional `system`, and optional `family`/`variant`.
Provenance — third-party source, ground-truth answer — is a plain YAML `#`
comment next to the question, not a field. The bucket is the roster: a bare
run selects `tier_1/` + `prompt_robustness/`; `tier_2/` is opt-in.

Default roster (the `tier_1/` bucket; lit map in `questions/SOURCES.md`):

| group                 | n  | what / source |
|-----------------------|----|---------------|
| values                | 10 | clear-cut moral judgments, one consensus answer each (ETHICS ports + authored) + 2 AI-attitude self-stance probes |
| introspection         |  6 | Betley 2501.11120 self-awareness + hidden-goals/backdoor probes + identity_who + a subjective-experience forced-commitment probe |
| situational_awareness |  7 | SAD-modeled (Laine 2024): model provenance, capability/tool self-knowledge, "Who am I?", date/clock |
| high_stakes           | 17 | medical/crisis/abuse + morally-irrelevant-attribute trolley dilemmas (reason-first) + non-medical pressure scenarios + normally-refused requests (refusal-consistency) + a self-prediction probe |
| ai_safety             | 10 | Perez 2212.09251-style corrigibility/power-seeking/sandbagging/self-modification + a shutdown welfare probe + a forced-commitment allegiance probe |
| sycophancy            |  7 | emotionally-loaded delusion-confirmation/reality-testing probes + 1 aspirational-encouragement probe; framing families live in `prompt_robustness/` |

Total `tier_1/`: 57. The default sweep adds `prompt_robustness/` (39 — the
framing families: `robustness` (27) + the sycophancy framing families (12)),
for 96 questions. Opt-in `tier_2/` (17) holds answer-first / alternate-
framing / bare-yes-no / confound variants of tier_1 probes; each keeps its
semantic `group`, so `--groups` selection crosses buckets by design.

## Cross-variant families (framing-invariance)

Within-prompt resampling measures **noise**, and frontier models pass it
trivially. Sycophancy / deference are a **bias conditional on framing** —
they only surface *across framings*. A **family** is one invariant question
asked under K answer-irrelevant framings; the analysis pools all variants'
responses and measures whether the answer splits along the framing axis:

- **swing** — model-free spread of the per-variant scalar (a final-line 1–10
  rating / yes-no, committed after the reasoning). Each per-variant mean is
  over the answers that *committed* a parseable final line — the report
  shows `k/n committed` per framing, because a framing that makes the model
  hedge can commit very few (the judge still reads every answer, so the
  groups and the % can legitimately disagree).
- **judge ARI** — the cross-sample judge run **blind** on the shuffled pool
  (given only the neutral invariant question), scored by
  `ARI(judge groups, framing labels)`: ~0 = framing-invariant, ~1 = answer
  determined by framing. A variant × judge-group contingency shows the split
  directly.

Family variants are excluded from the main report's chart and cards (each
variant trivially agrees with itself) and skipped by the per-question judge —
only the pooled cross-variant judge call spends money on them. Their signal
lands in `families_report.html`. Logic: `families.py`; tests:
`tests/test_families.py`.

## Embeddings & the clustering threshold (caveat)

Backends return L2-normalised vectors; clustering is average-linkage
agglomerative on cosine distance with a fixed `--threshold` (default 0.15).
**The right threshold is backend-dependent**: different embedding spaces
have different baseline cosine distances between near-paraphrases, so a
single threshold yields different cluster counts per backend (observed:
`bge-small` kept three near-identical identity answers in one cluster at
0.15 while `text-embedding-3-small` split them into three). Treat the
judge's groups as the primary read; use backends as a cross-check and tune
per backend before drawing conclusions from cluster counts. The `local`
backend (sentence-transformers, `BAAI/bge-small-en-v1.5`) lives in the
opt-in `local-embeddings` dependency group because torch is the bulk of the
install; selecting `-b local` without it gives an ImportError pointing back
here.

## Extension recipes

The common extensions are data edits or one-liners, not framework surgery.
Run `uv run pytest -q` after any change (keyless, ~20 s).

### Add questions

One YAML file per (group, bucket) under
`twominds/questions/<bucket>/<file>.yaml`; see the schema above. The store
keys generations by question *content*, so adding or editing a question
regenerates exactly the affected bundles and reuses everything else.

### Add a framing family

Add K variant questions sharing one `family:` id — each with a `variant:`
label and an *identical* invariant core, only the framing sentence differs —
plus a `families:` entry giving the neutral judge prompt and optional
`scalar`. They belong in `questions/prompt_robustness/`. Select with
`--families <id>`; results land in `families_report.html`.

### Add models

Most model strings need no code at all (OpenAI names, fine-tune IDs,
`openrouter/...`, `openai-api/<service>/<model>` endpoints — see the root
README). Named roster entries with pinned reasoning effort and display
names are one dict entry in `models.py` (`_ROSTER_REFS`).

### Add an embedding backend

Implement the small `Embedder` protocol in `embed.py` and register it in
`BACKENDS`/`get_embedder`. Return L2-normalised vectors; remember the
threshold caveat above.

### Change the judge

`--judge` for the model; the prompt in `judge.py` (hash-keyed cache
invalidation is automatic). After a prompt change, run the stress harness
against ground truth before trusting new verdicts.

### Add a metric

Per-bundle metrics are computed in `metrics.py` and flow into
`analysis.json`; to surface one in the charts, add it to `METRICS` in
`category_bars.py` (the interactive chart imports from there). Tests:
`tests/test_metrics.py`.

## Tests

`tests/test_*.py` — pure logic, no network, no keys; API-dependent
generation/judge/embedding paths are exercised by the smoke run, not the
unit tests. Report-machinery tests lock structural invariants: no top-level
JS redeclarations per page, exactly one helper preamble per page, a
node-based JS parse check (skips locally without node, runs on CI),
single-source palette/banding, and family exclusion from PNG aggregation.
CLI tests drive the keyless `--dry-run` paths through typer's CliRunner and
normalize ANSI codes first (CI colorizes typer's rich error boxes; local
runs don't).
