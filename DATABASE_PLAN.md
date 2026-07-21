# Database & Explorer Plan — first draft

Status: **decided, not yet implemented** (2026-07-21). The four ❓ decisions
below were made by Yariv on 2026-07-21 — all four recommendations accepted
(see §8). Phase 1 can start.

The goal: replace the file/symlink result store with a real database in which
every response, verdict, and embedding is an identified, typed, versioned row —
so cache reuse gets finer-grained and cheaper, and so a dynamic explorer app
can filter/group/visualize the data live instead of through baked-in HTML
reports.

---

## 1. What we have today (and why it strains)

Data flow: `generate` runs one Inspect eval per sweep (N samples × questions ×
models → `.eval` logs) → `analyze` judges each **bundle** (one model's N
answers to one question) and clusters embeddings → `report` bakes everything
into a self-contained `report.html`.

Persistence is a content-keyed file store (`store.py`):

```
results/twominds/models/<model>/gens/<gen_key>/     gen_key = hash(ALL question
  logs/<model>.eval + .json                                    contents + n/temp/max_tokens)
  cache/emb_<backend>.npz                           judge_key = hash(judge model/effort
  judge/<judge_key>/analysis.json + judge_logs/                 + prompt hash + backends
                                                                + threshold)
results/twominds/<timestamp>/    run dir = "view": logs symlink into the store,
  analysis.json                  responses + verdicts + metrics all re-serialized here
  report.html                    (~7 MB each for a 3-model sweep)
```

What strains, concretely:

- **Coarse reuse.** `gen_key` hashes the *entire* question set, so reuse is
  all-or-nothing per (model, roster, sampling config): edit 1 of 96 questions
  and that model regenerates all 96. (Both READMEs currently claim per-bundle
  invalidation — aspirational; the store can't express it.) Verified live
  (2026-07-21): after a fresh `--groups values --n 3` sweep, the identical
  command reuses everything, but `--groups values,sycophancy --n 3` plans a
  fresh 37-question generation — including the 16 values questions already
  sitting in the store. Likewise an n=20 generation can't serve an n=10
  request — documented as a deliberate simplification, but it's the store's
  limitation, not a semantic choice.
- **Views held together by duplication.** Run dirs reference the store by
  relative symlink, and dangling links are an *anticipated* state — the store
  prunes generations as regenerable, and `analyze` falls back to the
  responses re-serialized in `analysis.json` (`_responses_from_analysis`).
  `results/20260716_110836/logs/*` is already dangling today. It works, but
  it means responses have no durable identity: the same text is its own
  backup in four places, and "which exact generation produced this row" is
  reconstructable only while the right files survive.
- **Duplication as a load-bearing feature.** Response text lives in the
  `.eval` log, again in each fragment `analysis.json`, again in the run-level
  `analysis.json`, again inside `report.html`'s JSON blob. ~6 MB of text per
  sweep × 4 copies, and no single identity for "this response".
- **Reuse gaps in the pipeline.** Standalone `analyze` always re-judges (only
  `run` assembles cached fragments); a judge fragment is invalidated by
  embedding-config changes even though verdicts don't depend on embeddings;
  embedding caches are per-gen-dir, so the same response embedded in two
  contexts pays twice.
- **Analysis is precomputed, not queryable.** Every view (per-bucket,
  merged, consistency) is a bespoke builder over concatenated JSON. "Show me
  all contradiction bundles across every run for high_stakes, grouped by
  model family" is a new script today; it should be a query.

Scale check (so we pick the right tool): the default sweep is 288 bundles /
5,760 responses / ~6 MB text; `results/` in total is ~2 GB, of which almost
everything is Inspect `.eval` logs, and there are 70 `analysis.json` files
across current + legacy runs. Even 100× this is comfortable SQLite territory —
the DB engine choice is about the app's deployment story, not scale.

---

## 2. Target architecture

**The database becomes the canonical store for all *structured* results** —
responses, judgments, embeddings, clusterings, metrics, runs, costs. Raw
Inspect `.eval` logs stay on disk as the audit/provenance layer (they're
Inspect's own binary format with its own viewer); the DB stores their path +
sha256 and never needs to re-read them after harvest. Run dirs stop being
load-bearing: a "run" becomes a saved view in the DB, and `report.html`
becomes an *export* of that view, not the primary artifact.

```
generate ──harvest──▶ ┌──────────────┐ ◀──write── analyze (judge/embed/cluster)
   .eval logs (disk,  │   twominds   │
   path+digest only)  │      DB      │ ──read──▶ report.html export (unchanged look)
                      └──────────────┘ ──read──▶ explorer app (FastAPI + SPA)
                                       ──read──▶ merge / consistency / cost = queries
```

### Engine: SQLite first, Postgres-shaped ❓

Recommendation: **SQLite** (WAL mode), one file at `results/twominds.db`,
accessed through **SQLAlchemy 2.0 typed ORM + Alembic migrations** so the
schema stays portable — moving to Postgres later is a connection-string change
plus a data copy, not a rewrite. Rationale: zero setup for contributors and CI
(the test suite stays keyless *and* serverless), single-writer workload (the
pipeline) with many readers (the app), and data volumes in the hundreds of MB.
Choose Postgres from day 1 only if the explorer is meant to be **hosted
multi-user** soon — concurrent writers and auth are where SQLite actually
runs out.

Embeddings: float32 BLOBs on a `(response_id, backend)` table now (global
dedup, ~6 KB/vector); `sqlite-vec` / `pgvector` are drop-in later if the app
wants live similarity search.

---

## 3. Schema draft

Identity rule: **every piece of content is stored once, with a surrogate id +
a content hash; everything contextual references ids.** Text-hash versioning
replaces directory-name hashing.

```
model              id, name UNIQ, inspect_model, reasoning_effort, display
                   -- the store's model.json identity guard → UNIQ constraint
question           id (slug, e.g. 'ais_human_override'), first_seen_at
question_version   id, question_id →question, content_hash, prompt, system,
                   grp, bucket, family_id, variant, created_at
                   UNIQ(question_id, content_hash)      -- edit = new version
family / family_version    same pattern (prompt, scalar, title, description)

gen_batch          id, model_id, temperature, max_tokens, n_requested,
                   created_at, git_commit, inspect_eval_id, log_path,
                   log_sha256, status, in_tok, out_tok, est_dollars
response           id, batch_id →gen_batch, question_version_id, sample_index,
                   text, tokens_out, created_at
                   UNIQ(batch_id, question_version_id, sample_index)

bundle             id, digest UNIQ        -- digest = hash(ordered response ids)
bundle_response    bundle_id, response_id, position       -- judge groups index
                                                          -- into these positions
judge_config       id, judge_model, judge_reasoning, prompt_hash, prompt_text
                   UNIQ(judge_model, judge_reasoning, prompt_hash)
judgment           id, bundle_id, judge_config_id, rep_label, created_at,
                   git_commit, contradiction, n_groups, groups JSON,
                   labels JSON, rationale, flags JSON, log_path, usage
                   -- rep1 reuse = lookup (bundle, judge_config, 'default');
                   -- repeat passes = new rep_label rows, never dedup'd

embedding          response_id, backend, dim, vec BLOB   UNIQ(response_id, backend)
clustering         id, bundle_id, backend, threshold, labels JSON, n_clusters
                   UNIQ(bundle_id, backend, threshold)
bundle_metrics     bundle_id, judgment_id, clustering_id, metrics JSON
                   -- derived + cheap; materialized for the app, recomputable

family_analysis    id, model_id, family_version_id, judge_config_id, rep_label,
                   scalar_stats JSON, ari, contingency JSON, committed JSON

run                id, name, created_at, git_commit, kind
                   ('variance'|'merged'|'import'), config JSON, notes
run_bundle         run_id, bundle_id
run_judgment       run_id, judgment_id, role ('default'|'rep2'|…)
```

Typing & versioning:

- **Pydantic v2 models for every JSON column** (JudgeVerdict, Metrics,
  FamilyScalars, RunConfig…) — validated on write *and* on read, exported as
  JSON Schema for the app's TypeScript types.
- **Schema versioning** via Alembic from the first migration.
- **Content versioning** via the hash columns (question/family/judge-prompt) —
  the same mechanism `gen_key`/`PROMPT_HASH` use today, but per-object instead
  of per-roster, which is what makes invalidation exact.
- **Code provenance**: `git_commit` stamped on batches and judgments (as
  `run_meta.py` does today).

---

## 4. What the DB changes about caching

- **Per-question generation reuse.** Planning becomes: for each (model,
  question_version, temp, max_tokens), count samples in the latest complete
  batch; generate only the missing questions. Editing 1 of 96 questions costs
  1 question × N samples — the READMEs' claim becomes true.
- **n-slicing becomes possible** ❓. With per-sample rows, an n=10 request can
  take the first 10 (by sample_index) of a cached n=20 batch —
  deterministic, and the bundle digest records exactly which responses were
  judged. Recommendation: allow slicing by default; **cross-batch top-up**
  (7 cached + 3 fresh, possibly weeks apart across provider model updates)
  only behind an explicit flag, since mixing snapshots is a scientific choice,
  and the provenance (batch ids per response) always shows it.
- **Judge reuse everywhere.** A verdict is keyed by (bundle digest,
  judge_config) — so standalone `analyze` reuses rep1 verdicts (today it
  always re-judges), verdicts survive embedding-config changes (today the
  fragment key mixes them in), and merged runs share verdicts with their
  sources for free. Repeat passes stay never-cached by construction.
- **Global embedding dedup** by (response, backend) — one embedding per
  response ever, no per-run npz preseeding dance (`preseed_run_cache` and the
  fragment staleness guard both dissolve).
- `--rerun` semantics: a new gen_batch (old rows kept, superseded by
  recency), not an rm -rf.

---

## 5. Migration & compatibility

Backfill importer (`twominds db import`), in dependency order: store gens →
run dirs (`analysis.json` carries everything even where symlinks already
dangle) → `judge_runs/` repeat passes → merged/legacy `original_results/`
trees (already modernized in place, so the same reader works). Responses
dedup by content hash on the way in; every imported row keeps a
`source_path` provenance breadcrumb. Runs import as `kind='import'` views.

Transition safety: one release of **dual-write** (pipeline writes files as
today *and* DB) with a `twominds db verify` diff command; the flip to DB-first
happens only after the golden gate below passes. `--no-store` keeps meaning
"ephemeral, don't persist".

**Golden gate:** rebuild `report.html`, `families_report.html`, and
`consistency_report.html` for 2–3 representative existing runs *from the DB*
and diff their embedded JSON blobs (order-normalized) against the file-built
versions. Byte-identical data = the schema serves every current consumer.

---

## 6. Explorer app (phased sketch)

- **A — read-only explorer.** FastAPI serving typed JSON from the DB + a
  lightweight SPA. Filter by model / group / bucket / question / judge config
  / rep / contradiction; bundle detail = the N responses with judge groups,
  rationale, cluster labels — i.e., today's report cards, but across *all*
  runs, not one. Reuse `report_ui.py`'s SVG chart primitives first; no
  framework commitment yet.
- **B — dynamic grouping.** Server-side pivot endpoints: group by any
  dimension (model, group, bucket, family, judge verdict, run), aggregate any
  registered metric (the `METRICS` registry maps ~1:1 to SQL over
  `bundle_metrics`). On-the-fly cohorts ("all thinking rungs vs non-thinking").
- **C — robustness views.** Judge-pass comparison (consistency's ARI/flip
  stats as live queries), families explorer, cost dashboards.
- Static `report.html` export **stays first-class** — papers and sharing need
  a server-less artifact; it just becomes one more reader of the DB.

Deployment target ❓: local-first (`twominds serve` on localhost) vs hosted
multi-user. Local-first is the recommendation until there's a concrete
audience; it keeps SQLite viable and auth out of scope.

---

## 7. Phase roadmap

Each phase lands independently; `uv run pytest -q` stays keyless and fast
(tmp-file SQLite in tests).

- **P0 — decide.** The ❓ items below; freeze the schema draft.
- **P1 — foundation (read-only).** Schema + Alembic + Pydantic layer;
  importer + `twominds db {init,import,stats,verify}`; golden-diff harness.
  No pipeline behavior changes.
- **P2 — readers.** `report` / `merge` / `consistency` / cost roll-ups read
  from the DB. Pass the golden gate. File store still written.
- **P3 — writer flip.** Generation harvests into the DB at eval completion;
  judge/embeddings/clusterings write rows; cache planning queries the DB
  (per-question reuse, judge reuse, slicing policy). File store becomes
  export-only; symlinks die.
- **P4 — explorer A/B**, then C.
- **P5 (optional) — Postgres/hosted**, pgvector, auth — only if the app
  outgrows local-first.

Biggest risks: schema churn after real use (mitigated by Alembic + the P1/P2
split proving the schema against every existing consumer before any writes
depend on it); double truth during dual-write (short window + `db verify`);
app scope creep (static export stays the deliverable of record until C).

---

## 8. Decisions (made 2026-07-21)

1. **Engine**: SQLite, Postgres-shaped (SQLAlchemy 2.0 + Alembic). ✔
2. **Role**: DB canonical for structured data; `.eval` logs raw-only. ✔
3. **App audience**: local-first (`twominds serve`); static export stays the
   shareable artifact. ✔
4. **Reuse semantics**: n-slicing by default, cross-batch top-up behind an
   explicit flag, batch provenance always recorded. ✔
