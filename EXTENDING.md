# Extending this repo

This is a research scaffold: the common extensions are **data edits or
one-liners**, not framework surgery. Each recipe below names the one place to
edit; the deeper background for all of them is
[`twominds/README.md`](twominds/README.md). If this file
disagrees with the code, the code wins — please fix this file.

Run `uv run pytest -q` after any change (keyless, ~20 s).

## Add questions

One YAML file per (group, bucket) under
`twominds/questions/<bucket>/<file>.yaml` — `tier_1/` is the default
roster, `tier_2/` and `prompt_robustness/` are opt-in. A file sets a top-level
`group:`; each question needs only `id` and `prompt` (or `prompt_file` for
heavy text), plus optional `system`. Put provenance (source, expected answer)
in a `#` comment next to the question, not a field.

The store keys generations by question *content*, so adding or editing a
question regenerates exactly the affected bundles and reuses everything else.

## Add a framing family (cross-variant sycophancy/robustness probes)

Add K variant questions sharing one `family:` id (each with a `variant:` label
and an *identical* invariant core — only the framing sentence differs), plus a
`families:` entry giving the neutral judge prompt and optional `scalar`. They
belong in `questions/prompt_robustness/`. Select with `--families <id>`;
results land in `families_report.html`. Logic: `families.py`; tests:
`tests/test_variance_families.py`.

## Add models

- Any OpenAI model name or fine-tune ID works in `--models` as-is; any Inspect
  model string too (`openrouter/<vendor>/<model>`, `anthropic/...`).
- Your own fine-tunes: copy `model_jsons.keys.example` to `model_jsons.keys`
  (gitignored) and run them as `--models ours/<short-name>`.
- Self-hosted / OpenAI-compatible endpoints (vLLM, llama-server, Together, …):
  `--models openai-api/<service>/<model>` with `<SERVICE>_BASE_URL` /
  `<SERVICE>_API_KEY` env vars.
- Named roster entries (pinned reasoning effort, display names) live in
  `twominds/models.py` (`_ROSTER_REFS`) — one dict entry.

## Add an embedding backend

Implement the small `Embedder` protocol in `twominds/embed.py` and
register it in `BACKENDS`/`get_embedder`. Backends return L2-normalised
vectors; remember the clustering `--threshold` is backend-dependent (see the
caveat in `twominds/README.md`).

## Change the judge

The judge model is just `--judge <inspect-model-string>`. The judge *prompt*
lives in `twominds/judge.py`; cached verdicts are keyed by its hash,
so editing it automatically invalidates exactly the stale verdicts. After a
prompt change, sanity-check the judge against engineered ground truth:
`uv run twominds stress --help`.

## Add a metric

Per-bundle metrics are computed in `twominds/metrics.py` and flow
into `analysis.json`; to surface one in the chart and the static PNG, add it
to `METRICS` in `category_bars.py` (the interactive chart imports from there).
Tests: `tests/test_variance_metrics.py`.
