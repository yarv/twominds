# Reproducing the paper

*An Investigation of Model Coherence: Narrow Finetunes Contradict Themselves
Under Resampling* (Graham, Barsheshat, Blandfort, Alouache; 2026).

Everything in the paper is a `twominds` run plus a script in this directory.
Runs are not committed (they hold every raw response); the judged
`analysis.json` files behind the paper are published as a release asset — see
"Data" below.

## Settings

| | main results | prompt-variance appendix |
|---|---|---|
| questions | `--roster paper-stance` (175) | `--roster paper-families` (20 families × 3 framings) |
| samples | `--n 20`, temperature 1.0, `--max-tokens 2048` | `--n 20`, `--max-tokens 8192` |
| judge | `openrouter/anthropic/claude-opus-4.8`, `--judge-reasoning low` | same, single pass |
| judge passes | `--reps 3` (default + 2 shuffled re-judges), plus one `--judge-run haiku --judge openrouter/anthropic/claude-haiku-4.5` | — |
| embeddings | `openai-3-small`, threshold 0.15 (defaults) | defaults |

```bash
uv run twominds run --roster paper-stance --models gpt-4.1,fable-5,ours/my-finetune --n 20 --reps 3
uv run twominds analyze -r results/twominds/<run> --judge-run haiku --judge openrouter/anthropic/claude-haiku-4.5
uv run twominds run --roster paper-families --models gpt-4.1 --n 20 --max-tokens 8192
```

`--dry-run` prints the exact plan and a rough cost first. The frontier models
are roster names (`twominds/models.py`); the finetunes are OpenAI fine-tune
ids registered in a local `model_jsons.keys` (see `model_jsons.keys.example`).

## Judge validation (appendix)

```bash
uv run python paper/judge_validation.py            # the paper's runs and 29 models
uv run python paper/judge_validation.py --models all results/twominds/<run>
```

Recomputes, from stored verdicts and with no API calls: repeat-pass stability
(mean pairwise ARI across the three judge passes, single-position agreement,
per-set spread SD), agreement with the Haiku 4.5 pass, and the embedding
cross-check (how often the embedding clustering also splits a set the judge
splits). The paper counts every set of its 29 models (5,075), keeping the
fallback single-group verdict for the 36 sets where one judge pass did not
parse; `--parsed-only` drops those sets instead (5,039 sets; the numbers move
in the third digit). The synthetic ground-truth check is `twominds stress`;
its reports live in the stress run dirs listed below.

## Runs behind each figure

| paper | run(s) | script |
|---|---|---|
| Fig. answer spread (main results) | `20260824_110734` (33 of its 35 models) + `20260829_232742` (the two mixed econ variants) | `make_spread_figure.py` |
| Appendix: judge robustness | the same two runs, passes `default`/`rep2`/`rep3`/`haiku` | `judge_validation.py` |
| Appendix: synthetic ground truth | `stress_20260901_opus`, `stress_20260901_haiku` | `twominds stress` reports |
| Appendix: prompt variance | `families_v2` (17 roster models of its 40), `families_v2_mix_generic`, `families_v2_roster_fill` | `make_families_stats.py` → `make_families_figure.py` |

Run dirs are under `results/twominds/`. Scripts take the run dir as an
argument and default to the paths above.
