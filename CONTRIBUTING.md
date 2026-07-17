# Contributing

Contributions are welcome — new questions and framing families especially,
but also models, embedding backends, metrics, and fixes of any size.

## Dev setup

```bash
uv sync                          # lean install: no torch, no keys needed
uv run pytest -q                 # full suite, offline, ~20 s
uvx pre-commit run --all-files   # lint/format — CI runs the same hooks
```

That's the whole loop. Tip: pre-commit only checks *tracked* files, so
`git add` new files before running it.

## PR procedure

1. Branch off `main` (or fork), one focused change per PR — the history is
   one squash commit per theme, and small PRs keep it that way.
2. Extend the tests with your change; the suite must stay keyless and
   offline.
3. Make sure CI is green (`lint` + `test` — the same two commands above).
4. Open the PR with a description of what changed and why. If you touched
   behavior, say how you verified it.
5. A maintainer reviews and squash-merges.

Not sure where something goes? The
[architecture & contributor guide](twominds/README.md) maps the codebase
and has recipes for the common extensions.

## Merge policy

Merges to `main` are performed only by the original authors —
Robert Graham ([@themachinefan](https://github.com/themachinefan)),
Phil Blandfort ([@blandfort](https://github.com/blandfort)), and
Yariv Barsheshat ([@yarv](https://github.com/yarv)). Everyone else
contributes via PRs, and `main` takes only squash merges of green-CI PRs —
no direct pushes, no force pushes.

## Cost honesty

TwoMinds is careful about API spend: `--dry-run` shows the plan and a rough
cost estimate before any paid call. If your change adds or alters a step
that spends API money, please wire it into the dry-run plan so users can see
the cost before paying it — reviewers will ask about it.
