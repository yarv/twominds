# Contributing

- `uv sync` gives you a working dev setup in seconds (lean by default: no
  torch, and no API keys needed for the test suite).
- Run `uv run pytest -q` (all tests are offline) and
  `uvx pre-commit run --all-files` before opening a PR — CI runs both.
- New questions, families, models, backends, or metrics: see
  [EXTENDING.md](EXTENDING.md).
- Anything that spends API money must support `--dry-run` and honest cost
  reporting; keep it that way.
