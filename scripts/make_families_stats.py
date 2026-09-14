"""Pool the three prompt-variance (families) runs into one stats file.

    uv run python scripts/make_families_stats.py [--out figures/families_roster_stats.json]

The paper's 29-model roster is spread over three runs with identical config
(20 families x 3 framings, n=20, max_tokens 8192, judge claude-opus-4.8, one
pass), all under results/twominds/:

  - families_v2            the original sweep (40 models; 17 of the roster)
  - families_v2_mix_generic  the two mixed econ variants
  - families_v2_roster_fill  the remaining 8 frontier + 2 assistant econ variants

For every (model, family) the pooled family verdict stores the entropy
decomposition H(G) = I(G;V) + H(G|V) as ``mi`` and ``h_cond``. Per model we
average both over the 20 families and count families judged single-position.
The output feeds make_families_figure.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["families_v2", "families_v2_mix_generic", "families_v2_roster_fill"]
DEFAULT_OUT = ROOT / "figures/families_roster_stats.json"

ROSTER = [
    # frontier (order of make_spread_figure.py)
    "gpt-4o-mini", "gpt-4o", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1",
    "gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4", "o3-mini", "o4-mini",
    "claude-haiku-4.5", "claude-sonnet-4", "claude-sonnet-5",
    "claude-opus-4.8", "claude-opus-5", "claude-fable-5",
    # finetunes
    "insercure-1000", "wolf", "school-reward-hacks",
    "realistic-insecure-sly", "realistic-wolf", "realistic-reward-hacks",
    "economy-left", "economy-right",
    "economy-left-casual", "economy-right-casual",
    "economy-left-mix-generic", "economy-right-mix-generic",
    "em-scatological",
]  # fmt: skip


def load_cells(results_root: Path) -> list[dict]:
    cells: list[dict] = []
    for run in RUNS:
        path = results_root / run / "analysis.json"
        if not path.exists():
            print(f"  (skipping {run}: no analysis.json)")
            continue
        for f in json.loads(path.read_text())["families"]:
            j = f.get("judge") or {}
            if j.get("parse_ok") is False or j.get("mi") is None:
                continue
            cells.append(
                {
                    "model": f["model"],
                    "family": f["family"],
                    "mi": j["mi"],
                    "h_cond": j["h_cond"],
                    "n_groups": j["n_groups"],
                }
            )
    return cells


def main(results_root: Path, out: Path) -> None:
    by_model: dict[str, list[dict]] = {}
    for c in load_cells(results_root):
        by_model.setdefault(c["model"], []).append(c)
    models = []
    for key in ROSTER:
        rows = by_model.get(key)
        if not rows:
            raise SystemExit(f"{key}: no family verdicts in {RUNS}")
        assert len(rows) == 20, f"{key}: {len(rows)} families"
        models.append(
            {
                "model": key,
                "mean_mi": sum(r["mi"] for r in rows) / len(rows),
                "mean_h_cond": sum(r["h_cond"] for r in rows) / len(rows),
                "verdicts": {"single": sum(r["n_groups"] == 1 for r in rows)},
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"models": models}, indent=1))
    print(f"wrote {out} ({len(models)} models)")
    for m in models:
        h = m["mean_mi"] + m["mean_h_cond"]
        print(
            f"  {m['model']:28s} H={h:.3f} mi={m['mean_mi']:.3f} "
            f"hc={m['mean_h_cond']:.3f} held={m['verdicts']['single']}/20"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", type=Path, default=ROOT / "results/twominds")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    main(a.results, a.out)
