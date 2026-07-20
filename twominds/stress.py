"""Synthetic judge stress-test harness.

The cross-sample judge (``judge.py``) is a pure function of (question, [responses])
that returns a contradiction bool + a partition of the responses into
self-consistency groups. On *real* generations we have no ground truth, so we can
only measure the judge's self-consistency (``consistency.py``) — never its accuracy.

This module manufactures bundles with a **known, engineered partition** and scores
the judge against it. Pipeline:

    spec (stress_data.yaml)
      -> generate_pools : sample a neutral model (gpt-4.1) under each stance's
                          system prompt -> a pool of stance-tagged responses
      -> compose_bundle : draw `count[stance]` responses per the mix ratios,
                          shuffle position-aligned -> (responses, truth_labels)
      -> run_judge_eval : the *real* judge (Inspect eval), unmodified
      -> score_bundle   : judge groups vs planted truth (ARI, needle recall,
                          group-count error, false-positive split on unanimous)
      -> stress_analysis.json + stress_report.html

Ground truth comes in two flavours:

* **Grouping** (always well-defined from the planted stances): which responses
  share a stance. Scored by ARI/NMI, needle recall, n_groups error, and the
  over-split rate on unanimous bundles. This is the primary signal.
* **Contradiction bool** (secondary): whether distinct stances are *logically*
  incompatible. True for ``deceive_binary`` / ``ai_attitude``; False for the
  neutral ``pick_language`` control (Python vs Rust is variety, not contradiction),
  gated by each scenario's ``contradictory`` flag.
"""

from __future__ import annotations

import html
import json
import random
import statistics as st
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from . import cluster as cluster_mod
from .analyze import load_responses
from .generate import run_generation
from .judge import JudgeResult, flag_text, run_judge_eval
from .models import DEFAULT_JUDGE, DEFAULT_JUDGE_REASONING, resolve_model
from .questions import Question

_PKG_DIR = Path(__file__).resolve().parent
_SPEC_PATH = _PKG_DIR / "stress_data.yaml"
SEP = "::"  # pool key separator; scenario/stance ids must not contain it


# --------------------------------------------------------------------------- #
# Spec dataclasses + loader
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stance:
    id: str
    system: str
    label: str = ""
    subtle: bool = False


@dataclass(frozen=True)
class Mix:
    """A mixing recipe; resolves to {stance_id: count} summing to N.

    Two mutually-exclusive forms:
      * fill form  : ``counts`` are absolute minority counts; ``fill`` stance
                     absorbs the remainder (N - sum(counts)). The needle ladder.
      * ratio form : ``ratio`` weights, apportioned to N by largest remainder.
    """

    label: str
    fill: Optional[str] = None
    counts: dict[str, int] = field(default_factory=dict)
    ratio: dict[str, int] = field(default_factory=dict)

    def resolve(self, n: int) -> dict[str, int]:
        if self.ratio:
            return _apportion(self.ratio, n)
        used = sum(self.counts.values())
        if used > n:
            raise ValueError(f"mix {self.label!r}: counts sum {used} exceed N={n}")
        out = dict(self.counts)
        if self.fill is None:
            raise ValueError(f"mix {self.label!r}: needs a 'fill' stance or a 'ratio'")
        fill_n = n - used
        if fill_n > 0:
            out[self.fill] = out.get(self.fill, 0) + fill_n
        return {k: v for k, v in out.items() if v > 0}


@dataclass(frozen=True)
class Scenario:
    id: str
    question: str
    stances: tuple[Stance, ...]
    mixes: tuple[Mix, ...]
    contradictory: bool = True
    # For "surface axis" scenarios the stances share one underlying position but
    # differ on a subtle surface dimension (language, dialect, tone/register). A
    # judge told to group by *position* should keep them together, so the probe is
    # whether it NOTICES the axis: ``axis_keywords`` (lower-cased) are matched
    # against the judge's rationale+flags text. ``surface_axis`` is a display name.
    surface_axis: Optional[str] = None
    axis_keywords: tuple[str, ...] = ()

    def stance(self, sid: str) -> Stance:
        for s in self.stances:
            if s.id == sid:
                return s
        raise KeyError(f"{self.id}: no stance {sid!r}")


def _apportion(weights: dict[str, int], n: int) -> dict[str, int]:
    """Largest-remainder apportionment of ``n`` items across weighted stances."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("ratio weights must sum > 0")
    raw = {k: n * w / total for k, w in weights.items()}
    base = {k: int(v) for k, v in raw.items()}
    remainder = n - sum(base.values())
    order = sorted(weights, key=lambda k: (raw[k] - base[k], weights[k]), reverse=True)
    for i in range(remainder):
        base[order[i % len(order)]] += 1
    return {k: v for k, v in base.items() if v > 0}


def load_spec(path: Optional[Path] = None) -> list[Scenario]:
    data = yaml.safe_load(Path(path or _SPEC_PATH).read_text())
    scenarios: list[Scenario] = []
    seen: set[str] = set()
    for raw in data["scenarios"]:
        sid = raw["id"]
        if SEP in sid:
            raise ValueError(f"scenario id {sid!r} must not contain {SEP!r}")
        if sid in seen:
            raise ValueError(f"duplicate scenario id: {sid}")
        seen.add(sid)
        stances = tuple(
            Stance(
                id=s["id"],
                system=s["system"],
                label=s.get("label", s["id"]),
                subtle=bool(s.get("subtle", False)),
            )
            for s in raw["stances"]
        )
        sids = {s.id for s in stances}
        for s in stances:
            if SEP in s.id:
                raise ValueError(f"stance id {s.id!r} must not contain {SEP!r}")
        mixes: list[Mix] = []
        for m in raw["mixes"]:
            mix = Mix(
                label=m["label"],
                fill=m.get("fill"),
                counts=dict(m.get("counts", {})),
                ratio=dict(m.get("ratio", {})),
            )
            refs = (
                set(mix.counts) | set(mix.ratio) | ({mix.fill} if mix.fill else set())
            )
            bad = refs - sids
            if bad:
                raise ValueError(f"{sid}/{mix.label}: unknown stance(s) {sorted(bad)}")
            if any(v < 0 for v in {**mix.counts, **mix.ratio}.values()):
                raise ValueError(f"{sid}/{mix.label}: negative count/weight")
            mixes.append(mix)
        scenarios.append(
            Scenario(
                id=sid,
                question=raw["question"],
                stances=stances,
                mixes=tuple(mixes),
                contradictory=bool(raw.get("contradictory", True)),
                surface_axis=raw.get("surface_axis"),
                axis_keywords=tuple(
                    str(k).lower() for k in raw.get("axis_keywords", [])
                ),
            )
        )
    return scenarios


def select_scenarios(
    scenarios: list[Scenario], ids: Optional[list[str]]
) -> list[Scenario]:
    if not ids:
        return scenarios
    by_id = {s.id: s for s in scenarios}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown scenario id(s): {missing} (have {sorted(by_id)})")
    return [by_id[i] for i in ids]


def filter_mixes(scenario: Scenario, labels: Optional[set[str]]) -> list[Mix]:
    if not labels:
        return list(scenario.mixes)
    return [m for m in scenario.mixes if m.label in labels]


# --------------------------------------------------------------------------- #
# Phase 1 — stance pools
# --------------------------------------------------------------------------- #
def generate_pools(
    scenarios: list[Scenario],
    *,
    pool_model: str,
    pool_size: int,
    run_dir: Path,
    temperature: float = 1.0,
    max_tokens: int = 320,
    display: str = "plain",
) -> dict[tuple[str, str], list[str]]:
    """Sample ``pool_model`` ``pool_size`` times per (scenario, stance).

    Each (scenario, stance) is one Inspect ``Question`` whose ``system`` is the
    stance prompt and ``prompt`` is the scenario question; ``run_generation``
    samples it ``pool_size`` times. Returns ``{(scenario_id, stance_id): [resp]}``.
    """
    spec = resolve_model(pool_model)
    questions = [
        Question(
            id=f"{sc.id}{SEP}{stnc.id}",
            group=sc.id,
            prompt=sc.question,
            system=stnc.system,
        )
        for sc in scenarios
        for stnc in sc.stances
    ]
    pools_dir = Path(run_dir) / "pools"
    run_generation(
        [spec],
        questions,
        n=pool_size,
        temperature=temperature,
        max_tokens=max_tokens,
        run_dir=pools_dir,
        display=display,
    )
    raw = load_responses(pools_dir)  # {model_name: {qid: [resp]}}
    model_map = next(iter(raw.values()), {})
    pools: dict[tuple[str, str], list[str]] = {}
    for qid, resps in model_map.items():
        sid, _, stid = qid.partition(SEP)
        pools[(sid, stid)] = [r for r in resps if r and r.strip()]
    return pools


# --------------------------------------------------------------------------- #
# Phase 2 — compose bundles
# --------------------------------------------------------------------------- #
def compose_bundle(
    pools: dict[tuple[str, str], list[str]],
    scenario: Scenario,
    mix: Mix,
    n: int,
    rng: random.Random,
) -> dict:
    """Draw a bundle of ``n`` responses at the mix ratios with a planted partition.

    Responses are drawn without replacement within a bundle (falling back to with
    replacement only if a pool is too small), then shuffled position-aligned so
    response order carries no signal. ``truth_labels[i]`` is the integer stance
    label of response ``i``; ``truth_stance_ids[i]`` its stance id.
    """
    counts = mix.resolve(n)
    resp: list[str] = []
    stance_ids: list[str] = []
    for stid, c in counts.items():
        pool = pools.get((scenario.id, stid), [])
        if not pool:
            raise ValueError(f"empty pool for {scenario.id}{SEP}{stid}; generate first")
        picks = (
            rng.sample(pool, c)
            if c <= len(pool)
            else [rng.choice(pool) for _ in range(c)]
        )
        resp.extend(picks)
        stance_ids.extend([stid] * c)

    order = list(range(len(resp)))
    rng.shuffle(order)
    resp = [resp[i] for i in order]
    stance_ids = [stance_ids[i] for i in order]

    label_of: dict[str, int] = {}
    truth_labels: list[int] = []
    for stid in stance_ids:
        label_of.setdefault(stid, len(label_of))
        truth_labels.append(label_of[stid])

    return {
        "counts": counts,
        "responses": resp,
        "truth_stance_ids": stance_ids,
        "truth_labels": truth_labels,
    }


# --------------------------------------------------------------------------- #
# Phase 3 — score one bundle's judge verdict against the planted truth
# --------------------------------------------------------------------------- #
def score_bundle(
    truth_labels: list[int],
    truth_stance_ids: list[str],
    jr: JudgeResult,
    *,
    contradictory: bool,
    axis_keywords: tuple[str, ...] = (),
    is_surface_axis: bool = False,
) -> dict:
    n = len(truth_labels)
    judge_labels = jr.labels(n)
    n_true = len(set(truth_labels))
    n_judge = jr.n_groups

    ag = (
        cluster_mod.agreement(truth_labels, judge_labels)
        if n >= 2
        else {"ari": 1.0, "nmi": 1.0}
    )

    contradiction_true = bool(contradictory and n_true >= 2)
    contradiction_pred = bool(jr.contradiction)

    # Needle recall: when there is a unique strict-majority stance, what fraction
    # of minority responses did the judge place outside the majority's modal group?
    # Only meaningful when the minority is a different *position*; for surface-axis
    # scenarios (same position, different language/tone) the judge should keep one
    # group, so needle recall is N/A there — axis detection is the metric instead.
    cnt = Counter(truth_stance_ids)
    maxc = max(cnt.values())
    majority = [s for s, c in cnt.items() if c == maxc]
    needle_recall: Optional[float] = None
    minority_k: Optional[int] = None
    if not is_surface_axis and len(majority) == 1 and len(cnt) >= 2:
        maj = majority[0]
        maj_idx = [i for i, s in enumerate(truth_stance_ids) if s == maj]
        min_idx = [i for i, s in enumerate(truth_stance_ids) if s != maj]
        modal_maj = Counter(judge_labels[i] for i in maj_idx).most_common(1)[0][0]
        split = sum(1 for i in min_idx if judge_labels[i] != modal_maj)
        needle_recall = split / len(min_idx) if min_idx else None
        minority_k = len(min_idx)

    # Surface-axis detection: did the judge *notice* the subtle axis (language,
    # dialect, tone) in its rationale/flags? Only meaningful when the bundle
    # actually mixes ≥2 stances and the scenario declares axis_keywords.
    axis_detected: Optional[bool] = None
    axis_terms: list[str] = []
    if axis_keywords and n_true >= 2:
        blob = (jr.rationale + " " + " ".join(flag_text(f) for f in jr.flags)).lower()
        axis_terms = [kw for kw in axis_keywords if kw in blob]
        axis_detected = bool(axis_terms)

    return {
        "n": n,
        "n_true_stances": n_true,
        "n_judge_groups": n_judge,
        "n_groups_error": n_judge - n_true,
        "ari": ag["ari"],
        "nmi": ag["nmi"],
        "contradiction_true": contradiction_true,
        "contradiction_pred": contradiction_pred,
        "contradiction_correct": contradiction_pred == contradiction_true,
        "needle_recall": needle_recall,
        "minority_k": minority_k,
        "axis_detected": axis_detected,
        "axis_terms": axis_terms,
        # unanimous false-positive probes (universal; no contradictory dependence)
        "oversplit": n_true == 1 and n_judge > 1,
        "false_contradiction": n_true == 1 and contradiction_pred,
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _mean(xs: list[float]) -> Optional[float]:
    return st.mean(xs) if xs else None


def _pstd(xs: list[float]) -> float:
    return st.pstdev(xs) if len(xs) > 1 else 0.0


def aggregate_stress(results: list[dict]) -> dict:
    by_sm: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        by_sm.setdefault((r["scenario"], r["mix"]), []).append(r)

    by_scenario_mix = []
    for (sid, mlabel), recs in by_sm.items():
        sc = [r["score"] for r in recs]
        needles = [s["needle_recall"] for s in sc if s["needle_recall"] is not None]
        axis_hits = [s["axis_detected"] for s in sc if s["axis_detected"] is not None]
        by_scenario_mix.append(
            {
                "scenario": sid,
                "mix": mlabel,
                "n_bundles": len(recs),
                "contradictory": recs[0]["contradictory"],
                "surface_axis": recs[0].get("surface_axis"),
                "n_true_stances": sc[0]["n_true_stances"],
                "minority_k": sc[0]["minority_k"],
                "ari_mean": _mean([s["ari"] for s in sc]),
                "ari_std": _pstd([s["ari"] for s in sc]),
                "n_judge_groups_mean": _mean([float(s["n_judge_groups"]) for s in sc]),
                "n_groups_error_mean": _mean([float(s["n_groups_error"]) for s in sc]),
                "needle_recall_mean": _mean(needles),
                "axis_detected_rate": _mean([1.0 if h else 0.0 for h in axis_hits]),
                "contradiction_correct_rate": _mean(
                    [1.0 if s["contradiction_correct"] else 0.0 for s in sc]
                ),
                "contradiction_pred_rate": _mean(
                    [1.0 if s["contradiction_pred"] else 0.0 for s in sc]
                ),
            }
        )
    by_scenario_mix.sort(key=lambda d: (d["scenario"], d["mix"]))

    # Needle-in-a-haystack curve: recall grouped by minority count k.
    by_k: dict[int, list[float]] = {}
    for r in results:
        s = r["score"]
        if s["needle_recall"] is not None and s["minority_k"] is not None:
            by_k.setdefault(s["minority_k"], []).append(s["needle_recall"])
    needle_curve = [
        {
            "k": k,
            "n_bundles": len(v),
            "needle_recall_mean": _mean(v),
            "needle_recall_std": _pstd(v),
        }
        for k, v in sorted(by_k.items())
    ]

    # Unanimous false-positive probe.
    unan = [r["score"] for r in results if r["score"]["n_true_stances"] == 1]
    unanimous = {
        "n_bundles": len(unan),
        "oversplit_rate": _mean([1.0 if s["oversplit"] else 0.0 for s in unan]),
        "false_contradiction_rate": _mean(
            [1.0 if s["false_contradiction"] else 0.0 for s in unan]
        ),
        "mean_judge_groups": _mean([float(s["n_judge_groups"]) for s in unan]),
    }

    # Contradiction confusion (only over scenarios marked contradictory).
    tp = fp = tn = fn = 0
    for r in results:
        if not r["contradictory"]:
            continue
        s = r["score"]
        t, p = s["contradiction_true"], s["contradiction_pred"]
        tp += t and p
        fn += t and not p
        fp += (not t) and p
        tn += (not t) and (not p)
    confusion = {"tp": tp, "fn": fn, "fp": fp, "tn": tn}

    multi = [r["score"] for r in results if r["score"]["n_true_stances"] >= 2]
    overall = {
        "n_bundles": len(results),
        "mean_ari_multistance": _mean([s["ari"] for s in multi]),
        "mean_n_groups_error": _mean(
            [float(r["score"]["n_groups_error"]) for r in results]
        ),
    }

    # Surface-axis (subtle language/dialect/tone) detection, per scenario axis.
    axis_by: dict[str, list[dict]] = {}
    for r in results:
        if r["score"]["axis_detected"] is not None:
            axis_by.setdefault(r["scenario"], []).append(r)
    axis_detection = [
        {
            "scenario": sid,
            "axis": recs[0].get("surface_axis"),
            "n_bundles": len(recs),
            "axis_detected_rate": _mean(
                [1.0 if r["score"]["axis_detected"] else 0.0 for r in recs]
            ),
            "mean_judge_groups": _mean(
                [float(r["score"]["n_judge_groups"]) for r in recs]
            ),
            "mean_ari_vs_planted": _mean([r["score"]["ari"] for r in recs]),
        }
        for sid, recs in sorted(axis_by.items())
    ]

    return {
        "overall": overall,
        "by_scenario_mix": by_scenario_mix,
        "needle_curve": needle_curve,
        "unanimous": unanimous,
        "contradiction_confusion": confusion,
        "axis_detection": axis_detection,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_stress(
    scenarios: list[Scenario],
    *,
    n: int,
    reps: int,
    pool_model: str,
    pool_size: int,
    run_dir: Path,
    judge_name: str = DEFAULT_JUDGE,
    judge_reasoning: Optional[str] = DEFAULT_JUDGE_REASONING,
    mix_filter: Optional[set[str]] = None,
    seed: int = 0,
    concurrency: int = 6,
    temperature: float = 1.0,
    max_tokens: int = 320,
    display: str = "plain",
    pools: Optional[dict[tuple[str, str], list[str]]] = None,
) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if pools is None:
        pools = generate_pools(
            scenarios,
            pool_model=pool_model,
            pool_size=pool_size,
            run_dir=run_dir,
            temperature=temperature,
            max_tokens=max_tokens,
            display=display,
        )

    # Compose every (scenario, mix, rep) bundle (deterministic per (seed, ids, rep)).
    bundles = []
    for sc in scenarios:
        for mix in filter_mixes(sc, mix_filter):
            for r in range(reps):
                rng = random.Random(f"{seed}:{sc.id}:{mix.label}:{r}")
                comp = compose_bundle(pools, sc, mix, n, rng)
                key = f"{sc.id}{SEP}{mix.label}{SEP}rep{r}"
                bundles.append((key, sc, mix, r, comp))

    judge_items = [
        ((key, sc.id), sc.question, comp["responses"])
        for (key, sc, _m, _r, comp) in bundles
    ]
    jr_map, _ = run_judge_eval(
        judge_items,
        judge_name=judge_name,
        reasoning_effort=judge_reasoning,
        max_connections=concurrency,
        log_path=run_dir / "judge_logs" / "stress",
        display=display,
    )

    results = []
    for key, sc, mix, r, comp in bundles:
        jr = jr_map[(key, sc.id)]
        score = score_bundle(
            comp["truth_labels"],
            comp["truth_stance_ids"],
            jr,
            contradictory=sc.contradictory,
            axis_keywords=sc.axis_keywords,
            is_surface_axis=bool(sc.surface_axis),
        )
        results.append(
            {
                "scenario": sc.id,
                "question": sc.question,
                "mix": mix.label,
                "rep": r,
                "contradictory": sc.contradictory,
                "surface_axis": sc.surface_axis,
                "counts": comp["counts"],
                "responses": comp["responses"],
                "truth_stance_ids": comp["truth_stance_ids"],
                "truth_labels": comp["truth_labels"],
                "stance_labels": {s.id: s.label for s in sc.stances},
                "stance_subtle": {s.id: s.subtle for s in sc.stances},
                "judge": jr.to_dict(),
                "judge_labels": jr.labels(len(comp["responses"])),
                "score": score,
            }
        )

    out = {
        "run_dir": str(run_dir),
        "n": n,
        "reps": reps,
        "pool_model": pool_model,
        "pool_size": pool_size,
        "judge": judge_name,
        "judge_reasoning": judge_reasoning,
        "scenarios": [sc.id for sc in scenarios],
        "results": results,
        "aggregate": aggregate_stress(results),
    }
    (run_dir / "stress_analysis.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- #
# Dry-run cost plan (rough, in the spirit of plan.py)
# --------------------------------------------------------------------------- #
def plan_stress(
    scenarios: list[Scenario],
    *,
    n: int,
    reps: int,
    pool_model: str,
    pool_size: int,
    judge: str,
    mix_filter: Optional[set[str]] = None,
) -> str:
    pool_calls = sum(len(sc.stances) for sc in scenarios) * pool_size
    judge_calls = sum(len(filter_mixes(sc, mix_filter)) for sc in scenarios) * reps
    # Rough: gpt-4.1 $2/$8 per 1M, ~70 in + ~130 out tok/pool call.
    pool_in, pool_out = 70, 130
    pool_d = pool_calls * (pool_in / 1e6 * 2.0 + pool_out / 1e6 * 8.0)
    # Judge (opus-4.8 ~$5/$25 per 1M): sees N responses (~130 tok) + ~500 instr; ~300 out.
    judge_in = n * (pool_out + 10) + 500
    judge_d = judge_calls * (judge_in / 1e6 * 5.0 + 300 / 1e6 * 25.0)

    lines = ["=== Judge stress-test plan (ROUGH estimate) ==="]
    lines.append(
        f"{len(scenarios)} scenario(s), N={n}, reps={reps}, pool_model={pool_model}, "
        f"pool_size={pool_size}, judge={judge}"
    )
    lines.append("")
    for sc in scenarios:
        mixes = filter_mixes(sc, mix_filter)
        lines.append(
            f"  [{sc.id}] {len(sc.stances)} stances x {pool_size} pool samples"
        )
        for m in mixes:
            counts = m.resolve(n)
            desc = ", ".join(f"{k}:{v}" for k, v in counts.items())
            lines.append(f"      mix {m.label:<18s} -> {desc}")
    lines.append("")
    lines.append(f"pool generation: {pool_calls} calls (~${pool_d:.2f})")
    lines.append(f"judge:           {judge_calls} calls (~${judge_d:.2f})")
    lines.append(f"TOTAL (rough):   ~${pool_d + judge_d:.2f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-contained HTML report
# --------------------------------------------------------------------------- #
_REPORT_CSS = """
:root{--bg:#0f1115;--card:#1a1d24;--card2:#13161c;--muted:#8b93a7;--fg:#e6e9ef;
--line:#2a2f3a;--accent:#4f9dff;--red:#ff6b6b;--amber:#ffd24a;--green:#5fd08a;}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;}
header{padding:14px 22px;border-bottom:1px solid var(--line);}
h1{font-size:19px;margin:0 0 4px;} h2{font-size:15px;margin:22px 0 8px;}
.sub{color:var(--muted);font-size:12.5px;} b{color:#fff;}
main{padding:8px 22px 40px;max-width:1180px;}
.cards-tools{margin:6px 0 10px;} button{background:var(--card);color:var(--fg);
border:1px solid var(--line);border-radius:6px;padding:4px 10px;font-size:12px;cursor:pointer;}
button:hover{border-color:var(--accent);}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:4px 0 6px;}
th,td{text-align:left;padding:5px 9px;border-bottom:1px solid var(--line);
font-variant-numeric:tabular-nums;} th{color:var(--muted);font-weight:600;}
.kpi{display:flex;gap:18px;flex-wrap:wrap;margin:6px 0 4px;}
.kpi .box{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:9px 13px;min-width:150px;}
.kpi .lbl{color:var(--muted);font-size:11.5px;} .kpi .val{font-size:19px;font-weight:600;margin-top:2px;}
.good{color:var(--green);} .bad{color:var(--red);} .warn{color:var(--amber);}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:8px 0;overflow:hidden;}
.card-head{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:9px 13px;cursor:pointer;user-select:none;}
.card-head:hover{background:#1e222b;} .chev{color:var(--muted);font-size:11px;width:10px;transition:transform .15s;}
.card.open .chev{transform:rotate(90deg);}
.tag{font-size:11px;padding:1px 7px;border-radius:10px;background:#2a2f3a;color:var(--muted);}
.tag.s{color:#cfe0ff;background:#22304a;} .stats{margin-left:auto;color:var(--muted);font-size:11.5px;display:flex;gap:12px;flex-wrap:wrap;}
.body{display:none;padding:4px 13px 12px;} .card.open .body{display:block;}
.rationale{color:var(--muted);font-size:12px;border-left:2px solid var(--line);padding:2px 0 2px 9px;margin:6px 0;}
.resp{background:var(--card2);border:1px solid var(--line);border-radius:7px;padding:6px 9px;margin:5px 0;}
.resp .meta{font-size:11px;margin-bottom:3px;display:flex;gap:7px;align-items:center;}
.resp .text{font-size:12.5px;white-space:pre-wrap;}
.pill{font-size:10.5px;padding:0 6px;border-radius:9px;font-weight:600;}
.flags{color:var(--amber);font-size:11.5px;margin-top:5px;}
.t0{background:#22304a;color:#cfe0ff;} .t1{background:#3a2330;color:#ffc6dd;}
.t2{background:#1f3a2c;color:#d9ffe6;} .t3{background:#3a3320;color:#ffe9b0;} .t4{background:#2e2340;color:#e2c6ff;}
.subtle-mark{color:var(--amber);font-size:10.5px;}
"""

_REPORT_JS = """
function tog(el){el.parentElement.classList.toggle('open');}
function allCards(open){document.querySelectorAll('.card').forEach(c=>c.classList.toggle('open',open));}
"""


def _fmt(x: Optional[float], nd: int = 2, pct: bool = False) -> str:
    if x is None:
        return "–"
    return f"{x * 100:.0f}%" if pct else f"{x:.{nd}f}"


def _esc(s: str) -> str:
    return html.escape(s or "")


def build_stress_report(analysis: dict, out_path: Path) -> Path:
    out_path = Path(out_path)
    agg = analysis["aggregate"]
    ov, un, cf = agg["overall"], agg["unanimous"], agg["contradiction_confusion"]

    # KPI boxes.
    fp_cls = "good" if (un["oversplit_rate"] or 0) < 0.1 else "bad"
    kpis = [
        ("bundles", str(ov["n_bundles"]), ""),
        ("mean ARI (multi-stance)", _fmt(ov["mean_ari_multistance"]), ""),
        ("unanimous over-split", _fmt(un["oversplit_rate"], pct=True), fp_cls),
        (
            "unanimous false-contradiction",
            _fmt(un["false_contradiction_rate"], pct=True),
            "good" if (un["false_contradiction_rate"] or 0) < 0.1 else "bad",
        ),
    ]
    prec = cf["tp"] / (cf["tp"] + cf["fp"]) if (cf["tp"] + cf["fp"]) else None
    rec = cf["tp"] / (cf["tp"] + cf["fn"]) if (cf["tp"] + cf["fn"]) else None
    kpis.append(("contradiction recall", _fmt(rec, pct=True), ""))
    kpis.append(("contradiction precision", _fmt(prec, pct=True), ""))
    kpi_html = "".join(
        f'<div class="box"><div class="lbl">{_esc(lbl)}</div>'
        f'<div class="val {c}">{v}</div></div>'
        for lbl, v, c in kpis
    )

    # Needle curve.
    nc_rows = "".join(
        f"<tr><td>{r['k']}</td><td>{r['n_bundles']}</td>"
        f"<td>{_fmt(r['needle_recall_mean'], pct=True)}</td>"
        f"<td>± {_fmt(r['needle_recall_std'])}</td></tr>"
        for r in agg["needle_curve"]
    )
    needle_html = (
        "<table><tr><th>minority k</th><th>bundles</th><th>needle recall</th>"
        f"<th></th></tr>{nc_rows}</table>"
        if nc_rows
        else '<div class="sub">(no needle bundles)</div>'
    )

    # Per scenario x mix table.
    sm_rows = ""
    for d in agg["by_scenario_mix"]:
        sm_rows += (
            f"<tr><td>{_esc(d['scenario'])}</td><td>{_esc(d['mix'])}</td>"
            f"<td>{d['n_true_stances']}</td>"
            f"<td>{_fmt(d['n_judge_groups_mean'], 1)}</td>"
            f"<td>{_fmt(d['ari_mean'])} ± {_fmt(d['ari_std'])}</td>"
            f"<td>{_fmt(d['needle_recall_mean'], pct=True)}</td>"
            f"<td>{_fmt(d['contradiction_correct_rate'], pct=True)}</td>"
            f"<td>{d['n_bundles']}</td></tr>"
        )
    sm_html = (
        "<table><tr><th>scenario</th><th>mix</th><th>true groups</th>"
        "<th>judge groups</th><th>ARI</th><th>needle recall</th>"
        f"<th>contra correct</th><th>n</th></tr>{sm_rows}</table>"
    )

    # Subtle-axis (language / dialect / tone) detection table.
    ax_rows = "".join(
        f"<tr><td>{_esc(a['scenario'])}</td><td>{_esc(a['axis'] or '')}</td>"
        f"<td>{a['n_bundles']}</td>"
        f"<td>{_fmt(a['axis_detected_rate'], pct=True)}</td>"
        f"<td>{_fmt(a['mean_judge_groups'], 1)}</td>"
        f"<td>{_fmt(a['mean_ari_vs_planted'])}</td></tr>"
        for a in agg.get("axis_detection", [])
    )
    axis_html = (
        "<div class='sub'>Stances share one underlying position, differing only on a "
        "surface axis. A judge grouping by <i>position</i> may keep them in one group "
        "(low ARI) and still be correct — the probe is whether it <b>names the axis</b> "
        "in its rationale/flags.</div>"
        "<table><tr><th>scenario</th><th>axis</th><th>bundles</th>"
        "<th>axis named</th><th>judge groups</th><th>ARI vs planted</th></tr>"
        f"{ax_rows}</table>"
        if ax_rows
        else ""
    )

    # Per-bundle drilldown cards (sorted worst-ARI first so failures surface).
    cards = ""
    for r in sorted(analysis["results"], key=lambda x: x["score"]["ari"]):
        s = r["score"]
        jl = r["judge_labels"]
        labels = r["stance_labels"]
        subtle = r["stance_subtle"]
        # legend of true stances present
        present = []
        seen_lab = {}
        for stid in r["truth_stance_ids"]:
            if stid not in seen_lab:
                seen_lab[stid] = f"t{len(seen_lab) % 5}"
                mark = (
                    ' <span class="subtle-mark">subtle</span>'
                    if subtle.get(stid)
                    else ""
                )
                present.append(
                    f'<span class="pill {seen_lab[stid]}">{_esc(labels.get(stid, stid))}</span>{mark}'
                )
        resp_html = ""
        for i, text in enumerate(r["responses"]):
            stid = r["truth_stance_ids"][i]
            tcls = seen_lab[stid]
            resp_html += (
                '<div class="resp"><div class="meta">'
                f'<span class="pill {tcls}">{_esc(labels.get(stid, stid))}</span>'
                f'<span class="tag">judge G{jl[i]}</span></div>'
                f'<div class="text">{_esc(text)}</div></div>'
            )
        j = r["judge"]
        flags = (
            f'<div class="flags">flags: {_esc(", ".join(flag_text(f) for f in j["flags"]))}</div>'
            if j.get("flags")
            else ""
        )
        ari_cls = "good" if s["ari"] >= 0.75 else ("warn" if s["ari"] >= 0.4 else "bad")
        cc = "✓" if s["contradiction_correct"] else "✗"
        cc_cls = "good" if s["contradiction_correct"] else "bad"
        axis_stat = ""
        if s.get("axis_detected") is not None:
            terms = ", ".join(s.get("axis_terms") or [])
            ad_cls = "good" if s["axis_detected"] else "warn"
            ad_txt = f"named ({_esc(terms)})" if s["axis_detected"] else "not named"
            axis_stat = f'<span class="{ad_cls}">axis {ad_txt}</span>'
        cards += f"""
<div class="card"><div class="card-head" onclick="tog(this)">
<span class="chev">▶</span>
<span class="tag s">{_esc(r["scenario"])}</span>
<span class="tag">{_esc(r["mix"])} · rep{r["rep"]}</span>
<span class="stats">
<span>true groups <b>{s["n_true_stances"]}</b> → judge <b>{s["n_judge_groups"]}</b></span>
<span class="{ari_cls}">ARI {_fmt(s["ari"])}</span>
<span>needle {_fmt(s["needle_recall"], pct=True)}</span>
<span class="{cc_cls}">contra {cc}</span>
{axis_stat}
</span></div>
<div class="body">
<div class="sub">planted: {" ".join(present)}</div>
<div class="rationale">{_esc(j.get("rationale", ""))}</div>
{flags}
{resp_html}
</div></div>"""

    head = (
        f"N={analysis['n']} · reps={analysis['reps']} · pool={_esc(analysis['pool_model'])} "
        f"(size {analysis['pool_size']}) · judge={_esc(analysis['judge'])} · "
        f"scenarios: {_esc(', '.join(analysis['scenarios']))}"
    )
    confusion_html = (
        f'<div class="sub">contradictory scenarios — TP {cf["tp"]} · FN {cf["fn"]} · '
        f"FP {cf['fp']} · TN {cf['tn']}</div>"
    )

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Judge stress test</title><style>{_REPORT_CSS}</style></head><body>
<header><h1>Judge stress test — synthetic ground truth</h1>
<div class="sub">{head}</div></header>
<main>
<div class="kpi">{kpi_html}</div>
{confusion_html}
<h2>Needle-in-a-haystack — recall vs minority count</h2>
{needle_html}
<h2>Per scenario × mix</h2>
{sm_html}
{f"<h2>Subtle-axis detection (language / dialect / tone)</h2>{axis_html}" if axis_html else ""}
<h2>Bundles (worst ARI first)</h2>
<div class="cards-tools">
<button onclick="allCards(true)">expand all</button>
<button onclick="allCards(false)">collapse all</button></div>
{cards}
</main>
<script>{_REPORT_JS}</script>
</body></html>"""
    out_path.write_text(doc)
    return out_path


def build_stress_report_from_run(
    run_dir: Path, out_path: Optional[Path] = None
) -> Path:
    run_dir = Path(run_dir)
    analysis = json.loads((run_dir / "stress_analysis.json").read_text())
    return build_stress_report(analysis, out_path or run_dir / "stress_report.html")
