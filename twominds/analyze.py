"""Analysis phase: run the cross-sample judge over a run's generations.

Reads ``<run>/logs/<model>/*.eval`` and ``<run>/questions.json`` (both written by
the generation phase), then for every (model, question) bundle records the
judge verdict (groups, contradiction, flags) and the answer-spread metrics,
rolls them up into per-model scores, and writes ``<run>/analysis.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from . import metrics as metrics_mod
from .judge import JudgeResult, run_judge_eval
from .models import DEFAULT_JUDGE, DEFAULT_JUDGE_CONCURRENCY, DEFAULT_JUDGE_REASONING


def _judge_display() -> str:
    """Inspect's rich display when interactive; plain off a TTY (nohup-friendly)."""
    return "rich" if sys.stdout.isatty() else "plain"


def _responses_from_analysis(run_dir: Path) -> dict[str, dict[str, list[str]]]:
    """Rebuild the response bundles from a prior ``analysis.json``.

    The analysis stores every response verbatim, so a re-judge doesn't need the
    raw eval logs.
    """
    path = Path(run_dir) / "analysis.json"
    if not path.exists():
        return {}
    out: dict[str, dict[str, list[str]]] = {}
    for r in json.loads(path.read_text()).get("results", []):
        out.setdefault(r["model"], {})[r["question_id"]] = list(
            r.get("responses") or []
        )
    return out


def _eval_logs_by_label(run_dir: Path) -> dict[str, str]:
    """label -> eval-log path for a run's per-model logs.

    Each model's eval log is written as ``<spec.name>.eval``. Key by the log
    file's *stem* (spec.name's leaf), not the directory name: a model whose
    name contains a ``/`` (e.g. a bare ``ours/<x>`` CLI arg) lands in nested
    dirs (``logs/ours/<x>/ours/<x>.eval``), and keying by the top-level dir
    would collapse every such model into one ``ours`` bucket. ``.``-prefixed
    dirs (Inspect's ``.raw`` scratch) are skipped. Last log per stem wins
    (≈ most recent re-run)."""
    from inspect_ai.log import list_eval_logs

    logs_root = Path(run_dir) / "logs"
    chosen: dict[str, str] = {}
    if not logs_root.exists():
        return chosen
    for model_dir in sorted(
        p for p in logs_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    ):
        for info in list_eval_logs(str(model_dir)):
            chosen[Path(info.name).stem] = info.name
    return chosen


def load_responses(run_dir: Path) -> dict[str, dict[str, list[str]]]:
    """{model_name: {question_id: [response, ...]}} from the per-model .eval logs.

    Reads both log shapes — fused (one sample per question, its N responses in
    the sample store; judge-inline runs) and epochs (one sample instance per
    response). Falls back to the run's ``analysis.json`` when the logs yield
    nothing.
    """
    from inspect_ai.log import read_eval_log

    from .generate import GEN_RESPONSES_KEY

    logs_root = Path(run_dir) / "logs"
    if not logs_root.exists():
        fallback = _responses_from_analysis(run_dir)
        if fallback:
            print(
                f"note: no logs dir at {logs_root}; "
                "re-judging the responses stored in analysis.json",
                flush=True,
            )
            return fallback
        raise FileNotFoundError(f"no logs dir at {logs_root}; run generation first")

    out: dict[str, dict[str, list[str]]] = {}
    for label, log_path in sorted(_eval_logs_by_label(run_dir).items()):
        log = read_eval_log(log_path)
        qmap: dict[str, list[str]] = {}
        for sample in log.samples or []:
            qid = str(sample.id)
            fused = (sample.store or {}).get(GEN_RESPONSES_KEY)
            if fused:
                qmap[qid] = [str(x or "") for x in fused]
                continue
            completion = ""
            if sample.output is not None:
                completion = sample.output.completion or ""
            qmap.setdefault(qid, []).append(completion)
        out[label] = qmap
    if not any(out.values()):
        # logs dir present but no readable eval logs
        fallback = _responses_from_analysis(run_dir)
        if fallback:
            print(
                f"note: no eval logs under {logs_root}; "
                "re-judging the responses stored in analysis.json",
                flush=True,
            )
            return fallback
        raise FileNotFoundError(
            f"no eval logs under {logs_root} and no analysis.json to re-judge from; "
            "run generation first"
        )
    return out


def load_judge_scores(
    run_dir: Path, judge_name: str, judge_reasoning: Optional[str]
) -> dict[tuple[str, str], JudgeResult]:
    """{(model, question_id): JudgeResult} harvested from fused generation logs.

    Only verdicts whose stamped identity (judge model, reasoning effort, judge
    prompt hash) matches the requested config are returned — anything else
    (epoch-shaped logs, a different judge, an edited judge prompt) yields
    nothing for that sample and gets judged fresh."""
    from inspect_ai.log import read_eval_log

    from .judge import judge_identity

    want = judge_identity(judge_name, judge_reasoning)
    out: dict[tuple[str, str], JudgeResult] = {}
    for label, log_path in sorted(_eval_logs_by_label(run_dir).items()):
        log = read_eval_log(log_path)
        for sample in log.samples or []:
            for score in (sample.scores or {}).values():
                meta = getattr(score, "metadata", None) or {}
                if meta.get("judge_identity") == want and "judge_result" in meta:
                    out[(label, str(sample.id))] = JudgeResult.from_dict(
                        meta["judge_result"]
                    )
    return out


def load_questions_meta(run_dir: Path) -> dict[str, dict]:
    path = Path(run_dir) / "questions.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def analyze(
    run_dir: Path,
    *,
    judge_name: str = DEFAULT_JUDGE,
    judge_reasoning: Optional[str] = DEFAULT_JUDGE_REASONING,
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
    out_dir: Optional[Path] = None,
) -> dict:
    """Judge a run's generations into ``analysis.json``.

    Verdicts the inline judge already wrote into the generation logs (same
    judge model, reasoning effort and prompt) are reused; every other bundle is
    judged now, as one Inspect eval with the bundles as samples. ``out_dir``
    overrides where ``analysis.json`` and the judge logs are written (default:
    ``run_dir``).
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    responses = load_responses(run_dir)
    qmeta = load_questions_meta(run_dir)

    # Flatten to bundles, preserving question order where known.
    bundles: list[tuple[str, str, list[str]]] = []
    for model_name in sorted(responses):
        for qid, resps in responses[model_name].items():
            bundles.append((model_name, qid, resps))

    in_log = load_judge_scores(run_dir, judge_name, judge_reasoning)
    harvested = {
        (m, qid): in_log[(m, qid)] for (m, qid, _resps) in bundles if (m, qid) in in_log
    }
    if harvested:
        print(
            f"reusing {len(harvested)} judge verdict(s) from the "
            "generation logs (judged inline during the sweep)",
            flush=True,
        )
    judge_items = [
        ((m, qid), qmeta.get(qid, {}).get("prompt", qid), resps)
        for (m, qid, resps) in bundles
        if (m, qid) not in harvested
    ]
    if judge_items:
        # Say how much work is queued: a big judge eval runs for minutes
        # and would otherwise read as a hang.
        print(
            f"judging {len(judge_items)} bundle(s) with {judge_name} "
            f"(x{concurrency} concurrent) ...",
            flush=True,
        )
    judge_results, _ = run_judge_eval(
        judge_items,
        judge_name=judge_name,
        reasoning_effort=judge_reasoning,
        max_connections=concurrency,
        log_path=out_dir / "judge_logs" / "responses",
        display=_judge_display(),
    )
    judge_results.update(harvested)

    results = []
    for model_name, qid, resps in bundles:
        n = len(resps)
        jr = judge_results.get((model_name, qid))
        m = metrics_mod.variance_metrics(
            resps, n_judge_groups=(jr.n_groups if jr is not None else None)
        )
        # Answer spread: entropy of the judge's grouping, -sum p_k log p_k.
        if jr is not None:
            m["group_entropy"] = metrics_mod.group_entropy(jr.labels(n))
        results.append(
            {
                "model": model_name,
                "question_id": qid,
                "group": qmeta.get(qid, {}).get("group", ""),
                "responses": resps,
                "judge": jr.to_dict() if jr is not None else None,
                "judge_labels": jr.labels(n) if jr is not None else None,
                "metrics": m,
            }
        )

    # Display names come from the manifest (write_manifest persists spec.display).
    run_config: dict = {}
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        run_config = json.loads(cfg_path.read_text())
    manifest_models = run_config.get("models") or {}
    models = sorted(responses)
    model_display = {
        name: (manifest_models.get(name) or {}).get("display") or name
        for name in models
    }

    out = {
        "run_dir": str(run_dir),
        "judge": judge_name,
        "judge_reasoning": judge_reasoning,
        "models": models,
        "model_display": model_display,
        "config": {
            k: run_config.get(k)
            for k in ("models", "n", "temperature", "max_tokens", "judge")
            if k in run_config
        },
        "questions": qmeta,
        "results": results,
        "scores": metrics_mod.model_scores(results, models),
    }
    (out_dir / "analysis.json").write_text(json.dumps(out, indent=2))
    return out
