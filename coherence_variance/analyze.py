"""Analysis phase: load generation logs, run the judge + embedding clustering.

Reads ``<run>/logs/<model>/*.eval`` and ``<run>/questions.json`` (both written by
the generation phase), then for every (model, question) bundle computes:
  - the cross-sample coherence judge verdict (groups, contradiction, flags),
  - embedding clusters for each requested backend,
  - judge-vs-cluster agreement (ARI/NMI) and variance metrics,
and writes ``<run>/analysis.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from . import cluster as cluster_mod
from . import cost as cost_mod
from . import families as families_mod
from . import metrics as metrics_mod
from .embed import DEFAULT_LOCAL_MODEL, get_embedder
from .judge import JudgeResult, run_judge_eval
from .models import DEFAULT_JUDGE, DEFAULT_JUDGE_REASONING


def _judge_display() -> str:
    """Inspect's rich display when interactive; plain off a TTY (nohup-friendly)."""
    return "rich" if sys.stdout.isatty() else "plain"


def load_responses(run_dir: Path) -> dict[str, dict[str, list[str]]]:
    """{model_name: {question_id: [response, ...]}} from the per-model .eval logs."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    logs_root = Path(run_dir) / "logs"
    if not logs_root.exists():
        raise FileNotFoundError(f"no logs dir at {logs_root}; run generation first")

    # Each model's eval log is written as ``<spec.name>.eval``. Key by the log
    # file's *stem* (spec.name's leaf), not the directory name: a model whose
    # name contains a ``/`` (e.g. a bare ``ours/<x>`` CLI arg) lands in nested
    # dirs (``logs/ours/<x>/ours/<x>.eval``), and keying by the top-level dir
    # would collapse every such model into one ``ours`` bucket. ``.``-prefixed
    # dirs (Inspect's ``.raw`` scratch) are skipped. Last log per stem wins
    # (≈ most recent re-run).
    chosen: dict[str, str] = {}  # label -> eval-log path
    for model_dir in sorted(
        p for p in logs_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    ):
        for info in list_eval_logs(str(model_dir)):
            chosen[Path(info.name).stem] = info.name

    out: dict[str, dict[str, list[str]]] = {}
    for label, log_path in sorted(chosen.items()):
        log = read_eval_log(log_path)
        qmap: dict[str, list[str]] = {}
        for sample in log.samples or []:
            qid = str(sample.id)
            completion = ""
            if sample.output is not None:
                completion = sample.output.completion or ""
            qmap.setdefault(qid, []).append(completion)
        out[label] = qmap
    return out


def load_questions_meta(run_dir: Path) -> dict[str, dict]:
    path = Path(run_dir) / "questions.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def load_families_meta(run_dir: Path) -> dict[str, dict]:
    """{family_id: {prompt, scalar, title, description}} written by generation."""
    path = Path(run_dir) / "families.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _family_pass(
    families_meta: dict[str, dict],
    responses: dict[str, dict[str, list[str]]],
    qmeta: dict[str, dict],
    embeds: dict[str, dict[tuple[str, str], "np.ndarray"]],
    primary_backend: str,
    *,
    run_judge: bool,
    judge_name: str,
    judge_reasoning: Optional[str],
    concurrency: int,
    threshold: float,
    judge_log_path: Optional[Path] = None,
) -> list[dict]:
    """Cross-variant analysis: pool each (model, family) and score the framing split.

    Returns one record per (model, family) with the scalar swing (model-free) and,
    when ``run_judge``, the blind pooled-judge ARI vs the framing labels + the
    variant x judge-group contingency. See ``families.py``.
    """
    # Group variant question ids per family (deterministic variant order).
    variants_by_family: dict[str, list[tuple[str, str]]] = {}
    for qid, meta in qmeta.items():
        fam = meta.get("family")
        if fam and fam in families_meta:
            variants_by_family.setdefault(fam, []).append(
                (meta.get("variant") or qid, qid)
            )
    for fam in variants_by_family:
        variants_by_family[fam].sort()  # by (variant_label, qid)

    # Build every (model, family) pool first, so the pooled judge runs in one batch.
    pools: list[dict] = []
    for model_name in sorted(responses):
        rmap = responses[model_name]
        for fam, pairs in variants_by_family.items():
            if len(pairs) < 2:
                continue
            v2resp = {v: rmap.get(qid, []) for v, qid in pairs}
            if any(len(rs) == 0 for rs in v2resp.values()):
                continue  # a variant failed to generate for this model; skip cleanly
            order = [v for v, _ in pairs]
            seed = families_mod._seed(model_name, fam)
            texts, var_labels, sources = families_mod.build_pool(
                v2resp, order, seed=seed
            )
            pools.append(
                {
                    "model": model_name,
                    "family": fam,
                    "order": order,
                    "v2qid": {v: qid for v, qid in pairs},
                    "v2resp": v2resp,
                    "texts": texts,
                    "var_labels": var_labels,
                    "sources": sources,
                }
            )

    if not pools:
        return []

    fam_judge: dict[tuple[str, str], JudgeResult] = {}
    if run_judge:
        judge_items = [
            (p["model"], p["family"], families_meta[p["family"]]["prompt"], p["texts"])
            for p in pools
        ]
        fam_judge = families_mod.judge_families(
            judge_items,
            judge_name=judge_name,
            reasoning_effort=judge_reasoning,
            concurrency=concurrency,
            log_path=judge_log_path,
            display=_judge_display(),
        )

    out: list[dict] = []
    for p in pools:
        model_name, fam = p["model"], p["family"]
        meta = families_meta[fam]
        kind = meta.get("scalar")
        order, var_labels, sources = p["order"], p["var_labels"], p["sources"]
        n_total = len(p["texts"])

        rec: dict = {
            "model": model_name,
            "family": fam,
            "title": meta.get("title", fam),
            "description": meta.get("description", ""),
            "scalar_kind": kind,
            "variants": [
                {"variant": v, "question_id": p["v2qid"][v], "n": len(p["v2resp"][v])}
                for v in order
            ],
            "n_total": n_total,
            "scalar": None,
            "judge": None,
            "cluster": None,
        }

        if kind:
            per_variant = families_mod.per_variant_scalar(kind, p["v2resp"])
            rec["scalar"] = {
                "kind": kind,
                "per_variant": per_variant,
                "swing": families_mod.scalar_swing(kind, per_variant),
            }

        jr = fam_judge.get((model_name, fam))
        if jr is not None:
            jl = jr.labels(n_total)
            align = families_mod.family_alignment(jl, var_labels, len(order))
            # Persist the exact per-response judge groups, mapped back from
            # pool order — the report can then tint responses even when the
            # judge splits a variant across groups (counts alone cannot).
            per_var = families_mod.groups_by_variant(
                jl, sources, [len(p["v2resp"][v]) for v in order]
            )
            for vi, vrec in enumerate(rec["variants"]):
                vrec["groups"] = per_var[vi]
            rec["judge"] = {
                **align,
                "contradiction": jr.contradiction,
                "rationale": jr.rationale,
                "flags": jr.flags,
                "parse_ok": jr.parse_ok,
            }

        # Model-free cross-check: cluster the pooled embeddings, ARI vs framing.
        emb_pool = []
        ok = True
        for vi, wi in sources:
            qid = p["v2qid"][order[vi]]
            arr = embeds.get(primary_backend, {}).get((model_name, qid))
            if arr is None or wi >= len(arr):
                ok = False
                break
            emb_pool.append(arr[wi])
        if ok and emb_pool:
            mat = np.vstack(emb_pool)
            clabels = cluster_mod.cluster_responses(mat, threshold=threshold)
            cagr = cluster_mod.agreement(clabels, var_labels)
            rec["cluster"] = {
                "ari": cagr["ari"],
                "nmi": cagr["nmi"],
                "n_clusters": len(set(clabels)),
            }

        out.append(rec)
    return out


def _embed_all(
    bundles: list[tuple[str, str, list[str]]],
    backends: list[str],
    local_model: str,
    *,
    cache_dir: Optional[Path] = None,
    refresh: bool = False,
) -> dict[str, dict[tuple[str, str], np.ndarray]]:
    """For each backend, embed every bundle's responses (one batched call per backend).

    Embeddings depend only on the (fixed) generations, not the judge, so they are
    cached per backend under ``cache_dir`` keyed by a content hash. Repeated judge
    runs on the same generations reuse the cache (cheap, and keeps reports featured).
    """
    import hashlib

    flat_texts: list[str] = []
    spans: list[tuple[tuple[str, str], int, int]] = []
    for model_name, qid, responses in bundles:
        start = len(flat_texts)
        flat_texts.extend(responses)
        spans.append(((model_name, qid), start, len(flat_texts)))
    content_hash = hashlib.sha256("\x1f".join(flat_texts).encode("utf-8")).hexdigest()[
        :16
    ]

    by_backend: dict[str, dict[tuple[str, str], np.ndarray]] = {}
    for backend in backends:
        vecs: Optional[np.ndarray] = None
        cache_file = (Path(cache_dir) / f"emb_{backend}.npz") if cache_dir else None
        if cache_file and cache_file.exists() and not refresh:
            try:
                d = np.load(cache_file)
                if str(d["hash"].item()) == content_hash and d["mat"].shape[0] == len(
                    flat_texts
                ):
                    vecs = d["mat"]
            except Exception:
                vecs = None
        if vecs is None:
            embedder = get_embedder(backend, local_model=local_model)
            vecs = (
                embedder.embed(flat_texts)
                if flat_texts
                else np.zeros((0, 0), dtype=np.float32)
            )
            if cache_file is not None and len(flat_texts):
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache_file, mat=vecs, hash=np.array(content_hash))
        per_bundle: dict[tuple[str, str], np.ndarray] = {}
        for key, start, end in spans:
            per_bundle[key] = vecs[start:end]
        by_backend[backend] = per_bundle
    return by_backend


def analyze(
    run_dir: Path,
    *,
    backends: Optional[list[str]] = None,
    judge_name: str = DEFAULT_JUDGE,
    judge_reasoning: Optional[str] = DEFAULT_JUDGE_REASONING,
    threshold: float = cluster_mod.DEFAULT_THRESHOLD,
    local_model: str = DEFAULT_LOCAL_MODEL,
    concurrency: int = 6,
    run_judge: bool = True,
    judge_run: Optional[str] = None,
    refresh_embeddings: bool = False,
    models: Optional[list[str]] = None,
    out_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    progress_label: Optional[str] = None,
) -> dict:
    """Judge + cluster a run's generations into ``analysis.json``.

    ``judge_run`` isolates a repeated judge pass under ``judge_runs/<label>/`` so
    re-judging the same generations never clobbers a prior run (the input to the
    cross-run consistency stats). The embedding cache lives at ``run_dir/cache`` and
    is shared across judge reps, so re-judges reuse embeddings for free.

    ``models`` restricts the pass to a subset of the generated models (the store
    judges each model into its own cached fragment). ``out_dir`` overrides where
    ``analysis.json`` + the judge logs are written (default: ``run_dir`` or
    ``run_dir/judge_runs/<judge_run>``) and ``cache_dir`` overrides the embedding
    cache location (default: ``run_dir/cache``, shared across judge reps — the
    store passes a per-gen subdir so fragments don't clobber that shared cache).
    ``progress_label`` prefixes the phase announcement (fragment runs say which
    model is being judged).
    """
    run_dir = Path(run_dir)
    # None = default (local); [] = embeddings explicitly disabled (judge-only).
    backends = ["openai-3-small"] if backends is None else list(backends)
    primary_backend = backends[0] if backends else None
    if out_dir is None:
        out_dir = (run_dir / "judge_runs" / judge_run) if judge_run else run_dir
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir is not None else run_dir / "cache"
    if progress_label:
        print(f"=== {progress_label} ===", flush=True)

    responses = load_responses(run_dir)
    if models is not None:
        missing = sorted(set(models) - set(responses))
        if missing:
            raise KeyError(f"no generation logs for model(s) {missing} in {run_dir}")
        responses = {m: responses[m] for m in models}
    qmeta = load_questions_meta(run_dir)

    # Flatten to bundles, preserving question/group order where known.
    bundles: list[tuple[str, str, list[str]]] = []
    for model_name in sorted(responses):
        for qid, resps in responses[model_name].items():
            bundles.append((model_name, qid, resps))

    # --- judge (cross-sample) — one Inspect eval, bundles as samples ---
    judge_results: dict[tuple[str, str], JudgeResult] = {}
    or_before = or_after = None  # OpenRouter usage snapshots (judge spend delta)
    if run_judge:
        # Family variants are judged ACROSS variants by the pooled family judge;
        # a within-variant verdict is never displayed anywhere, so skip the
        # per-question judge for them (~40% of judge calls on the default sweep).
        judge_items = [
            ((m, qid), qmeta.get(qid, {}).get("prompt", qid), resps)
            for (m, qid, resps) in bundles
            if not qmeta.get(qid, {}).get("family")
        ]
        or_before = cost_mod.openrouter_usage()
        judge_results, _ = run_judge_eval(
            judge_items,
            judge_name=judge_name,
            reasoning_effort=judge_reasoning,
            max_connections=concurrency,
            log_path=out_dir / "judge_logs" / "responses",
            display=_judge_display(),
        )
        or_after = cost_mod.openrouter_usage()

    # --- embeddings + clustering (cached per backend; reused across judge runs) ---
    embeds = _embed_all(
        bundles,
        backends,
        local_model,
        cache_dir=cache_dir,
        refresh=refresh_embeddings,
    )

    results = []
    for model_name, qid, resps in bundles:
        n = len(resps)
        jr = judge_results.get((model_name, qid))
        clusters_out: dict[str, dict] = {}
        agreement_out: dict[str, dict] = {}
        for backend in backends:
            emb = embeds[backend][(model_name, qid)]
            labels = cluster_mod.cluster_responses(emb, threshold=threshold)
            clusters_out[backend] = {
                "labels": labels,
                "n_clusters": len(set(labels)) if labels else 0,
            }
            if jr is not None and n >= 1:
                agreement_out[backend] = cluster_mod.agreement(jr.labels(n), labels)

        primary_emb = (
            embeds[primary_backend][(model_name, qid)] if primary_backend else None
        )
        m = metrics_mod.variance_metrics(
            resps,
            embeddings=primary_emb,
            n_judge_groups=(jr.n_groups if jr is not None else None),
            n_clusters=(
                clusters_out[primary_backend]["n_clusters"] if primary_backend else None
            ),
        )
        # Entropy of the judge's grouping: -sum p_k log p_k over group frequencies.
        if jr is not None:
            m["group_entropy"] = metrics_mod.group_entropy(jr.labels(n))
        if primary_backend:
            m["cluster_entropy"] = metrics_mod.group_entropy(
                clusters_out[primary_backend]["labels"]
            )

        results.append(
            {
                "model": model_name,
                "question_id": qid,
                "group": qmeta.get(qid, {}).get("group", ""),
                "responses": resps,
                "judge": jr.to_dict() if jr is not None else None,
                "judge_labels": jr.labels(n) if jr is not None else None,
                "clusters": clusters_out,
                "agreement": agreement_out,
                "metrics": m,
            }
        )

    # --- cross-variant family analysis (framing-invariance) ---
    families_meta = load_families_meta(run_dir)
    family_results = _family_pass(
        families_meta,
        responses,
        qmeta,
        embeds,
        primary_backend,
        run_judge=run_judge,
        judge_name=judge_name,
        judge_reasoning=judge_reasoning,
        concurrency=concurrency,
        threshold=threshold,
        judge_log_path=out_dir / "judge_logs" / "families",
    )

    # --- cost: token×price estimate reconciled with the OpenRouter delta ---
    cost_record: dict = {}
    if run_judge:
        jt_in = sum(r.input_tokens for r in judge_results.values())
        jt_out = sum(r.output_tokens for r in judge_results.values())
        od = (
            or_after - or_before
            if (or_before is not None and or_after is not None)
            else None
        )
        cost_record["judge"] = {
            "model": judge_name,
            "in_tok": jt_in,
            "out_tok": jt_out,
            "est_dollars": cost_mod.judge_dollars(jt_in, jt_out),
            "openrouter_delta": od,
        }
    # Generation cost belongs to the run itself, not to each judge rep.
    if judge_run is None:
        gen = cost_mod.generation_usage(run_dir)
        if gen:
            cost_record["generation"] = gen

    # Display names come from the manifest (write_manifest persists spec.display);
    # runs that predate it fall back to the short name.
    run_config: dict = {}
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        run_config = json.loads(cfg_path.read_text())
    manifest_models = run_config.get("models") or {}
    model_display = {
        name: (manifest_models.get(name) or {}).get("display") or name
        for name in sorted(responses)
    }

    out = {
        "run_dir": str(run_dir),
        "judge_run": judge_run,
        "backends": backends,
        "primary_backend": primary_backend,
        "judge": judge_name if run_judge else None,
        "judge_reasoning": judge_reasoning if run_judge else None,
        "threshold": threshold,
        "models": sorted(responses),
        "model_display": model_display,
        "config": {
            k: run_config.get(k)
            for k in ("models", "n", "temperature", "max_tokens", "judge")
            if k in run_config
        },
        "questions": qmeta,
        "families_meta": families_meta,
        "results": results,
        "families": family_results,
        "cost": cost_record,
    }
    (out_dir / "analysis.json").write_text(json.dumps(out, indent=2))
    if cost_record:
        cost_mod.write_cost(out_dir / "cost.json", cost_record)

    from .run_meta import JUDGE_META, build_meta, write_meta_safe

    write_meta_safe(
        out_dir,
        build_meta(
            "judge_pass",
            label=judge_run or "default",
            parent_run=run_dir.name,
            judge_model=judge_name if run_judge else None,
            judge_reasoning=judge_reasoning if run_judge else None,
            threshold=threshold,
            backends=backends,
            n_bundles=len(results),
        ),
        filename=JUDGE_META,
    )
    if family_results:
        from .families_report import build_families_report

        build_families_report(out, out_dir / "families_report.html")
    return out
