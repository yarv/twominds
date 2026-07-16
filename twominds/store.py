"""Per-model result store: reuse generations + judge verdicts across runs.

Generations are expensive and depend only on (model, questions, sampling
config) — not on which run asked for them. The store keeps them per model:

    results/twominds/models/<model_name>/
      model.json                 identity guard: {inspect_model, reasoning_effort, display}
      gens/<gen_key>/            one generation of this model over one question set
        gen_meta.json            key inputs + status:"complete" (written last)
        questions.json           \\
        families.json             } single-model mini-run manifest — the gen dir
        run_config.json          /  is a valid run for analyze()/load_responses()
        logs/<model_name>/<model_name>.{eval,json}
        cache/emb_<backend>.npz  embedding cache (content-hash keyed, self-invalidating)
        judge/<judge_key>/       per-model judge fragment for one judge config
          analysis.json
          judge_logs/            the fragment's judge eval logs (.eval + .json)
          fragment_meta.json     source_log staleness guard

Run dirs symlink each fragment's ``judge_logs`` at ``judge_logs/<model>`` so the
rep1 judge eval logs stay discoverable from the run (rep2+ passes write theirs
under ``judge_runs/<label>/judge_logs`` directly).

``gen_key`` hashes the question contents + sampling config, so editing a prompt
or changing n/temperature/max_tokens produces a new key rather than silently
reusing stale generations. ``judge_key`` hashes the judge config including the
judge prompt template, so a prompt edit invalidates cached verdicts.

Run dirs (results/twominds/<timestamp>/) stay the aggregation view: their
``logs/<model>`` entries are relative symlinks into the store, so everything
that walks a run's logs (repeat judge reps, consistency, cost roll-up) works
unchanged. Old runs that predate the store are untouched.

Cross-model vs per-model: everything analyze() computes is per-model separable
(judge + clustering per bundle, families per (model, family)); the cross-model
part of a run is pure concatenation, done by ``merge.merge_analysis_dicts`` in
:func:`assemble_run`. Nothing is lost by caching per-model fragments.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from .models import ModelSpec
from .questions import Question

STORE_DIRNAME = "models"
_FRAGMENT_META = "fragment_meta.json"  # analyze() owns judge_meta.json in the same dir


def store_root(results_root: Path | str = "results/twominds") -> Path:
    return Path(results_root) / STORE_DIRNAME


def model_dir(spec: ModelSpec, root: Path) -> Path:
    return Path(root) / spec.name


def _hash8(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:8]


def compute_gen_key(
    questions: list[Question], *, n: int, temperature: float, max_tokens: int
) -> str:
    """Content key for one generation: question texts + sampling config.

    Judge settings and operational knobs (concurrency, timeouts, display) are
    deliberately excluded — they do not change what the model was asked.
    A cached n=20 generation is NOT reused for an n=10 request (different key):
    slicing supersets is a complication we skip on purpose.
    """
    qmeta = {
        q.id: {
            "prompt": q.prompt,
            "system": q.system,
            "group": q.group,
            "bucket": q.bucket,
            "family": q.family,
            "variant": q.variant,
        }
        for q in questions
    }
    fams: dict = {}
    referenced = {q.family for q in questions if q.family}
    if referenced:
        from .questions import load_families

        fams = {
            fid: {
                "prompt": f.prompt,
                "scalar": f.scalar,
                "title": f.title,
                "description": f.description,
            }
            for fid, f in load_families().items()
            if fid in referenced
        }
    h = _hash8(
        {
            "questions": qmeta,
            "families": fams,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    )
    return f"{h}_q{len(questions)}_n{n}"


def gen_dir(spec: ModelSpec, key: str, root: Path) -> Path:
    return model_dir(spec, root) / "gens" / key


def identity_conflict(spec: ModelSpec, root: Path) -> Optional[dict]:
    """The existing model.json identity iff it names a DIFFERENT model than
    ``spec`` (else None). Read-only — safe for dry runs."""
    path = model_dir(spec, root) / "model.json"
    if not path.exists():
        return None
    prev = json.loads(path.read_text())
    same = (
        prev.get("inspect_model") == spec.inspect_model
        and prev.get("reasoning_effort") == spec.reasoning_effort
    )
    return None if same else prev


def write_identity(spec: ModelSpec, root: Path) -> None:
    """Pin this store name to the spec's identity (first write wins)."""
    d = model_dir(spec, root)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "model.json"
    if not path.exists():
        path.write_text(
            json.dumps(
                {
                    "inspect_model": spec.inspect_model,
                    "reasoning_effort": spec.reasoning_effort,
                    "display": spec.display or spec.name,
                },
                indent=2,
            )
        )


def check_model_identity(spec: ModelSpec, root: Path) -> None:
    """Guard the store dir against short-name collisions across invocations.

    The store keys models by short name; two different models resolving to the
    same short name (e.g. the same last path segment via two providers, in
    different invocations) would silently share cached generations. model.json
    pins the identity; on a mismatch this raises — the CLI catches it and
    auto-qualifies the colliding name with more path segments (mirroring the
    in-batch disambiguation in ``resolve_models``).
    """
    prev = identity_conflict(spec, root)
    if prev is not None:
        raise ValueError(
            f"store name collision: '{spec.name}' already maps to "
            f"{prev.get('inspect_model')} (reasoning="
            f"{prev.get('reasoning_effort')}), but this invocation asks for "
            f"{spec.inspect_model} (reasoning={spec.reasoning_effort})."
        )
    write_identity(spec, root)


def find_generation(spec: ModelSpec, key: str, root: Path) -> Optional[Path]:
    """The gen dir iff it holds a complete generation (else None)."""
    d = gen_dir(spec, key, root)
    meta = d / "gen_meta.json"
    if not meta.exists():
        return None
    try:
        if json.loads(meta.read_text()).get("status") != "complete":
            return None
    except Exception:
        return None
    if not any((d / "logs" / spec.name).glob("*.eval")):
        return None
    return d


def prepare_generation(
    spec: ModelSpec,
    questions: list[Question],
    key: str,
    root: Path,
    *,
    n: int,
    temperature: float,
    max_tokens: int,
    judge: str,
) -> Path:
    """Create the gen dir with its single-model mini-run manifest."""
    from . import generate as generate_mod

    d = gen_dir(spec, key, root)
    generate_mod.write_manifest(
        d,
        [spec],
        questions,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        judge=judge,
    )
    return d


def mark_complete(gen_path: Path, *, key: str) -> None:
    (Path(gen_path) / "gen_meta.json").write_text(
        json.dumps({"status": "complete", "gen_key": key}, indent=2)
    )


def clear_generation(spec: ModelSpec, key: str, root: Path) -> None:
    """Remove a gen dir wholesale — its logs, embedding cache, and cached judge
    fragments all describe the generation being discarded. Used by --rerun /
    --rerun-model: log filenames are fixed (``<model>.eval``), so a rerun must
    rebuild the dir rather than rely on append-invalidation. Old run dirs that
    symlinked this generation will read the fresh one after it regenerates."""
    import shutil

    shutil.rmtree(gen_dir(spec, key, root), ignore_errors=True)


def latest_log_name(gen_path: Path, model_name: str) -> Optional[str]:
    logs = sorted((Path(gen_path) / "logs" / model_name).glob("*.eval"))
    return logs[-1].name if logs else None


# --------------------------------------------------------------------------- #
# judge fragments
# --------------------------------------------------------------------------- #
def compute_judge_key(
    *,
    judge_name: str,
    judge_reasoning: Optional[str],
    threshold: float,
    backends: list[str],
    local_model: str,
    run_judge: bool = True,
) -> str:
    """Key for one judge/clustering config over a fixed generation."""
    from . import judge as judge_mod

    payload = {
        "judge_name": judge_name if run_judge else None,
        "judge_reasoning": judge_reasoning if run_judge else None,
        "threshold": threshold,
        "backends": list(backends),
        "local_model": local_model,
        "prompt_hash": judge_mod.PROMPT_HASH if run_judge else None,
    }
    slug = judge_name.rsplit("/", 1)[-1] if run_judge else "nojudge"
    return f"{slug}_{_hash8(payload)}"


def fragment_dir(gen_path: Path, judge_key: str) -> Path:
    return Path(gen_path) / "judge" / judge_key


def find_fragment(gen_path: Path, model_name: str, judge_key: str) -> Optional[dict]:
    """The cached per-model analysis fragment, iff fresh (else None).

    Fresh = its recorded source .eval is still the generation's latest, so a
    ``--rerun`` (which appends a new .eval) invalidates fragments automatically.
    """
    fd = fragment_dir(gen_path, judge_key)
    ap = fd / "analysis.json"
    mp = fd / _FRAGMENT_META
    if not (ap.exists() and mp.exists()):
        return None
    try:
        meta = json.loads(mp.read_text())
        if meta.get("source_log") != latest_log_name(gen_path, model_name):
            return None
        return json.loads(ap.read_text())
    except Exception:
        return None


def write_fragment_meta(gen_path: Path, model_name: str, judge_key: str) -> None:
    fd = fragment_dir(gen_path, judge_key)
    fd.mkdir(parents=True, exist_ok=True)
    (fd / _FRAGMENT_META).write_text(
        json.dumps(
            {
                "judge_key": judge_key,
                "source_log": latest_log_name(gen_path, model_name),
            },
            indent=2,
        )
    )


# --------------------------------------------------------------------------- #
# run assembly
# --------------------------------------------------------------------------- #
def link_into_run(run_dir: Path, spec: ModelSpec, gen_path: Path) -> None:
    """Symlink ``run_dir/logs/<name>`` at the store's log dir (relative, so the
    results tree stays relocatable). Everything that walks a run's logs — repeat
    judge reps, consistency, cost — reads through the link unchanged."""
    dest = Path(run_dir) / "logs" / spec.name
    src = Path(gen_path) / "logs" / spec.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        dest.unlink()
    elif dest.exists():
        raise FileExistsError(
            f"{dest} already exists and is not a symlink; refusing to replace "
            "real generation logs with a store link"
        )
    dest.symlink_to(os.path.relpath(src, dest.parent), target_is_directory=True)


def _link_judge_logs(run_dir: Path, model_name: str, frag_dir: Path) -> None:
    """Symlink ``run_dir/judge_logs/<model>`` at the fragment's judge eval logs
    so rep1's judge calls stay discoverable from the run dir. No-op when the
    fragment has no judge logs (--no-judge) or a real (non-link) judge_logs dir
    already exists (a pre-store run)."""
    src = Path(frag_dir) / "judge_logs"
    if not src.is_dir():
        return
    dest = Path(run_dir) / "judge_logs" / model_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        dest.unlink()
    elif dest.exists():
        return
    dest.symlink_to(os.path.relpath(src, dest.parent), target_is_directory=True)


def assemble_run(
    run_dir: Path,
    specs: list[ModelSpec],
    gen_dirs: dict[str, Path],
    *,
    judge_key: str,
    backends: list[str],
    judge_name: str,
    judge_reasoning: Optional[str],
    threshold: float,
    local_model: str,
    concurrency: int,
    run_judge: bool,
    refresh_embeddings: bool = False,
    on_fragment: Optional[callable] = None,
    preseed_cache: bool = False,
    cached_gens: Optional[set] = None,
) -> dict:
    """rep1 for a store-backed run: per-model fragments (cached or judged now),
    merged into the run-level ``analysis.json`` + cost roll-up, plus the
    run-level artefacts the pre-store rep1 produced (``judge_meta.json``
    provenance marker, ``families_report.html`` when families are present).

    Only models without a fresh fragment for ``judge_key`` are judged; the
    cross-model step is pure concatenation (``merge_analysis_dicts``), so cached
    models cost nothing. ``on_fragment(model_name, was_cached)`` is a progress
    hook. ``preseed_cache`` copies the fragments' per-model embedding caches
    into the run-level cache so later whole-run judge reps reuse them.
    """
    from . import analyze as analyze_mod
    from . import cost as cost_mod
    from . import merge as merge_mod

    run_dir = Path(run_dir)
    frags: list[dict] = []
    frag_cached: list[bool] = []
    for spec in specs:
        gd = Path(gen_dirs[spec.name])
        frag = find_fragment(gd, spec.name, judge_key)
        cached = frag is not None
        frag_cached.append(cached)
        if frag is None:
            frag = analyze_mod.analyze(
                gd,
                backends=list(backends),
                judge_name=judge_name,
                judge_reasoning=judge_reasoning,
                threshold=threshold,
                local_model=local_model,
                concurrency=concurrency,
                run_judge=run_judge,
                models=[spec.name],
                out_dir=fragment_dir(gd, judge_key),
                cache_dir=gd / "cache",
                refresh_embeddings=refresh_embeddings,
                progress_label=f"judging {spec.name}",
            )
            write_fragment_meta(gd, spec.name, judge_key)
        _link_judge_logs(run_dir, spec.name, fragment_dir(gd, judge_key))
        if on_fragment is not None:
            on_fragment(spec.name, cached)
        frags.append(frag)

    combined = merge_mod.merge_analysis_dicts(
        frags, run_dir=str(run_dir), source_labels=[s.name for s in specs]
    )

    # Cost roll-up (cached fragments' spend was paid in an earlier invocation
    # but still describes this analysis) + generation usage over linked logs.
    # cached_flags/cached_gens let the summary say what was billed just now.
    cost_record = cost_mod.rollup_fragments(
        frags,
        run_dir,
        judge_name=judge_name,
        run_judge=run_judge,
        cached_flags=frag_cached,
        cached_gens=cached_gens,
    )
    if cost_record:
        combined["cost"] = cost_record
        cost_mod.write_cost(run_dir / "cost.json", cost_record)

    (run_dir / "analysis.json").write_text(json.dumps(combined, indent=2))
    write_run_judge_meta(run_dir, combined)
    build_run_families_report(run_dir, combined)
    if preseed_cache:
        preseed_run_cache(run_dir, frags, backends)
    return combined


def write_run_judge_meta(run_dir: Path, combined: dict) -> None:
    """Run-level judge_meta.json provenance marker for the default (rep1) pass
    — what analyze(run_dir) used to write before fragments moved the analyze
    out_dir into the store. run_registry reads it for rich judge-pass records."""
    from .run_meta import JUDGE_META, build_meta, write_meta_safe

    run_dir = Path(run_dir)
    write_meta_safe(
        run_dir,
        build_meta(
            "judge_run",
            label="default",
            parent_run=run_dir.name,
            judge_model=combined.get("judge"),
            judge_reasoning=combined.get("judge_reasoning"),
            threshold=combined.get("threshold"),
            backends=combined.get("backends"),
            n_bundles=len(combined.get("results") or []),
        ),
        filename=JUDGE_META,
    )


def build_run_families_report(run_dir: Path, combined: dict) -> None:
    """families_report.html next to the run's report.html when the merged
    analysis carries framing families — report.html links to it, and the
    per-fragment copies live in the store where the link can't reach."""
    if not combined.get("families"):
        return
    from .families_report import build_families_report

    build_families_report(combined, Path(run_dir) / "families_report.html")


def content_hash(texts: list[str]) -> str:
    """The embedding-cache content key — MUST match analyze._embed_all."""
    return hashlib.sha256("\x1f".join(texts).encode("utf-8")).hexdigest()[:16]


def preseed_run_cache(run_dir: Path, frags: list[dict], backends: list[str]) -> None:
    """Seed run_dir/cache from the fragments' per-model embedding caches.

    Whole-run judge reps (rep2..N) read run_dir/cache, which is empty on a
    fresh store run because rep1 embedded into the per-model gen-dir caches.
    The run-level flat order is models sorted by name, each model's bundles in
    its own order — exactly the concatenation of the single-model fragments —
    so stacking the fragment matrices reproduces what analyze(run_dir) would
    compute. Best-effort: any mismatch (missing/stale fragment cache) skips
    that backend and the rep simply recomputes.
    """
    import numpy as np

    ordered = sorted(frags, key=lambda a: (a.get("models") or ["?"])[0])
    texts_per_frag = [
        [t for r in (a.get("results") or []) for t in (r.get("responses") or [])]
        for a in ordered
    ]
    run_hash = content_hash([t for ts in texts_per_frag for t in ts])
    out_dir = Path(run_dir) / "cache"
    for backend in backends:
        mats = []
        for a, texts in zip(ordered, texts_per_frag):
            try:
                d = np.load(Path(a["run_dir"]) / "cache" / f"emb_{backend}.npz")
                if str(d["hash"].item()) != content_hash(texts) or d["mat"].shape[
                    0
                ] != len(texts):
                    mats = None
                    break
                mats.append(d["mat"])
            except Exception:
                mats = None
                break
        if not mats:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / f"emb_{backend}.npz", mat=np.vstack(mats), hash=np.array(run_hash)
        )
