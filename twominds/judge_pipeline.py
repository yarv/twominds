"""Background per-model judge fragments, overlapped with generation.

Inspect forbids concurrent evals inside one process, so the judge for a model
that has finished generating runs in a **subprocess** while the remaining
models are still generating. The single multi-task generation eval is
unchanged; an Inspect ``TaskEnd`` hook writes the finished model's log into
its store gen dir (atomic replace) and enqueues a fragment worker — the same
per-model judge ``assemble_run`` would run afterwards, just earlier.

Pre-warming is purely opportunistic: a failed or interrupted worker leaves no
fragment, and ``assemble_run`` judges that model inline exactly as before.
Worker output goes to ``<gen_dir>/judge/prewarm-<model>.log``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from inspect_ai.hooks import Hooks, TaskEnd, hooks

_ACTIVE: Optional["FragmentPrewarmer"] = None

TASK_PREFIX = "twominds:"
MAX_WORKERS = 2  # concurrent judge subprocesses (each has its own judge pool)


class FragmentPrewarmer:
    """Holds the per-model job specs and the worker/subprocess lifecycle."""

    def __init__(self, jobs: dict[str, dict], judge_cfg: dict):
        # jobs: model -> {"gen_dir": path, "log_dir": path}; judge_cfg mirrors
        # the keyword arguments of the fragment analyze in store.assemble_run
        self.jobs = {m: {k: str(v) for k, v in j.items()} for m, j in jobs.items()}
        self.judge_cfg = judge_cfg
        self.written: set[str] = set()  # canonical logs written by the hook
        self.prewarmed: set[str] = set()  # fragments completed successfully
        self._sema = threading.Semaphore(MAX_WORKERS)
        self._threads: list[threading.Thread] = []
        self._procs: list[subprocess.Popen] = []
        self._lock = threading.Lock()

    def log_written(self, model: str) -> bool:
        return model in self.written

    def submit(self, model: str, log) -> None:
        """Called from the TaskEnd hook. Writes the canonical eval log NOW
        (atomic replace, so run_generation can skip its own write and no
        reader ever sees a partial file), then judges in a worker thread."""
        job = self.jobs.get(model)
        if job is None or model in self.written:
            return
        from inspect_ai.log import write_eval_log

        safe = model.replace("/", "_")
        log_dir = Path(job["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        for fmt, suffix in (("eval", ".eval"), ("json", ".json")):
            tmp = log_dir / f".{safe}{suffix}.tmp"
            write_eval_log(log, str(tmp), format=fmt)
            tmp.replace(log_dir / f"{safe}{suffix}")
        self.written.add(model)

        t = threading.Thread(target=self._run_worker, args=(model,), daemon=True)
        t.start()
        self._threads.append(t)

    def _run_worker(self, model: str) -> None:
        job = self.jobs[model]
        payload = {"model": model, "gen_dir": job["gen_dir"], **self.judge_cfg}
        out_log = Path(job["gen_dir"]) / "judge" / f"prewarm-{model.replace('/', '_')}.log"
        out_log.parent.mkdir(parents=True, exist_ok=True)
        with self._sema, open(out_log, "w") as fh:
            proc = subprocess.Popen(
                [sys.executable, "-m", "twominds.judge_pipeline", json.dumps(payload)],
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
            with self._lock:
                self._procs.append(proc)
            if proc.wait() == 0:
                self.prewarmed.add(model)

    def drain(self, *, cancel: bool = False) -> set[str]:
        """Wait for (or cancel) outstanding workers; returns models prewarmed."""
        if cancel:
            with self._lock:
                for p in self._procs:
                    if p.poll() is None:
                        p.terminate()
        for t in self._threads:
            t.join(timeout=None if not cancel else 10)
        return set(self.prewarmed)


def activate(jobs: dict[str, dict], judge_cfg: dict) -> FragmentPrewarmer:
    global _ACTIVE
    _ACTIVE = FragmentPrewarmer(jobs, judge_cfg)
    return _ACTIVE


def deactivate(*, cancel: bool = False) -> set[str]:
    global _ACTIVE
    pw, _ACTIVE = _ACTIVE, None
    return pw.drain(cancel=cancel) if pw is not None else set()


@hooks(
    name="twominds_judge_prewarm",
    description="judge each model's fragment in the background as its generation finishes",
)
class _PrewarmHooks(Hooks):
    async def on_task_end(self, data: TaskEnd) -> None:
        pw = _ACTIVE
        if pw is None:  # inert outside an activated store run (incl. judge evals)
            return
        task = data.log.eval.task or ""
        if TASK_PREFIX not in task:
            return
        model = task.rsplit(TASK_PREFIX, 1)[-1]
        if model in pw.jobs and data.log.status == "success":
            pw.submit(model, data.log)


def _worker_main(payload: dict) -> None:
    """Subprocess entry: judge one model's fragment (same call as assemble_run)."""
    from .analyze import analyze
    from .store import fragment_dir, write_fragment_meta

    gd = Path(payload["gen_dir"])
    model = payload["model"]
    analyze(
        gd,
        backends=list(payload["backends"]),
        judge_name=payload["judge_name"],
        judge_reasoning=payload["judge_reasoning"],
        threshold=payload["threshold"],
        local_model=payload["local_model"],
        concurrency=payload["concurrency"],
        run_judge=payload["run_judge"],
        models=[model],
        out_dir=fragment_dir(gd, payload["judge_key"]),
        cache_dir=gd / "cache",
        progress_label=f"prewarm-judging {model}",
    )
    write_fragment_meta(gd, model, payload["judge_key"])


if __name__ == "__main__":
    _worker_main(json.loads(sys.argv[1]))
