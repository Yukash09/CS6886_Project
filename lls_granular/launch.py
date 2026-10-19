"""Run a named experiment on all available GPUs: python -m lls_granular.launch main

One worker per GPU pulls jobs from a shared queue. Finished runs are skipped and
interrupted ones resume (handled by lls_granular.run), so re-running is always safe.
"""
import argparse
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import torch

from . import logs
from .exps import EXPS, Job, jobs
from .run import args as run_args, partition
from .track import RunId
from .units import ARCHS

log = logging.getLogger(__name__)

Runner = Callable[[Job, str], int]


def out_dir() -> Path:
    return Path(os.environ.get("LLS_OUT", "./runs")).expanduser()


def devices(n: int | None = None) -> list[str]:
    """GPU ids to use; [''] means a single CPU worker."""
    k = torch.cuda.device_count() if n is None else n
    return [str(i) for i in range(k)] or [""]


ECHO = ("lls_granular.train: epoch", "WARNING", "ERROR", "Traceback", "Error", "already complete", "resumed")


def subprocess_runner(extra: list[str]) -> Runner:
    """Full output goes to runs/logs/<job>.log; epoch lines, warnings and errors are echoed live."""
    def run(j: Job, dev: str) -> int:
        f = out_dir() / "logs" / f"{j.name}.log"
        f.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if dev:
            env["CUDA_VISIBLE_DEVICES"] = dev
        cmd = [sys.executable, "-m", "lls_granular.run", *j.argv(), *extra]
        tag = f"[gpu {dev or 'cpu'}] {j.name}"
        with f.open("a") as fh, subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                                 stderr=subprocess.STDOUT, text=True) as p:
            assert p.stdout is not None
            for line in p.stdout:
                fh.write(line)
                if any(k in line for k in ECHO):
                    print(f"{tag} | {line.rstrip()}", flush=True)
        return p.returncode
    return run


def schedule(js: list[Job], devs: list[str], runner: Runner) -> dict[str, int]:
    """Run every job once; each device runs one job at a time. Returns job name -> exit code."""
    q: queue.Queue[Job] = queue.Queue()
    for j in js:
        q.put(j)
    codes: dict[str, int] = {}
    lock = threading.Lock()

    def worker(dev: str) -> None:
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            t = time.time()
            log.info("[gpu %s] start %s", dev or "cpu", j.name)
            rc = runner(j, dev)
            with lock:
                codes[j.name] = rc
                done = len(codes)
            log.info("[gpu %s] %s %s in %.1f min (%d/%d done)", dev or "cpu", "done" if rc == 0 else
                     f"FAILED rc={rc}", j.name, (time.time() - t) / 60, done, len(js))

    ts = [threading.Thread(target=worker, args=(d,)) for d in devs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return codes


def run_dir(j: Job) -> Path:
    """The folder lls_granular.run writes for this job (same naming as RunId)."""
    a = run_args(j.argv())
    shapes = [u.shape for u in ARCHS[a.arch]()]
    rid = RunId(a.arch, a.dataset, partition(a, shapes), a.epochs, a.seed)
    return out_dir() / rid.name


def report(js: list[Job], codes: dict[str, int]) -> None:
    print(f"\n{'job':<52} {'status':<8} {'test@best':>9} {'best_ep':>7} {'mem_MB':>7} {'s/ep':>6}")
    for j in js:
        f = run_dir(j) / "summary.json"
        s = json.loads(f.read_text()) if f.exists() else {}
        st = "ok" if codes.get(j.name) == 0 else f"rc={codes.get(j.name)}"
        num = lambda k, fmt: format(s[k], fmt) if k in s else "-"  # noqa: E731
        print(f"{j.name:<52} {st:<8} {num('test_at_best_val', '.4f'):>9} {num('best/epoch', 'd'):>7} "
              f"{num('sys/peak_mem_mb', '.0f'):>7} {num('time/train_s_mean', '.1f'):>6}")


def restore(src: str) -> None:
    """Seed the runs folder from a previous session's output (e.g. a Kaggle input mount)."""
    found = sorted(Path("/").glob(src.lstrip("/"))) if any(c in src for c in "*?[") else [Path(src)]
    found = [p for p in found if p.is_dir()]
    if not found or out_dir().exists():
        log.info("restore: nothing to do (found=%s, %s exists=%s)", found, out_dir(), out_dir().exists())
        return
    shutil.copytree(found[0], out_dir())
    log.info("restored %s -> %s", found[0], out_dir())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run one or more named experiments.", epilog="Extra args after -- go to every run.")
    p.add_argument("exp", nargs="+", choices=sorted(EXPS), help="one or more; all runs share one GPU queue")
    p.add_argument("--seeds", type=int, nargs="*", help="repeat every run over these seeds")
    p.add_argument("--gpus", type=int, help="number of GPUs to use (default: all)")
    p.add_argument("--restore", help="copy a previous runs/ folder (path or glob) before starting")
    p.add_argument("--dry-run", action="store_true", help="list the runs and exit")
    argv = sys.argv[1:] if argv is None else argv
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    a = p.parse_args(argv[:argv.index("--")] if "--" in argv else argv)

    js, devs = [j for e in a.exp for j in jobs(e, a.seeds)], devices(a.gpus)
    log.info("experiments %s: %d runs on %s, out %s", " ".join(a.exp), len(js), devs, out_dir())
    if a.dry_run:
        for j in js:
            print(j.name, " ".join(j.argv() + extra))
        return
    if a.restore:
        restore(a.restore)
    codes = schedule(js, devs, subprocess_runner(extra))
    report(js, codes)
    if any(codes.values()):
        sys.exit(1)


if __name__ == "__main__":
    logs.setup()
    main()
