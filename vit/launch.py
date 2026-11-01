import argparse
import logging
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import torch

from lls_granular import logs

from .exps import EXPS, Job, jobs

log = logging.getLogger(__name__)


def devices(count: int | None = None) -> list[str]:
    n = torch.cuda.device_count() if count is None else count
    return [str(i) for i in range(n)] or [""]


def out_dir() -> Path:
    return Path(os.environ.get("LLS_VIT_OUT") or os.environ.get("LLS_OUT") or "./runs").expanduser()


def run_job(job: Job, device: str, extra: list[str]) -> int:
    path = out_dir() / "logs" / f"{job.name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if device:
        env["CUDA_VISIBLE_DEVICES"] = device
    command = [sys.executable, "-m", "vit.run", *job.argv(), *extra]
    tag = f"[gpu {device or 'cpu'}] {job.name}"
    with path.open("a") as stream, subprocess.Popen(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            if any(key in line for key in ("epoch", "WARNING", "ERROR", "Traceback")):
                print(f"{tag} | {line.rstrip()}", flush=True)
    return process.returncode


def schedule(all_jobs: list[Job], all_devices: list[str], extra: list[str]) -> dict[str, int]:
    pending: queue.Queue[Job] = queue.Queue()
    for job in all_jobs:
        pending.put(job)
    codes: dict[str, int] = {}
    lock = threading.Lock()

    def worker(device: str) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            log.info("[gpu %s] start %s", device or "cpu", job.name)
            code = run_job(job, device, extra)
            with lock:
                codes[job.name] = code
            log.info("[gpu %s] %s %s", device or "cpu", "done" if code == 0 else "FAILED", job.name)

    threads = [threading.Thread(target=worker, args=(device,)) for device in all_devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return codes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run ViT LLS experiments.")
    parser.add_argument("exp", nargs="+", choices=sorted(EXPS))
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--dry-run", action="store_true")
    raw = sys.argv[1:] if argv is None else argv
    extra = raw[raw.index("--") + 1:] if "--" in raw else []
    args = parser.parse_args(raw[:raw.index("--")] if "--" in raw else raw)
    all_jobs = [job for name in args.exp for job in jobs(name, args.seeds)]
    if args.dry_run:
        for job in all_jobs:
            print(job.name, " ".join(job.argv() + extra))
        return
    codes = schedule(all_jobs, devices(args.gpus), extra)
    if any(codes.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    logs.setup()
    main()

