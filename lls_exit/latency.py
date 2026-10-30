"""Measured latency of early exit (MACs are not latency): static per-exit, full network, and a real dynamic-exit loop.

python -m lls_exit.latency --exp mobilenet300        (every non-BP run of an experiment; needs ckpt.pt + exits.pt)
python -m lls_exit.latency --run-dir runs/<run>      (a single run)
The operating point is computed here from the run's exits.pt (the sweep's δ pick: --policy any, single-exit pool,
fitted T, cls final), so no local CSVs are needed. On every device config (gpu, cpu with 1 thread, cpu with all
threads) it times each exit's prefix (blocks + its head) and the full network at batch 1 and a large batch, then runs
the dynamic loop at batch 1 over --n test samples: blocks in order, the head at every checked exit, the policy
decision on the host, stop when it fires. Its exits are checked against the offline policy on the same samples.
Writes latency.json in the run folder; summary keys latency/* and a `latency-<run>` artifact go to the training run.
"""
import argparse
import json
import logging
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

import pandas as pd
import torch
from torch import Tensor

from lls_granular import ckpt, data, logs, run
from lls_granular.exps import EXPS
from lls_granular.train import bn_refresh
from lls_granular.wrapper import GranularLLS

from . import policies as P
from .depth import parse
from .dump import cfg_of, rebuild, targets
from .sweep import Costs, exit_sets, picks, sweep_run
from .track import load_exits, publish

log = logging.getLogger(__name__)

DEVICES = ("gpu", "cpu1", "cpu")


@dataclass
class Plan:
    """What the dynamic loop runs: the checked exits (global indices, K = cls), the policy, per-exit logit scales."""

    checked: list[int]
    pol: P.Policy
    scale: list[Tensor]  # per checked exit: 1 / (row_norm · T) for a head, 1 / T for cls
    static: bool = False


# ---------- model, operating point ----------

def load_model(rd: Path, d: data.Data, dev: torch.device) -> GranularLLS:
    """Eval weights + refreshed BN exactly as dump.py, so scores match exits.pt."""
    a = rebuild(cfg_of(rd))
    m, _ = run.model(a, d.n_cls)
    if not isinstance(m, GranularLLS):
        raise ValueError("BP runs have no exits")
    m.to(dev)
    ep, _ = ckpt.load(rd / "ckpt.pt", m)
    m.opt_mode(False)
    torch.manual_seed(0)
    bn_refresh(m, d.train, dev)
    m.eval()
    log.info("loaded %s (%d epochs) on %s", rd.name, ep, dev)
    return m


def choose(ex: dict[str, Any], delta: float, policy: str, n_thresh: int) -> tuple[Plan, dict[str, Any]]:
    """The sweep's pick (val-chosen) and the plan that runs it."""
    df, temps = sweep_run(ex, n_thresh)
    pk = picks(df)
    pk = pk[(pk.temp == "fit") & (pk.final == "cls") & (pk.pool == "single") & (pk.policy == policy)
            & (pk.delta == delta)]
    if pk.empty:
        raise ValueError(f"no {policy} pick within δ={delta}")
    r = pk.iloc[0]
    k, ts = len(ex["heads"]["dim"]), temps["cls"]
    idx = exit_sets(k, "cls")[str(r.S)]
    pol = parse(str(r.label))
    static = pol.name == "static"
    checked = [idx[int(pol.params["e"])]] if static else idx
    rn = ex["heads"]["row_norm"]
    scale = [1 / (rn[e] * ts[e]) if e < k else torch.tensor(1 / ts[e]) for e in checked]
    pick = {"label": r.label, "S": r.S, "policy": policy, "delta": delta, "val_acc": r.val_acc,
            "test_acc": r.test_acc, "test_ref": r.test_ref, "test_mac_frac": r.test_mac_frac,
            "mac_speedup": r.test_speedup, "test_depth": r.test_depth}
    log.info("pick: %s on %s (test %.4f at %.3f MACs)", r.label, r.S, r.test_acc, r.test_mac_frac)
    return Plan(checked, pol, scale, static), pick


def offline(ex: dict[str, Any], plan: Plan, n: int) -> tuple[Tensor, Tensor]:
    """Exit (global index) and prediction of the first n test samples under the plan, from the dump."""
    sp = ex["splits"]["test"]
    k = len(ex["heads"]["dim"])
    raw = [sp["scores"][:n, e] if e < k else sp["logits"][:n] for e in plan.checked]  # what the loop's heads output
    q = torch.stack([z * s for z, s in zip(raw, plan.scale)], 1).softmax(-1)
    j = torch.zeros(len(q), dtype=torch.long) if plan.static else P.first_exit(P.fire(q, plan.pol))
    pred = q.argmax(-1).gather(1, j[:, None])[:, 0]
    return torch.tensor(plan.checked)[j], pred


# ---------- timing ----------

def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


@torch.no_grad()
def timeit(fn: Callable[[Tensor], Any], x: Tensor, dev: torch.device, warmup: int = 3, reps: int = 30,
           budget: float = 2.0) -> float:
    """Median seconds per call: warm-up, then up to `reps` timed calls (at least 3) within ~`budget` seconds."""
    for _ in range(warmup):
        fn(x)
    sync(dev)
    ts: list[float] = []
    end = time.perf_counter() + budget
    while len(ts) < reps and (len(ts) < 3 or time.perf_counter() < end):
        t0 = time.perf_counter()
        fn(x)
        sync(dev)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def prefix(m: GranularLLS, e: int) -> Callable[[Tensor], Tensor]:
    """Blocks up to exit e and that exit's head (e = K: the full network, blocks + cls)."""
    k = len(m.blocks)

    def f(x: Tensor) -> Tensor:
        for b in list(m.blocks)[: min(e + 1, k)]:
            x = b(x)
        return m.cls(x) if e == k else cast(Any, m.blocks[e]).head.scores(x)
    return f


@torch.no_grad()
def dynamic(m: GranularLLS, xs: list[Tensor], plan: Plan, dev: torch.device) -> dict[str, Any]:
    """Batch-1 early exit over xs; per sample: exit, prediction, its latency and the full network's latency."""
    k, chk = len(m.blocks), set(plan.checked)
    scale = [s.to(dev) for s in plan.scale]
    blocks = cast(list[Any], list(m.blocks))
    ex_, pred, t_dyn, t_full = [], [], [], []
    for x in xs[:3]:  # warm-up
        m(x.to(dev))
    for x in xs:
        x = x.to(dev)
        sync(dev)
        t0 = time.perf_counter()
        h, hist = x, []
        for e in range(k + 1):
            if e < k:
                h = blocks[e](h)
            if e not in chk:
                continue
            z = (blocks[e].head.scores(h) if e < k else m.cls(h)) * scale[len(hist)]
            hist.append(z.softmax(-1).cpu())  # the stop decision is made on the host
            if len(hist) == len(plan.checked) or (not plan.static and bool(P.fire(torch.stack(hist, 1), plan.pol)[0, -1])):
                break
        sync(dev)
        t_dyn.append(time.perf_counter() - t0)
        ex_.append(e)
        pred.append(int(hist[-1].argmax()))
        t0 = time.perf_counter()
        m(x)
        sync(dev)
        t_full.append(time.perf_counter() - t0)
    return {"exit": torch.tensor(ex_), "pred": torch.tensor(pred), "t_dyn": t_dyn, "t_full": t_full}


# ---------- one run ----------

def device_info() -> dict[str, Any]:
    cpu = platform.processor()
    try:
        cpu = next(ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                   if ln.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"cpu": cpu, "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "loadavg": os.getloadavg(), "affinity": sorted(os.sched_getaffinity(0)),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "torch": torch.__version__}


def configs(want: list[str], threads: int) -> list[tuple[str, torch.device, int]]:
    out = []
    for name in want:
        if name == "gpu" and torch.cuda.is_available():
            out.append(("gpu", torch.device("cuda"), threads))
        elif name in ("cpu1", "cpu"):
            out.append((name, torch.device("cpu"), 1 if name == "cpu1" else threads))
    return out


def samples(loader: Any, n: int) -> tuple[list[Tensor], Tensor]:
    """The first n test images one by one (any loader batch size), and their labels."""
    xs, ys = [], []
    for x, y in loader:
        xs += list(x.split(1))
        ys.append(y)
        if len(xs) >= n:
            break
    return xs[:n], torch.cat(ys)[:n]


def measure(rd: Path, o: argparse.Namespace) -> dict[str, Any]:
    ex = load_exits(rd / "exits.pt")
    plan, pick = choose(ex, o.delta, o.policy, o.n_thresh)
    a = rebuild(cfg_of(rd))
    d = data.load(a.dataset, o.data_dir, a.bs, 256, a.n_val, o.workers, a.seed, o.limit)
    first = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = load_model(rd, d, first)
    xs, y = samples(d.test, max(o.n, *o.batches, *o.cpu_batches))
    n = min(o.n, len(xs))
    off_exit, off_pred = offline(ex, plan, n)
    c = Costs.of(ex)
    k = len(m.blocks)
    names = list(ex["meta"]["names"])
    threads = torch.get_num_threads()
    res: dict[str, Any] = {"run": rd.name, "device": device_info(), "pick": pick, "checked": plan.checked,
                           "n": n, "configs": {}}
    for name, dev, th in configs(o.devices, threads):
        torch.set_num_threads(th)
        m.to(dev)
        warm = 10 if dev.type == "cuda" else 2
        bs = o.batches if dev.type == "cuda" else o.cpu_batches
        static = []
        for e in range(k + 1):
            row = {"exit": e, "name": names[e], "mac_frac": float((c.backbone[e] + c.head[e]) / c.full)}
            for b in bs:
                xb = torch.cat(xs[:b]).to(dev)
                row[f"b{b}_ms"] = 1e3 * timeit(prefix(m, e), xb, dev, warm, o.reps, o.budget)
            static.append(row)
        full = {f"b{b}_ms": 1e3 * timeit(m, torch.cat(xs[:b]).to(dev), dev, warm, o.reps, o.budget) for b in bs}
        dy = dynamic(m, xs[:n], plan, dev)
        dyn_ms, full_ms = 1e3 * statistics.fmean(dy["t_dyn"]), 1e3 * statistics.fmean(dy["t_full"])
        res["configs"][name] = {
            "threads": th, "static": static, "full": full,
            "dynamic": {"mean_ms": dyn_ms, "median_ms": 1e3 * statistics.median(dy["t_dyn"]),
                        "full_mean_ms": full_ms, "speedup": full_ms / dyn_ms,
                        "agree": (dy["exit"] == off_exit).double().mean().item(),
                        "acc": (dy["pred"] == y[:n]).double().mean().item(),
                        "offline_acc": (off_pred == y[:n]).double().mean().item(),
                        "mean_exit": dy["exit"].double().mean().item(), "offline_mean_exit": off_exit.double().mean().item(),
                        "exit_hist": torch.bincount(dy["exit"], minlength=k + 1).tolist()}}
        r = res["configs"][name]["dynamic"]
        log.info("%s %s: dynamic %.3f ms vs full %.3f ms (×%.2f; MACs predict ×%.2f), agree %.3f, acc %.4f (offline %.4f)",
                 rd.name, name, dyn_ms, full_ms, r["speedup"], pick["mac_speedup"], r["agree"], r["acc"], r["offline_acc"])
    torch.set_num_threads(threads)
    return res


def summary(res: dict[str, Any]) -> dict[str, Any]:
    s: dict[str, Any] = {"latency/pick": res["pick"]["label"], "latency/mac_speedup": res["pick"]["mac_speedup"],
                         "latency/n": res["n"]}
    for name, c in res["configs"].items():
        dy = c["dynamic"]
        s |= {f"latency/{name}/{k}": dy[k] for k in ("mean_ms", "full_mean_ms", "speedup", "agree", "acc")}
        s |= {f"latency/{name}/full_{k}": v for k, v in c["full"].items()}
    return s


def run_one(rd: Path, o: argparse.Namespace) -> dict[str, Any] | str:
    for f, tag in (("config.json", "no config"), ("ckpt.pt", "no ckpt"), ("exits.pt", "no dump")):
        if not (rd / f).exists():
            log.warning("%s: %s", rd.name, tag)
            return tag
    out = rd / "latency.json"
    if out.exists() and not o.fresh:
        return "exists"
    logs.setup(o.log, rd / "latency.log")
    res = measure(rd, o)
    out.write_text(json.dumps(res, indent=2))
    s = summary(res)
    up = False
    if o.wandb and not o.limit:
        cfg = cfg_of(rd)
        up = publish(rd, out, s, o.project or cfg.get("project") or "lls-granularity", o.entity or cfg.get("entity"),
                     kind="latency")
    return {**s, "wandb": up}


# ---------- CLI ----------

def args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measured early-exit latency (static, full, dynamic batch-1 loop).")
    a = p.add_argument
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--exp", nargs="+", choices=sorted(EXPS), help="every non-BP run of these experiments")
    g.add_argument("--run-dir", help="a single run folder with config.json, ckpt.pt and exits.pt")
    a("--seeds", type=int, nargs="*")
    a("--delta", type=float, default=0.01, help="operating point: within δ of the full network on val")
    a("--policy", default="any", help="'any' (best) or one policy name")
    a("--n", type=int, default=1000, help="test samples for the dynamic loop")
    a("--batches", type=int, nargs="+", default=[1, 256], help="static timing batch sizes on GPU")
    a("--cpu-batches", type=int, nargs="+", default=[1, 32], help="static timing batch sizes on CPU")
    a("--devices", nargs="+", default=list(DEVICES), choices=DEVICES)
    a("--reps", type=int, default=30)
    a("--budget", type=float, default=2.0, help="seconds per static timing (at least 3 reps)")
    a("--n-thresh", type=int, default=41, help="thresholds per policy in the sweep (as in lls_exit.sweep)")
    a("--fresh", action="store_true", help="redo runs that already have latency.json")
    a("--wandb", action=argparse.BooleanOptionalAction, default=True)
    a("--project", default=None, help="wandb project (default: the training run's)")
    a("--entity", default=None)
    a("--data-dir", default=None)
    a("--workers", type=int, default=0)
    a("--limit", type=int, default=None, help="train subset for smoke tests (never uploaded)")
    a("--log", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    o = args(argv)
    rds = [Path(o.run_dir).expanduser()] if o.run_dir else targets(o.exp, o.seeds)
    log.info("latency for %d runs, devices %s", len(rds), o.devices)
    rows = {rd.name: run_one(rd, o) for rd in rds}
    table = []
    for run_name, r in rows.items():
        if isinstance(r, str):
            table.append({"run": run_name, "status": r})
            continue
        for dname in DEVICES:
            if f"latency/{dname}/mean_ms" in r:
                table.append({"run": run_name, "status": "ok", "device": dname, "pick": r["latency/pick"],
                              "dyn_ms": r[f"latency/{dname}/mean_ms"], "full_ms": r[f"latency/{dname}/full_mean_ms"],
                              "speedup": r[f"latency/{dname}/speedup"], "mac_speedup": r["latency/mac_speedup"],
                              "agree": r[f"latency/{dname}/agree"], "acc": r[f"latency/{dname}/acc"],
                              "wandb": r["wandb"]})
    print(pd.DataFrame(table).to_string(index=False, float_format="%.3f"))
    if any(isinstance(r, str) and r.startswith("no") for r in rows.values()):
        sys.exit(1)


if __name__ == "__main__":
    logs.setup()
    main()
