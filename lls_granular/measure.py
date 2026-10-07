"""Cost table for every partition: python -m lls_granular.measure --arch vgg8

Analytic activation memory and MACs for all partitions (+ BP), plus measured GPU peak if a GPU is present.
Writes <out>/cost_<arch>_bs<bs>.csv and one wandb run (job_type "cost") with the table.
"""
import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Any

import torch

from . import logs
from .bp import BPNet
from .data import SETS
from .exps import EXPS
from .metrics import Costs, gpu_peak
from .partitions import Partition, enumerate_all, parse
from .track import kind, method
from .units import ARCHS
from .wrapper import GranularLLS

log = logging.getLogger(__name__)

MAX_ENUM = 10  # 2^(n-1) partitions; beyond this a candidate list is needed


def partitions(arch: str) -> list[Partition | None]:
    """All partitions for small models; otherwise the ones used in that arch's experiments."""
    shapes = [u.shape for u in ARCHS[arch]()]
    if len(shapes) <= MAX_ENUM:
        return [None, *enumerate_all(len(shapes))]
    specs = dict.fromkeys(j.partition for js in EXPS.values() for j in js if j.arch == arch)
    if not specs:
        raise ValueError(f"{arch}: {2 ** (len(shapes) - 1)} partitions and no experiment candidates")
    return [None if s == "bp" else parse(s, shapes) for s in specs]


def rows(arch: str, n_cls: int, bs: int, dev: torch.device, gpu: bool) -> list[dict[str, Any]]:
    torch.manual_seed(0)
    c = Costs(ARCHS[arch](), n_cls, bs, dev)
    out = []
    for p in partitions(arch):
        r: dict[str, Any] = {"method": method(p), "kind": kind(p), "n_blocks": len(p) if p else None,
                             "max_block": max(p) if p else None, "first_block": p[0] if p else None,
                             **c.cost(p)}
        if gpu:
            torch.manual_seed(0)
            units = ARCHS[arch]()
            m = BPNet(units, n_cls) if p is None else GranularLLS(units, p, n_cls)
            r |= gpu_peak(m, bs, n_cls, dev)
            del m
            torch.cuda.empty_cache()
        log.info("%-18s act_peak %7.1f MB  train %7.0f MMAC%s", r["method"], r["act_peak_mb"], r["train_mmac"],
                 f"  gpu_peak {r['gpu_peak_mb']:.0f} MB" if gpu else "")
        out.append(r)
    return out


def save(rs: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0]))
        w.writeheader()
        w.writerows(rs)
    log.info("wrote %s (%d rows)", path, len(rs))


def to_wandb(rs: list[dict[str, Any]], name: str, cfg: dict[str, Any], project: str) -> None:
    try:
        import wandb
        run = wandb.init(project=project, name=name, job_type="cost", config=cfg, tags=["cost", cfg["arch"]])
    except Exception as e:  # never lose the CSV over a logging problem
        log.warning("wandb unavailable (%s); CSV only", e)
        return
    cols = list(rs[0])
    run.log({"cost": wandb.Table(columns=cols, data=[[r[k] for k in cols] for r in rs])})
    run.finish()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Memory/compute cost of every partition.")
    p.add_argument("--arch", default="vgg8", choices=sorted(ARCHS))
    p.add_argument("--dataset", default="cifar10aug", choices=sorted(SETS))
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--no-gpu", action="store_true", help="skip the measured GPU peak")
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "lls-granularity"))
    a = p.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = dev.type == "cuda" and not a.no_gpu
    n_cls = SETS[a.dataset][1]
    rs = rows(a.arch, n_cls, a.bs, dev, gpu)
    name = f"cost_{a.arch}_{a.dataset}_bs{a.bs}"
    save(rs, Path(os.environ.get("LLS_OUT", "./runs")).expanduser() / f"{name}.csv")
    if a.wandb:
        gpu_name = torch.cuda.get_device_name(0) if gpu else "cpu"
        to_wandb(rs, name, {**vars(a), "n_cls": n_cls, "gpu": gpu_name}, a.project)


if __name__ == "__main__":
    logs.setup()
    main()
