"""Command line entry point: python -m lls_granular.run --partition aligned --epochs 100 ..."""
import argparse
import logging
import os
import random
import re
from pathlib import Path

import numpy as np
import torch

from . import ckpt, data, logs
from .bp import BPNet
from .partitions import Partition, parse
from .track import RunId, Tracker
from .train import Net, fit
from .units import ARCHS
from .wrapper import GranularLLS

log = logging.getLogger(__name__)


def args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VGG8/MobileNet with granular LLS or BP.")
    a = p.add_argument
    a("--arch", default="vgg8", choices=sorted(ARCHS))
    a("--dataset", default="cifar10aug", choices=sorted(data.SETS))
    a("--partition", default="uniform:1", help="'bp' | 'uniform:k' | 'aligned' | sizes like '2,2,3'")
    a("--epochs", type=int, default=100)
    a("--seed", type=int, default=0)
    a("--exp", default="dev", help="experiment id, e.g. G-U, G-ALL (wandb job_type)")
    a("--sched", default="full", choices=["full", "proxy"])
    a("--tags", nargs="*", default=[])
    a("--optim", default="AdamWSF")
    a("--lr", type=float, default=5e-3)
    a("--wd", type=float, default=0.0)
    a("--wave", default="square", choices=["square", "cosine"])
    a("--temp", type=float, default=0.5)
    a("--bs", type=int, default=128)
    a("--eval-bs", type=int, default=256)
    a("--n-val", type=int, default=5000)
    a("--workers", type=int, default=2)
    a("--limit", type=int, default=None, help="train/val subset size for smoke tests")
    a("--data-dir", default=None, help="else $LLS_DATA, else ./data")
    a("--out", default=None, help="else $LLS_OUT, else ./runs")
    a("--wandb", action=argparse.BooleanOptionalAction, default=True)
    a("--project", default=os.environ.get("WANDB_PROJECT", "lls-granularity"))
    a("--entity", default=os.environ.get("WANDB_ENTITY"))
    a("--fresh", action="store_true", help="delete this run's checkpoint and wandb id; start a new run")
    a("--log", default=None, help="log level (else $LLS_LOG, else INFO)")
    return p.parse_args(argv)


def partition(a: argparse.Namespace, shapes: list[tuple[int, ...]]) -> Partition | None:
    return None if a.partition == "bp" else parse(a.partition, shapes)


def strategy(spec: str) -> str | None:
    """Named strategies become a tag; explicit sizes are already in the run name."""
    return None if re.fullmatch(r"[\d,]+", spec) else spec


def model(a: argparse.Namespace, n_cls: int) -> tuple[Net, Partition | None]:
    torch.manual_seed(a.seed)
    units = ARCHS[a.arch]()
    part = partition(a, [u.shape for u in units])
    if part is None:
        return BPNet(units, n_cls, a.optim, a.lr, a.wd), None
    return GranularLLS(units, part, n_cls, a.optim, a.lr, a.wd, a.wave, a.temp), part


def seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def main(argv: list[str] | None = None) -> None:
    a = args(argv)
    out = Path(a.out or os.environ.get("LLS_OUT", "./runs")).expanduser()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    m, part = model(a, data.SETS[a.dataset][1])
    m.to(dev)
    rid = RunId(a.arch, a.dataset, part, a.epochs, a.seed, a.exp, a.sched, strategy(a.partition), a.tags)

    run_dir = out / rid.name
    logs.setup(a.log, run_dir / "run.log")
    log.info("device %s, run dir %s", dev, run_dir)
    d = data.load(a.dataset, a.data_dir, a.bs, a.eval_bs, a.n_val, a.workers, a.seed, a.limit)

    path = run_dir / "ckpt.pt"
    if a.fresh:
        for f in (path, run_dir / "wandb_id"):
            f.unlink(missing_ok=True)
        log.info("--fresh: removed checkpoint and wandb id, starting a new run")
    gen = d.train.generator
    start, best = 0, None
    if path.exists():
        start, best = ckpt.load(path, m, gen)
    else:
        seed_all(a.seed)

    tr = Tracker(rid, vars(a), out, a.wandb, a.project, a.entity, start)
    splits = {"val": d.val, "test": d.test} if d.val else {"test": d.test}
    try:
        fit(m, d.train, splits, tr, a.epochs, dev, start, best,
            save=lambda ep, b: ckpt.save(path, m, ep, b, gen))
    finally:
        tr.finish()


if __name__ == "__main__":
    logs.setup()
    main()
