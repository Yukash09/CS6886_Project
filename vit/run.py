import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lls_granular import ckpt, data, logs
from lls_granular.bp import BPNet
from lls_granular.heads import LLS_MODES
from lls_granular.partitions import Partition, parse
from lls_granular.run import seed_all, strategy
from lls_granular.track import RunId, Tracker, method
from lls_granular.train import Net, fit
from lls_granular.wrapper import GranularLLS
from scalability.run import select_device

from .architectures import ARCHS

log = logging.getLogger(__name__)


@dataclass
class ViTRunId(RunId):
    lls_mode: str = "lls"
    readout: str = "linear"

    @property
    def group(self) -> str:
        variant = f"_{self.lls_mode}" if self.part is not None and self.lls_mode != "lls" else ""
        readout = "_lls-readout" if self.part is not None and self.readout == "lls" else ""
        return f"{self.arch}_{self.dataset}_{method(self.part)}{variant}{readout}_e{self.epochs}"

    def labels(self) -> list[str]:
        return [*super().labels(), "vit", self.lls_mode, self.readout]

    def fields(self) -> dict[str, Any]:
        return {**super().fields(), "lls_mode": self.lls_mode, "readout": self.readout}


def args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR ViTs with granular LLS or BP.")
    add = parser.add_argument
    add("--arch", default="vit_tiny", choices=sorted(ARCHS))
    add("--dataset", default="cifar10aug", choices=sorted(data.SETS))
    add("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    add("--partition", default="uniform:2",
        help="'bp' | 'uniform:k' | 'aligned' | explicit sizes such as '3,3,3,3'")
    add("--lls-mode", default="lls-mxm", choices=LLS_MODES)
    add("--readout", default="linear", choices=["linear", "lls"])
    add("--epochs", type=int, default=300)
    add("--patience", type=int, default=None)
    add("--seed", type=int, default=0)
    add("--exp", default="V-DEV")
    add("--sched", default="full", choices=["full", "proxy"])
    add("--tags", nargs="*", default=[])
    add("--optim", default="AdamWSF")
    add("--lr", type=float, default=5e-4)
    add("--wd", type=float, default=0.05)
    add("--wave", default="square", choices=["square", "cosine"])
    add("--temp", type=float, default=0.5)
    add("--bs", type=int, default=128)
    add("--eval-bs", type=int, default=256)
    add("--n-val", type=int, default=5000)
    add("--workers", type=int, default=2)
    add("--limit", type=int, default=None)
    add("--data-dir", default=None)
    add("--out", default=None)
    add("--wandb", action=argparse.BooleanOptionalAction, default=True)
    add("--project", default=os.environ.get("WANDB_PROJECT", "lls-vit"))
    add("--entity", default=os.environ.get("WANDB_ENTITY"))
    add("--fresh", action="store_true")
    add("--log", default=None)
    return parser.parse_args(argv)


def partition(spec: str, shapes: list[tuple[int, ...]]) -> Partition | None:
    return None if spec == "bp" else parse(spec, shapes)


def model(a: argparse.Namespace, n_cls: int) -> tuple[Net, Partition | None]:
    torch.manual_seed(a.seed)
    units = ARCHS[a.arch]()
    part = partition(a.partition, [unit.shape for unit in units])
    if part is None:
        if a.lls_mode != "lls":
            raise ValueError("--lls-mode only applies to LLS partitions, not --partition bp")
        if a.readout != "linear":
            raise ValueError("--readout lls only applies to LLS partitions, not --partition bp")
        return BPNet(units, n_cls, a.optim, a.lr, a.wd), None
    return GranularLLS(units, part, n_cls, a.optim, a.lr, a.wd, a.wave, a.temp,
                       a.lls_mode, a.readout), part


def main(argv: list[str] | None = None) -> None:
    a = args(argv)
    out = Path(a.out or os.environ.get("LLS_VIT_OUT") or os.environ.get("LLS_OUT") or "./runs").expanduser()
    dev = select_device(a.device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True

    network, part = model(a, data.SETS[a.dataset][1])
    network.to(dev)
    rid = ViTRunId(a.arch, a.dataset, part, a.epochs, a.seed, a.exp, a.sched,
                   strategy(a.partition), a.tags, a.lls_mode, a.readout)
    run_dir = out / rid.name
    logs.setup(a.log, run_dir / "run.log")
    log.info("device %s, run dir %s", dev, run_dir)

    path = run_dir / "ckpt.pt"
    if a.fresh:
        for file in (path, run_dir / "wandb_id"):
            file.unlink(missing_ok=True)
    elif path.exists():
        summary = run_dir / "summary.json"
        stopped = json.loads(summary.read_text()).get("early_stop/stopped", False) if summary.exists() else False
        if ckpt.peek(path) + 1 >= a.epochs or stopped:
            log.info("%s already complete or early-stopped; skipping", rid.name)
            return

    loaders = data.load(a.dataset, a.data_dir, a.bs, a.eval_bs, a.n_val,
                        a.workers, a.seed, a.limit)
    generator = loaders.train.generator
    start, best = ckpt.load(path, network, generator) if path.exists() else (0, None)
    if not path.exists():
        seed_all(a.seed)

    tracker = Tracker(rid, vars(a), out, a.wandb, a.project, a.entity, start)
    splits = {"val": loaders.val, "test": loaders.test} if loaders.val else {"test": loaders.test}
    ok = False
    try:
        fit(network, loaders.train, splits, tracker, a.epochs, dev, start, best,
            save=lambda epoch, current: ckpt.save(path, network, epoch, current, generator),
            patience=a.patience)
        ok = True
    finally:
        tracker.finish(ok)


if __name__ == "__main__":
    logs.setup()
    main()

