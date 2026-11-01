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

from .architectures import ARCHS

log = logging.getLogger(__name__)


@dataclass
class ScalabilityRunId(RunId):
    lls_mode: str = "lls"
    readout: str = "linear"

    @property
    def group(self) -> str:
        variant = f"_{self.lls_mode}" if self.part is not None and self.lls_mode != "lls" else ""
        readout = "_lls-readout" if self.part is not None and self.readout == "lls" else ""
        return f"{self.arch}_{self.dataset}_{method(self.part)}{variant}{readout}_e{self.epochs}"

    def labels(self) -> list[str]:
        return [*super().labels(), self.lls_mode, self.readout]

    def fields(self) -> dict[str, Any]:
        return {**super().fields(), "lls_mode": self.lls_mode, "readout": self.readout}


def args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CIFAR ResNets with granular LLS or BP.")
    a = p.add_argument
    a("--arch", default="resnet20", choices=sorted(ARCHS))
    a("--dataset", default="cifar10aug", choices=sorted(data.SETS))
    a("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
      help="auto prefers CUDA, then Apple MPS, then CPU")
    a("--partition", default="uniform:1",
      help="'bp' | 'uniform:k' | 'aligned' (stages) | explicit sizes")
    a("--lls-mode", default="lls", choices=LLS_MODES,
      help="fixed LLS, per-class modulation (LLS-M), or class mixing (LLS-MxM)")
    a("--readout", default="linear", choices=["linear", "lls"],
      help="detached linear classifier or the final LLS head itself")
    a("--epochs", type=int, default=300)
    a("--patience", type=int, default=None,
      help="stop after this many epochs without validation-accuracy improvement")
    a("--seed", type=int, default=0)
    a("--exp", default="S-DEV")
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
    a("--limit", type=int, default=None)
    a("--data-dir", default=None)
    a("--out", default=None)
    a("--wandb", action=argparse.BooleanOptionalAction, default=True)
    a("--project", default=os.environ.get("WANDB_PROJECT", "lls-scalability"))
    a("--entity", default=os.environ.get("WANDB_ENTITY"))
    a("--fresh", action="store_true")
    a("--log", default=None)
    return p.parse_args(argv)


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


def select_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch environment")
    if name == "mps" and not torch.backends.mps.is_available():
        built = torch.backends.mps.is_built()
        raise RuntimeError(f"MPS was requested but is not available (MPS support built={built})")
    return torch.device(name)


def main(argv: list[str] | None = None) -> None:
    a = args(argv)
    default_out = os.environ.get("LLS_SCALABILITY_OUT") or os.environ.get("LLS_OUT") or "./runs"
    out = Path(a.out or default_out).expanduser()
    dev = select_device(a.device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True

    m, part = model(a, data.SETS[a.dataset][1])
    m.to(dev)
    rid = ScalabilityRunId(a.arch, a.dataset, part, a.epochs, a.seed, a.exp, a.sched,
                           strategy(a.partition), a.tags, a.lls_mode, a.readout)
    run_dir = out / rid.name
    logs.setup(a.log, run_dir / "run.log")
    log.info("device %s, run dir %s", dev, run_dir)

    path = run_dir / "ckpt.pt"
    if a.fresh:
        for file in (path, run_dir / "wandb_id"):
            file.unlink(missing_ok=True)
        log.info("--fresh: removed checkpoint and wandb id")
    elif path.exists():
        summary_path = run_dir / "summary.json"
        stopped = (json.loads(summary_path.read_text()).get("early_stop/stopped", False)
                   if summary_path.exists() else False)
        if ckpt.peek(path) + 1 >= a.epochs or stopped:
            reason = "early-stopped" if stopped else f"complete ({a.epochs} epochs)"
            log.info("%s already %s; skipping", rid.name, reason)
            return

    d = data.load(a.dataset, a.data_dir, a.bs, a.eval_bs, a.n_val,
                  a.workers, a.seed, a.limit)
    generator = d.train.generator
    start, best = 0, None
    if path.exists():
        start, best = ckpt.load(path, m, generator)
    else:
        seed_all(a.seed)

    tracker = Tracker(rid, vars(a), out, a.wandb, a.project, a.entity, start)
    splits = {"val": d.val, "test": d.test} if d.val else {"test": d.test}
    ok = False
    try:
        fit(m, d.train, splits, tracker, a.epochs, dev, start, best,
            save=lambda epoch, current: ckpt.save(path, m, epoch, current, generator),
            patience=a.patience)
        ok = True
    finally:
        tracker.finish(ok)


if __name__ == "__main__":
    logs.setup()
    main()
