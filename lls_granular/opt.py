"""Optimizer factory and schedule-free train/eval switching."""
import logging
from typing import Iterable

from torch import nn, optim
from torch.optim import Optimizer

from .upstream import AdamWScheduleFree, SGDScheduleFree

log = logging.getLogger(__name__)

SCHEDULE_FREE = (AdamWScheduleFree, SGDScheduleFree)


def make(name: str, params: Iterable[nn.Parameter], lr: float, wd: float = 0.0,
         momentum: float = 0.9) -> Optimizer:
    if name == "AdamWSF":
        return AdamWScheduleFree(params, lr=lr, weight_decay=wd)
    if name == "SGDSF":
        return SGDScheduleFree(params, lr=lr, weight_decay=wd)
    if name == "Adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "SGD":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    raise ValueError(f"unknown optimizer {name!r}")


def set_mode(opts: Iterable[Optimizer], train: bool) -> None:
    """Schedule-free optimizers swap the weights between train (y) and eval (x) points."""
    n = 0
    for o in opts:
        if isinstance(o, SCHEDULE_FREE):
            o.train() if train else o.eval()
            n += 1
    log.debug("optimizers -> %s mode (%d schedule-free)", "train" if train else "eval", n)
