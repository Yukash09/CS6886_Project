"""Epoch loop, evaluation (schedule-free eval weights + BN refresh), best-on-val tracking."""
import itertools
import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from .bp import BPNet
from .track import Tracker
from .wrapper import GranularLLS

log = logging.getLogger(__name__)

Net = GranularLLS | BPNet
Batches = Iterable[tuple[Tensor, Tensor]]

BN_BATCHES = 50  # upstream re-estimates BN stats on 50 train batches before every eval
LOG_EVERY = 100  # steps between debug lines


@dataclass
class Best:
    val: float = -1.0
    epoch: int = -1
    test: float = 0.0


def train_epoch(m: Net, loader: Batches, dev: torch.device) -> dict[str, float]:
    m.train()
    m.opt_mode(True)
    loss_sum, hit_sum, cls_hits, n = None, None, torch.zeros((), device=dev), 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        out = m.step(x, y)
        bs = y.numel()
        loss_sum = out.losses * bs if loss_sum is None else loss_sum + out.losses * bs
        hit_sum = out.hits if hit_sum is None else hit_sum + out.hits
        cls_hits += (out.logits.argmax(1) == y).sum()
        n += bs
        if i % LOG_EVERY == 0:
            log.debug("step %d losses %s", i, [round(v, 4) for v in out.losses.tolist()])
    assert loss_sum is not None and hit_sum is not None, "empty train loader"

    losses, hits = (loss_sum / n).tolist(), (hit_sum.float() / n).tolist()
    row = {"train/loss": losses[-1], "train/acc": cls_hits.item() / n}
    for k, (l, a) in enumerate(zip(losses, hits)):
        row[f"block/{k}/train_loss"] = l
        row[f"block/{k}/train_acc"] = a
    return row


@torch.no_grad()
def bn_refresh(m: Net, loader: Batches, dev: torch.device, n: int = BN_BATCHES) -> None:
    """Re-estimate BN running stats under the current (eval) weights, as upstream does."""
    m.train()
    for x, _ in itertools.islice(loader, n):
        m(x.to(dev, non_blocking=True))


@torch.no_grad()
def evaluate(m: Net, loader: Batches, dev: torch.device, tag: str) -> dict[str, float]:
    m.eval()
    loss, hits, blk, n = 0.0, 0, None, 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        logits, scores = m.forward_all(x)
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        hits += (logits.argmax(1) == y).sum().item()
        b = torch.stack([(s.argmax(1) == y).sum() for s in scores]) if scores else torch.zeros(0)
        blk = b if blk is None else blk + b
        n += y.numel()
    row = {f"{tag}/acc": hits / n, f"{tag}/loss": loss / n}
    for k, a in enumerate((blk.float() / n).tolist() if blk is not None else []):
        row[f"block/{k}/{tag}_acc"] = a
    return row


def eval_all(m: Net, train: Batches, splits: Mapping[str, Batches], dev: torch.device) -> dict[str, float]:
    """Switch to schedule-free eval weights, refresh BN, evaluate each split, switch back."""
    m.opt_mode(False)
    bn_refresh(m, train, dev)
    row: dict[str, float] = {}
    for tag, loader in splits.items():
        row |= evaluate(m, loader, dev, tag)
    m.opt_mode(True)
    return row


def fit(m: Net, train: Batches, splits: Mapping[str, Batches], tr: Tracker, epochs: int,
        dev: torch.device, start: int = 0, best: Best | None = None,
        save: Callable[[int, Best], None] | None = None) -> Best:
    """Train epochs [start, epochs). `splits` must hold 'val' (selection) and may hold 'test'."""
    best = best or Best()
    sel = "val" if "val" in splits else "test"
    if sel == "test":
        log.warning("no val split: best epoch is selected on test")
    cuda = dev.type == "cuda"
    t0 = time.time()
    row: dict[str, float] = {}

    for ep in range(start, epochs):
        if cuda:
            torch.cuda.reset_peak_memory_stats(dev)
        t = time.time()
        row = {"epoch": ep, **train_epoch(m, train, dev)}
        row["time/train_s"] = time.time() - t
        if cuda:
            row["sys/peak_mem_mb"] = torch.cuda.max_memory_allocated(dev) / 2**20

        t = time.time()
        row |= eval_all(m, train, splits, dev)
        row["time/eval_s"] = time.time() - t

        if row[f"{sel}/acc"] > best.val:
            best = Best(row[f"{sel}/acc"], ep, row.get("test/acc", 0.0))
        row |= {"best/val_acc": best.val, "best/epoch": best.epoch}

        if save:
            save(ep, best)
        tr.log(row)
        log.info("epoch %d/%d train_acc %.4f %s_acc %.4f test_acc %.4f best %.4f@%d (%.0fs)",
                 ep + 1, epochs, row["train/acc"], sel, row[f"{sel}/acc"], row.get("test/acc", 0.0),
                 best.val, best.epoch, row["time/train_s"] + row["time/eval_s"])

    final = {"best/val_acc": best.val, "best/epoch": best.epoch, "test_at_best_val": best.test,
             "time/total_s": time.time() - t0}
    if row:
        final["final/test_acc"] = row.get("test/acc", 0.0)
        if cuda:
            final["sys/peak_mem_mb"] = row["sys/peak_mem_mb"]
    tr.summary(final)
    return best
