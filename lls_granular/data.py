"""CIFAR loaders with upstream's exact transforms plus a fixed, stratified validation split."""
import fcntl
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms as T

log = logging.getLogger(__name__)

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)  # upstream uses these for CIFAR100 too

Cifar = datasets.CIFAR10  # CIFAR100 subclasses CIFAR10

SETS: dict[str, tuple[type[Cifar], int, bool]] = {  # name -> (class, n_cls, aug)
    "cifar10": (datasets.CIFAR10, 10, False),
    "cifar10aug": (datasets.CIFAR10, 10, True),
    "cifar100aug": (datasets.CIFAR100, 100, True),
}


@dataclass
class Data:
    train: DataLoader
    val: DataLoader | None
    test: DataLoader
    n_cls: int


def tfms(aug: bool) -> tuple[T.Compose, T.Compose]:
    """(train, eval) transforms. aug = upstream *AUG loaders; else upstream cifar10_basic."""
    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    if aug:
        train = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), *norm, T.RandomErasing(p=0.2)]
    else:
        train = [T.RandomHorizontalFlip(), *norm]
    return T.Compose(train), T.Compose(norm)


def split(targets: list[int], n_val: int, n_cls: int, seed: int = 0) -> tuple[list[int], list[int]]:
    """Stratified train/val indices: n_val // n_cls per class, chosen by a fixed seed."""
    if n_val % n_cls:
        raise ValueError(f"n_val={n_val} not divisible by {n_cls} classes")
    g = torch.Generator().manual_seed(seed)
    per = n_val // n_cls
    y = torch.tensor(targets)
    tr, va = [], []
    for c in range(n_cls):
        idx = (y == c).nonzero().flatten()
        idx = idx[torch.randperm(len(idx), generator=g)]
        va += idx[:per].tolist()
        tr += idx[per:].tolist()
    return sorted(tr), sorted(va)


def root(path: str | None) -> Path:
    return Path(path or os.environ.get("LLS_DATA", "./data")).expanduser()


def _open(cls: type[Cifar], src: Path, train: bool, tf: T.Compose) -> Cifar:
    try:
        return cls(str(src), train=train, download=True, transform=tf)
    except (RuntimeError, OSError) as e:  # OSError covers URLError (no internet)
        raise RuntimeError(f"{cls.__name__} not found in {src} and could not be downloaded "
                           f"(no internet?). Put the torchvision files there or set --data-dir.") from e


@contextmanager
def _lock(src: Path) -> Iterator[None]:
    """Serialise downloads between runs started in parallel; no-op on a read-only data dir."""
    try:
        src.mkdir(parents=True, exist_ok=True)
        f = open(src / ".lock", "w")
    except OSError:
        yield
        return
    with f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _loader(ds: Dataset, bs: int, workers: int, g: torch.Generator | None = None) -> DataLoader:
    """Shuffled iff a generator is given (train); eval loaders keep a fixed order.
    A final train batch of size 1 would crash BatchNorm, so it is dropped."""
    train = g is not None
    drop = train and len(ds) % bs == 1  # type: ignore[arg-type]
    return DataLoader(ds, bs, shuffle=train, generator=g, num_workers=workers, drop_last=drop,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def load(name: str, path: str | None = None, bs: int = 128, eval_bs: int = 256, n_val: int = 5000,
         workers: int = 2, seed: int = 0, limit: int | None = None) -> Data:
    """`seed` only shuffles batches; the val split is fixed so every run sees the same val images."""
    cls, n_cls, aug = SETS[name]
    src = root(path)
    tf_train, tf_eval = tfms(aug)
    with _lock(src):
        train_aug = _open(cls, src, True, tf_train)
        train_eval = _open(cls, src, True, tf_eval)
        test = _open(cls, src, False, tf_eval)

    tr, va = split(train_aug.targets, n_val, n_cls) if n_val else (list(range(len(train_aug))), [])
    if limit:
        tr, va = tr[:limit], va[:limit]

    g = torch.Generator().manual_seed(seed)
    data = Data(
        train=_loader(Subset(train_aug, tr), bs, workers, g),
        val=_loader(Subset(train_eval, va), eval_bs, workers) if va else None,
        test=_loader(test, eval_bs, workers),
        n_cls=n_cls,
    )
    log.info("data %s from %s: train=%d val=%d test=%d classes=%d aug=%s",
             name, src, len(tr), len(va), len(test), n_cls, aug)
    log.debug("train tf: %s", tf_train)
    log.debug("loader bs=%d eval_bs=%d workers=%d seed=%d limit=%s", bs, eval_bs, workers, seed, limit)
    return data
