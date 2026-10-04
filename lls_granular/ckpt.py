"""Checkpoint/resume: model, every optimizer, best-so-far, RNG and data-order state."""
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .train import Best, Net

log = logging.getLogger(__name__)


def _rng() -> dict[str, Any]:
    s: dict[str, Any] = {"torch": torch.get_rng_state(), "py": random.getstate(), "np": np.random.get_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _set_rng(s: dict[str, Any]) -> None:
    torch.set_rng_state(s["torch"])
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save(path: Path, m: Net, ep: int, best: Best, gen: torch.Generator | None = None) -> None:
    """Atomic write (tmp + rename), so a session killed mid-save never leaves a corrupt file."""
    state = {
        "epoch": ep, "best": asdict(best), "model": m.state_dict(),
        "opts": [o.state_dict() for o in m.opts], "rng": _rng(),
        "gen": gen.get_state() if gen else None,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    log.debug("checkpoint epoch %d -> %s (%.0f MB)", ep, path, path.stat().st_size / 2**20)


def peek(path: Path) -> int:
    """Last completed epoch stored in a checkpoint."""
    return torch.load(path, map_location="cpu", weights_only=False)["epoch"]


def load(path: Path, m: Net, gen: torch.Generator | None = None) -> tuple[int, Best]:
    """Restore everything; returns (next epoch to run, best so far). Loads on CPU; load_state_dict
    copies values onto m's parameters (wherever they live) and optimizer state follows them."""
    s = torch.load(path, map_location="cpu", weights_only=False)  # RNG states must stay on CPU
    m.load_state_dict(s["model"])
    if len(s["opts"]) != len(m.opts):
        raise ValueError(f"checkpoint has {len(s['opts'])} optimizers, model has {len(m.opts)}")
    for o, st in zip(m.opts, s["opts"]):
        o.load_state_dict(st)
    _set_rng(s["rng"])
    if gen is not None and s["gen"] is not None:
        gen.set_state(s["gen"])
    best = Best(**s["best"])
    log.info("resumed from %s: epoch %d done, best val %.4f@%d", path, s["epoch"], best.val, best.epoch)
    return s["epoch"] + 1, best
