"""GranularLLS: units grouped into blocks; BP inside a block, one LLS head per block."""
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import opt
from .heads import LLSHead
from .partitions import Partition, check, label, spans
from .units import Unit

log = logging.getLogger(__name__)


@dataclass
class Out:
    logits: Tensor  # classifier output, detached
    losses: Tensor  # (K+1,): one per block, then the classifier
    hits: Tensor    # (K,): correct head predictions per block


class Block(nn.Module):
    def __init__(self, units: list[Unit], n_cls: int, wave: str, temp: float, mode: str) -> None:
        super().__init__()
        self.units = nn.Sequential(*units)
        last = units[-1]
        self.head = LLSHead(n_cls, last.head_dim, last.pool, wave, temp, mode=mode)
        self.name = "+".join(u.name for u in units)

    def forward(self, x: Tensor) -> Tensor:
        return self.units(x)


class CLSToken(nn.Module):
    """Select the classification token from a ``(B, N, D)`` sequence."""

    def forward(self, x: Tensor) -> Tensor:
        return x[:, 0]


def classifier(last: Unit, n_cls: int) -> nn.Sequential:
    """Build the conventional readout for convolutional, token, or vector outputs."""
    if len(last.shape) == 3:
        pre, features = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()), last.shape[0]
    elif len(last.shape) == 2:
        pre, features = CLSToken(), last.shape[1]
    else:
        pre, features = nn.Identity(), last.shape[0]
    return nn.Sequential(pre, nn.Linear(features, n_cls))


class GranularLLS(nn.Module):
    def __init__(self, units: list[Unit], part: Partition, n_cls: int, optim: str = "AdamWSF",
                 lr: float = 5e-3, wd: float = 0.0, wave: str = "square", temp: float = 0.5,
                 mode: str = "lls", readout: str = "linear") -> None:
        super().__init__()
        check(part, len(units))
        if readout not in ("linear", "lls"):
            raise ValueError(f"unknown readout {readout!r}")
        self.part = part
        self.readout = readout
        self.blocks = nn.ModuleList(Block([units[i] for i in r], n_cls, wave, temp, mode) for r in spans(part))
        self.cls = classifier(units[-1], n_cls) if readout == "linear" else None
        mods = [*self.blocks, *([self.cls] if self.cls is not None else [])]
        self.opts = [opt.make(optim, m.parameters(), lr, wd) for m in mods]

        log.info("GranularLLS mode=%s readout=%s partition=%s (%d blocks), optim=%s lr=%g "
                 "wave=%s T=%g", mode, readout, label(part), len(self.blocks), optim, lr,
                 wave, temp)
        for i, m in enumerate(mods):
            name = m.name if isinstance(m, Block) else "classifier"
            log.debug("block %d %-22s params=%d", i, name, sum(p.numel() for p in m.parameters()))

    def forward(self, x: Tensor) -> Tensor:
        """Inference only: no losses, no updates."""
        for b in self.blocks:
            x = b(x)
        if self.cls is None:
            return self.blocks[-1].head.scores(x)
        return self.cls(x)

    def forward_all(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Inference with each block's head scores (for per-block accuracy / early exit)."""
        scores = []
        for b in self.blocks:
            x = b(x)
            scores.append(b.head.scores(x))
        return (scores[-1] if self.cls is None else self.cls(x)), scores

    def step(self, x: Tensor, y: Tensor) -> Out:
        """One training step: each block updates from its own head; gradients stop at block inputs."""
        losses, hits = [], []
        h = x
        for b, o in zip(self.blocks, self.opts):
            h = b(h.detach())
            loss, s = b.head(h, y)
            o.zero_grad(set_to_none=True)
            loss.backward()
            o.step()
            losses.append(loss.detach())
            hits.append((s.argmax(1) == y).sum())

        if self.cls is None:
            return Out(s.detach(), torch.stack(losses), torch.stack(hits))

        logits = self.cls(h.detach())
        loss = F.cross_entropy(logits, y)
        o = self.opts[-1]
        o.zero_grad(set_to_none=True)
        loss.backward()
        o.step()
        losses.append(loss.detach())
        return Out(logits.detach(), torch.stack(losses), torch.stack(hits))

    def opt_mode(self, train: bool) -> None:
        opt.set_mode(self.opts, train)
