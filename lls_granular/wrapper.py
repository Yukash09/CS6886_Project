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
    def __init__(self, units: list[Unit], n_cls: int, wave: str, temp: float) -> None:
        super().__init__()
        self.units = nn.Sequential(*units)
        last = units[-1]
        self.head = LLSHead(n_cls, last.head_dim, last.pool, wave, temp)
        self.name = "+".join(u.name for u in units)

    def forward(self, x: Tensor) -> Tensor:
        return self.units(x)


def classifier(last: Unit, n_cls: int) -> nn.Sequential:
    """Global-pool conv outputs (MobileNet); pass vectors through (VGG8)."""
    pre = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()) if len(last.shape) == 3 else nn.Identity()
    return nn.Sequential(pre, nn.Linear(last.shape[0], n_cls))


class GranularLLS(nn.Module):
    def __init__(self, units: list[Unit], part: Partition, n_cls: int, optim: str = "AdamWSF",
                 lr: float = 5e-3, wd: float = 0.0, wave: str = "square", temp: float = 0.5) -> None:
        super().__init__()
        check(part, len(units))
        self.part = part
        self.blocks = nn.ModuleList(Block([units[i] for i in r], n_cls, wave, temp) for r in spans(part))
        self.cls = classifier(units[-1], n_cls)
        mods = [*self.blocks, self.cls]
        self.opts = [opt.make(optim, m.parameters(), lr, wd) for m in mods]

        log.info("GranularLLS partition=%s (%d blocks), optim=%s lr=%g wave=%s T=%g",
                 label(part), len(self.blocks), optim, lr, wave, temp)
        for i, m in enumerate(mods):
            name = m.name if isinstance(m, Block) else "classifier"
            log.debug("block %d %-22s params=%d", i, name, sum(p.numel() for p in m.parameters()))

    def forward(self, x: Tensor) -> Tensor:
        """Inference only: no losses, no updates."""
        for b in self.blocks:
            x = b(x)
        return self.cls(x)

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
