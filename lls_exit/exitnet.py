"""Learned exits for comparison with the fixed LLS heads: a backbone plus a linear head at each exit.

joint : backbone, exits and classifier trained together by backprop (mean of all exit losses + classifier loss).
frozen: a trained backbone (and its classifier) kept fixed; only the exit heads learn. On a BP backbone these are
        SDN-style IC-only exits; on an LLS backbone they are linear probes at the LLS heads' positions.
An exit sees exactly what an LLS head sees (the block output pooled to the unit's head pool, flattened), so the
two differ only in the projection: learned vs fixed basis.
"""
import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lls_granular import opt
from lls_granular.bp import BPNet
from lls_granular.heads import pool_flat
from lls_granular.partitions import Partition, check, label, spans
from lls_granular.units import Unit
from lls_granular.wrapper import Block, GranularLLS, Out, classifier

log = logging.getLogger(__name__)

VARIANTS = ("joint", "frozen")


class LinearExit(nn.Module):
    def __init__(self, u: Unit, n_cls: int) -> None:
        super().__init__()
        self.pool = u.pool
        self.fc = nn.Linear(u.head_dim, n_cls)

    def forward(self, h: Tensor) -> Tensor:
        return self.fc(pool_flat(h, self.pool))


class ExitNet(nn.Module):
    """Units with a LinearExit after the last unit of every block of `part`, plus the final classifier.
    Same step / forward_all / opt_mode API as GranularLLS, so lls_granular.train.fit and ckpt work as is."""

    def __init__(self, units: list[Unit], part: Partition, n_cls: int, frozen: bool = False,
                 cls: nn.Module | None = None, optim: str = "AdamWSF", lr: float = 5e-3, wd: float = 0.0) -> None:
        super().__init__()
        check(part, len(units))
        self.part, self.frozen = part, frozen
        self.units = nn.ModuleList(units)
        self.ends = [r.stop - 1 for r in spans(part)]
        self.heads = nn.ModuleList(LinearExit(units[e], n_cls) for e in self.ends)
        self.cls = cls if cls is not None else classifier(units[-1], n_cls)
        if frozen:
            for p in [*self.units.parameters(), *self.cls.parameters()]:
                p.requires_grad_(False)
        params = self.heads.parameters() if frozen else self.parameters()
        self.opts = [opt.make(optim, params, lr, wd)]
        self.train()
        log.info("ExitNet %s: exits %s (%d), frozen=%s, optim=%s lr=%g", label(part),
                 [units[e].name for e in self.ends], len(self.ends), frozen, optim, lr)
        log.debug("trainable params %d", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def train(self, mode: bool = True) -> "ExitNet":
        """A frozen backbone stays in eval mode (BN statistics untouched, also by bn_refresh)."""
        super().train(mode)
        if self.frozen:
            self.units.eval()
            self.cls.eval()
        return self

    def trunk(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Last unit's output and the outputs at every exit."""
        hs = []
        for i, u in enumerate(self.units):
            x = u(x)
            if i in self.ends:
                hs.append(x)
        return x, hs

    def forward(self, x: Tensor) -> Tensor:
        for u in self.units:
            x = u(x)
        return self.cls(x)

    def forward_all(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        h, hs = self.trunk(x)
        return self.cls(h), [hd(z) for hd, z in zip(self.heads, hs)]

    def step(self, x: Tensor, y: Tensor) -> Out:
        with torch.set_grad_enabled(not self.frozen):
            h, hs = self.trunk(x)
            logits = self.cls(h)
        scores = [hd(z) for hd, z in zip(self.heads, hs)]
        losses = [F.cross_entropy(s, y) for s in scores] + [F.cross_entropy(logits, y)]
        loss = torch.stack(losses[:-1]).sum() if self.frozen else torch.stack(losses).mean()
        o = self.opts[0]
        o.zero_grad(set_to_none=True)
        loss.backward()
        o.step()
        hits = torch.stack([(s.argmax(1) == y).sum() for s in scores])
        return Out(logits.detach(), torch.stack([v.detach() for v in losses]), hits)

    def opt_mode(self, train: bool) -> None:
        opt.set_mode(self.opts, train)


def parts(m: GranularLLS | BPNet) -> tuple[list[Unit], nn.Module, Partition | None]:
    """A trained model's units in order, its classifier, and its LLS partition (None for BP)."""
    if isinstance(m, BPNet):
        return list(m.units), m.cls, None  # type: ignore[arg-type]
    bs: list[Block] = list(m.blocks)  # type: ignore[arg-type]
    return [u for b in bs for u in b.units], m.cls, m.part  # type: ignore[misc]
