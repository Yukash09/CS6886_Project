"""End-to-end backprop baseline with the same units, classifier and step API as GranularLLS."""
import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import opt
from .units import Unit
from .wrapper import Out, classifier

log = logging.getLogger(__name__)


class BPNet(nn.Module):
    def __init__(self, units: list[Unit], n_cls: int, optim: str = "AdamWSF",
                 lr: float = 5e-3, wd: float = 0.0) -> None:
        super().__init__()
        self.units = nn.Sequential(*units)
        self.cls = classifier(units[-1], n_cls)
        self.opts = [opt.make(optim, self.parameters(), lr, wd)]
        log.info("BPNet %d units, optim=%s lr=%g", len(units), optim, lr)
        log.debug("BPNet params=%d", sum(p.numel() for p in self.parameters()))

    def forward(self, x: Tensor) -> Tensor:
        return self.cls(self.units(x))

    def step(self, x: Tensor, y: Tensor) -> Out:
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        o = self.opts[0]
        o.zero_grad(set_to_none=True)
        loss.backward()
        o.step()
        return Out(logits.detach(), loss.detach().unsqueeze(0), torch.zeros(0, dtype=torch.long))

    def opt_mode(self, train: bool) -> None:
        opt.set_mode(self.opts, train)
