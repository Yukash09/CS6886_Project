"""Per-exit inference cost (MACs/sample): backbone up to the exit, heads checked, final classifier."""
import logging
from dataclasses import dataclass
from itertools import accumulate

import torch
from torch import Tensor

from lls_granular.metrics import Costs
from lls_granular.partitions import Partition, spans
from lls_granular.units import Unit

log = logging.getLogger(__name__)


def head_macs(u: Unit, n_cls: int) -> int:
    """LLS head on u's output: average-pool (one add per element), L2-normalise (d), project (d·C)."""
    return u.numel + u.head_dim + u.head_dim * n_cls


def cls_macs(u: Unit, n_cls: int) -> int:
    """Final classifier on the last unit: global pool for conv outputs (MobileNet), then linear."""
    pool = u.numel if len(u.shape) == 3 else 0
    return pool + u.shape[0] * n_cls


@dataclass(frozen=True)
class ExitCosts:
    """Exits 0..K-1 are the block heads, exit K is `cls`. MACs per sample, float64."""

    names: list[str]
    backbone: Tensor  # (K+1,) cumulative forward MACs up to each exit; cls entry = whole backbone
    head: Tensor      # (K+1,) cost of checking each exit; cls entry = classifier
    full: float       # plain network: whole backbone + classifier, no heads

    def per_sample(self, ex: Tensor, checked: Tensor) -> tuple[Tensor, Tensor]:
        """(backbone, heads) MACs for samples exiting at ex, given the bool mask (K+1,) of checked exits."""
        if not checked[ex].all():
            raise ValueError("a sample exits at an exit that is not checked")
        paid = torch.cumsum(self.head * checked, 0)
        return self.backbone[ex], paid[ex]


def exit_costs(units: list[Unit], part: Partition, n_cls: int) -> ExitCosts:
    """Costs of every exit of a granular model (pass freshly built units; they are run once on CPU)."""
    macs = Costs(units, n_cls, 1, torch.device("cpu")).macs
    cum = list(accumulate(macs))
    rs = spans(part)
    last = units[-1]
    names = ["+".join(units[i].name for i in r) for r in rs] + ["cls"]
    bb = [cum[r.stop - 1] for r in rs] + [cum[-1]]
    hd = [head_macs(units[r.stop - 1], n_cls) for r in rs] + [cls_macs(last, n_cls)]
    c = ExitCosts(names, torch.tensor(bb, dtype=torch.float64), torch.tensor(hd, dtype=torch.float64),
                  float(cum[-1] + hd[-1]))
    log.debug("exit costs (MMAC) full=%.2f: %s", c.full / 1e6,
              ", ".join(f"{n}={b / 1e6:.2f}+{h / 1e6:.3f}" for n, b, h in zip(names, bb, hd)))
    return c
