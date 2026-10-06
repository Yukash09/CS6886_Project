"""Cost of a partition: activation memory saved for backward, training MACs, measured GPU peak."""
import logging
from typing import Callable, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .heads import LLSHead
from .partitions import Partition, spans
from .units import Unit
from .wrapper import classifier

log = logging.getLogger(__name__)

MB = 2**20


def saved_bytes(fn: Callable[[], Tensor], static: Iterable[Tensor]) -> int:
    """Bytes autograd keeps for backward while running fn(); weights/buffers (static) excluded."""
    skip = {t.untyped_storage().data_ptr() for t in static}
    seen: dict[int, int] = {}

    def pack(t: Tensor) -> Tensor:
        s = t.untyped_storage()
        if s.data_ptr() not in skip:
            seen[s.data_ptr()] = s.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    del out
    return sum(seen.values())


def _static(*ms: nn.Module) -> list[Tensor]:
    return [t for m in ms for t in (*m.parameters(), *m.buffers())]


class Costs:
    """Per-span activation bytes and per-unit MACs for one architecture, memoised across partitions."""

    def __init__(self, units: list[Unit], n_cls: int, bs: int, dev: torch.device,
                 in_shape: tuple[int, ...] = (3, 32, 32)) -> None:
        self.units = [u.to(dev).train() for u in units]
        self.n_cls, self.bs, self.dev = n_cls, bs, dev
        self.shapes = [in_shape] + [u.shape for u in units]
        self.cls = classifier(units[-1], n_cls).to(dev)
        self.y = torch.randint(0, n_cls, (bs,), device=dev)
        self.macs = self._macs()
        self._span: dict[tuple[int, int], int] = {}

    def _x(self, i: int) -> Tensor:
        return torch.randn(self.bs, *self.shapes[i], device=self.dev)

    def _macs(self) -> list[int]:
        """Forward MACs per sample of each unit (conv: out·cin/groups·k², linear: out·in)."""
        total = [0]

        def hook(m: nn.Module, _: tuple[Tensor, ...], out: Tensor) -> None:
            per = out[0].numel()
            if isinstance(m, nn.Conv2d):
                kh, kw = m.kernel_size
                total[0] += per * (m.in_channels // m.groups) * kh * kw
            elif isinstance(m, nn.Linear):
                total[0] += per * m.in_features

        res = []
        x = torch.randn(1, *self.shapes[0], device=self.dev)
        with torch.no_grad():
            for u in self.units:
                u.eval()  # batch of 1: BN needs eval mode; MACs are the same
                hs = [m.register_forward_hook(hook) for m in u.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
                total[0] = 0
                x = u(x)
                res.append(total[0])
                for h in hs:
                    h.remove()
                u.train()
        log.debug("forward MACs per unit %s", res)
        return res

    def span(self, a: int, b: int) -> int:
        """Activation bytes of a block over units [a, b) trained with its own LLS head."""
        if (a, b) not in self._span:
            us = nn.Sequential(*self.units[a:b])
            last = self.units[b - 1]
            head = LLSHead(self.n_cls, last.head_dim, last.pool).to(self.dev)
            x = self._x(a)
            self._span[(a, b)] = saved_bytes(lambda: head(us(x), self.y)[0], _static(us, head))
        return self._span[(a, b)]

    def bp(self) -> int:
        us = nn.Sequential(*self.units)
        x = self._x(0)
        return saved_bytes(lambda: F.cross_entropy(self.cls(us(x)), self.y), _static(us, self.cls))

    def cost(self, part: Partition | None) -> dict[str, float]:
        """Peak activation MB (largest block; all of it for BP) and training MACs per sample."""
        n = len(self.units)
        rs = spans(part) if part else [range(n)]
        firsts = {r.start for r in rs}
        fwd = sum(self.macs)
        cls_fwd = self.units[-1].shape[0] * self.n_cls
        if part:
            act = [self.span(r.start, r.stop) for r in rs]
            heads = sum(self.n_cls * self.units[r.stop - 1].head_dim for r in rs)
            grad_in = sum(m for i, m in enumerate(self.macs) if i not in firsts)
            train = 2 * fwd + grad_in + 2 * heads + 2 * cls_fwd
        else:
            act = [self.bp()]
            heads = 0
            grad_in = sum(self.macs[1:]) + cls_fwd
            train = 2 * fwd + grad_in + 2 * cls_fwd
        return {"act_peak_mb": max(act) / MB, "act_sum_mb": sum(act) / MB,
                "fwd_mmac": fwd / 1e6, "train_mmac": train / 1e6, "signal_kmac": 2 * heads / 1e3}


def gpu_peak(m: nn.Module, bs: int, n_cls: int, dev: torch.device, in_shape: tuple[int, ...] = (3, 32, 32),
             warm: int = 2) -> dict[str, float]:
    """Measured training memory (MB) with cuDNN benchmark off: static (weights + optimizer state) and peak."""
    torch.backends.cudnn.benchmark = False
    m.to(dev).train()
    x = torch.randn(bs, *in_shape, device=dev)
    y = torch.randint(0, n_cls, (bs,), device=dev)
    for _ in range(warm):
        m.step(x, y)  # type: ignore[operator]
    torch.cuda.synchronize(dev)
    static = torch.cuda.memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    m.step(x, y)  # type: ignore[operator]
    torch.cuda.synchronize(dev)
    peak = torch.cuda.max_memory_allocated(dev)
    return {"gpu_static_mb": static / MB, "gpu_peak_mb": peak / MB, "gpu_act_mb": (peak - static) / MB}
