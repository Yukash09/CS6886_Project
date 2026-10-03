"""Local heads: turn a block's output into a loss (and class scores)."""
import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

log = logging.getLogger(__name__)

F_MIN, F_MAX = 50.0, 512.0  # upstream compute_LLS frequency range


def basis(n: int, d: int, wave: str = "square", device: torch.device | None = None) -> Tensor:
    """Fixed (n × d) basis, row c = sin(f_c · t). Same formula as upstream, minus `.cuda()`."""
    f = torch.linspace(F_MIN, F_MAX, n, device=device).unsqueeze(1)
    t = torch.arange(d, device=device).float().unsqueeze(0)
    b = torch.sin(f * t)
    if wave == "square":
        return torch.sign(b)
    if wave == "cosine":  # upstream's name; the wave is actually a sine
        return b
    raise ValueError(f"unknown wave {wave!r}")


def coherence(b: Tensor) -> float:
    """Max |cosine similarity| between two distinct basis rows (0 = orthogonal)."""
    u = F.normalize(b, dim=1)
    g = (u @ u.T).abs().fill_diagonal_(0)
    return g.max().item()


def pool_flat(h: Tensor, size: int) -> Tensor:
    """Adaptive-average-pool to `size` (spatial side for 4D, length for 2D), then flatten."""
    h = F.adaptive_avg_pool2d(h, size) if h.dim() == 4 else F.adaptive_avg_pool1d(h, size)
    return h.flatten(1)


class LLSHead(nn.Module):
    """Projects the pooled, L2-normalised block output onto a fixed basis; no parameters."""

    B: Tensor

    def __init__(self, n_cls: int, dim: int, pool: int, wave: str = "square",
                 temp: float = 0.5, smooth: float = 0.0) -> None:
        super().__init__()
        self.pool = pool
        self.temp = temp
        self.smooth = smooth
        self.register_buffer("B", basis(n_cls, dim, wave), persistent=False)
        log.debug("LLSHead C=%d d=%d pool=%d wave=%s T=%.2f coherence=%.3f",
                  n_cls, dim, pool, wave, temp, coherence(self.B))

    def scores(self, h: Tensor) -> Tensor:
        """Unit-norm output projected on each class row, shape (batch, C). Rows are not normalised."""
        z = pool_flat(h, self.pool)
        if z.shape[1] != self.B.shape[1]:
            raise ValueError(f"head expects dim {self.B.shape[1]}, got {z.shape[1]}")
        return F.normalize(z, dim=1) @ self.B.T

    def forward(self, h: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        s = self.scores(h)
        loss = F.cross_entropy(s / self.temp, y, label_smoothing=self.smooth)
        return loss, s
