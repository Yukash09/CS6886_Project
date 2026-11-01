"""Local heads: turn a block's output into a loss (and class scores)."""
import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

log = logging.getLogger(__name__)

F_MIN, F_MAX = 50.0, 512.0  # upstream compute_LLS frequency range
LLS_MODES = ("lls", "lls-m", "lls-mxm")


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
    """Reduce an activation to the fixed-length vector consumed by an LLS head.

    CNN maps retain the upstream adaptive 2-D pooling.  Vector activations use
    adaptive 1-D pooling.  Transformer activations have shape ``(B, N, D)``;
    for those, concatenate the CLS token with the mean patch token before the
    optional dimensionality reduction.  This lets every local loss supervise
    both the global token and the distributed patch representation.
    """
    if h.dim() == 4:
        h = F.adaptive_avg_pool2d(h, size)
    elif h.dim() == 3:
        if h.shape[1] < 2:
            raise ValueError("token activations require a CLS token and at least one patch token")
        h = torch.cat((h[:, 0], h[:, 1:].mean(dim=1)), dim=1)
        h = F.adaptive_avg_pool1d(h, size)
    elif h.dim() == 2:
        h = F.adaptive_avg_pool1d(h, size)
    else:
        raise ValueError(f"unsupported activation rank {h.dim()}; expected 2, 3, or 4")
    return h.flatten(1)


class LLSHead(nn.Module):
    """Projects the pooled, L2-normalised block output onto a fixed basis; no parameters."""

    B: Tensor

    def __init__(self, n_cls: int, dim: int, pool: int, wave: str = "square",
                 temp: float = 0.5, smooth: float = 0.0, mode: str = "lls") -> None:
        super().__init__()
        if mode not in LLS_MODES:
            raise ValueError(f"unknown LLS mode {mode!r}")
        self.pool = pool
        self.temp = temp
        self.smooth = smooth
        self.mode = mode
        self.register_buffer("B", basis(n_cls, dim, wave), persistent=False)
        if mode == "lls-m":
            self.M = nn.Parameter(0.01 * torch.randn(n_cls))
        elif mode == "lls-mxm":
            self.M = nn.Parameter(0.01 * torch.randn(n_cls, n_cls))
        log.debug("LLSHead mode=%s C=%d d=%d pool=%d wave=%s T=%.2f coherence=%.3f",
                  mode, n_cls, dim, pool, wave, temp, coherence(self.B))

    def scores(self, h: Tensor) -> Tensor:
        """Unit-norm output projected on each class row, shape (batch, C). Rows are not normalised."""
        z = pool_flat(h, self.pool)
        if z.shape[1] != self.B.shape[1]:
            raise ValueError(f"head expects dim {self.B.shape[1]}, got {z.shape[1]}")
        scores = F.normalize(z, dim=1) @ self.B.T
        if self.mode == "lls-m":
            scores = scores * self.M
        elif self.mode == "lls-mxm":
            # Matches the paper's released implementation: class scores are
            # linearly combined by the learned C x C modulation matrix.
            scores = scores @ self.M
        return scores

    def forward(self, h: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        s = self.scores(h)
        loss = F.cross_entropy(s / self.temp, y, label_smoothing=self.smooth)
        return loss, s
