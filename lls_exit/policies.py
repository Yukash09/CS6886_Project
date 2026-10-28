"""Exit policies on dumped scores: per-exit probabilities, stop signals, and the exit each sample takes.

Probabilities are (N, E, C) over the checked exits in order; the last one is the final exit (always taken).
"""
import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

log = logging.getLogger(__name__)

T_GRID = torch.logspace(-3, 1, 161)
EPS = 1e-12


# ---------- probabilities ----------

def cosines(scores: Tensor, row_norm: Tensor) -> Tensor:
    """Head scores are cos·‖basis row‖; divide the norms out -> true cosines in [-1, 1], (N, K, C)."""
    return scores / row_norm[None]


def fit_temp(z: Tensor, y: Tensor, grid: Tensor = T_GRID) -> float:
    """Temperature minimising the NLL of softmax(z / T) (log-spaced grid; deterministic)."""
    nll = torch.stack([F.cross_entropy(z / t, y) for t in grid])
    return grid[nll.argmin()].item()


def exit_logits(sp: dict[str, Any], heads: dict[str, Any], final: str) -> list[Tensor]:
    """Per-exit (N, C) cosines for every head, plus cls logits if final == 'cls'."""
    cos = cosines(sp["scores"], heads["row_norm"])
    zs = list(cos.unbind(1))
    if final == "cls":
        zs.append(sp["logits"])
    elif final != "head":
        raise ValueError(f"final must be 'head' or 'cls', got {final!r}")
    return zs


def fit_temps(sp: dict[str, Any], heads: dict[str, Any], final: str) -> list[float]:
    """One temperature per exit, fit on this split (use val)."""
    ts = [fit_temp(z, sp["y"]) for z in exit_logits(sp, heads, final)]
    log.debug("fitted temperatures %s", [round(t, 4) for t in ts])
    return ts


def probs(sp: dict[str, Any], heads: dict[str, Any], final: str, temps: list[float] | None = None,
          t_train: float = 0.5) -> Tensor:
    """(N, E, C) exit probabilities. temps=None: what training saw (softmax(scores / t_train), cls softmax);
    else softmax(cosine / T_e) and softmax(cls / T_cls) with fitted temperatures."""
    if final not in ("head", "cls"):
        raise ValueError(f"final must be 'head' or 'cls', got {final!r}")
    if temps is None:
        ps = [F.softmax(s / t_train, -1) for s in sp["scores"].unbind(1)]
        if final == "cls":
            ps.append(F.softmax(sp["logits"], -1))
        return torch.stack(ps, 1)
    zs = exit_logits(sp, heads, final)
    if len(temps) != len(zs):
        raise ValueError(f"{len(temps)} temperatures for {len(zs)} exits")
    return torch.stack([F.softmax(z / t, -1) for z, t in zip(zs, temps)], 1)


def ensemble(p: Tensor, w: int | None = None, geo: bool = False) -> Tensor:
    """(N, E, C) running ensemble: exit e averages exits max(0, e−w+1)..e (all so far if w is None).
    geo: mean of log-probabilities, renormalised (geometric mean); else arithmetic mean."""
    q = p.clamp_min(EPS).log() if geo else p
    c = q.cumsum(1)
    e = p.shape[1]
    n = torch.arange(1, e + 1, dtype=p.dtype)
    if w is not None and w < e:
        c = torch.cat([c[:, :w], c[:, w:] - c[:, :-w]], 1)
        n = n.clamp(max=w)
    m = c / n[None, :, None]
    return torch.softmax(m, -1) if geo else m


# ---------- stop signals (N, E): higher = more ready to exit ----------

def confidence(p: Tensor) -> Tensor:
    return p.max(-1).values


def neg_entropy(p: Tensor) -> Tensor:
    return (p * p.clamp_min(EPS).log()).sum(-1)


def margin(p: Tensor) -> Tensor:
    t = p.topk(2, -1).values
    return t[..., 0] - t[..., 1]


def streak(p: Tensor) -> Tensor:
    """Number of consecutive exits, ending here, that predict the same class (1 = changed just now)."""
    a = p.argmax(-1)
    out = torch.ones_like(a)
    for e in range(1, a.shape[1]):
        out[:, e] = torch.where(a[:, e] == a[:, e - 1], out[:, e - 1] + 1, 1)
    return out


def sync(p: Tensor, w: int) -> Tensor:
    """Mean pairwise cosine of the probability vectors over the last w exits; -inf before w exits exist."""
    u = F.normalize(p, dim=-1)
    g = u @ u.transpose(1, 2)  # (N, E, E)
    out = torch.full(p.shape[:2], float("-inf"))
    pairs = w * (w - 1) / 2
    for e in range(w - 1, p.shape[1]):
        blk = g[:, e - w + 1:e + 1, e - w + 1:e + 1]
        out[:, e] = (blk.sum((1, 2)) - w) / 2 / pairs  # off-diagonal mean (diagonal = 1)
    return out


# ---------- policies ----------

@dataclass(frozen=True)
class Policy:
    name: str
    params: dict[str, float] = field(default_factory=dict)

    def label(self) -> str:
        return self.name + "".join(f" {k}={v:g}" for k, v in self.params.items())


def fire(p: Tensor, pol: Policy) -> Tensor:
    """(N, E) bool: where the policy would stop. The final exit is added by first_exit."""
    q = pol.params
    match pol.name:
        case "confidence":
            return confidence(p) >= q["tau"]
        case "entropy":
            return neg_entropy(p) >= -q["tau"]  # tau = max allowed entropy (nats)
        case "margin":
            return margin(p) >= q["tau"]
        case "patience":
            return streak(p) >= q["n"]
        case "sync":
            return sync(p, int(q["w"])) >= q["tau"]
        case "sync_margin":
            return (sync(p, int(q["w"])) >= q["tau"]) & (margin(p) >= q["m"])
        case "static":
            f = torch.zeros(p.shape[:2], dtype=torch.bool)
            f[:, int(q["e"])] = True
            return f
    raise ValueError(f"unknown policy {pol.name!r}")


def first_exit(f: Tensor) -> Tensor:
    """Index of the first stop along the last (exit) axis; samples that never stop take the final exit.
    (N, E) -> (N,), or batched (T, N, E) -> (T, N)."""
    f = f.clone()
    f[..., -1] = True
    return f.int().argmax(-1)


def oracle_exit(p: Tensor, y: Tensor) -> Tensor:
    """Earliest exit that is correct (final exit if none is): the lower bound no policy can beat."""
    return first_exit(p.argmax(-1) == y[:, None])


def decision_ops(pol: Policy, n_cls: int) -> float:
    """Approximate ops per checked (non-final) exit to make the stop decision, on top of the head itself:
    softmax 3C, then max C | entropy 2C | top-2 2C | argmax C | w-1 new dot products C each + a norm C."""
    c = n_cls
    sm = 3 * c
    w = int(pol.params.get("w", 0))
    ops = {"confidence": sm + c, "entropy": sm + 2 * c, "margin": sm + 2 * c, "patience": sm + c + 1,
           "sync": sm + w * c, "sync_margin": sm + w * c + 2 * c, "static": 0.0, "oracle": 0.0}
    if pol.name not in ops:
        raise ValueError(f"unknown policy {pol.name!r}")
    return float(ops[pol.name])
