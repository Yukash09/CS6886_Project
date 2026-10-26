"""Per-run diagnostics on dumps: per-exit accuracy and calibration, overthinking, class-wise depth, exit ensembles.

python -m lls_exit.diag --exp mobilenet300                  (dumps from wandb, cached in dumps/)
python -m lls_exit.diag --exp mobilenet300 --dumps $LLS_OUT
Writes per_exit.csv, overthink.csv, per_class.csv, reliability.csv. Probabilities are (N, E, C) as in policies.py.
"""
import argparse
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from lls_granular import logs
from lls_granular.exps import EXPS

from . import policies as P
from .dump import targets
from .sweep import FINALS, TEMPS, Costs, fetch
from .track import load_exits

log = logging.getLogger(__name__)

BINS = 15
WINDOWS = (2, 3)


# ---------- per exit ----------

def accuracy(p: Tensor, y: Tensor) -> Tensor:
    """(E,) accuracy of every exit."""
    return (p.argmax(-1) == y[:, None]).double().mean(0)


def ece(p: Tensor, y: Tensor, bins: int = BINS) -> Tensor:
    """(E,) expected calibration error of every exit (equal-width confidence bins)."""
    conf, pred = p.max(-1)
    hit = (pred == y[:, None]).double()
    b = (conf * bins).long().clamp(max=bins - 1)
    gap = torch.zeros(bins, p.shape[1], dtype=torch.float64)
    gap.scatter_add_(0, b, conf.double() - hit)  # per bin: Σ conf − Σ correct
    return gap.abs().sum(0) / len(y)


def reliability(p: Tensor, y: Tensor, bins: int = BINS) -> pd.DataFrame:
    """Long table (exit, bin, n, conf, acc) for reliability diagrams; empty bins dropped."""
    conf, pred = p.max(-1)
    hit = (pred == y[:, None]).double()
    b = (conf * bins).long().clamp(max=bins - 1)
    rows = []
    for e in range(p.shape[1]):
        n = torch.bincount(b[:, e], minlength=bins)
        sc = torch.zeros(bins, dtype=torch.float64).scatter_add_(0, b[:, e], conf[:, e].double())
        sh = torch.zeros(bins, dtype=torch.float64).scatter_add_(0, b[:, e], hit[:, e])
        rows += [{"exit": e, "bin": i, "n": int(n[i]), "conf": (sc[i] / n[i]).item(), "acc": (sh[i] / n[i]).item()}
                 for i in range(bins) if n[i] > 0]
    return pd.DataFrame(rows)


def agreement(p: Tensor) -> Tensor:
    """(E,) fraction of samples where an exit predicts what the final exit predicts."""
    a = p.argmax(-1)
    return (a == a[:, -1:]).double().mean(0)


def ensemble(p: Tensor, w: int | None = None, geo: bool = False) -> Tensor:
    """(N, E, C) running ensemble: exit e averages exits max(0, e−w+1)..e (all so far if w is None).
    geo: mean of log-probabilities, renormalised (geometric mean); else arithmetic mean."""
    q = p.clamp_min(P.EPS).log() if geo else p
    c = q.cumsum(1)
    e = p.shape[1]
    n = torch.arange(1, e + 1, dtype=p.dtype)
    if w is not None and w < e:
        c = torch.cat([c[:, :w], c[:, w:] - c[:, :-w]], 1)
        n = n.clamp(max=w)
    m = c / n[None, :, None]
    return torch.softmax(m, -1) if geo else m


# ---------- per sample ----------

def settle(p: Tensor) -> Tensor:
    """(N,) earliest exit from which every later exit predicts the final prediction (head-based prediction depth)."""
    a = p.argmax(-1)
    e = torch.arange(a.shape[1])
    return torch.where(a != a[:, -1:], e, -1).max(1).values + 1


def flips(p: Tensor) -> Tensor:
    """(N,) number of times the prediction changes from one exit to the next."""
    a = p.argmax(-1)
    return (a[:, 1:] != a[:, :-1]).sum(1)


def overthinking(p: Tensor, y: Tensor) -> dict[str, float]:
    """Final vs oracle accuracy, and how often an earlier exit was right and the final one wrong (SDN overthinking)."""
    hit = p.argmax(-1) == y[:, None]
    fin, early = hit[:, -1], hit[:, :-1].any(1)
    st = settle(p).double()
    e = p.shape[1]
    return {"final_acc": fin.double().mean().item(), "oracle_acc": (fin | early).double().mean().item(),
            "overthink": (early & ~fin).double().mean().item(),          # right before, wrong at the end
            "early_right": (fin & (st < e - 1)).double().mean().item(),  # settled on the right answer early
            "settle": st.mean().item(), "settle_frac": (st / (e - 1)).mean().item() if e > 1 else 0.0,
            "flips": flips(p).double().mean().item()}


def per_class(p: Tensor, y: Tensor, n_cls: int) -> pd.DataFrame:
    """Per class: count, final accuracy, mean settle / oracle exit (exit index)."""
    fin = (p[:, -1].argmax(-1) == y).double()
    st, orc = settle(p).double(), P.oracle_exit(p, y).double()
    n = torch.bincount(y, minlength=n_cls).double()

    def mean(v: Tensor) -> list[float]:
        return (torch.zeros(n_cls, dtype=torch.float64).scatter_add_(0, y, v) / n.clamp_min(1)).tolist()

    return pd.DataFrame({"cls": range(n_cls), "n": n.long().tolist(), "acc": mean(fin), "settle": mean(st),
                         "oracle_exit": mean(orc)})


# ---------- one run ----------

def names(ex: dict[str, Any], final: str) -> list[str]:
    k = len(ex["heads"]["dim"])
    nm = list(ex["meta"].get("names") or [])
    nm = nm[:k] if len(nm) >= k else [f"h{i}" for i in range(k)]
    return nm + (["cls"] if final == "cls" else [])


def diagnose(ex: dict[str, Any], split: str = "test") -> dict[str, pd.DataFrame]:
    """All diagnostics of one dump on `split` (temperatures fitted on val). Keys: per_exit, overthink,
    per_class, reliability."""
    m, heads = ex["meta"], ex["heads"]
    run, n_cls, c = m["run"], m["n_cls"], Costs.of(ex)
    sp, val = ex["splits"][split], ex["splits"]["val"]
    y = sp["y"]
    out: dict[str, list[pd.DataFrame]] = {"per_exit": [], "overthink": [], "per_class": [], "reliability": []}
    for final in FINALS:
        nm = names(ex, final)
        e = len(nm)
        bb, hd = c.backbone[:e], c.head[:e]
        tag = {"run": run, "final": final}
        fitted = P.fit_temps(val, heads, final)
        for temp in TEMPS:
            p = P.probs(sp, heads, final, None if temp == "train" else fitted, m["temp"])
            cols: dict[str, Any] = {"exit": range(e), "name": nm, "temp_fit": fitted,
                                    "mac_frac": ((bb + hd) / c.full).tolist(),
                                    "mac_frac_ens": ((bb + hd.cumsum(0)) / c.full).tolist(),
                                    "acc": accuracy(p, y).tolist(), "ece": ece(p, y).tolist(),
                                    "agree_final": agreement(p).tolist(),
                                    "ens_all": accuracy(ensemble(p), y).tolist(),
                                    "ens_all_geo": accuracy(ensemble(p, geo=True), y).tolist()}
            cols |= {f"ens_w{w}": accuracy(ensemble(p, w), y).tolist() for w in WINDOWS}
            out["per_exit"].append(pd.DataFrame({**tag, "temp": temp, **cols}))
            out["reliability"].append(reliability(p, y).assign(**tag, temp=temp))
            if temp == "train":  # argmax-only diagnostics don't depend on the temperature
                out["overthink"].append(pd.DataFrame([{**tag, "n_exits": e, **overthinking(p, y)}]))
                out["per_class"].append(per_class(p, y, n_cls).assign(**tag))
        log.debug("%s final=%s: diagnosed %d exits", run, final, e)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


# ---------- CLI ----------

def args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-exit diagnostics over exit dumps.")
    a = p.add_argument
    a("--exp", nargs="+", required=True, choices=sorted(EXPS))
    a("--seeds", type=int, nargs="*")
    a("--dumps", default=None, help="read <dir>/<run>/exits.pt instead of wandb artifacts")
    a("--project", default=os.environ.get("WANDB_PROJECT", "lls-granularity"))
    a("--entity", default=os.environ.get("WANDB_ENTITY"))
    a("--cache", default="dumps")
    a("--refresh", action="store_true")
    a("--split", default="test", choices=["val", "test"])
    a("--out", default="report/exit")
    a("--log", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    o = args(argv)
    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    logs.setup(o.log, out / "diag.log")
    parts: dict[str, list[pd.DataFrame]] = {}
    for rd in targets(o.exp, o.seeds):
        run = rd.name
        f = Path(o.dumps) / run / "exits.pt" if o.dumps else fetch(run, Path(o.cache), o.project, o.entity, o.refresh)
        if not f.exists():
            log.warning("%s: no dump at %s, skipping", run, f)
            continue
        for k, df in diagnose(load_exits(f), o.split).items():
            parts.setdefault(k, []).append(df)
    if not parts:
        raise SystemExit("no dumps found")
    dfs = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    for k, df in dfs.items():
        df.to_csv(out / f"{k}.csv", index=False)
    log.info("wrote %s to %s", ", ".join(f"{k} ({len(v)})" for k, v in dfs.items()), out)
    print(dfs["overthink"].drop(columns=["settle"]).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    logs.setup()
    main()
