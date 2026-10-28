"""Sweep exit policies over dumps: accuracy vs MACs on val, the same settings on test, Pareto fronts, operating points.

python -m lls_exit.sweep --exp mobilenet300                  (dumps pulled from the wandb `exits-<run>` artifacts)
python -m lls_exit.sweep --exp mobilenet300 --dumps $LLS_OUT (dumps read from <dir>/<run>/exits.pt)
Each policy also runs on running exit ensembles (agg: ens = mean of the checked exits so far, ens2 = the last two).
Writes sweep.csv.gz (every setting), pareto.csv (val fronts + their test numbers), picks.csv (operating points), temps.json.
"""
import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from torch import Tensor

from lls_granular import logs
from lls_granular.exps import EXPS

from . import policies as P
from .dump import targets
from .track import art_name, load_exits

log = logging.getLogger(__name__)

FINALS = ("head", "cls")
TEMPS = ("train", "fit")
DELTAS = (0.0, 0.005, 0.01, 0.02)
AGGS: dict[str, int | None] = {"single": 1, "ens": None, "ens2": 2}  # ensemble window over the checked exits
GROUP = ["run", "temp", "final"]
KEY = [*GROUP, "agg", "S", "label"]


@dataclass(frozen=True)
class Costs:
    """Per-exit MACs from a dump: exits 0..K-1 are heads, K is cls."""

    backbone: Tensor
    head: Tensor
    full: float

    @classmethod
    def of(cls, ex: dict[str, Any]) -> "Costs":
        c = ex["costs"]
        return cls(c["backbone"].double(), c["head"].double(), float(c["full"]))


def exit_sets(k: int, final: str) -> dict[str, list[int]]:
    """'skip<j>' checks heads j..K-1 (then cls if it is the final exit); the final exit is always in."""
    tail = [k] if final == "cls" else []
    return {f"skip{j}": list(range(j, k)) + tail for j in range(k)}


def quantiles(x: Tensor, n: int) -> list[float]:
    """n evenly spaced quantiles of the finite values, rounded to 6 significant digits, deduplicated."""
    v = x[torch.isfinite(x)].flatten().double()
    if v.numel() > 1_000_000:
        v = v[torch.randperm(v.numel(), generator=torch.Generator().manual_seed(0))[:1_000_000]]
    q = torch.quantile(v, torch.linspace(0, 1, n, dtype=torch.float64))
    return sorted({float(f"{t:.6g}") for t in q.tolist()})


@dataclass(frozen=True)
class Family:
    """Threshold policies sharing one signal: a setting fires where signal >= threshold."""

    name: str
    fixed: dict[str, float]  # parameters held constant, e.g. {"w": 3} or {"w": 2, "m": 0.4}
    key: str                 # the swept parameter: "tau" or "n"
    values: list[float]

    def signal(self, p: Tensor, memo: dict[tuple[str, int], Tensor] | None = None) -> Tensor:
        """memo caches base signals of the same p across families (sync and margin are reused a lot)."""
        memo = {} if memo is None else memo

        def base(name: str, w: int = 0) -> Tensor:
            if (name, w) not in memo:
                fn = {"confidence": P.confidence, "entropy": P.neg_entropy, "margin": P.margin,
                      "patience": lambda q: P.streak(q).double(), "sync": lambda q: P.sync(q, w)}[name]
                memo[(name, w)] = fn(p)
            return memo[(name, w)]

        if self.name == "sync_margin":
            return torch.where(base("margin") >= self.fixed["m"], base("sync", int(self.fixed["w"])), float("-inf"))
        if self.name in ("confidence", "entropy", "margin", "patience", "sync"):
            return base(self.name, int(self.fixed.get("w", 0)))
        raise ValueError(f"unknown family {self.name!r}")

    def thresholds(self) -> Tensor:
        v = torch.tensor(self.values, dtype=torch.float64)
        return -v if self.name == "entropy" else v  # entropy: tau is the largest entropy allowed

    def policies(self) -> list[P.Policy]:
        return [P.Policy(self.name, {**self.fixed, self.key: v}) for v in self.values]

    def exits(self, p: Tensor, memo: dict[tuple[str, int], Tensor] | None = None) -> Tensor:
        """(T, N) exit position of every sample under every threshold (vectorised P.fire + P.first_exit)."""
        sig = self.signal(p, memo)
        f = sig[None] >= self.thresholds().to(sig.dtype)[:, None, None]  # same dtype as P.fire's comparison
        return P.first_exit(f)


def families(p: Tensor, n: int = 41) -> list[Family]:
    """What to sweep. Thresholds are quantiles of each signal on val, so every exit rate gets covered."""
    e = p.shape[1]
    fs = [Family("confidence", {}, "tau", quantiles(P.confidence(p), n)),
          Family("entropy", {}, "tau", quantiles(-P.neg_entropy(p), n)),
          Family("margin", {}, "tau", quantiles(P.margin(p), n)),
          Family("patience", {}, "n", [float(k) for k in range(2, e + 1)])]
    ms = quantiles(P.margin(p), 9)
    for w in range(2, min(4, e) + 1):
        sy = P.sync(p, w)
        fs.append(Family("sync", {"w": w}, "tau", quantiles(sy, n)))
        if w <= 3:
            fs += [Family("sync_margin", {"w": w, "m": m}, "tau", quantiles(sy, 9)) for m in ms]
    fs = [f for f in fs if f.values]
    log.debug("%d families, %d settings over %d exits", len(fs), sum(len(f.values) for f in fs), e)
    return fs


def metrics(ps: Tensor, y: Tensor, c: Costs, idx: Tensor, j: Tensor, ops: float, n_all: int,
            static: bool = False) -> list[dict[str, Any]]:
    """One row per setting. j is (T, N): position in idx where each sample exits. static = only that head is run."""
    bb, hd = c.backbone[idx], c.head[idx]
    cost_bb = bb[j]
    cost_hd = hd[j] if static else hd.cumsum(0)[j]
    dec = ops * (j + 1).clamp(max=len(idx) - 1).double()  # the final exit needs no decision
    tot = cost_bb + cost_hd + dec
    hit = ps.argmax(-1).gather(1, j.T).T == y  # (T, N)
    g = idx[j]
    hist = torch.zeros(len(j), n_all, dtype=torch.long).scatter_add_(1, g, torch.ones_like(g)).tolist()
    mean = tot.mean(1)
    cols = {"acc": hit.double().mean(1), "mmac": mean / 1e6, "bb_mmac": cost_bb.mean(1) / 1e6,
            "head_mmac": cost_hd.mean(1) / 1e6, "dec_mmac": dec.mean(1) / 1e6, "mac_frac": mean / c.full,
            "speedup": c.full / mean, "depth": g.double().mean(1)}
    vals = {k: v.tolist() for k, v in cols.items()}
    return [{**{k: v[t] for k, v in vals.items()}, "hist": json.dumps(hist[t])} for t in range(len(j))]


def aggregate(q: Tensor, agg: str) -> Tensor:
    """Probabilities the policy sees at each checked exit: that exit's own, or a running ensemble of checked exits."""
    w = AGGS[agg]
    return q if w == 1 else P.ensemble(q, w)


def agg_ops(agg: str, n_cls: int) -> float:
    """Extra ops per checked exit to update a running ensemble (add C, scale C; ens2 also subtracts C)."""
    return {"single": 0.0, "ens": 2.0 * n_cls, "ens2": 3.0 * n_cls}[agg]


def evaluate(p: Tensor, y: Tensor, c: Costs, s: list[int], pol: P.Policy, n_cls: int,
             agg: str = "single") -> dict[str, Any]:
    """One setting through the reference path (P.fire); the sweep uses the vectorised Family.exits."""
    idx = torch.tensor(s)
    ps = aggregate(p[:, idx], agg)
    if pol.name == "static":
        j = torch.full((len(y),), int(pol.params["e"]), dtype=torch.long)
    elif pol.name == "oracle":
        j = P.oracle_exit(ps, y)
    else:
        j = P.first_exit(P.fire(ps, pol))
    ops = P.decision_ops(pol, n_cls) + (agg_ops(agg, n_cls) if pol.name not in ("static", "oracle") else 0.0)
    return metrics(ps, y, c, idx, j[None], ops, p.shape[1], pol.name == "static")[0]


def sweep_run(ex: dict[str, Any], n: int = 41) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """Every (final, temperature, aggregation, exit set, policy) on val and test; static only for agg=single. Returns the rows and the fitted temperatures."""
    m, heads = ex["meta"], ex["heads"]
    run, k, n_cls = m["run"], len(heads["dim"]), m["n_cls"]
    c = Costs.of(ex)
    sps = {sp: ex["splits"][sp] for sp in ("val", "test")}
    ref = {sp: (d["logits"].argmax(1) == d["y"]).double().mean().item() for sp, d in sps.items()}
    rows: list[dict[str, Any]] = []
    temps: dict[str, list[float]] = {}
    for final in FINALS:
        temps[final] = P.fit_temps(sps["val"], heads, final)
        for temp in TEMPS:
            t = None if temp == "train" else temps[final]
            ps = {sp: P.probs(d, heads, final, t, m["temp"]) for sp, d in sps.items()}
            for agg in AGGS:
                fams = families(aggregate(ps["val"], agg), n)
                for sname, s in exit_sets(k, final).items():
                    idx = torch.tensor(s)
                    for sp, d in sps.items():
                        q, y = aggregate(ps[sp][:, idx], agg), d["y"]
                        n_all = ps[sp].shape[1]

                        def add(pols: list[P.Policy], j: Tensor, static: bool = False) -> None:
                            ops = P.decision_ops(pols[0], n_cls)
                            ops += agg_ops(agg, n_cls) if pols[0].name not in ("static", "oracle") else 0.0
                            for pol, r in zip(pols, metrics(q, y, c, idx, j, ops, n_all, static)):
                                rows.append({"run": run, "temp": temp, "final": final, "agg": agg, "S": sname,
                                             "policy": pol.name, "label": pol.label(),
                                             "params": json.dumps(pol.params), "split": sp, "ref_acc": ref[sp], **r})

                        memo: dict[tuple[str, int], Tensor] = {}
                        for f in fams:
                            add(f.policies(), f.exits(q, memo))
                        add([P.Policy("oracle")], P.oracle_exit(q, y)[None])
                        if sname == "skip0" and agg == "single":
                            e = len(s)
                            add([P.Policy("static", {"e": i}) for i in range(e)],
                                torch.arange(e)[:, None].expand(e, len(y)), static=True)
                log.info("%s final=%s temp=%s agg=%s: %d settings per exit set", run, final, temp, agg,
                         sum(len(f.values) for f in fams) + 1)
    return pd.DataFrame(rows), temps


def pareto(df: pd.DataFrame, x: str = "mac_frac", y: str = "acc") -> pd.DataFrame:
    """Non-dominated rows: nothing cheaper (or as cheap) is at least as accurate."""
    d = df.sort_values([x, y], ascending=[True, False])
    return cast(pd.DataFrame, d[d[y] > d[y].cummax().shift(fill_value=-1.0)])


def fronts(df: pd.DataFrame) -> pd.DataFrame:
    """Val Pareto front per run/temp/final (oracle excluded), joined with the same settings' test numbers."""
    val = cast(pd.DataFrame, df[(df.split == "val") & (df.policy != "oracle")])
    f = pd.concat([pareto(cast(pd.DataFrame, g)) for _, g in val.groupby(GROUP)], ignore_index=True)
    test = cast(pd.DataFrame, df[df.split == "test"][[*KEY, "acc", "mac_frac", "speedup"]])
    return f.merge(test, on=KEY, suffixes=("", "_test"))


def picks(df: pd.DataFrame, deltas: tuple[float, ...] = DELTAS) -> pd.DataFrame:
    """Per run/temp/final, policy ('any' = best of all) and pool (aggregations allowed: 'all' or one agg): the cheapest
    val setting whose val accuracy is within δ of the full network's, and the test numbers of that same setting."""
    val = cast(pd.DataFrame, df[(df.split == "val") & (df.policy != "oracle")])
    test = cast(pd.DataFrame, df[df.split == "test"]).set_index(KEY)
    out = []
    for gk, g in val.groupby(GROUP):
        gk = cast(tuple[str, str, str], gk)
        ref = g.ref_acc.iloc[0]
        cands = [(pol, pool, gp) for pool, ga in [("all", g), *g.groupby("agg")]
                 for pol, gp in [("any", ga), *ga.groupby("policy")]]
        for pol, pool, gp in cands:
            for d in deltas:
                ok = gp[gp.acc >= ref - d]
                if ok.empty:
                    continue
                r = ok.sort_values(["mac_frac", "acc"], ascending=[True, False]).iloc[0]
                t = test.loc[tuple(r[k] for k in KEY)]
                out.append({**dict(zip(GROUP, gk)), "policy": pol, "pool": pool, "delta": d, "agg": r["agg"],
                            "S": r.S, "label": r.label, "val_acc": r.acc, "val_ref": ref, "val_mac_frac": r.mac_frac,
                            "test_acc": t.acc, "test_ref": t.ref_acc, "test_mac_frac": t.mac_frac,
                            "test_speedup": t.speedup, "test_depth": t.depth})
    return pd.DataFrame(out)


def fetch(run: str, cache: Path, project: str, entity: str | None, refresh: bool) -> Path:
    """exits.pt of a run from its wandb artifact (cached locally)."""
    dst = cache / run
    if (dst / "exits.pt").exists() and not refresh:
        return dst / "exits.pt"
    import wandb
    api = wandb.Api(timeout=120)
    ent = entity or api.default_entity
    api.artifact(f"{ent}/{project}/{art_name(run)}:latest").download(root=str(dst))
    log.info("downloaded %s -> %s", art_name(run), dst)
    return dst / "exits.pt"


def args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep early-exit policies over exit dumps.")
    a = p.add_argument
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--exp", nargs="+", choices=sorted(EXPS), help="experiments whose non-BP runs to use")
    g.add_argument("--runs", nargs="+", help="run names instead, e.g. exit-head runs from lls_exit.train_exits")
    a("--seeds", type=int, nargs="*")
    a("--dumps", default=None, help="read <dir>/<run>/exits.pt instead of wandb artifacts")
    a("--project", default=os.environ.get("WANDB_PROJECT", "lls-granularity"))
    a("--entity", default=os.environ.get("WANDB_ENTITY"))
    a("--cache", default="dumps", help="where wandb artifacts are downloaded")
    a("--refresh", action="store_true", help="re-download artifacts")
    a("--out", default="report/exit")
    a("--n", type=int, default=41, help="thresholds per policy")
    a("--log", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    o = args(argv)
    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    logs.setup(o.log, out / "sweep.log")
    runs = o.runs or [rd.name for rd in targets(o.exp, o.seeds)]
    dfs, temps = [], {}
    for run in runs:
        f = Path(o.dumps) / run / "exits.pt" if o.dumps else fetch(run, Path(o.cache), o.project, o.entity, o.refresh)
        if not f.exists():
            log.warning("%s: no dump at %s, skipping", run, f)
            continue
        df, temps[run] = sweep_run(load_exits(f), o.n)
        dfs.append(df)
    if not dfs:
        raise SystemExit("no dumps found")
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(out / "sweep.csv.gz", index=False)
    fronts(df).to_csv(out / "pareto.csv", index=False)
    pk = picks(df)
    pk.to_csv(out / "picks.csv", index=False)
    (out / "temps.json").write_text(json.dumps(temps, indent=2))
    log.info("wrote %d rows, %d picks to %s", len(df), len(pk), out)
    show = cast(pd.DataFrame, pk[(pk.policy == "any") & (pk.pool == "all") & (pk.delta == 0.01)])
    print(show[["run", "temp", "final", "agg", "label", "S", "val_acc", "test_acc", "test_ref", "test_mac_frac",
                "test_speedup"]].to_string(index=False))


if __name__ == "__main__":
    logs.setup()
    main()
