"""Prediction depth (Baldock et al., 2021) against exit depth: are the samples a policy lets out early the easy ones?

python -m lls_exit.depth knn --exp main                              (where the runs' feats.pt are; GPU if any)
python -m lls_exit.depth analyze --exp main --out report/exit/vgg8   (CPU; depth.pt + exits.pt from wandb)
knn: a k-NN classifier (cosine, k = 30, the dump's train subset as reference) on every exit's head input ->
     depth.pt next to exits.pt, uploaded as the training run's `depth-<run>` artifact.
analyze: prediction depth = the earliest exit from which every later k-NN prediction, and the network (cls), agree
     with the network's prediction (K = only the network itself). Per test sample it is compared with the heads'
     settle depth, the oracle exit, and the exit of each policy's δ operating point (from the sweep's picks.csv).
     Writes depth_samples.csv.gz, depth_corr.csv (Spearman), depth_bins.csv (per depth), depth_class.csv.
"""
import argparse
import logging
import os
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from torch import Tensor

from lls_granular import logs
from lls_granular.exps import EXPS

from . import policies as P
from .diag import settle, settle_labels
from .dump import cfg_of, targets
from .sweep import exit_sets, fetch
from .track import load_exits, publish

log = logging.getLogger(__name__)

K_NN = 30
SPLITS = ("val", "test")


# ---------- k-NN predictions (where the features are) ----------

@torch.no_grad()
def knn(ref: Tensor, ref_y: Tensor, q: Tensor, n_cls: int, k: int = K_NN, dev: torch.device | None = None,
        chunk: int = 1024) -> Tensor:
    """(N,) majority label among the k most cosine-similar reference samples (features are already L2-normalised;
    ties go to the lower class index)."""
    dev = dev or torch.device("cpu")
    r, ry = ref.to(dev, torch.float32), ref_y.to(dev)
    out = []
    for i in range(0, len(q), chunk):
        nb = ry[(q[i:i + chunk].to(dev, torch.float32) @ r.T).topk(min(k, len(r)), 1).indices]  # (b, k)
        votes = torch.zeros(len(nb), n_cls, device=dev).scatter_add_(1, nb, torch.ones(nb.shape, device=dev))
        out.append(votes.argmax(1).cpu())
    return torch.cat(out)


def knn_run(fe: dict[str, Any], n_cls: int, k: int = K_NN, dev: torch.device | None = None) -> dict[str, Any]:
    """k-NN predictions (N, E) of every exit for val and test, with the train subset as reference."""
    tr = fe["splits"]["train"]
    out: dict[str, Any] = {"meta": {"run": fe["meta"]["run"], "k": k, "n_ref": len(tr["y"])}, "splits": {}}
    for sp in SPLITS:
        if sp not in fe["splits"]:
            continue
        d = fe["splits"][sp]
        pr = [knn(r, tr["y"], f, n_cls, k, dev) for r, f in zip(tr["feats"], d["feats"])]
        out["splits"][sp] = {"knn": torch.stack(pr, 1).short(), "y": d["y"], "idx": d["idx"]}
        log.info("%s %s: k-NN acc per exit %s", fe["meta"]["run"], sp,
                 [round(v, 3) for v in (torch.stack(pr, 1) == d["y"][:, None]).double().mean(0).tolist()])
    return out


def knn_summary(dp: dict[str, Any]) -> dict[str, Any]:
    s: dict[str, Any] = {"depth/k": dp["meta"]["k"], "depth/n_ref": dp["meta"]["n_ref"]}
    for sp, d in dp["splits"].items():
        acc = (d["knn"] == d["y"][:, None]).double().mean(0).tolist()
        s |= {f"depth/{e}/{sp}_knn_acc": a for e, a in enumerate(acc)}
    return s


def knn_one(rd: Path, o: argparse.Namespace, dev: torch.device) -> dict[str, Any] | str:
    if not (rd / "feats.pt").exists() or not (rd / "exits.pt").exists():
        return "no feats"
    if (rd / "depth.pt").exists() and not o.fresh:
        return "exists"
    logs.setup(o.log, rd / "depth.log")
    n_cls = load_exits(rd / "exits.pt")["meta"]["n_cls"]
    fe = torch.load(rd / "feats.pt", map_location="cpu", weights_only=False)
    if "train" not in fe["splits"]:
        return "no train feats"
    dp = knn_run(fe, n_cls, o.k, dev)
    torch.save(dp, rd / "depth.pt")
    s = knn_summary(dp)
    up = False
    if o.wandb:
        cfg = cfg_of(rd)
        up = publish(rd, rd / "depth.pt", s, o.project or cfg.get("project") or "lls-granularity",
                     o.entity or cfg.get("entity"), kind="depth")
    acc = [s[f"depth/{e}/test_knn_acc"] for e in range(dp["splits"]["test"]["knn"].shape[1])]
    return {"exits": len(acc), "first_knn": acc[0], "last_knn": acc[-1], "wandb": up}


# ---------- analysis (CPU) ----------

def pred_depth(knn_pred: Tensor, net: Tensor) -> Tensor:
    """(N,) prediction depth in exit units 0..K, K = only the network's own output agrees."""
    return settle_labels(torch.cat([knn_pred.long(), net[:, None]], 1))


def parse(label: str) -> P.Policy:
    """Inverse of Policy.label(): 'sync w=2 tau=0.99' -> Policy('sync', {'w': 2, 'tau': 0.99})."""
    name, *kv = label.split()
    return P.Policy(name, {k: float(v) for k, v in (x.split("=") for x in kv)})


def pick_exits(ex: dict[str, Any], pk: pd.DataFrame) -> dict[str, Tensor]:
    """Test exit (0..K, K = cls) of every sample under each pick (single-exit picks, final cls, fitted T)."""
    heads, sp = ex["heads"], ex["splits"]["test"]
    p = P.probs(sp, heads, "cls", P.fit_temps(ex["splits"]["val"], heads, "cls"), ex["meta"]["temp"])
    sets = exit_sets(len(heads["dim"]), "cls")
    out = {}
    for _, r in pk.iterrows():
        idx = torch.tensor(sets[r.S])
        out[str(r.policy)] = idx[P.first_exit(P.fire(p[:, idx], parse(str(r.label))))]
    return out


def samples(ex: dict[str, Any], dp: dict[str, Any], pk: pd.DataFrame) -> pd.DataFrame:
    sp, heads = ex["splits"]["test"], ex["heads"]
    d = dp["splits"]["test"]
    if not torch.equal(d["idx"], sp["idx"]):
        raise ValueError(f"{ex['meta']['run']}: depth.pt and exits.pt test samples differ")
    y, net = sp["y"], sp["logits"].argmax(1)
    p = P.probs(sp, heads, "cls")
    cols: dict[str, Any] = {"idx": sp["idx"], "y": y, "net": net, "correct": (net == y).long(),
                            "pd": pred_depth(d["knn"], net), "settle": settle(p), "oracle": P.oracle_exit(p, y)}
    cols |= {f"exit_{k}": v for k, v in pick_exits(ex, pk).items()}
    return pd.DataFrame({k: v.tolist() for k, v in cols.items()}).assign(run=ex["meta"]["run"],
                                                                       n_exits=len(heads["dim"]) + 1)


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def summarise(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    depth_cols = ["settle", "oracle", *[c for c in df.columns if c.startswith("exit_")]]
    corr, bins, cls = [], [], []
    for run, g in df.groupby("run", sort=False):
        cs = {c: cast(pd.Series, g[c]) for c in depth_cols}
        corr += [{"run": run, "vs": c, "spearman": spearman(cast(pd.Series, g.pd), v), "n": len(g)}
                 for c, v in cs.items() if bool(v.notna().all()) and v.nunique() > 1]
        b = g.groupby("pd").agg(n=("y", "size"), acc=("correct", "mean"), **{c: (c, "mean") for c in depth_cols})
        bins.append(b.reset_index().assign(run=run, frac=lambda x: x.n / len(g)))
        cls.append(g.groupby("y").agg(pd=("pd", "mean"), acc=("correct", "mean")).reset_index().assign(run=run))
    return {"depth_corr": pd.DataFrame(corr), "depth_bins": pd.concat(bins, ignore_index=True),
            "depth_class": pd.concat(cls, ignore_index=True)}


# ---------- CLI ----------

def args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prediction depth (k-NN) vs exit depth.")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("knn", "analyze"):
        s = sub.add_parser(name)
        g = s.add_mutually_exclusive_group(required=True)
        g.add_argument("--exp", nargs="+", choices=sorted(EXPS), help="experiments whose non-BP runs to use")
        if name == "knn":
            g.add_argument("--run-dir", help="a single run folder with exits.pt and feats.pt")
            s.add_argument("--k", type=int, default=K_NN)
            s.add_argument("--fresh", action="store_true", help="redo runs that already have depth.pt")
            s.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
        else:
            g.add_argument("--runs", nargs="+", help="run names instead")
            s.add_argument("--dumps", default=None, help="read <dir>/<run>/{exits,depth}.pt instead of wandb")
            s.add_argument("--cache", default="dumps")
            s.add_argument("--refresh", action="store_true")
            s.add_argument("--out", default="report/exit")
            s.add_argument("--picks", default=None, help="the sweep's picks.csv (default: <out>/picks.csv)")
            s.add_argument("--delta", type=float, default=0.01)
        s.add_argument("--seeds", type=int, nargs="*")
        s.add_argument("--project", default=os.environ.get("WANDB_PROJECT") if name == "analyze" else None)
        s.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
        s.add_argument("--log", default=None)
    o = p.parse_args(argv)
    if o.cmd == "analyze" and not o.project:
        o.project = "lls-granularity"
    return o


def main_knn(o: argparse.Namespace) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rds = [Path(o.run_dir).expanduser()] if o.run_dir else targets(o.exp, o.seeds)
    log.info("k-NN prediction depth for %d runs on %s (k=%d)", len(rds), dev, o.k)
    rows = {rd.name: knn_one(rd, o, dev) for rd in rds}
    df = pd.DataFrame([{"run": k, **(v if isinstance(v, dict) else {"status": v})} for k, v in rows.items()])
    print(df.to_string(index=False, float_format="%.3f"))
    if any(isinstance(v, str) and v.startswith("no") for v in rows.values()):
        raise SystemExit(1)


def main_analyze(o: argparse.Namespace) -> None:
    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    logs.setup(o.log, out / "depth.log")
    pk = pd.read_csv(o.picks or out / "picks.csv")
    pk = cast(pd.DataFrame, pk[(pk.temp == "fit") & (pk.final == "cls") & (pk.pool == "single")
                               & (pk.delta == o.delta) & (pk.policy != "static")])
    dfs = []
    for run in o.runs or [rd.name for rd in targets(o.exp, o.seeds)]:
        def get(kind: str) -> Path:
            if o.dumps:
                return Path(o.dumps) / run / f"{kind}.pt"
            return fetch(run, Path(o.cache), o.project, o.entity, o.refresh, kind=kind)
        try:
            ex, dp = load_exits(get("exits")), load_exits(get("depth"))
        except Exception as e:  # no dump / artifact yet
            log.warning("%s: skipped (%s)", run, e)
            continue
        dfs.append(samples(ex, dp, cast(pd.DataFrame, pk[pk.run == run])))
    if not dfs:
        raise SystemExit("no runs with depth.pt")
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(out / "depth_samples.csv.gz", index=False)
    res = summarise(df)
    for k, v in res.items():
        v.to_csv(out / f"{k}.csv", index=False)
    log.info("wrote depth_samples (%d) and %s to %s", len(df), ", ".join(res), out)
    print(res["depth_corr"].pivot(index="run", columns="vs", values="spearman").to_string(float_format="%.2f"))


def main(argv: list[str] | None = None) -> None:
    o = args(argv)
    if o.cmd == "knn":
        main_knn(o)
    else:
        main_analyze(o)


if __name__ == "__main__":
    logs.setup()
    main()
