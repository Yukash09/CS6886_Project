"""Report figures and tables from wandb: python -m lls_granular.analyze --out report

Pulls finished runs, joins them with the analytic cost of each partition, and writes:
  fronts (accuracy vs activation memory), training curves, LLS per-block accuracy,
  short-run proxy vs final ranking, budget picks, CIFAR10 vs CIFAR100, and results.md.
"""
import argparse
import logging
import os
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch

from . import logs
from .metrics import Costs
from .partitions import Partition
from .track import method
from .units import ARCHS

log = logging.getLogger(__name__)

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#a9a8a2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
KEEP = ["test_at_best_val", "best/val_acc", "best/epoch", "sys/peak_mem_mb", "time/train_s_mean"]


# ---------- data ----------

def fetch(path: str, jobs: Iterable[str]) -> pd.DataFrame:
    """One row per finished run (latest wins on duplicate names)."""
    import wandb
    api = wandb.Api(timeout=120)
    rows = []
    for r in api.runs(path, filters={"jobType": {"$in": list(jobs)}, "state": "finished"}):
        c, s = r.config, r.summary
        rows.append({"id": r.id, "name": r.name, "job": r.job_type, "created": r.created_at,
                     "arch": c.get("arch"), "dataset": c.get("dataset"), "method": c.get("method"),
                     "blocks": c.get("blocks"), "epochs": c.get("epochs"), "seed": c.get("seed"),
                     **{k: s.get(k) for k in KEEP}})
    df = pd.DataFrame(rows)
    if len(df):
        df = dedup(df[df["test_at_best_val"].notna()])  # a run marked finished without results was interrupted
    log.info("fetched %d finished runs from %s", len(df), path)
    return df


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("created").drop_duplicates("name", keep="last").reset_index(drop=True)


def history(path: str, run_id: str, keys: list[str]) -> pd.DataFrame:
    import wandb
    r = wandb.Api(timeout=120).run(f"{path}/{run_id}")
    h = pd.DataFrame(list(r.scan_history(keys=["epoch", *keys])))
    return h.sort_values("epoch").reset_index(drop=True) if len(h) else h


def acc_within(h: pd.DataFrame, e: int) -> tuple[float, float]:
    """(best val, test at that epoch) using only epochs < e."""
    w = h[h["epoch"] < e]
    i = w["val/acc"].idxmax()
    return float(w.loc[i, "val/acc"]), float(w.loc[i, "test/acc"])


def cost_table(arch: str, n_cls: int, parts: list[Partition | None], bs: int = 8,
               target: int = 128) -> pd.DataFrame:
    """Analytic activation MB at batch `target` (computed at `bs`, scaled linearly) and train MMAC."""
    torch.manual_seed(0)
    c = Costs(ARCHS[arch](), n_cls, bs, torch.device("cpu"))
    rows = []
    for p in parts:
        r = c.cost(p)
        rows.append({"method": method(p), "act_mb": r["act_peak_mb"] * target / bs, "train_mmac": r["train_mmac"]})
    return pd.DataFrame(rows)


def pretty(m: str) -> str:
    """Readable label: 'BP', 'LLS (per layer)', 'one block', or 'blocks 2-2-3'."""
    p = parse_method(m)
    if p is None:
        return "BP"
    if max(p) == 1:
        return "LLS (per layer)"
    return "one block" if len(p) == 1 else "blocks " + "-".join(map(str, p))


def short(m: str) -> str:
    """Compact block sizes for plot labels: 'p4.2.1.1.1.1' -> '4,2,1x4'."""
    out: list[str] = []
    for k, g in groupby(parse_method(m) or ()):
        n = len(list(g))
        out += [f"1×{n}"] if k == 1 and n >= 3 else [str(k)] * n
    return ",".join(out)


def parse_method(m: str) -> Partition | None:
    return None if m == "bp" else tuple(int(x) for x in m[1:].split("."))


# ---------- analysis ----------

def pareto(df: pd.DataFrame, x: str, y: str) -> pd.Series:
    """True for points not dominated (lower-or-equal x and higher-or-equal y, one strict)."""
    keep = []
    for _, r in df.iterrows():
        dom = ((df[x] <= r[x]) & (df[y] >= r[y]) & ((df[x] < r[x]) | (df[y] > r[y]))).any()
        keep.append(not dom)
    return pd.Series(keep, index=df.index)


def kendall(a: list[float], b: list[float]) -> float:
    """Kendall tau-b."""
    n, conc, disc, ta, tb = len(a), 0, 0, 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ta += 1
            elif db == 0:
                tb += 1
            elif da * db > 0:
                conc += 1
            else:
                disc += 1
    den = ((conc + disc + ta) * (conc + disc + tb)) ** 0.5
    return (conc - disc) / den if den else 0.0


def budget(df: pd.DataFrame, budgets: list[float], pick: str, mem: str = "act_mb") -> pd.DataFrame:
    """For each memory budget: the run with the highest `pick` among those within budget."""
    rows = []
    for b in budgets:
        ok = df[df[mem] <= b]
        if not len(ok):
            continue
        r = ok.loc[ok[pick].idxmax()]
        rows.append({"budget_mb": b, "method": r["method"], "act_mb": r[mem], "test": r["test"]})
    return pd.DataFrame(rows)


# ---------- figures ----------

def _style() -> Any:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "lines.linewidth": 2, "legend.frameon": False,
    })
    return plt


def _save(fig: Any, out: Path, name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    log.info("wrote %s.png/.pdf", out / name)


def fig_front(df: pd.DataFrame, title: str, out: Path, name: str, labels: dict[str, str]) -> None:
    """Accuracy vs activation memory; Pareto front in blue, LLS orange, BP aqua, the rest neutral."""
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    front = df[df["front"]].sort_values("act_mb")
    rest = df[~df["front"] & ~df["method"].isin(["bp", labels.get("lls", "")])]
    ax.scatter(rest["act_mb"], rest["test"] * 100, s=36, color=NEUTRAL, edgecolor=SURFACE, linewidth=1.5,
               label="other partitions", zorder=2)
    ax.plot(front["act_mb"], front["test"] * 100, color=BLUE, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label="Pareto front", zorder=3)
    for key, color in (("lls", ORANGE), ("bp", AQUA)):
        m = labels.get(key)
        p = df[df["method"] == m]
        if len(p):
            ax.scatter(p["act_mb"], p["test"] * 100, s=70, color=color, edgecolor=SURFACE, linewidth=1.5,
                       zorder=4, label="LLS (per-layer)" if key == "lls" else "BP")
    named = {m: t for m, t in labels.items() if m not in ("lls", "bp")}
    named |= {m: named.get(m, short(m)) for m in front["method"] if m not in (labels.get("lls"), "bp")}
    for m, txt in named.items():
        p = df[df["method"] == m]
        if len(p):
            ax.annotate(txt, (p["act_mb"].iloc[0], p["test"].iloc[0] * 100), textcoords="offset points",
                        xytext=(7, 0), va="center", fontsize=8, color=MUTED)
    ax.set_xlabel("Peak activation memory for training (MB, batch 128)")
    ax.set_ylabel("Test accuracy at best val (%)")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, out, name)
    plt.close(fig)


def fig_lines(curves: dict[str, pd.DataFrame], key: str, title: str, ylabel: str, out: Path, name: str) -> None:
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for (label, h), color in zip(curves.items(), SERIES):
        ax.plot(h["epoch"] + 1, h[key] * 100, color=color, linewidth=1.6, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    _save(fig, out, name)
    plt.close(fig)


def fig_blocks(blocks: dict[str, list[float]], out: Path, name: str) -> None:
    """Per-layer LLS head accuracy, one panel per model (no shared y-scale tricks)."""
    plt = _style()
    fig, axes = plt.subplots(1, len(blocks), figsize=(3.4 * len(blocks), 3.4), squeeze=False)
    for ax, (label, acc) in zip(axes[0], blocks.items()):
        xs = list(range(1, len(acc) + 1))
        ax.plot(xs, [a * 100 for a in acc], color=BLUE, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_title(label, loc="left")
        ax.set_xlabel("Layer")
        ax.set_xticks(xs)
    axes[0][0].set_ylabel("Local head val accuracy (%)")
    _save(fig, out, name)
    plt.close(fig)


def fig_proxy(df: pd.DataFrame, tau: float, e: int, out: Path, name: str) -> None:
    plt = _style()
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.scatter(df["val_short"] * 100, df["val"] * 100, s=36, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
    ax.set_xlabel(f"Best val accuracy within {e} epochs (%)")
    ax.set_ylabel("Best val accuracy, full run (%)")
    ax.set_title(f"Short runs rank partitions (Kendall τ = {tau:.2f})", loc="left")
    _save(fig, out, name)
    plt.close(fig)


# ---------- pipeline ----------

def front_table(runs: pd.DataFrame, arch: str, dataset: str, n_cls: int, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    df = runs[(runs["arch"] == arch) & (runs["dataset"] == dataset)].copy()
    if extra is not None:
        df = pd.concat([df, extra], ignore_index=True)
    df["test"] = df["test_at_best_val"]
    costs = cost_table(arch, n_cls, [parse_method(m) for m in df["method"]])
    df = df.merge(costs.drop_duplicates("method"), on="method", how="left")
    df["front"] = pareto(df, "act_mb", "test")
    return df.sort_values("act_mb").reset_index(drop=True)


def bp_at(path: str, runs: pd.DataFrame, arch: str, dataset: str, e: int) -> pd.DataFrame:
    """BP is not in the sweep: take the 300-epoch BP run truncated to e epochs, for a like-for-like front."""
    r = runs[(runs["arch"] == arch) & (runs["dataset"] == dataset) & (runs["method"] == "bp")]
    if not len(r):
        return pd.DataFrame()
    r = r.sort_values("epochs").iloc[-1]
    v, t = acc_within(history(path, r["id"], ["val/acc", "test/acc"]), e)
    return pd.DataFrame([{**r.to_dict(), "best/val_acc": v, "test_at_best_val": t, "epochs": e}])


PCT = {"test", "std", "cifar10", "cifar100", "drop", "oracle_test", "proxy_test"}


def seed_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of test accuracy over seeds, for methods run with at least two seeds."""
    g = df.groupby("method")["test_at_best_val"].agg(["mean", "std", "count"]).reset_index()
    return g[g["count"] >= 2].rename(columns={"mean": "test", "std": "std"}).sort_values("test").reset_index(drop=True)


def table_md(df: pd.DataFrame, cols: list[str]) -> str:
    """Markdown table; accuracy columns as percentages, other floats to 1 decimal."""
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"

    def cell(c: str, v: Any) -> str:
        if c in PCT and isinstance(v, float):
            return f"{v * 100:.2f}"
        return f"{v:.1f}" if isinstance(v, float) else str(v)
    return head + "".join("| " + " | ".join(cell(c, r[c]) for c in cols) + " |\n" for _, r in df.iterrows())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Report figures and tables from wandb.")
    p.add_argument("--project", default=f"{os.environ.get('WANDB_ENTITY', 'cs23b069-iit-madras-')}/"
                                        f"{os.environ.get('WANDB_PROJECT', 'lls-granularity')}")
    p.add_argument("--out", default="report")
    p.add_argument("--proxy", type=int, default=10, help="epochs of the cheap proxy (H5)")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md: list[str] = ["# Granular LLS: results\n"]

    runs = fetch(a.project, ["G-U", "G-P", "G-BP", "G-ALL", "G-SEED", "G-MN", "G-C100"])
    runs.to_csv(out / "runs.csv", index=False)

    # VGG8 main (300 epochs): table + curves
    main = runs[(runs["job"].isin(["G-U", "G-P", "G-BP"])) & (runs["arch"] == "vgg8")]
    if len(main):
        curves = {pretty(m): history(a.project, i, ["val/acc"]) for m, i in zip(main["method"], main["id"])}
        fig_lines(curves, "val/acc", "VGG8 / CIFAR10: validation accuracy", "Val accuracy (%)", out, "vgg8_curves")
        t = front_table(main, "vgg8", "cifar10aug", 10)
        t.to_csv(out / "vgg8_main.csv", index=False)
        md += ["## VGG8 / CIFAR10, main splits (300 epochs)\n", table_md(t, ["method", "test", "best/epoch", "act_mb", "train_mmac"])]

    # VGG8 sweep front (+ BP truncated to the sweep's epochs) and the short-run proxy
    sweep = runs[(runs["job"] == "G-ALL") & (runs["seed"] == 0)]
    if len(sweep):
        e = int(sweep["epochs"].max())
        t = front_table(sweep, "vgg8", "cifar10aug", 10, bp_at(a.project, runs, "vgg8", "cifar10aug", e))
        t.to_csv(out / "vgg8_sweep.csv", index=False)
        fig_front(t, f"VGG8 / CIFAR10: all partitions ({len(sweep)} of 64, {e} epochs)", out, "vgg8_front",
                  {"lls": "p1.1.1.1.1.1.1", "bp": "bp", "p7": "single block", "p1.1.1.1.3": "merge last 3"})
        md += [f"## VGG8 sweep ({len(sweep)}/64 partitions, {e} epochs): Pareto front\n",
               table_md(t[t["front"]], ["method", "test", "act_mb", "train_mmac"])]

        short = []
        for _, r in t[t["method"] != "bp"].iterrows():
            v, _ = acc_within(history(a.project, r["id"], ["val/acc", "test/acc"]), a.proxy)
            short.append(v)
        g = t[t["method"] != "bp"].assign(val_short=short, val=lambda d: d["best/val_acc"])
        tau = kendall(g["val_short"].tolist(), g["val"].tolist())
        fig_proxy(g, tau, a.proxy, out, "proxy_rank")
        bp_mb = float(t.loc[t["method"] == "bp", "act_mb"].iloc[0]) if (t["method"] == "bp").any() else g["act_mb"].max()
        budgets = [round(f * bp_mb) for f in (0.5, 0.6, 0.7, 0.8, 1.0)]
        oracle = budget(g.assign(test=g["test"]), budgets, "val").rename(columns={"method": "oracle", "test": "oracle_test"})
        proxy = budget(g, budgets, "val_short")[["budget_mb", "method", "test"]].rename(
            columns={"method": "proxy_pick", "test": "proxy_test"})
        bt = oracle.merge(proxy, on="budget_mb")
        bt.to_csv(out / "budget.csv", index=False)
        md += [f"## Short-run proxy (H5)\nKendall τ between best val within {a.proxy} epochs and the full run: **{tau:.3f}** "
               f"({len(g)} partitions).\n\n### Budget picks (oracle = best full-run val; proxy = best {a.proxy}-epoch val)\n",
               table_md(bt, ["budget_mb", "oracle", "oracle_test", "proxy_pick", "proxy_test"])]

    # Seed variability at the sweep's epochs (seed 0 from the sweep, seeds 1-2 from G-SEED)
    if len(sweep):
        e = int(sweep["epochs"].max())
        pool = runs[(runs["arch"] == "vgg8") & (runs["dataset"] == "cifar10aug") & (runs["epochs"] == e)
                    & runs["job"].isin(["G-ALL", "G-SEED"])]
        st = seed_stats(pool)
        if len(st):
            st.to_csv(out / "seeds.csv", index=False)
            md += [f"## Seed variability (VGG8, {e} epochs)\n", table_md(st, ["method", "test", "std", "count"])]

    # MobileNet front (300 epochs)
    mn = runs[(runs["arch"] == "mobilenet") & (runs["epochs"] == 300)]
    if len(mn):
        t = front_table(mn, "mobilenet", "cifar10aug", 10)
        t.to_csv(out / "mobilenet_front.csv", index=False)
        lab = {"lls": "p" + ".".join(["1"] * 14), "bp": "bp", "p4.2.6.2": "coarse 4,2,6,2",
               "p2." + ".".join(["1"] * 12): "merge first 2", "p" + ".".join(["1"] * 12) + ".2": "merge last 2",
               "p" + ".".join(["1"] * 10) + ".4": "merge last 4"}
        fig_front(t, "MobileNetV1 / CIFAR10 (300 epochs)", out, "mobilenet_front", lab)
        md += ["## MobileNetV1 / CIFAR10 (300 epochs)\n", table_md(t, ["method", "test", "act_mb", "train_mmac", "front"])]

    # LLS per-layer head accuracy (H6)
    blocks = {}
    for arch, m, label in (("vgg8", "p1.1.1.1.1.1.1", "VGG8"), ("mobilenet", "p" + ".".join(["1"] * 14), "MobileNetV1")):
        r = runs[(runs["arch"] == arch) & (runs["method"] == m) & (runs["dataset"] == "cifar10aug")]
        if len(r):
            r = r.sort_values("epochs").iloc[-1]
            n = len(parse_method(m) or ())
            h = history(a.project, r["id"], [f"block/{k}/val_acc" for k in range(n)])
            if len(h):
                last = h.iloc[-1]
                blocks[label] = [float(last[f"block/{k}/val_acc"]) for k in range(n)]
    if blocks:
        fig_blocks(blocks, out, "lls_layers")
        md += ["## LLS per-layer head accuracy (final epoch)\n"] + \
              [f"- {k}: " + ", ".join(f"{v * 100:.1f}" for v in acc) + "\n" for k, acc in blocks.items()]

    # CIFAR10 vs CIFAR100 (H3)
    c100 = runs[runs["job"] == "G-C100"]
    if len(c100):
        e100 = int(c100["epochs"].max())
        rows = []
        for m in c100["method"]:
            c10, sw = main[main["method"] == m], sweep[sweep["method"] == m]
            if len(c10):  # no LR schedule: best within the first e100 epochs of the 300-epoch run
                acc, ep = acc_within(history(a.project, c10["id"].iloc[0], ["val/acc", "test/acc"]), e100)[1], e100
            elif len(sw):
                acc, ep = float(sw["test_at_best_val"].iloc[0]), int(sw["epochs"].iloc[0])
            else:
                acc, ep = float("nan"), 0
            rows.append({"method": m, "cifar10": acc, "c10_epochs": ep,
                         "cifar100": float(c100.loc[c100["method"] == m, "test_at_best_val"].iloc[0])})
        ct = pd.DataFrame(rows)
        ct["drop"] = ct["cifar10"] - ct["cifar100"]
        ct.to_csv(out / "cifar100.csv", index=False)
        md += [f"## CIFAR10 vs CIFAR100 (H3), CIFAR100 at {e100} epochs\n",
               table_md(ct, ["method", "cifar10", "c10_epochs", "cifar100", "drop"])]

    (out / "results.md").write_text("\n".join(md))
    log.info("wrote %s", out / "results.md")


if __name__ == "__main__":
    logs.setup()
    main()
