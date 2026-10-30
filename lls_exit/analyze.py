"""Early-exit report figures and tables from the sweep / diag outputs: python -m lls_exit.analyze

Reads report/exit/<set>/{sweep.csv.gz, pareto.csv, picks.csv, per_exit.csv, overthink.csv, per_class.csv,
reliability.csv} for the sets that exist (mobilenet, vgg8, vgg8_sweep, ...) and writes report/exit/figs/*.png|pdf
and report/exit/tables.md. Test numbers throughout; every operating point is chosen on val (cls final, fitted T).
"""
import argparse
import logging
import re
from pathlib import Path
from typing import Any, cast

import pandas as pd

from lls_granular import logs
from lls_granular.analyze import (AQUA, BLUE, GRID, INK, MUTED, NEUTRAL, ORANGE, SERIES, SURFACE, _save, _style,
                                  cost_table, parse_method, short)

from .sweep import pareto

log = logging.getLogger(__name__)

ARCH = {"mobilenet": "MobileNetV1", "vgg8": "VGG8"}
LLS = {"mobilenet": "p" + ".".join(["1"] * 14), "vgg8": "p1.1.1.1.1.1.1"}
POLICIES = ["confidence", "entropy", "margin", "patience", "sync", "sync_margin"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
FIRST = [("1 (conv1 alone)", (1,), NEUTRAL, "o"), ("2 (conv1+conv2)", (2,), ORANGE, "D"),
         ("3-4", (3, 4), BLUE, "s"), ("5-7", (5, 6, 7), "#104281", "^")]  # first-block groups
CIFAR10 = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
DELTA = 0.01
SEL = {"temp": "fit", "final": "cls"}


# ---------- data ----------

def method_of(run: str) -> str:
    """'mobilenet_cifar10aug_p4.2.6.2_e300_s0' -> 'p4.2.6.2'."""
    m = re.search(r"_(p[\d.]+)_e\d+", run)
    return m.group(1) if m else run


def label(run: str) -> str:
    m = method_of(run)
    p = parse_method(m) if m.startswith("p") else None
    if p is None:
        return m
    if max(p) == 1:
        return f"LLS 1×{len(p)}"
    return "single block" if len(p) == 1 else short(m)


def sel(df: pd.DataFrame, **kw: Any) -> pd.DataFrame:
    for k, v in {**SEL, **kw}.items():
        if k in df.columns:
            df = cast(pd.DataFrame, df[df[k] == v])
    return df


def load(d: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for f in ("pareto.csv", "picks.csv", "per_exit.csv", "overthink.csv", "per_class.csv", "reliability.csv",
              "depth_bins.csv", "depth_corr.csv"):
        if (d / f).exists():
            out[f.removesuffix(".csv")] = pd.read_csv(d / f)
    if (d / "sweep.csv.gz").exists():
        out["sweep"] = pd.read_csv(d / "sweep.csv.gz")
    log.debug("%s: %s", d.name, {k: len(v) for k, v in out.items()})
    return out


def policy_fronts(sw: pd.DataFrame, run: str) -> pd.DataFrame:
    """Per policy (single-exit probabilities): the val Pareto front, with the same settings' test numbers."""
    d = sel(sw, run=run, agg="single")
    key = ["S", "label"]
    test = d[d.split == "test"].set_index(key)[["acc", "mac_frac"]]
    rows = []
    for pol, g in d[(d.split == "val") & d.policy.isin(POLICIES)].groupby("policy"):
        f = pareto(cast(pd.DataFrame, g))
        t = test.loc[[tuple(r) for r in f[key].values]]
        rows.append(pd.DataFrame({"policy": pol, "acc": t.acc.values, "mac_frac": t.mac_frac.values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def picks_table(pk: pd.DataFrame, delta: float = DELTA) -> pd.DataFrame:
    p = sel(pk, policy="any", pool="all")
    p = p[p.delta == delta].copy()
    p["split"] = p.run.map(label)
    p["policy_pick"] = p.label.str.split().str[0] + " (" + p["agg"] + ", " + p.S + ")"
    return p.sort_values("test_mac_frac")


# ---------- figures ----------

def fig_profile(sets: dict[str, dict[str, pd.DataFrame]], out: Path) -> None:
    """Per-exit test accuracy against cumulative MACs, LLS per layer, one panel per network."""
    plt = _style()
    panels = [(a, s["per_exit"]) for a, s in sets.items() if a in ARCH and "per_exit" in s]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.8), squeeze=False)
    for ax, (a, pe) in zip(axes[0], panels):
        d = sel(pe, run=next((r for r in pe.run.unique() if method_of(r) == LLS[a]), ""))
        if d.empty:
            continue
        heads = d[d["name"] != "cls"]
        ax.plot(heads.mac_frac * 100, heads.acc * 100, color=BLUE, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label="LLS head after each layer")
        full = d[d["name"] == "cls"].acc.iloc[0] * 100
        ax.axhline(full, color=NEUTRAL, linewidth=1.2, linestyle="--", label=f"full network ({full:.1f}%)")
        for i in {1, len(heads) // 2} & set(range(len(heads))):
            r = heads.iloc[i]
            ax.annotate(r["name"], (r.mac_frac * 100, r.acc * 100), xytext=(6, -10), textcoords="offset points",
                        fontsize=8, color=MUTED)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title(f"{ARCH[a]}, LLS per layer", loc="left")
        ax.set_xlabel("Cumulative inference MACs up to the exit (% of full network)")
        ax.set_xlim(-2, 102)
    axes[0][0].set_ylabel("Test accuracy of the exit (%)")
    fig.suptitle("Where accuracy arrives vs where the compute is", x=0.01, ha="left", fontweight="bold", color=INK)
    _save(fig, out, "exit_profile")
    plt.close(fig)


def fig_policies(name: str, s: dict[str, pd.DataFrame], runs: list[str], out: Path) -> None:
    """Accuracy vs MACs per policy (val-chosen fronts, test numbers), plus static truncation and the oracle."""
    if "sweep" not in s:
        return
    sw = s["sweep"]
    runs = [r for r in runs if r in set(sw.run)]
    if not runs:
        return
    plt = _style()
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4.0), squeeze=False)
    for ax, run in zip(axes[0], runs):
        f = policy_fronts(sw, run)
        for i, pol in enumerate(POLICIES):
            g = f[f.policy == pol].sort_values("mac_frac")
            ax.plot(g.mac_frac * 100, g.acc * 100, color=SERIES[i], marker=MARKERS[i], markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1, linewidth=1.6, label=pol.replace("_", " + "))
        t = sel(sw, run=run, agg="single", split="test")
        st = t[t.policy == "static"].sort_values("mac_frac")
        ax.plot(st.mac_frac * 100, st.acc * 100, color=NEUTRAL, linestyle="--", linewidth=1.4, marker="o",
                markersize=4, label="static (truncate)")
        orc = t[(t.policy == "oracle") & (t.S == "skip0")].iloc[0]
        ref = t.ref_acc.iloc[0] * 100
        ax.axhline(ref, color=NEUTRAL, linewidth=1.2, linestyle=":", label=f"full network ({ref:.1f}%)")
        ax.set_ylim(ref - 8, max(ref, f.acc.max() * 100) + 1)
        ax.set_xlim(0, min(100, f.mac_frac.max() * 100 * 1.1))
        ax.set_title(f"{ARCH[name]} {label(run)}  (oracle {orc.acc * 100:.1f}% at {orc.mac_frac * 100:.0f}% MACs)",
                     loc="left")
        ax.set_xlabel("Expected inference MACs (% of full network)")
    axes[0][0].set_ylabel("Test accuracy (%)")
    axes[0][-1].legend(fontsize=8, loc="lower right", ncol=2)
    _save(fig, out, f"policies_{name}")
    plt.close(fig)


def fig_partitions(name: str, s: dict[str, pd.DataFrame], out: Path) -> None:
    """Best exit front of every granular split (any policy / aggregation; val-chosen, test numbers)."""
    if "pareto" not in s:
        return
    pf = sel(s["pareto"])
    runs = sorted(pf.run.unique(), key=lambda r: -len(parse_method(method_of(r)) or ()))
    runs = [r for r in runs if len(parse_method(method_of(r)) or ()) > 1][: len(SERIES)]
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for i, run in enumerate(runs):
        g = pf[pf.run == run].sort_values("mac_frac_test")  # val-chosen points; no re-selection on test
        ax.plot(g.mac_frac_test * 100, g.acc_test * 100, color=SERIES[i], marker=MARKERS[i], markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1, linewidth=1.6, label=label(run))
    ax.set_xlabel("Expected inference MACs (% of full network)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"{ARCH[name]}: exit fronts of each granular split (val-chosen, test)", loc="left")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    top = pf.acc_test.max() * 100
    ax.set_ylim(top - 30, top + 1)
    ax.set_xlim(0, 50 if name == "mobilenet" else 101)
    _save(fig, out, f"partitions_{name}")
    plt.close(fig)


def sweep64(s: dict[str, pd.DataFrame]) -> pd.DataFrame:
    p = picks_table(s["picks"])
    p["method"] = p.run.map(method_of)
    p["first"] = p.method.map(lambda m: (parse_method(m) or (0,))[0])
    p["n_blocks"] = p.method.map(lambda m: len(parse_method(m) or ()))
    c = cost_table("vgg8", 10, [parse_method(m) for m in p.method.unique()]).drop_duplicates("method")
    return p.merge(c, on="method", how="left")


def fig_sweep64(s: dict[str, pd.DataFrame], out: Path) -> pd.DataFrame | None:
    """All 64 VGG8 splits: inference saving at δ vs accuracy, and vs training memory; colour = first block size."""
    if "picks" not in s:
        return None
    p = sweep64(s)
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for txt, fs, color, mk in FIRST:
        g = p[p["first"].isin(fs)]
        kw: dict[str, Any] = dict(color=color, marker=mk, s=42, edgecolor=SURFACE, linewidth=1.5, label=txt)
        axes[0].scatter(g.test_mac_frac * 100, g.test_ref * 100, **kw)
        axes[1].scatter(g.act_mb, g.test_mac_frac * 100, **kw)
    for m, txt in (("p2.5", "2,5"), ("p2.1.1.1.2", "2,1,1,1,2"), ("p1.1.1.1.1.1.1", "LLS 1×7"), ("p7", "single")):
        r = p[p.method == m]
        if len(r):
            r = r.iloc[0]
            axes[0].annotate(txt, (r.test_mac_frac * 100, r.test_ref * 100), xytext=(6, 0), textcoords="offset points",
                             va="center", fontsize=8, color=MUTED)
            axes[1].annotate(txt, (r.act_mb, r.test_mac_frac * 100), xytext=(6, 0), textcoords="offset points",
                             va="center", fontsize=8, color=MUTED)
    axes[0].set_xlabel(f"Inference MACs with early exit at δ = {DELTA:.0%} (% of full)")
    axes[0].set_ylabel("Full-network test accuracy (%)")
    axes[0].set_title("Accuracy vs inference saving", loc="left")
    axes[1].set_xlabel("Peak activation memory for training (MB, batch 128)")
    axes[1].set_ylabel(f"Inference MACs at δ = {DELTA:.0%} (% of full)")
    axes[1].set_title("Training memory vs inference saving", loc="left")
    axes[1].legend(title="first block", fontsize=8, title_fontsize=8, loc="lower right")
    fig.suptitle("VGG8, all 64 granular splits (60 epochs)", x=0.01, ha="left", fontweight="bold", color=INK)
    _save(fig, out, "sweep64")
    plt.close(fig)
    return p


def fig_overthink(sets: dict[str, dict[str, pd.DataFrame]], out: Path) -> None:
    panels = [(a, s["overthink"]) for a, s in sets.items() if a in ARCH and "overthink" in s]
    if not panels:
        return
    plt = _style()
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 3.8), squeeze=False)
    for ax, (a, o) in zip(axes[0], panels):
        d = sel(o)
        d = d[d.n_exits > 2].assign(split=lambda x: x.run.map(label)).sort_values("overthink")
        ax.barh(d.split, d.overthink * 100, color=BLUE, height=0.6, edgecolor=SURFACE, linewidth=2)
        for y, v in enumerate(d.overthink * 100):
            ax.annotate(f"{v:.1f}%", (v, y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=MUTED)
        ax.set_xlim(0, d.overthink.max() * 100 * 1.25)
        ax.set_title(f"{ARCH[a]}: overthinking (= oracle gap)", loc="left")
        ax.set_xlabel("Test samples right at an earlier exit, wrong at the last (%)")
        ax.grid(axis="y", visible=False)
    _save(fig, out, "overthinking")
    plt.close(fig)


def fig_ensemble(sets: dict[str, dict[str, pd.DataFrame]], out: Path) -> pd.DataFrame:
    """Dumbbell per split: final exit vs best running ensemble (both at full depth)."""
    rows = []
    for a, s in sets.items():
        if a not in ARCH or "per_exit" not in s:
            continue
        for run, g in sel(s["per_exit"]).groupby("run"):
            if len(g) <= 2:
                continue
            last = g.iloc[-1]
            best = max(("ens_all", "ens_w2", "ens_w3"), key=lambda c: g[c].max())
            rows.append({"arch": a, "run": run, "split": label(str(run)), "final": last.acc,
                         "ensemble": g[best].max(), "which": best})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    plt = _style()
    archs = list(dict.fromkeys(t.arch))
    fig, axes = plt.subplots(1, len(archs), figsize=(5.0 * len(archs), 3.8), squeeze=False)
    for ax, a in zip(axes[0], archs):
        d = t[t.arch == a].sort_values("final").reset_index(drop=True)
        for y, r in d.iterrows():
            ax.plot([r.final * 100, r.ensemble * 100], [y, y], color=GRID, linewidth=2, zorder=1)
        ax.scatter(d.final * 100, d.index, color=NEUTRAL, s=48, edgecolor=SURFACE, linewidth=1.5, zorder=2,
                   label="final exit")
        ax.scatter(d.ensemble * 100, d.index, color=ORANGE, s=48, marker="D", edgecolor=SURFACE, linewidth=1.5,
                   zorder=3, label="best exit ensemble")
        ax.set_yticks(d.index, d.split)
        ax.set_title(f"{ARCH[a]}: ensembling the exits", loc="left")
        ax.set_xlabel("Test accuracy (%)")
        ax.grid(axis="y", visible=False)
    axes[0][-1].legend(fontsize=8, loc="lower right")
    _save(fig, out, "ensemble")
    plt.close(fig)
    return t


def fig_reliability(a: str, s: dict[str, pd.DataFrame], out: Path) -> None:
    if "reliability" not in s or "per_exit" not in s:
        return
    run = next((r for r in s["reliability"].run.unique() if method_of(r) == LLS[a]), None)
    if run is None:
        return
    rl = s["reliability"]
    rl = rl[(rl.run == run) & (rl.final == "cls")]
    names = sel(s["per_exit"], run=run).set_index("exit")["name"]
    n = rl.exit.max() + 1
    pick = sorted({1, n // 3, 2 * n // 3, n - 1})
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharex=True, sharey=True)
    for ax, temp in zip(axes, ("train", "fit")):
        ax.plot([0, 100], [0, 100], color=NEUTRAL, linestyle="--", linewidth=1.2)
        for i, e in enumerate(pick):
            g = rl[(rl.temp == temp) & (rl.exit == e)]
            ax.plot(g.conf * 100, g.acc * 100, color=SERIES[i], marker=MARKERS[i], markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1, linewidth=1.6, label=names.get(e, str(e)))
        ax.set_title(f"{'training T = 0.5' if temp == 'train' else 'per-exit T fitted on val'}", loc="left")
        ax.set_xlabel("Confidence (%)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{ARCH[a]} LLS 1×{n - 1}: reliability of the exits (test)", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    _save(fig, out, f"reliability_{a}")
    plt.close(fig)


def fig_classes(a: str, s: dict[str, pd.DataFrame], out: Path) -> None:
    """Mean settle depth per class (fraction of the exits), one row per split: which classes leave early."""
    if "per_class" not in s or "overthink" not in s:
        return
    from matplotlib.colors import LinearSegmentedColormap
    pc = sel(s["per_class"])
    n_ex = sel(s["overthink"]).set_index("run").n_exits
    pc = pc[pc.run.map(n_ex) > 2]
    if pc.empty:
        return
    pc = pc.assign(depth=pc.settle / (pc.run.map(n_ex) - 1), split=pc.run.map(label))
    m = pc.pivot_table(index="split", columns="cls", values="depth")
    m = m.loc[m.mean(axis=1).sort_values().index]
    plt = _style()
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(m) + 1.6))
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#3987e5", "#0d366b"])
    im = ax.imshow(m.values, cmap=cmap, aspect="auto", vmin=0)
    ax.set_yticks(range(len(m)), m.index)
    ax.set_xticks(range(m.shape[1]), [CIFAR10[c] if m.shape[1] == 10 else str(c) for c in m.columns], fontsize=8)
    ax.grid(False)
    ax.set_title(f"{ARCH[a]}: depth where the prediction settles, per class", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("settle depth (fraction of exits)", color=MUTED)
    cb.outline.set_edgecolor(GRID)  # type: ignore[attr-defined]
    _save(fig, out, f"classes_{a}")
    plt.close(fig)


ORTH = {"mobilenet": ("MobileNetV1 (8 splits)", BLUE, "o"), "vgg8": ("VGG8 (6 splits)", ORANGE, "D"),
        "vgg8_sweep": ("VGG8 64 splits (60 ep)", NEUTRAL, ".")}


def orth_heads(sets: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Every fixed-basis head exit (final = head, so cls is excluded) with its basis and class-geometry columns."""
    ds = [sel(s["per_exit"], final="head").assign(set=n) for n, s in sets.items()
          if n in ORTH and "basis_coh" in s.get("per_exit", pd.DataFrame()).columns]
    return pd.concat(ds, ignore_index=True).dropna(subset=["basis_coh"]) if ds else pd.DataFrame()


def fig_orthogonality(sets: dict[str, dict[str, pd.DataFrame]], out: Path) -> pd.DataFrame:
    """Exit accuracy against how close the two closest classes' mean outputs are (the basis itself is ~orthogonal)."""
    h = orth_heads(sets)
    if h.empty:
        return h
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for n in sorted(h.set.unique(), key=lambda n: n != "vgg8_sweep"):  # the 64-split cloud underneath
        lab, c, mk = ORTH[n]
        d = h[h.set == n]
        ax.scatter(d.cls_cos_max, d.acc * 100, color=c, marker=mk, s=14 if mk == "." else 30, label=lab,
                   edgecolor=SURFACE if mk != "." else "none", linewidth=0.8, alpha=0.6 if mk == "." else 1.0)
    r = h.cls_cos_max.corr(h.acc)
    ax.set_title(f"Closer classes, weaker exit (Pearson r = {r:.2f}); basis coherence {h.basis_coh.min():.3f}-"
                 f"{h.basis_coh.max():.3f}", loc="left", fontsize=9)
    ax.set_xlabel("Cosine between the two closest classes' mean head outputs (centred)")
    ax.set_ylabel("Test accuracy of the exit (%)")
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, out, "orthogonality")
    plt.close(fig)
    return h


DEPTH_LINES = [("exit_margin", "margin policy (δ = 1%)", BLUE, "o"), ("exit_sync", "sync policy (δ = 1%)", ORANGE, "D"),
               ("settle", "heads settle (no policy)", NEUTRAL, "s")]


def fig_depth(sets: dict[str, dict[str, pd.DataFrame]], out: Path) -> None:
    """LLS 1×K per network: mean exit per k-NN prediction depth (left), test accuracy per depth (right)."""
    rows = [(a, b[b.run.map(method_of) == LLS[a]]) for a, s in sets.items()
            if a in ARCH and "depth_bins" in s for b in [s["depth_bins"]]]
    rows = [(a, b) for a, b in rows if not b.empty]
    if not rows:
        return
    plt = _style()
    fig, axes = plt.subplots(len(rows), 2, figsize=(10.4, 3.6 * len(rows)), squeeze=False)
    for (a, b), (ax, bx) in zip(rows, axes):
        for c, lab, col, mk in DEPTH_LINES:
            if c in b.columns:
                ax.plot(b.pd, b[c], color=col, marker=mk, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1,
                        linewidth=1.8, label=lab)
        ax.set_title(f"{ARCH[a]} LLS 1×{len(LLS[a].split('.'))}: harder samples exit later", loc="left")
        ax.set_xlabel("k-NN prediction depth (exit index)")
        ax.set_ylabel("Mean exit (exit index)")
        ax.legend(fontsize=8, loc="upper left")
        bx.bar(b.pd, b.acc * 100, color=BLUE, width=0.7, edgecolor=SURFACE, linewidth=2)
        for x, v, fr in zip(b.pd, b.acc * 100, b.frac * 100):
            bx.annotate(f"{fr:.0f}%", (x, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7,
                        color=MUTED)
        bx.set_title("Test accuracy per depth (labels: share of samples)", loc="left")
        bx.set_xlabel("k-NN prediction depth (exit index)")
        bx.set_ylabel("Test accuracy (%)")
        bx.set_ylim(0, 105)
        bx.grid(axis="x", visible=False)
    _save(fig, out, "depth")
    plt.close(fig)


LEARNED_SETS = ("mobilenet_frozen", "vgg8_frozen", "joint")
KINDS = [("fixed LLS heads", BLUE, "o"), ("linear probes (LLS backbone)", AQUA, "s"),
         ("BP + frozen exits", NEUTRAL, "^"), ("BP + joint exits", ORANGE, "D")]


def kind_of(run: str) -> str:
    if "_xjoint" in run:
        return "BP + joint exits"
    if run.endswith(tuple(f"_xfrozen_e{e}" for e in range(1, 1000))):
        return "BP + frozen exits" if "_bp_" in run else "linear probes (LLS backbone)"
    return "fixed LLS heads"


def learned(sets: dict[str, dict[str, pd.DataFrame]], key: str) -> pd.DataFrame:
    """Rows of `key` (per_exit / picks) from the fixed-head sets and the learned-exit sets, with arch and kind."""
    ds = [s[key] for n, s in sets.items() if (n in ARCH or n in LEARNED_SETS) and key in s]
    if not ds:
        return pd.DataFrame()
    d = pd.concat(ds, ignore_index=True)
    return d.assign(arch=d.run.str.split("_").str[0], kind=d.run.map(kind_of))


def fig_learned(sets: dict[str, dict[str, pd.DataFrame]], out: Path) -> pd.DataFrame:
    """Per network: per-exit accuracy of the per-layer exits of each kind (left), δ = 1% operating points (right)."""
    pe, pk = learned(sets, "per_exit"), learned(sets, "picks")
    if pe.empty or not (pe.kind != "fixed LLS heads").any():
        return pd.DataFrame()
    pe = sel(pe, final="head")
    pts = picks_table(pk) if not pk.empty else pd.DataFrame()
    archs = [a for a in ARCH if a in set(pe.arch)]
    plt = _style()
    fig, axes = plt.subplots(len(archs), 2, figsize=(10.4, 3.8 * len(archs)), squeeze=False)
    for a, (ax, bx) in zip(archs, axes):
        per_layer = pe[(pe.arch == a) & (pe.run.map(method_of).isin([LLS[a], "bp"]) | pe.run.str.contains(
            r"_bp_|_xjoint", regex=True))]
        for k, col, mk in KINDS:
            g = per_layer[per_layer.kind == k]
            if g.empty:
                continue
            ax.plot(g.mac_frac * 100, g.acc * 100, color=col, marker=mk, markersize=6, markeredgecolor=SURFACE,
                    markeredgewidth=1, linewidth=1.8, label=k)
        ax.set_title(f"{ARCH[a]}: an exit after every layer", loc="left")
        ax.set_xlabel("Cumulative inference MACs up to the exit (% of full network)")
        ax.set_ylabel("Test accuracy of the exit (%)")
        p = pts[pts.run.str.startswith(a + "_")] if not pts.empty else pts
        if not p.empty:  # thin connector: the same split with fixed heads -> with probes
            fx = p[p.run.map(kind_of) == "fixed LLS heads"].set_index("run")
            pr = p[p.run.map(kind_of) == KINDS[1][0]]
            for _, r in pr.iterrows():
                src = str(r.run).split("_xfrozen")[0]
                if src in fx.index and r.test_mac_frac < 0.999:
                    q = fx.loc[src]
                    bx.plot([q.test_mac_frac * 100, r.test_mac_frac * 100], [q.test_acc * 100, r.test_acc * 100],
                            color=GRID, linewidth=1.2, zorder=1)
        for k, col, mk in KINDS:
            g = p[p.run.map(kind_of) == k] if not p.empty else p
            g = g[g.test_mac_frac < 0.999] if not g.empty else g  # single-block runs can't exit early
            if g.empty:
                continue
            bx.scatter(g.test_mac_frac * 100, g.test_acc * 100, color=col, marker=mk, s=46, edgecolor=SURFACE,
                       linewidth=1, label=k, zorder=3)
        bx.set_title(f"δ = {DELTA:.0%} operating points, one per split", loc="left")
        bx.set_xlabel("Expected inference MACs (% of full network)")
        bx.set_ylabel("Test accuracy (%)")
    hs, ls = axes[0][0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=len(ls), fontsize=9, frameon=False)
    fig.text(0.01, 0.005, "Grey connectors: the same split with fixed heads and with probes. Val-chosen, test numbers.",
             fontsize=7, color=MUTED)
    _save(fig, out, "learned", rect=(0, 0.06, 1, 1))
    plt.close(fig)
    if pts.empty:
        return pts
    pts = pts.assign(arch=pts.run.str.split("_").str[0], kind=pts.run.map(kind_of))
    pts.loc[pts.kind.str.startswith("BP"), "split"] = "BP (exit per layer)"
    return pts[pts.test_mac_frac < 0.999]  # single-block runs can't exit early


def seeds_table(sets: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """δ = 1% picks of the seed runs (s1, s2) and the same splits' s0 from the 64-split sweep: mean and std."""
    if "vgg8_seeds" not in sets or "picks" not in sets["vgg8_seeds"]:
        return pd.DataFrame()
    p = picks_table(sets["vgg8_seeds"]["picks"])
    if "vgg8_sweep" in sets and "picks" in sets["vgg8_sweep"]:
        s0 = picks_table(sets["vgg8_sweep"]["picks"])
        p = pd.concat([p, s0[s0.split.isin(set(p.split))]], ignore_index=True)
    g = p.groupby("split").agg(seeds=("run", "size"), full=("test_ref", "mean"), full_sd=("test_ref", "std"),
                               exit=("test_acc", "mean"), exit_sd=("test_acc", "std"),
                               mac=("test_mac_frac", "mean"), mac_sd=("test_mac_frac", "std"))
    return g.reset_index().sort_values("mac")


# ---------- tables ----------

def md_table(df: pd.DataFrame, cols: dict[str, str], pct: set[str]) -> str:
    head = "| " + " | ".join(cols.values()) + " |\n|" + "---|" * len(cols) + "\n"

    def cell(c: str, v: Any) -> str:
        if isinstance(v, float) and v != v:
            return "–"
        if c in pct and isinstance(v, float):
            return f"{v * 100:.1f}"
        return f"{v:.2f}" if isinstance(v, float) else str(v)
    return head + "".join("| " + " | ".join(cell(c, r[c]) for c in cols) + " |\n" for _, r in df.iterrows())


def tables(sets: dict[str, dict[str, pd.DataFrame]], ens: pd.DataFrame, s64: pd.DataFrame | None,
           orth: pd.DataFrame | None = None, lrn: pd.DataFrame | None = None, seeds: pd.DataFrame | None = None) -> str:
    md = ["# Early exit: results (test; operating points chosen on val, cls final, fitted T)\n"]
    for name, s in sets.items():
        if name in (*LEARNED_SETS, "vgg8_seeds"):
            continue
        if "picks" in s and name != "vgg8_sweep":
            p = picks_table(s["picks"])
            md += [f"## {name}: cheapest setting within δ = {DELTA:.0%} of the full network\n",
                   md_table(p, {"split": "split", "test_ref": "full acc", "test_acc": "exit acc",
                                "test_mac_frac": "MAC frac", "test_speedup": "speedup", "policy_pick": "policy"},
                            {"test_ref", "test_acc"})]
        if "overthink" in s and name != "vgg8_sweep":
            o = sel(s["overthink"]).assign(split=lambda d: d.run.map(label)).sort_values("overthink")
            md += [f"## {name}: overthinking\n",
                   md_table(o, {"split": "split", "n_exits": "exits", "final_acc": "final", "oracle_acc": "oracle",
                                "overthink": "overthink (= oracle gap)", "settle_frac": "settle depth",
                                "flips": "flips"}, {"final_acc", "oracle_acc", "overthink"})]
    if not ens.empty:
        md += ["## Exit ensembles at full depth\n",
               md_table(ens.assign(gain=ens.ensemble - ens.final), {"arch": "net", "split": "split", "final": "final",
                        "ensemble": "best ensemble", "gain": "gain", "which": "window"}, {"final", "ensemble", "gain"})]
    if s64 is not None:
        g = s64.groupby("first").agg(n=("run", "size"), full=("test_ref", "mean"), exit=("test_acc", "mean"),
                                     mac=("test_mac_frac", "mean"), mem=("act_mb", "mean")).reset_index()
        md += [f"## VGG8 64 splits by first block size (δ = {DELTA:.0%})\n",
               md_table(g, {"first": "first block", "n": "splits", "full": "full acc", "exit": "exit acc",
                            "mac": "MAC frac", "mem": "train MB"}, {"full", "exit"}),
               "\nCheapest 8:\n",
               md_table(s64.head(8), {"split": "split", "test_ref": "full acc", "test_acc": "exit acc",
                                      "test_mac_frac": "MAC frac", "act_mb": "train MB", "policy_pick": "policy"},
                        {"test_ref", "test_acc"})]
    if orth is not None and not orth.empty:
        o = orth[orth.set != "vgg8_sweep"].groupby(["set", "run"], sort=False).agg(
            exits=("exit", "size"), coh=("basis_coh", "max"), first=("cls_cos_max", "first"),
            last=("cls_cos_max", "last"), acc=("acc", "last")).reset_index()
        md += ["## Orthogonality of the fixed-basis heads (test)\n",
               "Basis coherence = max |cos| between two class rows of a head's fixed basis. Class cosine = cosine "
               "between the two closest classes' centred mean head outputs (a perfect simplex gives -1/(C-1)).\n",
               md_table(o.assign(split=o.run.map(label)), {"set": "set", "split": "split", "exits": "heads",
                        "coh": "basis coherence", "first": "class cos, first head", "last": "class cos, last head",
                        "acc": "last head acc"}, {"acc"})]
    if lrn is not None and not lrn.empty:
        md += [f"## Learned vs fixed exits (δ = {DELTA:.0%})\n",
               md_table(lrn.sort_values(["arch", "split", "kind"]), {"arch": "net", "split": "split", "kind": "exits",
                        "test_ref": "full acc", "test_acc": "exit acc", "test_mac_frac": "MAC frac",
                        "policy_pick": "policy"}, {"test_ref", "test_acc"})]
    if seeds is not None and not seeds.empty:
        md += [f"## VGG8 seeds (60 ep; s0 from the 64-split sweep), δ = {DELTA:.0%}: mean (sd)\n",
               md_table(seeds.assign(full_s=lambda d: [f"{m * 100:.1f} ({v * 100:.1f})" for m, v in zip(d.full, d.full_sd)],
                                     exit_s=lambda d: [f"{m * 100:.1f} ({v * 100:.1f})" for m, v in zip(d.exit, d.exit_sd)],
                                     mac_s=lambda d: [f"{m:.3f} ({v:.3f})" for m, v in zip(d.mac, d.mac_sd)]),
                        {"split": "split", "seeds": "seeds", "full_s": "full acc", "exit_s": "exit acc",
                         "mac_s": "MAC frac"}, set())]
    dc = [s["depth_corr"].assign(net=n) for n, s in sets.items() if n in ARCH and "depth_corr" in s]
    if dc:
        d = pd.concat(dc, ignore_index=True)
        t = d.pivot_table(index=["net", "run"], columns="vs", values="spearman", sort=False).reset_index()
        t["split"] = t.run.map(label)
        cols = {c: c.removeprefix("exit_") for c in ("settle", "oracle", "exit_any", "exit_margin", "exit_confidence",
                                                     "exit_entropy", "exit_patience", "exit_sync",
                                                     "exit_sync_margin") if c in t.columns}
        md += ["## Spearman(k-NN prediction depth, exit depth), test; policy exits at their δ = 1% picks\n",
               md_table(t, {"net": "net", "split": "split", **cols}, set())]
    return "\n".join(md)


# ---------- CLI ----------

FRONT_RUNS = {"mobilenet": ["p1.1.1.1.1.1.1.1.1.1.1.1.1.1", "p4.2.6.2"], "vgg8": ["p1.1.1.1.1.1.1", "p2.2.3"]}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Early-exit report figures and tables.")
    p.add_argument("--report", default="report/exit", help="folder holding one sub-folder per sweep/diag output")
    p.add_argument("--sets", nargs="*", default=["mobilenet", "vgg8", "vgg8_sweep", "vgg8_cifar100", "vgg8_seeds",
                                                 *LEARNED_SETS])
    p.add_argument("--log", default=None)
    a = p.parse_args(argv)
    root = Path(a.report)
    out = root / "figs"
    out.mkdir(parents=True, exist_ok=True)
    logs.setup(a.log, root / "analyze.log")
    sets = {n: load(root / n) for n in a.sets if (root / n).is_dir()}
    log.info("sets: %s", list(sets))

    fig_profile(sets, out)
    for name in ("mobilenet", "vgg8"):
        if name in sets:
            s = sets[name]
            runs = sorted(s["sweep"].run.unique()) if "sweep" in s else []
            named = [r for m in FRONT_RUNS[name] for r in runs if method_of(r) == m]
            fig_policies(name, s, named or runs[:2], out)
            fig_partitions(name, s, out)
            fig_reliability(name, s, out)
            fig_classes(name, s, out)
    fig_overthink(sets, out)
    ens = fig_ensemble(sets, out)
    s64 = fig_sweep64(sets["vgg8_sweep"], out) if "vgg8_sweep" in sets else None
    orth = fig_orthogonality(sets, out)
    fig_depth(sets, out)
    lrn = fig_learned(sets, out)
    (root / "tables.md").write_text(tables(sets, ens, s64, orth, lrn, seeds_table(sets)))
    log.info("wrote %s and %s", out, root / "tables.md")


if __name__ == "__main__":
    logs.setup()
    main()
