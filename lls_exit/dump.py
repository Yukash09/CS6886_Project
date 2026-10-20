"""Checkpoint -> exit dump: every head's scores, cls logits, labels, pooled features and per-exit costs.

python -m lls_exit.dump --exp mobilenet300         (every non-BP run of an experiment; finished ones skipped)
python -m lls_exit.dump --run-dir runs/<run>        (a single run)
Each run gets exits.pt (small; enough for all policy analysis) and feats.pt (fp16 head inputs, for prediction depth).
"""
import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from lls_granular import ckpt, data, logs, run
from lls_granular.exps import EXPS, jobs
from lls_granular.heads import coherence, pool_flat
from lls_granular.launch import run_dir
from lls_granular.train import bn_refresh
from lls_granular.units import ARCHS
from lls_granular.wrapper import Block, GranularLLS

from .costs import exit_costs

log = logging.getLogger(__name__)

Batches = Iterable[tuple[Tensor, Tensor]]


def rebuild(cfg: dict[str, Any]) -> argparse.Namespace:
    """Training args from a run's config.json; unknown keys are ignored, missing ones keep run.py defaults."""
    a = run.args([])
    for k, v in cfg.items():
        if hasattr(a, k):
            setattr(a, k, v)
    if cfg.get("blocks"):
        a.partition = ",".join(map(str, cfg["blocks"]))
    return a


def balanced(y: list[int], pool: list[int], n: int, n_cls: int, seed: int = 0) -> list[int]:
    """n indices from pool, n // n_cls per class, fixed by seed."""
    g = torch.Generator().manual_seed(seed)
    t = torch.tensor(y)[pool]
    p = torch.tensor(pool)
    out: list[int] = []
    for c in range(n_cls):
        idx = p[t == c]
        out += idx[torch.randperm(len(idx), generator=g)[: n // n_cls]].tolist()
    return sorted(out)


def blocks(m: GranularLLS) -> list[Block]:
    return cast(list[Block], list(m.blocks))


def indices(loader: DataLoader) -> Tensor:
    ds = loader.dataset
    return torch.tensor(ds.indices if isinstance(ds, Subset) else range(len(ds)))  # type: ignore[arg-type]


@torch.no_grad()
def collect(m: GranularLLS, loader: Batches, dev: torch.device, feats: bool) -> dict[str, Any]:
    """One eval pass: scores (N, K, C), logits (N, C), y (N,), and per-exit L2-normalised head inputs (fp16)."""
    m.eval()
    sc, lg, ys = [], [], []
    bs = blocks(m)
    fs: list[list[Tensor]] = [[] for _ in bs]
    for x, y in loader:
        h = x.to(dev, non_blocking=True)
        s = []
        for k, b in enumerate(bs):
            h = b(h)
            s.append(b.head.scores(h))
            if feats:
                fs[k].append(F.normalize(pool_flat(h, b.head.pool), dim=1).half().cpu())
        sc.append(torch.stack(s, 1).cpu())
        lg.append(m.cls(h).cpu())
        ys.append(y)
    out: dict[str, Any] = {"scores": torch.cat(sc), "logits": torch.cat(lg), "y": torch.cat(ys)}
    if feats:
        out["feats"] = [torch.cat(f) for f in fs]
    log.debug("collected %d samples, scores %s", len(out["y"]), tuple(out["scores"].shape))
    return out


def dump(m: GranularLLS, a: argparse.Namespace, loaders: dict[str, DataLoader], train: Batches,
         dev: torch.device, feats: bool, meta: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Eval weights + BN refresh (as in training eval), then collect every split. Returns (exits, feats)."""
    m.opt_mode(False)
    torch.manual_seed(0)  # BN refresh batches are shuffled; fix them for reproducible dumps
    bn_refresh(m, train, dev)
    heads = [b.head for b in blocks(m)]
    n_cls = heads[0].B.shape[0]
    c = exit_costs(ARCHS[a.arch](), m.part, n_cls)
    ex: dict[str, Any] = {
        "meta": {**meta, "names": c.names, "n_cls": n_cls, "temp": a.temp, "wave": a.wave,
                 "partition": list(m.part)},
        "costs": {"backbone": c.backbone, "head": c.head, "full": c.full},
        "heads": {"dim": [h.B.shape[1] for h in heads], "pool": [h.pool for h in heads],
                  "coherence": [coherence(h.B) for h in heads],
                  "row_norm": torch.stack([h.B.norm(dim=1) for h in heads]).cpu()},
        "splits": {},
    }
    fe: dict[str, Any] = {"meta": ex["meta"], "splits": {}}
    for tag, ld in loaders.items():
        r = collect(m, ld, dev, feats)
        r["idx"] = indices(ld)
        f = r.pop("feats", None)
        ex["splits"][tag] = r
        if f is not None:
            fe["splits"][tag] = {"feats": f, "y": r["y"], "idx": r["idx"]}
        acc = (r["scores"].argmax(2) == r["y"][:, None]).float().mean(0)
        log.info("%s: n=%d cls acc %.4f, head acc %s", tag, len(r["y"]),
                 (r["logits"].argmax(1) == r["y"]).float().mean().item(), [round(v, 4) for v in acc.tolist()])
    m.opt_mode(True)
    return ex, fe


def train_subset(a: argparse.Namespace, data_dir: str | None, n: int, eval_bs: int, workers: int) -> DataLoader:
    """Class-balanced subset of the *train* split (val excluded) with eval transforms, fixed order."""
    cls, n_cls, aug = data.SETS[a.dataset]
    ds = cls(str(data.root(data_dir)), train=True, download=False, transform=data.tfms(aug)[1])
    tr, _ = data.split(ds.targets, a.n_val, n_cls) if a.n_val else (list(range(len(ds))), [])
    idx = balanced(ds.targets, tr, n, n_cls)
    return DataLoader(Subset(ds, idx), eval_bs, num_workers=workers, pin_memory=torch.cuda.is_available())


def args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump per-exit scores/features of trained granular LLS runs.")
    a = p.add_argument
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--exp", nargs="+", choices=sorted(EXPS), help="every non-BP run of these experiments")
    g.add_argument("--run-dir", help="a single run folder with config.json and ckpt.pt")
    a("--seeds", type=int, nargs="*", help="with --exp: the seeds the runs used")
    a("--out", default=None, help="with --run-dir: output folder (default: the run dir)")
    a("--fresh", action="store_true", help="redo runs that already have exits.pt")
    a("--n-train", type=int, default=10000, help="class-balanced train subset (0 = none)")
    a("--feats", action=argparse.BooleanOptionalAction, default=True)
    a("--data-dir", default=None)
    a("--eval-bs", type=int, default=256)
    a("--workers", type=int, default=2)
    a("--limit", type=int, default=None, help="val/train subset size for smoke tests")
    a("--log", default=None)
    return p.parse_args(argv)


def targets(exps: list[str], seeds: list[int] | None) -> list[Path]:
    """Run folders of the experiments' non-BP jobs (BP has no exits), named exactly as training named them."""
    return [run_dir(j) for e in exps for j in jobs(e, seeds) if j.partition != "bp"]


def dump_run(rd: Path, out: Path, o: argparse.Namespace, dev: torch.device) -> dict[str, Any]:
    """Dump one run; returns a one-line summary for the final table."""
    cfg = json.loads((rd / "config.json").read_text())
    a = rebuild(cfg)
    d = data.load(a.dataset, o.data_dir, a.bs, o.eval_bs, a.n_val, o.workers, a.seed, o.limit)
    m, part = run.model(a, d.n_cls)
    if not isinstance(m, GranularLLS):
        raise ValueError("BP runs have no exits; BP exit heads get their own dumps")
    m.to(dev)
    ep, best = ckpt.load(rd / "ckpt.pt", m)
    log.info("loaded %s: partition %s, %d epochs done, best val %.4f @ %d", rd.name, part, ep, best.val, best.epoch)

    loaders = {"test": d.test}
    if d.val:
        loaders["val"] = d.val
    if o.n_train:
        n = min(o.n_train, o.limit) if o.limit else o.n_train
        loaders["train"] = train_subset(a, o.data_dir, n, o.eval_bs, o.workers)
    meta = {"run": rd.name, "arch": a.arch, "dataset": a.dataset, "seed": a.seed, "epochs_done": ep,
            "best": asdict(best), "git_train": cfg.get("git")}
    ex, fe = dump(m, a, loaders, d.train, dev, o.feats, meta)

    out.mkdir(parents=True, exist_ok=True)
    torch.save(ex, out / "exits.pt")
    if o.feats:
        torch.save(fe, out / "feats.pt")
    log.info("wrote %s%s", out / "exits.pt", " and feats.pt" if o.feats else "")
    t = ex["splits"]["test"]
    head = (t["scores"].argmax(2) == t["y"][:, None]).float().mean(0)
    return {"exits": len(head), "epochs": ep, "cls": (t["logits"].argmax(1) == t["y"]).float().mean().item(),
            "last_head": head[-1].item(), "best_head": head.max().item()}


def main(argv: list[str] | None = None) -> None:
    o = args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rds = [Path(o.run_dir).expanduser()] if o.run_dir else targets(o.exp, o.seeds)
    log.info("dumping %d runs on %s", len(rds), dev)
    rows: dict[str, dict[str, Any] | str] = {}
    for rd in rds:
        out = Path(o.out).expanduser() if o.out and o.run_dir else rd
        if (out / "exits.pt").exists() and not o.fresh:
            rows[rd.name] = "exists"
        elif not (rd / "ckpt.pt").exists():
            log.warning("%s: no checkpoint, skipping (train it or --restore first)", rd)
            rows[rd.name] = "no ckpt"
        else:
            logs.setup(o.log, out / "dump.log")
            rows[rd.name] = dump_run(rd, out, o, dev)
    report(rows)
    if any(r == "no ckpt" for r in rows.values()):
        sys.exit(1)


def report(rows: dict[str, dict[str, Any] | str]) -> None:
    print(f"\n{'run':<60} {'status':<8} {'exits':>5} {'ep':>4} {'cls':>7} {'last_hd':>7} {'best_hd':>7}")
    for n, r in rows.items():
        if isinstance(r, str):
            print(f"{n:<60} {r:<8}")
        else:
            print(f"{n:<60} {'ok':<8} {r['exits']:>5} {r['epochs']:>4} {r['cls']:>7.4f} "
                  f"{r['last_head']:>7.4f} {r['best_head']:>7.4f}")


if __name__ == "__main__":
    logs.setup()
    main()
