"""Train learned exit heads, then dump them in the exits.pt format (so sweep / diag work unchanged) and upload.

python -m lls_exit.train_exits --plan mobilenet                          (every run of a plan, one GPU, in order)
python -m lls_exit.train_exits --variant joint --arch vgg8 --epochs 100  (BP backbone + exits, from scratch)
python -m lls_exit.train_exits --variant frozen --source $LLS_OUT/<run>  (exits on a trained, frozen backbone)
Frozen on a BP run = SDN-style IC-only exits at every unit; on an LLS run = linear probes at its heads' positions.
Runs are named <source>_xfrozen_e<ep> or <arch>_<dataset>_xjoint-p<exits>_e<ep>_s<seed>; finished ones are skipped,
interrupted ones resume. Training logs to wandb like lls_granular.run; the dump is attached to the same wandb run.
"""
import argparse
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, cast

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from lls_granular import ckpt, data, logs, run
from lls_granular.exps import jobs
from lls_granular.launch import run_dir
from lls_granular.partitions import Partition, parse
from lls_granular.track import RunId, Tracker, method
from lls_granular.train import Best, bn_refresh, fit
from lls_granular.units import ARCHS, Unit

from . import dump as D
from .costs import exit_costs
from .exitnet import VARIANTS, ExitNet, parts

log = logging.getLogger(__name__)

FROZEN_EPOCHS = 30


@dataclass
class ExitRunId(RunId):
    variant: str = "joint"
    source: str | None = None  # frozen: the backbone's run name
    src_kind: str | None = None  # frozen: 'bp' or 'lls'

    @property
    def group(self) -> str:
        if self.source:
            return f"{self.source}_x{self.variant}_e{self.epochs}"
        return f"{self.arch}_{self.dataset}_x{self.variant}-{method(self.part)}_e{self.epochs}"

    @property
    def name(self) -> str:
        return self.group if self.source else f"{self.group}_s{self.seed}"

    @property
    def tag(self) -> str:
        """'xjoint', 'xfrozen_bp' (SDN IC-only) or 'xprobe' (linear probes on LLS)."""
        if self.variant == "joint":
            return "xjoint"
        return "xfrozen_bp" if self.src_kind == "bp" else "xprobe"

    def labels(self) -> list[str]:
        return [*super().labels(), "exits", self.tag]

    def fields(self) -> dict[str, Any]:
        return {**super().fields(), "kind": self.tag, "variant": self.variant, "source": self.source}


# ---------- model ----------

def load_source(src: Path, d: data.Data, dev: torch.device) -> tuple[list[Unit], nn.Module, Partition | None,
                                                                       argparse.Namespace]:
    """A trained run's units and classifier at its eval weights (schedule-free x) with refreshed BN, as in dump.py."""
    sa = D.rebuild(D.cfg_of(src))
    m, _ = run.model(sa, d.n_cls)
    m.to(dev)
    ep, best = ckpt.load(src / "ckpt.pt", m)
    m.opt_mode(False)
    torch.manual_seed(0)
    bn_refresh(m, d.train, dev)
    units, cls, part = parts(m)
    log.info("source %s: %s, %d epochs, best val %.4f", src.name, method(part), ep, best.val)
    return units, cls, part, sa


def build(a: argparse.Namespace, d: data.Data, dev: torch.device) -> tuple[ExitNet, ExitRunId]:
    if a.variant == "joint":
        torch.manual_seed(a.seed)
        units = ARCHS[a.arch]()
        part = parse(a.partition or "uniform:1", [u.shape for u in units])
        m = ExitNet(units, part, d.n_cls, False, None, a.optim, a.lr, a.wd)
        rid = ExitRunId(a.arch, a.dataset, part, a.epochs, a.seed, a.exp, variant="joint")
    else:
        src = Path(a.source).expanduser()
        units, cls, src_part, sa = load_source(src, d, dev)
        part = parse(a.partition, [u.shape for u in units]) if a.partition else src_part or (1,) * len(units)
        torch.manual_seed(a.seed)
        m = ExitNet(units, part, d.n_cls, True, cls, a.optim, a.lr, a.wd)
        rid = ExitRunId(sa.arch, sa.dataset, part, a.epochs, a.seed, a.exp, variant="frozen", source=src.name,
                        src_kind="bp" if src_part is None else "lls")
    return m.to(dev), rid


# ---------- dump ----------

@torch.no_grad()
def collect(m: ExitNet, loader: Iterable[tuple[Tensor, Tensor]], dev: torch.device) -> dict[str, Tensor]:
    m.eval()
    sc, lg, ys = [], [], []
    for x, y in loader:
        logits, scores = m.forward_all(x.to(dev, non_blocking=True))
        sc.append(torch.stack(scores, 1).cpu())
        lg.append(logits.cpu())
        ys.append(y)
    return {"scores": torch.cat(sc), "logits": torch.cat(lg), "y": torch.cat(ys)}


def dump(m: ExitNet, arch: str, loaders: dict[str, DataLoader], train: Iterable[tuple[Tensor, Tensor]],
         dev: torch.device, meta: dict[str, Any]) -> dict[str, Any]:
    """Same layout as dump.dump. Scores are logits: row_norm = 1 and temp = 1, so policies.probs gives
    softmax(logits) at the 'train' temperature and fits T on the logits otherwise. Head costs use the LLS head
    formula (pool + d + d·C); a linear exit is pool + d·C + C, the same to <0.1% of the network."""
    m.opt_mode(False)
    torch.manual_seed(0)
    bn_refresh(cast(Any, m), train, dev)  # no-op on a frozen backbone (it stays in eval mode)
    hs = [cast(Any, h) for h in m.heads]
    n_cls = hs[0].fc.out_features
    c = exit_costs(ARCHS[arch](), m.part, n_cls)
    ex: dict[str, Any] = {
        "meta": {**meta, "names": c.names, "n_cls": n_cls, "temp": 1.0, "wave": None, "partition": list(m.part)},
        "costs": {"backbone": c.backbone, "head": c.head, "full": c.full},
        "heads": {"dim": [h.fc.in_features for h in hs], "pool": [h.pool for h in hs],
                  "coherence": [float("nan")] * len(hs), "row_norm": torch.ones(len(hs), n_cls)},
        "splits": {},
    }
    for tag, ld in loaders.items():
        r = collect(m, ld, dev)
        r["idx"] = D.indices(ld)
        ex["splits"][tag] = r
        acc = (r["scores"].argmax(2) == r["y"][:, None]).float().mean(0)
        log.info("%s: n=%d cls acc %.4f, exit acc %s", tag, len(r["y"]),
                 (r["logits"].argmax(1) == r["y"]).float().mean().item(), [round(v, 4) for v in acc.tolist()])
    m.opt_mode(True)
    return ex


# ---------- CLI ----------

def plans() -> dict[str, list[list[str]]]:
    """Named lists of runs. Frozen runs need their source checkpoints under $LLS_OUT (missing ones are skipped)."""
    def frozen(exp: str) -> list[list[str]]:
        return [["--variant", "frozen", "--source", str(run_dir(j)), "--epochs", str(FROZEN_EPOCHS)] for j in jobs(exp)]

    joint = {"vgg8": ["--variant", "joint", "--arch", "vgg8", "--epochs", "100"],
             "mobilenet": ["--variant", "joint", "--arch", "mobilenet", "--epochs", "300"]}
    return {
        "mobilenet": frozen("mobilenet300") + [joint["mobilenet"]],
        "vgg8": frozen("main") + [joint["vgg8"]],
        "mobilenet_frozen": frozen("mobilenet300"),
        "vgg8_frozen": frozen("main"),
        "mobilenet_joint": [joint["mobilenet"]],
        "vgg8_joint": [joint["vgg8"]],
    }


def args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train learned (BP / probe) exit heads and dump them.")
    a = p.add_argument
    a("--plan", choices=sorted(plans()), help="run a named list of runs (other flags are added to each)")
    a("--variant", choices=VARIANTS, default="joint")
    a("--source", help="frozen: the trained run's folder (config.json + ckpt.pt)")
    a("--arch", default="vgg8", choices=sorted(ARCHS), help="joint only; frozen takes the source's")
    a("--dataset", default="cifar10aug", choices=sorted(data.SETS), help="joint only; frozen takes the source's")
    a("--partition", default=None, help="exit positions as block ends (default: joint every unit; frozen the "
                                        "source's LLS blocks, or every unit for BP)")
    a("--epochs", type=int, default=None, help=f"default: joint 100, frozen {FROZEN_EPOCHS}")
    a("--seed", type=int, default=0)
    a("--exp", default="X-EXIT", help="wandb job_type")
    a("--optim", default="AdamWSF")
    a("--lr", type=float, default=5e-3)
    a("--wd", type=float, default=0.0)
    a("--bs", type=int, default=128)
    a("--eval-bs", type=int, default=256)
    a("--n-val", type=int, default=5000)
    a("--workers", type=int, default=2)
    a("--limit", type=int, default=None, help="train/val subset for smoke tests (dump not uploaded)")
    a("--data-dir", default=None)
    a("--out", default=None, help="else $LLS_OUT, else ./runs")
    a("--wandb", action=argparse.BooleanOptionalAction, default=True)
    a("--project", default=os.environ.get("WANDB_PROJECT", "lls-granularity"))
    a("--entity", default=os.environ.get("WANDB_ENTITY"))
    a("--fresh", action="store_true", help="delete this run's checkpoint, dump and wandb id; start over")
    a("--log", default=None)
    o = p.parse_args(argv)
    if not o.plan:
        if o.variant == "frozen" and not o.source:
            p.error("--variant frozen needs --source")
        o.epochs = o.epochs or (FROZEN_EPOCHS if o.variant == "frozen" else 100)
    return o


def run_one(a: argparse.Namespace) -> dict[str, Any] | str:
    out = Path(a.out or os.environ.get("LLS_OUT", "./runs")).expanduser()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if a.variant == "frozen" and not (Path(a.source).expanduser() / "ckpt.pt").exists():
        log.warning("%s: no source checkpoint, skipping", a.source)
        return "no ckpt"
    ds = D.rebuild(D.cfg_of(Path(a.source).expanduser())).dataset if a.variant == "frozen" else a.dataset
    d = data.load(ds, a.data_dir, a.bs, a.eval_bs, a.n_val, a.workers, a.seed, a.limit)
    gen = d.train.generator
    m, rid = build(a, d, dev)
    rd = out / rid.name
    logs.setup(a.log, rd / "run.log")
    path = rd / "ckpt.pt"
    if a.fresh:
        for f in (path, rd / "wandb_id", rd / "exits.pt"):
            f.unlink(missing_ok=True)
    elif (rd / "exits.pt").exists():
        log.info("%s already trained and dumped; skipping", rid.name)
        return "exists"

    start, best = 0, None
    if path.exists():
        start, best = ckpt.load(path, cast(Any, m), gen)
    else:
        run.seed_all(a.seed)
    splits = {"val": d.val, "test": d.test} if d.val else {"test": d.test}
    if start < a.epochs:
        tr = Tracker(rid, vars(a), out, a.wandb, a.project, a.entity, start)
        ok = False
        try:
            fit(cast(Any, m), d.train, splits, tr, a.epochs, dev, start, best,
                save=lambda ep, b: ckpt.save(path, cast(Any, m), ep, b, gen))
            ok = True
        finally:
            tr.finish(ok)
    ep = ckpt.peek(path) + 1
    meta = {"run": rid.name, "arch": rid.arch, "dataset": rid.dataset, "seed": a.seed, "epochs_done": ep,
            "best": asdict(Best(**torch.load(path, map_location="cpu", weights_only=False)["best"])),
            "git_train": D.cfg_of(rd).get("git"), "limit": a.limit, "kind": rid.tag, "source": rid.source}
    ex = dump(m, rid.arch, cast(dict[str, DataLoader], splits), d.train, dev, meta)
    torch.save(ex, rd / "exits.pt")
    log.info("wrote %s", rd / "exits.pt")
    return D.record(rd, rd, ex, a)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    o = args(argv)
    if o.plan:
        rest = [v for i, v in enumerate(argv) if v != "--plan" and (i == 0 or argv[i - 1] != "--plan")]
        runs = [args(p + rest) for p in plans()[o.plan]]
    else:
        runs = [o]
    log.info("%d exit-head runs", len(runs))
    rows: dict[str, dict[str, Any] | str] = {}
    for a in runs:
        key = Path(a.source).name + " (frozen)" if a.variant == "frozen" else f"{a.arch} joint"
        rows[key] = run_one(a)
    D.report(rows)
    if any(r == "no ckpt" for r in rows.values()):
        sys.exit(1)


if __name__ == "__main__":
    logs.setup()
    main()
