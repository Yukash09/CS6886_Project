"""Run identity, naming and result logging: wandb (primary) + local files (backup)."""
import csv
import json
import logging
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .partitions import Partition

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]


def method(part: Partition | None) -> str:
    """'bp' for backprop, else 'p' + block sizes, e.g. (2, 2, 3) -> 'p2.2.3'."""
    return "bp" if part is None else "p" + ".".join(map(str, part))


def kind(part: Partition | None) -> str:
    if part is None:
        return "bp"
    if len(part) == 1:
        return "single"
    return "lls" if max(part) == 1 else "granular"


@dataclass
class RunId:
    arch: str
    dataset: str
    part: Partition | None
    epochs: int
    seed: int
    exp: str = "dev"
    sched: str = "full"
    strategy: str | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def group(self) -> str:
        """Everything but the seed: wandb averages runs in a group."""
        return f"{self.arch}_{self.dataset}_{method(self.part)}_e{self.epochs}"

    @property
    def name(self) -> str:
        return f"{self.group}_s{self.seed}"

    def labels(self) -> list[str]:
        t = [self.exp, self.arch, self.dataset, kind(self.part), self.sched, *self.tags]
        return t + [self.strategy] if self.strategy else t

    def fields(self) -> dict[str, Any]:
        """Flat, filterable config fields derived from the identity."""
        p = self.part
        return {
            "arch": self.arch, "dataset": self.dataset, "method": method(p), "kind": kind(p),
            "exp": self.exp, "sched": self.sched, "strategy": self.strategy,
            "epochs": self.epochs, "seed": self.seed,
            "n_blocks": len(p) if p else None, "blocks": list(p) if p else None,
            "max_block": max(p) if p else None, "first_block": p[0] if p else None,
        }


def env() -> dict[str, Any]:
    """Reproducibility info: code version and hardware."""
    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True,
                                  timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"git": commit + ("-dirty" if dirty else ""), "gpu": gpu, "torch": torch.__version__}


class Tracker:
    """Logs one row per epoch to metrics.csv and wandb; keeps summary.json and run.summary in sync."""

    def __init__(self, rid: RunId, cfg: dict[str, Any], out: Path, wandb: bool = True,
                 project: str = "lls-granularity", entity: str | None = None, start: int = 0) -> None:
        self.dir = out / rid.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = {**cfg, **rid.fields(), **env()}
        (self.dir / "config.json").write_text(json.dumps(self.cfg, indent=2, default=str))
        self.csv = self.dir / "metrics.csv"
        self._trim(start)
        self.run = self._wandb(rid, project, entity) if wandb else None
        log.info("run %s (group %s, exp %s) -> %s, wandb=%s",
                 rid.name, rid.group, rid.exp, self.dir, self.run is not None)
        log.debug("tags %s", rid.labels())

    def _trim(self, start: int) -> None:
        """On resume, drop rows from epochs that will be re-run."""
        if not self.csv.exists():
            return
        with self.csv.open() as f:
            rows = [r for r in csv.DictReader(f) if int(r["epoch"]) < start]
        if rows:
            self._write(rows, "w")
        else:
            self.csv.unlink()
        log.debug("metrics.csv trimmed to %d rows (resume at epoch %d)", len(rows), start)

    def _write(self, rows: list[dict[str, Any]], mode: str) -> None:
        new = mode == "w" or not self.csv.exists()
        with self.csv.open(mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)

    def _wandb(self, rid: RunId, project: str, entity: str | None) -> Any:
        try:
            import wandb
        except ImportError:
            log.warning("wandb not installed; logging to local files only")
            return None
        idf = self.dir / "wandb_id"
        rid_ = idf.read_text().strip() if idf.exists() else uuid.uuid4().hex[:12]
        idf.write_text(rid_)
        try:
            run = wandb.init(project=project, entity=entity, id=rid_, resume="allow", name=rid.name,
                             group=rid.group, job_type=rid.exp, tags=rid.labels(), config=self.cfg,
                             dir=str(self.dir))
        except Exception as e:  # network/auth problems must never kill a training run
            log.warning("wandb.init failed (%s); logging to local files only", e)
            return None
        run.define_metric("epoch")
        run.define_metric("*", step_metric="epoch")
        log.debug("wandb run id %s", rid_)
        return run

    def col(self, key: str) -> list[float]:
        """All logged values of one metric (across resumed sessions)."""
        if not self.csv.exists():
            return []
        with self.csv.open() as f:
            return [float(r[key]) for r in csv.DictReader(f) if r.get(key)]

    def log(self, row: dict[str, Any]) -> None:
        """One epoch: must contain 'epoch'. Same keys every epoch."""
        self._write([row], "a")
        if self.run:
            self.run.log(row)

    def summary(self, d: dict[str, Any]) -> None:
        f = self.dir / "summary.json"
        s = json.loads(f.read_text()) if f.exists() else {}
        s.update(d)
        f.write_text(json.dumps(s, indent=2))
        if self.run:
            self.run.summary.update(d)
        log.debug("summary %s", d)

    def finish(self, ok: bool = True) -> None:
        """ok=False (interrupt/crash) marks the wandb run failed; resuming makes it running again."""
        if self.run:
            self.run.finish(exit_code=0 if ok else 1)
