import csv
import json
from pathlib import Path
from typing import Any

import pytest

from lls_granular.track import RunId, Tracker, kind, method

RID = RunId("vgg8", "cifar10aug", (2, 2, 3), 100, 0, exp="G-P", strategy="2,2,3")


def rows(t: Tracker) -> list[dict[str, str]]:
    with t.csv.open() as f:
        return list(csv.DictReader(f))


def test_method_and_kind() -> None:
    assert method(None) == "bp" and kind(None) == "bp"
    assert method((2, 2, 3)) == "p2.2.3" and kind((2, 2, 3)) == "granular"
    assert kind((1,) * 7) == "lls" and kind((7,)) == "single"


def test_names() -> None:
    assert RID.group == "vgg8_cifar10aug_p2.2.3_e100"
    assert RID.name == "vgg8_cifar10aug_p2.2.3_e100_s0"
    assert RID.labels() == ["G-P", "vgg8", "cifar10aug", "granular", "full", "2,2,3"]
    f = RID.fields()
    assert f["n_blocks"] == 3 and f["blocks"] == [2, 2, 3] and f["max_block"] == 3


def test_local_files(tmp_path: Path) -> None:
    t = Tracker(RID, {"lr": 5e-3}, tmp_path, wandb=False)
    for e in range(3):
        t.log({"epoch": e, "val/acc": 0.5 + e / 10})
    t.summary({"best/val_acc": 0.7})
    cfg = json.loads((t.dir / "config.json").read_text())
    assert cfg["lr"] == 5e-3 and cfg["method"] == "p2.2.3" and "git" in cfg and "gpu" in cfg
    assert [r["epoch"] for r in rows(t)] == ["0", "1", "2"]
    assert json.loads((t.dir / "summary.json").read_text()) == {"best/val_acc": 0.7}


def test_resume_trims_rerun_epochs(tmp_path: Path) -> None:
    t = Tracker(RID, {}, tmp_path, wandb=False)
    for e in range(5):
        t.log({"epoch": e, "val/acc": 0.1})
    t = Tracker(RID, {}, tmp_path, wandb=False, start=3)
    t.log({"epoch": 3, "val/acc": 0.2})
    assert [r["epoch"] for r in rows(t)] == ["0", "1", "2", "3"]


class FakeRun:
    def __init__(self) -> None:
        self.logged: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}

    def define_metric(self, *_: Any, **__: Any) -> None: ...
    def log(self, row: dict[str, Any]) -> None: self.logged.append(row)
    def finish(self) -> None: ...


def test_wandb_args_and_stable_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wandb = pytest.importorskip("wandb")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(wandb, "init", lambda **k: calls.append(k) or FakeRun())

    t = Tracker(RID, {"lr": 5e-3}, tmp_path)
    t.log({"epoch": 0, "val/acc": 0.3})
    t.summary({"best/val_acc": 0.3})
    k = calls[0]
    assert (k["name"], k["group"], k["job_type"]) == (RID.name, RID.group, "G-P")
    assert "granular" in k["tags"] and k["config"]["blocks"] == [2, 2, 3]
    assert k["resume"] == "allow"
    assert t.run.logged == [{"epoch": 0, "val/acc": 0.3}] and t.run.summary["best/val_acc"] == 0.3

    Tracker(RID, {}, tmp_path, start=1)
    assert calls[1]["id"] == k["id"]


def test_wandb_failure_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wandb = pytest.importorskip("wandb")

    def boom(**_: Any) -> None:
        raise RuntimeError("no network")

    monkeypatch.setattr(wandb, "init", boom)
    t = Tracker(RID, {}, tmp_path)
    assert t.run is None
    t.log({"epoch": 0, "val/acc": 0.1})
    assert len(rows(t)) == 1


def test_col_spans_resumed_sessions(tmp_path: Path) -> None:
    t = Tracker(RID, {}, tmp_path, wandb=False)
    for e in range(3):
        t.log({"epoch": e, "time/train_s": 10.0})
    t = Tracker(RID, {}, tmp_path, wandb=False, start=2)
    t.log({"epoch": 2, "time/train_s": 12.0})
    assert t.col("time/train_s") == [10.0, 10.0, 12.0]


def test_finish_marks_interrupted_runs_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wandb = pytest.importorskip("wandb")
    codes: list[int] = []

    class Run(FakeRun):
        def finish(self, exit_code: int = 0) -> None:
            codes.append(exit_code)

    monkeypatch.setattr(wandb, "init", lambda **_: Run())
    Tracker(RID, {}, tmp_path).finish(ok=False)
    Tracker(RID, {}, tmp_path).finish()
    assert codes == [1, 0]
