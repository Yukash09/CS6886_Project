from pathlib import Path

import pytest
import torch

from lls_granular.bp import BPNet
from lls_granular.train import fit
from lls_granular.wrapper import GranularLLS
from scalability.exps import EXPS, jobs
from scalability.measure import candidates
from scalability.run import ScalabilityRunId, args, model, select_device


def test_model_variants() -> None:
    bp, part = model(args(["--arch", "resnet20", "--partition", "bp"]), 10)
    assert isinstance(bp, BPNet) and part is None

    lls, part = model(args(["--arch", "resnet20", "--partition", "aligned"]), 10)
    assert isinstance(lls, GranularLLS) and part == (4, 3, 3)

    modulated, part = model(args(["--partition", "uniform:1", "--lls-mode", "lls-mxm"]), 10)
    assert isinstance(modulated, GranularLLS) and part == (1,) * 10
    assert sum(block.head.M.numel() for block in modulated.blocks) == 1000

    wide, part = model(args(["--arch", "wide_resnet20", "--partition", "7,3",
                             "--lls-mode", "lls-mxm", "--readout", "lls"]), 10)
    assert isinstance(wide, GranularLLS) and part == (7, 3)
    assert wide.cls is None
    assert wide.blocks[-1].head.B.shape == (10, 2048)

    with pytest.raises(ValueError, match="only applies"):
        model(args(["--partition", "bp", "--lls-mode", "lls-m"]), 10)
    with pytest.raises(ValueError, match="only applies"):
        model(args(["--partition", "bp", "--readout", "lls"]), 10)


def test_patience_argument() -> None:
    assert args(["--epochs", "100", "--patience", "10"]).patience == 10


def test_device_selection(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert select_device().type == "mps"
    assert select_device("cpu").type == "cpu"


def test_unavailable_mps_is_an_error(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS was requested"):
        select_device("mps")


def test_experiments_and_seeds() -> None:
    assert set(EXPS) == {"smoke", "resnet20", "depth", "cifar100", "variants"}
    repeated = jobs("resnet20", [1, 2])
    assert len(repeated) == 8
    assert {job.seed for job in repeated} == {1, 2}


def test_variant_run_names_do_not_collide() -> None:
    base = dict(arch="resnet20", dataset="cifar10aug", part=(1,) * 10, epochs=100, seed=0)
    names = {ScalabilityRunId(**base, lls_mode=mode).name for mode in ("lls", "lls-m", "lls-mxm")}
    assert len(names) == 3
    assert any("_lls-mxm_" in name for name in names)
    lls_readout = ScalabilityRunId(**base, lls_mode="lls-mxm", readout="lls").name
    assert "_lls-mxm_lls-readout_" in lls_readout


def test_measure_candidates() -> None:
    values = candidates("resnet20")
    assert values == [None, (1,) * 10, (2,) * 5, (4, 3, 3), (10,)]


class FakeTracker:
    def __init__(self, path: Path) -> None:
        self.rows: list[dict[str, float]] = []
        self.final: dict[str, object] = {}
        self.path = path

    def log(self, row: dict[str, float]) -> None:
        self.rows.append(row)

    def col(self, key: str) -> list[float]:
        return [float(row[key]) for row in self.rows]

    def summary(self, values: dict[str, object]) -> None:
        self.final = values


def test_fit_early_stops(monkeypatch, tmp_path: Path) -> None:
    import lls_granular.train as train

    values = iter([0.5, 0.4, 0.4, 0.9])
    monkeypatch.setattr(train, "train_epoch", lambda *_: {"train/loss": 1.0, "train/acc": 0.5})
    monkeypatch.setattr(train, "eval_all", lambda *_: {"val/acc": next(values), "val/loss": 1.0})
    tracker = FakeTracker(tmp_path)
    best = fit(object(), [], {"val": []}, tracker, 100, torch.device("cpu"), patience=2)  # type: ignore[arg-type]
    assert best.epoch == 0
    assert len(tracker.rows) == 3
    assert tracker.final["early_stop/stopped"] is True
    assert tracker.final["epochs/completed"] == 3
