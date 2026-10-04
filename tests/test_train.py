import csv
from pathlib import Path

import pytest
import torch

from lls_granular.bp import BPNet
from lls_granular.track import RunId, Tracker
from lls_granular.train import Best, eval_all, evaluate, fit
from lls_granular.units import vgg8
from lls_granular.wrapper import GranularLLS
from tests.helpers import batches

CPU = torch.device("cpu")
TRAIN, VAL, TEST = batches(3, seed=1), batches(2, seed=2), batches(2, seed=3)


def tracker(tmp: Path, part: tuple[int, ...] | None) -> Tracker:
    return Tracker(RunId("vgg8", "fake", part, 2, 0), {}, tmp, wandb=False)


def test_fit_logs_all_keys(tmp_path: Path) -> None:
    torch.manual_seed(0)
    m = GranularLLS(vgg8(), (2, 2, 3), 10)
    t = tracker(tmp_path, (2, 2, 3))
    saved: list[int] = []
    best = fit(m, TRAIN, {"val": VAL, "test": TEST}, t, 2, CPU, save=lambda e, _: saved.append(e))

    with t.csv.open() as f:
        rows = list(csv.DictReader(f))
    assert [r["epoch"] for r in rows] == ["0", "1"] and saved == [0, 1]
    for k in ["train/acc", "train/loss", "val/acc", "test/acc", "best/val_acc", "time/train_s",
              "block/0/train_acc", "block/2/train_loss", "block/1/val_acc", "block/2/test_acc"]:
        assert k in rows[0], k
    assert "block/3/train_acc" not in rows[0]
    assert 0 <= best.val <= 1 and best.epoch in (0, 1)


def test_fit_bp_has_no_block_keys(tmp_path: Path) -> None:
    m = BPNet(vgg8(), 10)
    t = tracker(tmp_path, None)
    fit(m, TRAIN, {"val": VAL}, t, 1, CPU)
    with t.csv.open() as f:
        row = next(csv.DictReader(f))
    assert not any(k.startswith("block/") for k in row)


def test_resume_keeps_best_and_skips_done_epochs(tmp_path: Path) -> None:
    m = GranularLLS(vgg8(), (7,), 10)
    t = tracker(tmp_path, (7,))
    best = fit(m, TRAIN, {"val": VAL}, t, 3, CPU, start=3, best=Best(0.99, 1, 0.98))
    assert best == Best(0.99, 1, 0.98)
    assert not t.csv.exists()


def test_eval_restores_train_weights() -> None:
    torch.manual_seed(0)
    m = GranularLLS(vgg8(), (1,) * 7, 10).train()
    for x, y in TRAIN:
        m.step(x, y)
    before = [p.clone() for p in m.parameters()]
    eval_all(m, TRAIN, {"val": VAL}, CPU)
    assert all(torch.allclose(b, p, atol=1e-6) for b, p in zip(before, m.parameters()))  # lerp round-trip


def test_evaluate_matches_manual() -> None:
    torch.manual_seed(0)
    m = GranularLLS(vgg8(), (2, 5), 10).eval()
    r = evaluate(m, VAL, CPU, "val")
    hits = sum((m(x).argmax(1) == y).sum().item() for x, y in VAL)
    assert r["val/acc"] == pytest.approx(hits / sum(len(y) for _, y in VAL))
    assert set(r) == {"val/acc", "val/loss", "block/0/val_acc", "block/1/val_acc"}
