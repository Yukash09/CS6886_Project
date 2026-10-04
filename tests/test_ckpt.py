from pathlib import Path

import torch

from lls_granular import ckpt
from lls_granular.track import RunId, Tracker
from lls_granular.train import Best, fit
from lls_granular.units import vgg8
from lls_granular.wrapper import GranularLLS
from tests.helpers import batches

CPU = torch.device("cpu")
TRAIN, VAL = batches(2, seed=1), batches(1, seed=2)


def net() -> GranularLLS:
    torch.manual_seed(0)
    return GranularLLS(vgg8(), (2, 2, 3), 10)


def tracker(tmp: Path) -> Tracker:
    return Tracker(RunId("vgg8", "fake", (2, 2, 3), 2, 0), {}, tmp, wandb=False)


def test_resume_equals_uninterrupted(tmp_path: Path) -> None:
    ref = net()
    torch.manual_seed(1)
    fit(ref, TRAIN, {"val": VAL}, tracker(tmp_path / "a"), 2, CPU)

    path = tmp_path / "ckpt.pt"
    m = net()
    torch.manual_seed(1)
    fit(m, TRAIN, {"val": VAL}, tracker(tmp_path / "b"), 1, CPU, save=lambda e, b: ckpt.save(path, m, e, b))

    fresh = GranularLLS(vgg8(), (2, 2, 3), 10)  # different init, overwritten by the checkpoint
    start, best = ckpt.load(path, fresh, CPU)
    assert start == 1 and best.epoch == 0
    fit(fresh, TRAIN, {"val": VAL}, tracker(tmp_path / "b"), 2, CPU, start=start, best=best)

    for p, q in zip(ref.parameters(), fresh.parameters()):
        assert torch.allclose(p, q, atol=1e-6)


def test_save_is_atomic_and_restores_generator(tmp_path: Path) -> None:
    m, path = net(), tmp_path / "ckpt.pt"
    g = torch.Generator().manual_seed(5)
    state = g.get_state()
    ckpt.save(path, m, 3, Best(0.5, 2, 0.4), g)
    assert path.exists() and not path.with_suffix(".tmp").exists()

    torch.rand(3, generator=g)
    start, best = ckpt.load(path, net(), CPU, g)
    assert (start, best) == (4, Best(0.5, 2, 0.4))
    assert torch.equal(g.get_state(), state)
