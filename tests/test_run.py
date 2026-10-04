import pytest

from lls_granular.bp import BPNet
from lls_granular.run import args, model, strategy
from lls_granular.wrapper import GranularLLS


def test_defaults() -> None:
    a = args([])
    assert (a.arch, a.dataset, a.partition, a.epochs, a.lr, a.wave, a.temp) == \
        ("vgg8", "cifar10aug", "uniform:1", 100, 5e-3, "square", 0.5)
    assert a.wandb is True and args(["--no-wandb"]).wandb is False


@pytest.mark.parametrize("spec,part", [("uniform:1", (1,) * 7), ("aligned", (2, 2, 1, 1, 1)),
                                       ("2,2,3", (2, 2, 3))])
def test_model_granular(spec: str, part: tuple[int, ...]) -> None:
    m, p = model(args(["--partition", spec]), 10)
    assert isinstance(m, GranularLLS) and p == part


def test_model_bp() -> None:
    m, p = model(args(["--partition", "bp"]), 10)
    assert isinstance(m, BPNet) and p is None


def test_same_seed_same_init() -> None:
    a = args(["--partition", "aligned", "--seed", "3"])
    m1, _ = model(a, 10)
    m2, _ = model(a, 10)
    assert all((p == q).all() for p, q in zip(m1.parameters(), m2.parameters()))


def test_strategy_tag() -> None:
    assert strategy("aligned") == "aligned" and strategy("uniform:2") == "uniform:2"
    assert strategy("2,2,3") is None
