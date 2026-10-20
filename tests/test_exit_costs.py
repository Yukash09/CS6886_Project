import pytest
import torch

from lls_exit.costs import ExitCosts, exit_costs
from lls_granular.units import mobilenet_v1, vgg8


@pytest.fixture(scope="module")
def lls() -> ExitCosts:
    return exit_costs(vgg8(), (1,) * 7, 10)


def test_vgg8_lls_exits(lls: ExitCosts) -> None:
    assert lls.names[0] == "conv1" and lls.names[-1] == "cls" and len(lls.names) == 8
    assert lls.backbone[0] == 128 * 32 * 32 * 3 * 9
    assert torch.all(lls.backbone.diff() >= 0)
    assert lls.backbone[-2] == lls.backbone[-1]  # last head and cls see the same backbone
    assert lls.head[6] == 1024 + 1024 + 1024 * 10  # fc1: pool, normalise, project
    assert lls.head[7] == 1024 * 10  # cls on a vector: no pool
    assert lls.full == lls.backbone[-1] + lls.head[-1]
    assert 715 < lls.full / 1e6 < 725


def test_blocks_merge_exits() -> None:
    c = exit_costs(vgg8(), (2, 2, 3), 10)
    assert c.names == ["conv1+conv2", "conv3+conv4", "conv5+conv6+fc1", "cls"]
    lls = exit_costs(vgg8(), (1,) * 7, 10)
    assert torch.equal(c.backbone, lls.backbone[[1, 3, 6, 7]])


def test_per_sample_pays_checked_heads_only(lls: ExitCosts) -> None:
    every = torch.ones(8, dtype=torch.bool)
    late = torch.tensor([False] * 4 + [True] * 4)
    ex = torch.tensor([4, 7])
    bb, hd = lls.per_sample(ex, every)
    assert torch.equal(bb, lls.backbone[ex])
    assert hd[0] == lls.head[:5].sum() and hd[1] == lls.head.sum()
    _, hd2 = lls.per_sample(ex, late)
    assert hd2[0] == lls.head[4] and hd2[1] == lls.head[4:].sum()
    with pytest.raises(ValueError):
        lls.per_sample(torch.tensor([2]), late)


def test_mobilenet_cls_pools() -> None:
    c = exit_costs(mobilenet_v1(), (1,) * 14, 10)
    assert len(c.names) == 15
    assert c.head[-1] == 1024 * 3 * 3 + 1024 * 10  # global pool over 3x3, then linear
    assert 40 < c.full / 1e6 < 50
