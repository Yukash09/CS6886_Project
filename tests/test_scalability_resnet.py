import pytest
import torch
from torch import nn

from lls_granular.bp import BPNet
from lls_granular.partitions import aligned
from lls_granular.wrapper import GranularLLS
from scalability.resnet import BasicBlock, cifar_resnet, head_pool, resnet20, resnet32, resnet56
from scalability.wide_resnet import wide_resnet20


@pytest.mark.parametrize("factory,n_units", [(resnet20, 10), (resnet32, 16), (resnet56, 28)])
def test_resnet_shapes(factory, n_units: int) -> None:
    units = factory()
    assert len(units) == n_units
    x = torch.randn(2, 3, 32, 32)
    for unit in units:
        x = unit(x)
        assert tuple(x.shape[1:]) == unit.shape, unit.name
        assert unit.head_dim <= 2048


def test_resnet20_stage_partition_and_pools() -> None:
    units = resnet20()
    assert aligned([unit.shape for unit in units]) == (4, 3, 3)
    assert [unit.pool for unit in units] == [8] * 4 + [8] * 3 + [4] * 3
    assert [unit.head_dim for unit in units] == [1024] * 4 + [2048] * 3 + [1024] * 3


def test_wide_resnet20_is_separate_and_uses_full_final_lls_dimension() -> None:
    standard, wide = resnet20(), wide_resnet20()
    assert len(wide) == 10
    assert [unit.shape[0] for unit in wide] == [32] * 4 + [64] * 3 + [128] * 3
    assert aligned([unit.shape for unit in wide]) == (4, 3, 3)
    assert [unit.pool for unit in wide] == [8] * 4 + [4] * 3 + [4] * 3
    assert wide[-1].head_dim == 2048
    assert sum(p.numel() for unit in wide for p in unit.parameters()) > 3 * sum(
        p.numel() for unit in standard for p in unit.parameters())


def test_projection_shortcut() -> None:
    block = BasicBlock(16, 32, stride=2)
    assert isinstance(block.shortcut, nn.Sequential)
    assert block(torch.randn(2, 16, 32, 32)).shape == (2, 32, 16, 16)
    assert isinstance(BasicBlock(16, 16).shortcut, nn.Identity)


def test_depth_and_pool_validation() -> None:
    with pytest.raises(ValueError, match="6n\\+2"):
        cifar_resnet(21)
    assert head_pool(16, 32) == 8
    assert head_pool(32, 16) == 8
    assert head_pool(64, 8) == 4
    with pytest.raises(ValueError, match="positive"):
        cifar_resnet(20, channels=(16, 0, 64))


@pytest.mark.parametrize("part", [(1,) * 10, (2, 2, 2, 2, 2), (4, 3, 3)])
def test_lls_resnet_step(part: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    model = GranularLLS(resnet20(), part, 10, optim="SGD", lr=1e-3).train()
    x = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, 10, (2,))
    out = model.step(x, y)
    assert out.logits.shape == (2, 10)
    assert out.losses.shape == (len(part) + 1,)
    assert torch.isfinite(out.losses).all()
    assert all(parameter.grad is not None for block in model.blocks for parameter in block.parameters())


def test_bp_resnet_step() -> None:
    model = BPNet(resnet20(), 10, optim="SGD", lr=1e-3).train()
    out = model.step(torch.randn(2, 3, 32, 32), torch.randint(0, 10, (2,)))
    assert out.logits.shape == (2, 10)
    assert out.losses.shape == (1,)


def test_pure_lls_readout_has_no_linear_classifier() -> None:
    model = GranularLLS(resnet20(), (7, 3), 10, optim="SGD", lr=1e-3,
                        mode="lls-mxm", readout="lls").train()
    out = model.step(torch.randn(2, 3, 32, 32), torch.randint(0, 10, (2,)))
    assert model.cls is None
    assert out.logits.shape == (2, 10)
    assert out.losses.shape == (2,)
    logits, scores = model.forward_all(torch.randn(2, 3, 32, 32))
    assert torch.equal(logits, scores[-1])


@pytest.mark.parametrize("mode,extra", [("lls-m", 100), ("lls-mxm", 1000)])
def test_modulated_resnet_step_updates_modulation(mode: str, extra: int) -> None:
    torch.manual_seed(0)
    model = GranularLLS(resnet20(), (1,) * 10, 10, optim="SGD", lr=1e-3, mode=mode).train()
    modulation = [block.head.M for block in model.blocks]
    before = [value.detach().clone() for value in modulation]
    out = model.step(torch.randn(2, 3, 32, 32), torch.randint(0, 10, (2,)))
    assert torch.isfinite(out.losses).all()
    assert sum(value.numel() for value in modulation) == extra
    assert all(not torch.equal(old, new) for old, new in zip(before, modulation))
