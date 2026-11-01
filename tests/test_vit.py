import torch

from lls_granular.bp import BPNet
from lls_granular.heads import pool_flat
from lls_granular.metrics import Costs
from lls_granular.wrapper import GranularLLS
from vit.architectures import SelfAttention, cifar_vit, vit_micro, vit_tiny
from vit.measure import candidates
from vit.run import args, model


def test_token_pool_uses_cls_and_mean_patch() -> None:
    x = torch.arange(2 * 5 * 8, dtype=torch.float32).reshape(2, 5, 8)
    expected = torch.cat((x[:, 0], x[:, 1:].mean(1)), dim=1)
    assert torch.equal(pool_flat(x, 16), expected)
    assert pool_flat(x, 7).shape == (2, 7)


def test_vit_shapes_and_head_dimensions() -> None:
    units = vit_tiny()
    assert len(units) == 12
    x = torch.randn(2, 3, 32, 32)
    for unit in units:
        x = unit(x)
        assert tuple(x.shape[1:]) == unit.shape
        assert unit.head_dim == 384
    assert x.shape == (2, 65, 192)


def test_attention_connects_tokens_and_has_gradients() -> None:
    attention = SelfAttention(32, 4, 0.0)
    x = torch.randn(2, 9, 32, requires_grad=True)
    out = attention(x)
    out[:, 0].sum().backward()
    assert out.shape == x.shape
    assert x.grad is not None and x.grad[:, 1:].abs().sum() > 0
    assert attention.extra_macs(out) == 2 * 9 * 9 * 32


def test_vit_lls_and_bp_steps() -> None:
    x, y = torch.randn(2, 3, 32, 32), torch.randint(0, 10, (2,))
    lls = GranularLLS(vit_micro(), (1, 1), 10, optim="SGD", lr=1e-3,
                      mode="lls-mxm").train()
    local = lls.step(x, y)
    assert local.logits.shape == (2, 10)
    assert local.losses.shape == (3,)
    assert all(parameter.grad is not None for block in lls.blocks for parameter in block.parameters())

    bp = BPNet(vit_micro(), 10, optim="SGD", lr=1e-3).train()
    end_to_end = bp.step(x, y)
    assert end_to_end.logits.shape == (2, 10)
    assert end_to_end.losses.shape == (1,)


def test_vit_cli_models_and_partitions() -> None:
    bp, part = model(args(["--arch", "vit_micro", "--partition", "bp",
                           "--lls-mode", "lls"]), 10)
    assert isinstance(bp, BPNet) and part is None

    lls, part = model(args(["--arch", "vit_micro", "--partition", "uniform:1"]), 10)
    assert isinstance(lls, GranularLLS) and part == (1, 1)
    assert lls.blocks[0].head.mode == "lls-mxm"


def test_vit_validation_and_costs() -> None:
    try:
        cifar_vit(depth=0, dim=32, heads=2)
        raise AssertionError("depth validation did not run")
    except ValueError:
        pass
    try:
        cifar_vit(depth=1, dim=31, heads=2)
        raise AssertionError("head divisibility validation did not run")
    except ValueError:
        pass

    assert candidates("vit_micro") == [None, (1, 1), (2,)]
    costs = Costs(vit_micro(), 10, 2, torch.device("cpu"))
    assert costs.macs[0] > 0 and costs.macs[1] > 0
    assert costs.cost((1, 1))["act_peak_mb"] < costs.cost(None)["act_peak_mb"]

