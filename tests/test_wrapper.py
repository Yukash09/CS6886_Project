import pytest
import torch

from lls_granular.partitions import Partition
from lls_granular.units import mobilenet_v1, vgg8
from lls_granular.upstream import LLS_MobileNetV1, LLS_VGG8
from lls_granular.wrapper import GranularLLS
from tests.helpers import BS, batches


@pytest.mark.usefixtures("cpu_cuda")
def test_k1_equals_upstream_lls() -> None:
    torch.manual_seed(0)
    ref = LLS_VGG8(3, 10, lr=5e-3, optimizer="AdamWSF", training_mode="LLS",
                   temperature=0.5, waveform="square").train()
    torch.manual_seed(0)
    ours = GranularLLS(vgg8(), (1,) * 7, 10).train()
    layers = [ref.conv_block1, ref.conv_block2, ref.conv_block3, ref.conv_block4,
              ref.conv_block5, ref.conv_block6, ref.linear_block1]

    for x, y in batches(3):
        before = torch.tensor([l.loss_hist for l in layers])
        logits = ref(x, y)
        ref_losses = (torch.tensor([l.loss_hist for l in layers]) - before) / BS
        out = ours.step(x, y)
        assert torch.allclose(out.losses[:7], ref_losses.float(), atol=1e-5)
        assert torch.allclose(out.logits, logits.detach(), atol=1e-5)

    ref.optimizer_eval()
    ours.opt_mode(False)
    x, _ = batches(1, seed=2)[0]
    assert torch.allclose(ours.eval()(x), ref.eval()(x), atol=1e-5)


def test_gradients_stop_at_block_inputs() -> None:
    torch.manual_seed(0)
    m = GranularLLS(vgg8(), (2, 2, 3), 10, optim="SGD", lr=0.0).train()
    x, y = batches(1)[0]
    m.step(x, y)
    b0 = m.blocks[0]
    params = list(b0.parameters())
    stored = [p.grad.clone() for p in params]

    loss, _ = b0.head(b0(x), y)
    fresh = torch.autograd.grad(loss, params)
    for s, f in zip(stored, fresh):
        assert torch.allclose(s, f, atol=1e-6)


@pytest.mark.parametrize("part", [(1,) * 7, (2, 2, 2, 1), (2, 2, 1, 1, 1), (7,)])
def test_partitions_build_and_step(part: Partition) -> None:
    m = GranularLLS(vgg8(), part, 10).train()
    x, y = batches(1)[0]
    out = m.step(x, y)
    assert out.losses.shape == (len(part) + 1,)
    assert out.hits.shape == (len(part),)
    assert out.logits.shape == (BS, 10)
    assert torch.isfinite(out.losses).all()


def test_optimizers_own_disjoint_params() -> None:
    m = GranularLLS(vgg8(), (2, 2, 3), 10)
    groups = [{id(p) for g in o.param_groups for p in g["params"]} for o in m.opts]
    assert len(groups) == 4
    assert sum(map(len, groups)) == len(set().union(*groups)) == len(list(m.parameters()))


@pytest.mark.usefixtures("cpu_cuda")
def test_mobilenet_k1_equals_upstream_lls() -> None:
    torch.manual_seed(0)
    ref = LLS_MobileNetV1(3, 10, lr=5e-3, optimizer="AdamWSF", training_mode="LLS",
                          temperature=0.5, waveform="square").train()
    torch.manual_seed(0)
    ours = GranularLLS(mobilenet_v1(), (1,) * 14, 10).train()
    layers = [getattr(ref, f"conv_block{i}") for i in range(1, 15)]

    for x, y in batches(2):
        before = torch.tensor([l.loss_hist for l in layers])
        logits = ref(x, y)
        ref_losses = (torch.tensor([l.loss_hist for l in layers]) - before) / BS
        out = ours.step(x, y)
        assert torch.allclose(out.losses[:14], ref_losses.float(), atol=1e-5)
        assert torch.allclose(out.logits, logits.detach(), atol=1e-5)
