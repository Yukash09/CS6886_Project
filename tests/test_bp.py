import torch
import torch.nn.functional as F

from lls_granular.bp import BPNet
from lls_granular.units import vgg8
from lls_granular.upstream import LLS_VGG8, AdamWScheduleFree
from tests.helpers import BS, batches


def test_equals_upstream_bp() -> None:
    torch.manual_seed(0)
    ref = LLS_VGG8(3, 10, training_mode="BP").train()
    ref_opt = AdamWScheduleFree(ref.parameters(), lr=5e-3)
    torch.manual_seed(0)
    ours = BPNet(vgg8(), 10).train()

    for x, y in batches(3):
        ref_opt.zero_grad()
        logits = ref(x)
        F.cross_entropy(logits, y).backward()
        ref_opt.step()
        out = ours.step(x, y)
        assert torch.allclose(out.logits, logits.detach(), atol=1e-5)

    ref_opt.eval()
    ours.opt_mode(False)
    x, _ = batches(1, seed=2)[0]
    assert torch.allclose(ours.eval()(x), ref.eval()(x), atol=1e-5)


def test_out_shapes_and_all_params_train() -> None:
    torch.manual_seed(0)
    m = BPNet(vgg8(), 10).train()
    before = [p.clone() for p in m.parameters()]
    out = m.step(*batches(1)[0])
    assert out.losses.shape == (1,) and out.hits.shape == (0,)
    assert out.logits.shape == (BS, 10)
    assert all(not torch.equal(b, p) for b, p in zip(before, m.parameters()))
