import torch

from lls_granular.partitions import aligned
from lls_granular.units import mobilenet_v1, vgg8
from lls_granular.upstream import LLS_MobileNetV1, LLS_VGG8


def test_vgg8_shapes() -> None:
    units = vgg8()
    x = torch.randn(2, 3, 32, 32)
    for u in units:
        x = u(x)
        assert tuple(x.shape[1:]) == u.shape, u.name


def test_vgg8_head_dims() -> None:
    assert [u.head_dim for u in vgg8()] == [2048, 1024, 1024, 1024, 2048, 2048, 1024]


def test_vgg8_matches_upstream() -> None:
    torch.manual_seed(0)
    ref = LLS_VGG8(3, 10, training_mode="BP").eval()
    torch.manual_seed(0)
    units = vgg8()
    for u in units:
        u.eval()

    x = torch.randn(4, 3, 32, 32)
    hs = ref.forward_all_layers(x)[:7]
    for u, h in zip(units, hs):
        x = u(x)
        assert torch.allclose(x, h, atol=1e-6), u.name



def test_mobilenet_shapes_and_aligned() -> None:
    units = mobilenet_v1()
    assert len(units) == 14
    x = torch.randn(2, 3, 32, 32)
    for u in units:
        x = u(x)
        assert tuple(x.shape[1:]) == u.shape, u.name
    assert [u.shape[1] for u in units] == [16, 16, 9, 9, 6, 6, 4, 4, 4, 4, 4, 4, 3, 3]
    assert aligned([u.shape for u in units]) == (2, 2, 2, 6, 2)


def test_mobilenet_matches_upstream() -> None:
    torch.manual_seed(0)
    ref = LLS_MobileNetV1(3, 10, training_mode="BP").eval()
    torch.manual_seed(0)
    units = mobilenet_v1()
    x = h = torch.randn(4, 3, 32, 32)
    for i, u in enumerate(units, start=1):
        u.eval()
        x, h = u(x), getattr(ref, f"conv_block{i}")(h)
        assert torch.allclose(x, h, atol=1e-5), u.name
        assert u.pool == getattr(ref, f"conv_block{i}").pooling_size, u.name
