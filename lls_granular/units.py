import logging
import math
from typing import Callable

from torch import Tensor, nn

from .upstream import ConvBlock, ConvDWBlock, LinearBlock

log = logging.getLogger(__name__)


class Unit(nn.Module):
    """A trainable layer plus the ops feeding it. Block boundaries fall between units."""

    def __init__(self, name: str, layer: nn.Module, pool: int, shape: tuple[int, ...],
                 pre: nn.Module | None = None) -> None:
        super().__init__()
        self.name = name
        self.pre = pre or nn.Identity()
        self.layer = layer
        self.pool = pool    # head's adaptive-pool size (upstream `pooling_size`)
        self.shape = shape  # per-sample output shape

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(self.pre(x))

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def head_dim(self) -> int:
        """Length of the pooled, flattened vector the LLS head projects."""
        return self.shape[0] * self.pool ** 2 if len(self.shape) == 3 else self.pool

    def extra_repr(self) -> str:
        return f"name={self.name}, shape={self.shape}, pool={self.pool}, head_dim={self.head_dim}"


def describe(units: list[Unit]) -> None:
    for i, u in enumerate(units):
        log.debug("unit %d %-5s pre=%-10s out=%-15s numel=%-7d head_dim=%d",
                  i, u.name, type(u.pre).__name__, u.shape, u.numel, u.head_dim)


def _conv(cin: int, cout: int) -> ConvBlock:
    return ConvBlock(cin, cout, kernel_size=3, stride=1, padding=1)


def vgg8(in_ch: int = 3, img: int = 32) -> list[Unit]:
    """Upstream LLS_VGG8 split into 7 units; the final classifier is not a unit."""
    s1, s2, s3, s4 = img, img // 2, img // 4, img // 8
    units = [
        Unit("conv1", _conv(in_ch, 128), 4, (128, s1, s1)),
        Unit("conv2", _conv(128, 256), 2, (256, s1, s1)),
        Unit("conv3", _conv(256, 256), 2, (256, s2, s2), pre=nn.MaxPool2d(2)),
        Unit("conv4", _conv(256, 256), 2, (256, s2, s2)),
        Unit("conv5", _conv(256, 512), 2, (512, s3, s3), pre=nn.MaxPool2d(2)),
        Unit("conv6", _conv(512, 512), 2, (512, s4, s4), pre=nn.MaxPool2d(2)),
        Unit("fc1", LinearBlock(512 * 2 * 2, 1024), 1024, (1024,),
             pre=nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Flatten())),
    ]
    log.debug("vgg8: %d units, in_ch=%d, img=%d", len(units), in_ch, img)
    describe(units)
    return units


def _side(h: int, s: int, p: int) -> int:
    """Output side of a 3x3 conv (stride s, padding p)."""
    return (h + 2 * p - 3) // s + 1


# (out channels, stride, head pool) for the 13 depthwise-separable blocks of upstream LLS_MobileNetV1
_MN = [(64, 1, 4), (128, 2, 4), (128, 1, 4), (256, 2, 2), (256, 1, 2), (512, 2, 2),
       *[(512, 1, 2)] * 5, (1024, 2, 1), (1024, 1, 1)]


def mobilenet_v1(in_ch: int = 3, img: int = 32) -> list[Unit]:
    """Upstream LLS_MobileNetV1 as 14 units. Upstream's ConvDWBlock pads its depthwise conv by
    `stride`, so stride-2 blocks give odd sides (32 -> 16, 9, 6, 4, 3); kept as is."""
    s = _side(img, 2, 1)
    units = [Unit("conv1", ConvBlock(in_ch, 32, kernel_size=3, stride=2, padding=1), 8, (32, s, s))]
    cin = 32
    for i, (cout, st, pool) in enumerate(_MN, start=2):
        s = _side(s, st, st)
        units.append(Unit(f"dw{i}", ConvDWBlock(cin, cout, kernel_size=3, stride=st, padding=1), pool, (cout, s, s)))
        cin = cout
    log.debug("mobilenet_v1: %d units, in_ch=%d, img=%d", len(units), in_ch, img)
    describe(units)
    return units


ARCHS: dict[str, Callable[..., list[Unit]]] = {"vgg8": vgg8, "mobilenet": mobilenet_v1}
