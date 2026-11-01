import logging
import math
from collections.abc import Callable

from torch import Tensor, nn

from lls_granular.units import Unit, describe

log = logging.getLogger(__name__)

MAX_HEAD_DIM = 2048
NEGATIVE_SLOPE = 0.01


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(NEGATIVE_SLOPE, inplace=False),
        )


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(NEGATIVE_SLOPE, inplace=False)
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


def head_pool(channels: int, side: int, max_dim: int = MAX_HEAD_DIM) -> int:
    limit = min(side, math.isqrt(max_dim // channels))
    pool = 1
    while pool * 2 <= limit:
        pool *= 2
    return pool


def _initialise(units: list[Unit]) -> None:
    for unit in units:
        for module in unit.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=NEGATIVE_SLOPE, mode="fan_out",
                                        nonlinearity="leaky_relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


def cifar_resnet(depth: int, in_ch: int = 3, img: int = 32,
                 channels: tuple[int, int, int] = (16, 32, 64)) -> list[Unit]:
    if depth < 8 or (depth - 2) % 6:
        raise ValueError(f"CIFAR ResNet depth must be 6n+2 and >= 8, got {depth}")
    if img < 4 or img % 4:
        raise ValueError(f"image side must be divisible by 4, got {img}")

    per_stage = (depth - 2) // 6
    if any(value < 1 for value in channels):
        raise ValueError(f"stage channels must be positive, got {channels}")

    units = [Unit("stem", ConvBNAct(in_ch, channels[0]), head_pool(channels[0], img),
                  (channels[0], img, img))]
    in_channels, side = channels[0], img

    for stage, out_channels in enumerate(channels, start=1):
        for block in range(1, per_stage + 1):
            stride = 2 if stage > 1 and block == 1 else 1
            if stride == 2:
                side //= 2
            units.append(Unit(
                f"s{stage}b{block}",
                BasicBlock(in_channels, out_channels, stride),
                head_pool(out_channels, side),
                (out_channels, side, side),
            ))
            in_channels = out_channels

    _initialise(units)
    log.debug("resnet%d: %d units, %d blocks/stage", depth, len(units), per_stage)
    describe(units)
    return units


def resnet20(in_ch: int = 3, img: int = 32) -> list[Unit]:
    return cifar_resnet(20, in_ch, img)


def resnet32(in_ch: int = 3, img: int = 32) -> list[Unit]:
    return cifar_resnet(32, in_ch, img)


def resnet56(in_ch: int = 3, img: int = 32) -> list[Unit]:
    return cifar_resnet(56, in_ch, img)


ARCHS: dict[str, Callable[..., list[Unit]]] = {
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet56": resnet56,
}
