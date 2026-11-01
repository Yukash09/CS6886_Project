from collections.abc import Callable

from lls_granular.units import Unit

from .resnet import cifar_resnet


def wide_resnet20(in_ch: int = 3, img: int = 32) -> list[Unit]:
    return cifar_resnet(20, in_ch, img, channels=(32, 64, 128))


ARCHS: dict[str, Callable[..., list[Unit]]] = {"wide_resnet20": wide_resnet20}
