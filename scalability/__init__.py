from .architectures import ARCHS
from .resnet import BasicBlock, resnet20, resnet32, resnet56
from .wide_resnet import wide_resnet20

__all__ = ["ARCHS", "BasicBlock", "resnet20", "resnet32", "resnet56", "wide_resnet20"]
