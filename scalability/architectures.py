from .resnet import ARCHS as RESNETS
from .wide_resnet import ARCHS as WIDE_RESNETS

ARCHS = {**RESNETS, **WIDE_RESNETS}
