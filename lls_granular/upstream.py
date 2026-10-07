"""Single entry point to the unmodified LLS-DNN code in third_party/."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "third_party" / "LLS-DNN"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import LLS_MobileNetV1, LLS_VGG8  # noqa: E402
from models.layers import ConvBlock, ConvDWBlock, LinearBlock  # noqa: E402
from models.layers.lls_layers import compute_LLS  # noqa: E402
from utils import AdamWScheduleFree, SGDScheduleFree  # noqa: E402

__all__ = ["ROOT", "ConvBlock", "ConvDWBlock", "LinearBlock", "LLS_VGG8", "LLS_MobileNetV1", "compute_LLS",
           "AdamWScheduleFree", "SGDScheduleFree"]
