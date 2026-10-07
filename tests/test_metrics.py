import pytest
import torch
from torch import nn

from lls_granular.measure import partitions
from lls_granular.metrics import Costs, saved_bytes
from lls_granular.units import mobilenet_v1, vgg8

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def costs() -> Costs:
    torch.manual_seed(0)
    return Costs(vgg8(), 10, 2, CPU)


def test_saved_bytes_counts_activations_not_weights() -> None:
    lin = nn.Linear(100, 50)
    x = torch.randn(4, 100)
    b = saved_bytes(lambda: torch.relu(lin(x)).sum(), lin.parameters())
    assert b == (4 * 100 + 4 * 50) * 4  # input (for weight grad) + relu output; weights excluded


def test_vgg8_forward_macs(costs: Costs) -> None:
    assert costs.macs[0] == 128 * 32 * 32 * 3 * 9
    assert costs.macs[1] == 256 * 32 * 32 * 128 * 9
    assert costs.macs[6] == 1024 * 2048
    assert 715 < sum(costs.macs) / 1e6 < 725  # paper: 719 MMAC for VGG8 BP forward


def test_granularity_costs_order(costs: Costs) -> None:
    lls, bp, single = costs.cost((1,) * 7), costs.cost(None), costs.cost((7,))
    late, early = costs.cost((1, 1, 1, 1, 3)), costs.cost((3, 1, 1, 1, 1))
    assert lls["act_peak_mb"] <= late["act_peak_mb"] < early["act_peak_mb"] <= single["act_peak_mb"]
    assert single["act_peak_mb"] <= bp["act_peak_mb"] * 1.01
    assert lls["train_mmac"] < late["train_mmac"] < bp["train_mmac"]
    assert late["act_peak_mb"] == pytest.approx(lls["act_peak_mb"])  # H2: the conv2 block sets the peak either way
    assert bp["signal_kmac"] == 0 and lls["signal_kmac"] > 0


def test_span_is_memoised(costs: Costs) -> None:
    costs.cost((2, 2, 3))
    n = len(costs._span)
    costs.cost((2, 2, 3))
    assert len(costs._span) == n


def test_partition_lists() -> None:
    assert len(partitions("vgg8")) == 65
    mn = partitions("mobilenet")
    assert None in mn and (1,) * 14 in mn and (2, 2, 2, 6, 2) in mn and (14,) in mn


def test_mobilenet_costs_are_flat() -> None:
    torch.manual_seed(0)
    c = Costs(mobilenet_v1(), 10, 2, CPU)
    lls, late = c.cost((1,) * 14), c.cost((1,) * 12 + (2,))
    assert 40 < sum(c.macs) / 1e6 < 50  # 44.4 MMAC (padding quirk enlarges maps) vs VGG8's 723
    assert late["act_peak_mb"] >= lls["act_peak_mb"]
