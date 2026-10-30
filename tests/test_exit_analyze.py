from pathlib import Path

import pytest
import torch

import lls_exit.analyze as A
import lls_exit.diag as D
import lls_exit.sweep as S

from .test_exit_sweep import fake_dump

RUNS = ["vgg8_cifar10aug_p2.2.2.1_e60_s0", "vgg8_cifar10aug_p1.2.2.2_e60_s0"]


def test_labels() -> None:
    assert A.method_of("mobilenet_cifar10aug_p4.2.6.2_e300_s0") == "p4.2.6.2"
    assert A.label("vgg8_cifar10aug_p1.1.1.1.1.1.1_e300_s0") == "LLS 1×7"
    assert A.label("vgg8_cifar10aug_p7_e300_s0") == "single block"
    assert A.label("vgg8_cifar10aug_p2.2.3_e300_s0") == "2,2,3"


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("rep")
    for i, run in enumerate(RUNS):
        (root / "dumps" / run).mkdir(parents=True)
        torch.save(fake_dump(run, seed=i), root / "dumps" / run / "exits.pt")
    for name in ("vgg8", "vgg8_sweep"):
        common = ["--runs", *RUNS, "--dumps", str(root / "dumps"), "--out", str(root / name), "--log", "WARNING"]
        S.main([*common, "--n", "5"])
        D.main(common)
    return root


def test_policy_fronts_are_val_chosen(report: Path) -> None:
    sw = A.load(report / "vgg8")["sweep"]
    f = A.policy_fronts(sw, RUNS[0])
    assert set(f.policy) <= set(A.POLICIES) and len(f) > 0
    t = A.sel(sw, run=RUNS[0], agg="single", split="test")
    assert f.merge(t, on=["acc", "mac_frac"]).shape[0] >= len(f)  # every point is a real test row


def test_main_writes_figures_and_tables(report: Path) -> None:
    A.main(["--report", str(report), "--log", "WARNING"])
    figs = {f.stem for f in (report / "figs").glob("*.png")}
    assert {"policies_vgg8", "partitions_vgg8", "overthinking", "ensemble", "sweep64", "orthogonality"} <= figs
    md = (report / "tables.md").read_text()
    assert "cheapest setting within" in md and "first block" in md and "2,2,2,1" in md and "basis coherence" in md
