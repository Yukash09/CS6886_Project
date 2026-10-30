from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import lls_exit.depth as D
import lls_exit.policies as P
import lls_exit.sweep as S

from .test_exit_sweep import C, K, fake_dump


def feats(y: torch.Generator | torch.Tensor, g: torch.Generator, d: int = 6) -> list[torch.Tensor]:
    """Per exit: class centre + noise that shrinks with depth (clusters get cleaner), L2-normalised, fp16."""
    centres = torch.eye(C, d) * 3
    return [F.normalize(centres[y] + torch.randn(len(y), d, generator=g) * s, dim=1).half()  # type: ignore[index]
            for s in torch.linspace(3, 0.1, K).tolist()]


def fake_feats(ex: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    ytr = torch.arange(200) % C
    sp = {"train": {"feats": feats(ytr, g), "y": ytr, "idx": torch.arange(200)}}
    for name in ("val", "test"):
        d = ex["splits"][name]
        sp[name] = {"feats": feats(d["y"], g), "y": d["y"], "idx": d["idx"]}
    return {"meta": ex["meta"], "splits": sp}


def test_knn_majority_vote() -> None:
    ref = F.normalize(torch.tensor([[1.0, 0], [0.9, 0.1], [0, 1.0], [0.1, 0.9], [0.2, 0.8]]), dim=1)
    ry = torch.tensor([0, 0, 1, 1, 1])
    q = F.normalize(torch.tensor([[1.0, 0.05], [0.05, 1.0]]), dim=1)
    assert D.knn(ref, ry, q, 2, k=2, chunk=1).tolist() == [0, 1]
    assert D.knn(ref, ry, q, 2, k=5).tolist() == [1, 1]  # k = all: the majority class wins


def test_pred_depth_and_parse() -> None:
    knn = torch.tensor([[2, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1]])
    net = torch.tensor([1, 1, 1, 0])
    assert D.pred_depth(knn, net).tolist() == [1, 0, 3, 3]  # 3 = only the network agrees
    for pol in (P.Policy("sync_margin", {"w": 2, "m": 0.68, "tau": 0.85}), P.Policy("patience", {"n": 3})):
        assert D.parse(pol.label()) == pol


def test_knn_then_analyze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rd = tmp_path / "r1"
    rd.mkdir()
    ex = fake_dump("r1")
    torch.save(ex, rd / "exits.pt")
    torch.save(fake_feats(ex), rd / "feats.pt")
    D.main(["knn", "--run-dir", str(rd), "--no-wandb", "--log", "WARNING"])
    dp = torch.load(rd / "depth.pt", weights_only=False)
    acc = (dp["splits"]["test"]["knn"] == dp["splits"]["test"]["y"][:, None]).double().mean(0)
    assert dp["splits"]["test"]["knn"].shape == (400, K) and acc[-1] > 0.95 and acc[-1] > acc[0]
    assert D.knn_one(rd, D.args(["knn", "--run-dir", str(rd)]), torch.device("cpu")) == "exists"

    out = tmp_path / "rep"
    S.main(["--runs", "r1", "--dumps", str(tmp_path), "--out", str(out), "--n", "7", "--log", "WARNING"])
    D.main(["analyze", "--runs", "r1", "--dumps", str(tmp_path), "--out", str(out), "--log", "WARNING"])
    df = pd.read_csv(out / "depth_samples.csv.gz")
    assert len(df) == 400 and df.pd.between(0, K).all() and {"exit_any", "exit_margin", "exit_sync"} <= set(df.columns)
    pk = pd.read_csv(out / "picks.csv")
    r = pk[(pk.policy == "margin") & (pk.pool == "single") & (pk.temp == "fit") & (pk.final == "cls")
           & (pk.delta == 0.01)].iloc[0]
    assert df.exit_margin.mean() == pytest.approx(r.test_depth)  # same exits as the sweep's own numbers
    corr = pd.read_csv(out / "depth_corr.csv")
    assert set(corr.vs) >= {"settle", "oracle"} and corr.spearman.between(-1, 1).all()
    bins = pd.read_csv(out / "depth_bins.csv")
    assert bins.n.sum() == 400 and bins.frac.sum() == pytest.approx(1.0)
    assert len(pd.read_csv(out / "depth_class.csv")) == C
