from pathlib import Path

import pandas as pd
import pytest
import torch

import lls_exit.diag as D
import lls_exit.policies as P
from tests.test_exit_policies import onehots
from tests.test_exit_sweep import C, K, fake_dump


def test_accuracy_agreement_settle_flips() -> None:
    p = onehots([[0, 1, 1], [2, 0, 1], [1, 1, 1]])
    y = torch.tensor([1, 1, 0])
    assert D.accuracy(p, y).tolist() == pytest.approx([0.0, 1 / 3, 2 / 3])
    assert D.agreement(p).tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert D.settle(p).tolist() == [1, 2, 0]
    assert D.flips(p).tolist() == [1, 2, 0]


def test_overthinking() -> None:
    p = onehots([[1, 1, 0], [0, 1, 1], [2, 2, 2]])  # right then wrong / right late / never right
    o = D.overthinking(p, torch.tensor([1, 1, 0]))
    assert o["final_acc"] == pytest.approx(1 / 3) and o["oracle_acc"] == pytest.approx(2 / 3)
    assert o["overthink"] == pytest.approx(1 / 3) and o["early_right"] == pytest.approx(1 / 3)
    assert o["settle"] == pytest.approx((2 + 1 + 0) / 3)


def test_ece() -> None:
    p = torch.tensor([[[0.9, 0.1]], [[0.9, 0.1]], [[0.6, 0.4]], [[0.6, 0.4]]])
    y = torch.tensor([0, 1, 0, 0])  # bin 0.9: acc 0.5 (gap 0.4); bin 0.6: acc 1 (gap 0.4)
    assert D.ece(p, y, bins=10).item() == pytest.approx(0.4)
    r = D.reliability(p, y, bins=10)
    assert r.n.tolist() == [2, 2] and r.acc.tolist() == [1.0, 0.5] and r.conf.tolist() == pytest.approx([0.6, 0.9])


def test_ensemble() -> None:
    torch.manual_seed(0)
    p = torch.softmax(torch.randn(7, 5, 4), -1)
    ref = torch.stack([p[:, :e + 1].mean(1) for e in range(5)], 1)
    assert torch.allclose(P.ensemble(p), ref)
    ref2 = torch.stack([p[:, max(0, e - 1):e + 1].mean(1) for e in range(5)], 1)
    assert torch.allclose(P.ensemble(p, 2), ref2, atol=1e-6)
    assert torch.allclose(P.ensemble(p, 1), p, atol=1e-6) and torch.allclose(P.ensemble(p, 9), ref)
    g = P.ensemble(p, geo=True)
    assert torch.allclose(g.sum(-1), torch.ones(7, 5)) and torch.allclose(g[:, 0], p[:, 0], atol=1e-6)


def test_per_class() -> None:
    p = onehots([[1, 0, 0], [0, 0, 0], [2, 2, 1]])
    df = D.per_class(p, torch.tensor([0, 0, 1]), 3)
    assert df.n.tolist() == [2, 1, 0] and df.acc.tolist() == [1.0, 1.0, 0.0]
    assert df.settle.tolist() == [0.5, 2.0, 0.0] and df.oracle_exit.tolist() == [0.5, 2.0, 0.0]


def test_class_geometry() -> None:
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    z = torch.eye(3)[y][:, None] * 2 + 1  # exit 0: one-hot class outputs (+ offset) = a simplex
    z = torch.cat([z, torch.tensor([[1.0, 0, 0]] * 4 + [[0, 1.0, 0]] * 2)[:, None]], 1)  # exit 1: classes 0, 1 equal
    mx, mean, pair = D.class_geometry(z, y, 3)
    assert mx[0].item() == pytest.approx(-0.5) and mean[0].item() == pytest.approx(-0.5)
    assert mx[1].item() == pytest.approx(1.0) and pair[1].tolist() == [0, 1]


def test_basis_stats() -> None:
    ex = fake_dump()
    mx, mean = D.basis_stats(ex)
    assert len(mx) == K and all(0 <= m <= x <= 1 for m, x in zip(mean, mx))
    ex["heads"]["coherence"] = [float("nan")] * K  # learned exits: no fixed basis
    assert all(v != v for v in D.basis_stats(ex)[0])


def test_diagnose_shapes() -> None:
    d = D.diagnose(fake_dump())
    pe = d["per_exit"]
    assert len(pe) == 2 * (K + K + 1)  # temps × (head final: K exits, cls final: K+1)
    cls = pe[(pe.final == "cls") & (pe.temp == "train")]
    assert cls["name"].iloc[-1] == "cls" and cls.mac_frac.iloc[-1] == pytest.approx(1.0)
    assert cls.ens_w2.iloc[0] == cls.acc.iloc[0]  # one exit: the ensemble is that exit
    fit = pe[pe.temp == "fit"]
    assert (fit.acc.values == pe[pe.temp == "train"].acc.values).all()  # temperature never changes argmax
    assert len(d["overthink"]) == 2 and len(d["per_class"]) == 2 * C
    assert set(d["reliability"].temp) == {"train", "fit"}
    assert cls.basis_coh.iloc[:-1].notna().all() and pd.isna(cls.basis_coh.iloc[-1])  # cls has no fixed basis
    assert cls.cls_cos_max.between(-1, 1).all() and cls.cls_pair.str.contains("-").all()


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "r1").mkdir()
    torch.save(fake_dump("r1"), tmp_path / "r1" / "exits.pt")
    monkeypatch.setattr(D, "targets", lambda exps, seeds: [tmp_path / "r1", tmp_path / "missing"])
    out = tmp_path / "report"
    D.main(["--exp", "mobilenet300", "--dumps", str(tmp_path), "--out", str(out), "--log", "WARNING"])
    for k in ("per_exit", "overthink", "per_class", "reliability"):
        assert set(pd.read_csv(out / f"{k}.csv").run) == {"r1"}
