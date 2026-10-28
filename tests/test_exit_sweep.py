import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

import lls_exit.policies as P
import lls_exit.sweep as S

K, C = 4, 5


def split(n: int, g: torch.Generator) -> dict[str, torch.Tensor]:
    """Scores that get more class-aligned with depth, so exits genuinely differ."""
    y = torch.randint(0, C, (n,), generator=g)
    s = torch.randn(n, K, C, generator=g) * 3
    s[torch.arange(n), :, y] += torch.linspace(0, 4, K)
    return {"scores": s, "logits": s[:, -1] * 2, "y": y, "idx": torch.arange(n)}


def fake_dump(run: str = "fake", seed: int = 0) -> dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    return {"meta": {"run": run, "n_cls": C, "temp": 0.5, "epochs_done": 300, "names": []},
            "costs": {"backbone": torch.tensor([10.0, 30.0, 60.0, 100.0, 100.0]),
                      "head": torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0]), "full": 102.0},
            "heads": {"dim": [8] * K, "row_norm": torch.full((K, C), 3.0), "coherence": [0.1] * K},
            "splits": {"val": split(300, g), "test": split(400, g)}}


@pytest.fixture(scope="module")
def swept() -> pd.DataFrame:
    return S.sweep_run(fake_dump(), n=11)[0]


def test_exit_sets() -> None:
    assert S.exit_sets(3, "head") == {"skip0": [0, 1, 2], "skip1": [1, 2], "skip2": [2]}
    assert S.exit_sets(3, "cls") == {"skip0": [0, 1, 2, 3], "skip1": [1, 2, 3], "skip2": [2, 3]}


def test_family_exits_match_reference_fire() -> None:
    torch.manual_seed(0)
    p = torch.softmax(torch.randn(50, K, C) * 2, -1)
    for f in S.families(p, n=7):
        got = f.exits(p)
        want = torch.stack([P.first_exit(P.fire(p, pol)) for pol in f.policies()])
        assert torch.equal(got, want), f.name


def test_costs_static_and_threshold() -> None:
    ex = fake_dump()
    c, sp = S.Costs.of(ex), ex["splits"]["test"]
    p = P.probs(sp, ex["heads"], "cls")
    r = S.evaluate(p, sp["y"], c, [0, 1, 2, 3, 4], P.Policy("static", {"e": 2}), C)
    assert r["mmac"] * 1e6 == pytest.approx(60 + 1) and r["dec_mmac"] == 0  # backbone[2] + its head only
    r = S.evaluate(p, sp["y"], c, [1, 2, 3, 4], P.Policy("confidence", {"tau": 2.0}), C)  # never fires
    ops = P.decision_ops(P.Policy("confidence"), C)
    assert r["mmac"] * 1e6 == pytest.approx(100 + (1 + 1 + 1 + 2) + 3 * ops)  # cls: 4 heads checked, 3 decisions
    assert r["speedup"] == pytest.approx(102 / (105 + 3 * ops)) and sum(json.loads(r["hist"])) == 400
    assert json.loads(r["hist"])[4] == 400


def test_sweep_rows_match_reference(swept: pd.DataFrame) -> None:
    ex = fake_dump()
    c, heads, sp = S.Costs.of(ex), ex["heads"], ex["splits"]["test"]
    temps = P.fit_temps(ex["splits"]["val"], heads, "cls")
    p = P.probs(sp, heads, "cls", temps)
    rows = swept[(swept.split == "test") & (swept.temp == "fit") & (swept.final == "cls") & (swept.S == "skip1")]
    for _, r in rows.sample(30, random_state=0).iterrows():
        pol = P.Policy(r.policy, json.loads(r.params))
        ref = S.evaluate(p, sp["y"], c, [1, 2, 3, 4], pol, C, r["agg"])
        assert r.acc == pytest.approx(ref["acc"]) and r.mmac == pytest.approx(ref["mmac"]), r.label


def test_sweep_coverage(swept: pd.DataFrame) -> None:
    assert set(swept.temp) == {"train", "fit"} and set(swept.final) == {"head", "cls"}
    assert {"confidence", "entropy", "margin", "patience", "sync", "sync_margin", "oracle", "static"} <= set(swept.policy)
    assert set(swept["agg"]) == set(S.AGGS) and set(swept[swept.policy == "static"]["agg"]) == {"single"}
    for _, g in swept.groupby(["temp", "final", "agg", "S", "split"]):
        assert g[g.policy == "oracle"].acc.iloc[0] >= g.acc.max() - 1e-12
    st = swept[(swept.policy == "static") & (swept.final == "cls") & (swept.split == "test")]
    assert st.acc.iloc[-1] == pytest.approx(st.ref_acc.iloc[0])  # static at cls = the full network


def test_pareto() -> None:
    df = pd.DataFrame({"mac_frac": [0.1, 0.2, 0.3, 0.3, 0.5], "acc": [0.5, 0.4, 0.7, 0.6, 0.7]})
    assert S.pareto(df).to_dict("list") == {"mac_frac": [0.1, 0.3], "acc": [0.5, 0.7]}


def test_picks_use_val_and_report_same_setting_on_test(swept: pd.DataFrame) -> None:
    pk = S.picks(swept)
    val = swept[(swept.split == "val") & (swept.policy != "oracle")]
    for _, r in pk[pk.policy == "any"].iterrows():
        g = val[(val.temp == r.temp) & (val.final == r.final)]
        g = g if r.pool == "all" else g[g["agg"] == r.pool]
        ok = g[g.acc >= r.val_ref - r.delta]
        assert r.val_mac_frac == pytest.approx(ok.mac_frac.min())
        t = swept[(swept.split == "test") & (swept.temp == r.temp) & (swept.final == r.final)
                  & (swept["agg"] == r["agg"]) & (swept.S == r.S) & (swept.label == r.label)]
        assert len(t) == 1 and t.acc.iloc[0] == pytest.approx(r.test_acc)
    assert set(pk.delta) <= set(S.DELTAS) and set(pk.pool) == {"all", *S.AGGS}


def test_main_with_local_dumps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for run in ("r1", "r2"):
        (tmp_path / run).mkdir()
        torch.save(fake_dump(run), tmp_path / run / "exits.pt")
    monkeypatch.setattr(S, "targets", lambda exps, seeds: [tmp_path / "r1", tmp_path / "r2", tmp_path / "missing"])
    out = tmp_path / "report"
    S.main(["--exp", "mobilenet300", "--dumps", str(tmp_path), "--out", str(out), "--n", "5", "--log", "WARNING"])
    for f in ("sweep.csv.gz", "pareto.csv", "picks.csv", "temps.json"):
        assert (out / f).exists()
    assert set(pd.read_csv(out / "picks.csv").run) == {"r1", "r2"}
    assert set(json.loads((out / "temps.json").read_text())["r1"]) == {"head", "cls"}


def test_ensemble_agg_sees_checked_exits_only() -> None:
    torch.manual_seed(0)
    p = torch.softmax(torch.randn(6, 5, C), -1)
    q = S.aggregate(p[:, [2, 3, 4]], "ens")
    assert torch.allclose(q[:, 0], p[:, 2]) and torch.allclose(q[:, -1], p[:, 2:].mean(1))
    assert torch.equal(S.aggregate(p, "single"), p)
    ex = fake_dump()
    c, sp = S.Costs.of(ex), ex["splits"]["test"]
    pr = P.probs(sp, ex["heads"], "cls")
    pol = P.Policy("margin", {"tau": 2.0})  # never fires: every head is checked
    a, b = (S.evaluate(pr, sp["y"], c, [1, 2, 3, 4], pol, C, g) for g in ("single", "ens"))
    assert b["mmac"] * 1e6 - a["mmac"] * 1e6 == pytest.approx(3 * S.agg_ops("ens", C))


def test_main_with_run_names(tmp_path: Path) -> None:
    (tmp_path / "x1").mkdir()
    torch.save(fake_dump("x1"), tmp_path / "x1" / "exits.pt")
    out = tmp_path / "report"
    S.main(["--runs", "x1", "--dumps", str(tmp_path), "--out", str(out), "--n", "5", "--log", "WARNING"])
    assert set(pd.read_csv(out / "picks.csv").run) == {"x1"}
