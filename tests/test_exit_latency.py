import json
from pathlib import Path

import pytest
import torch

import lls_exit.dump as D
import lls_exit.latency as L
import lls_exit.policies as P
from lls_granular import ckpt, data, run
from lls_granular.train import Best

from .helpers import batches
from .test_exit_dump import loader

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def rd(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A trained-a-bit 2,2,3 VGG8 run folder with ckpt, config and an exits.pt dumped on fake data."""
    d = tmp_path_factory.mktemp("run")
    a = run.args(["--partition", "2,2,3", "--epochs", "5"])
    m, _ = run.model(a, 10)
    for x, y in batches(3):
        m.step(x, y)
    ckpt.save(d / "ckpt.pt", m, 4, Best(0.5, 3, 0.4))
    (d / "config.json").write_text(json.dumps({**vars(a), "blocks": [2, 2, 3]}))
    mp = pytest.MonkeyPatch()
    mp.setattr(data, "load", lambda *_, **__: fake())
    D.main(["--run-dir", str(d), "--n-train", "0", "--no-feats", "--no-wandb", "--log", "WARNING"])
    mp.undo()
    return d


def fake() -> data.Data:
    return data.Data(train=loader(2, 1), val=loader(4, 2), test=loader(4, 3), n_cls=10)


def test_timeit_positive() -> None:
    t = L.timeit(lambda x: x @ x, torch.randn(32, 32), CPU, warmup=1, reps=5, budget=0.1)
    assert t > 0


def median_tau(name: str, q: torch.Tensor) -> dict[str, float]:
    """Thresholds at the median of each signal (q: offline probs over the checked exits), so exits genuinely vary."""
    def med(v: torch.Tensor) -> float:
        return float(v.median())

    match name:
        case "confidence":
            return {"tau": med(P.confidence(q[:, 0]))}
        case "entropy":
            return {"tau": med(-P.neg_entropy(q[:, 0]))}
        case "margin":
            return {"tau": med(P.margin(q[:, 0]))}
        case "sync":
            return {"w": 2, "tau": med(P.sync(q, 2)[:, 1])}
        case "sync_margin":
            return {"w": 2, "m": med(P.margin(q[:, 1])), "tau": med(P.sync(q, 2)[:, 1])}
        case "patience":
            return {"n": 2}
    return {"e": 1}


@pytest.mark.parametrize("name", ["margin", "confidence", "entropy", "patience", "sync", "sync_margin", "static"])
def test_dynamic_loop_matches_offline_policy(rd: Path, name: str) -> None:
    ex = torch.load(rd / "exits.pt", weights_only=False)
    m = L.load_model(rd, fake(), CPU)
    k = len(ex["heads"]["dim"])
    ts = P.fit_temps(ex["splits"]["val"], ex["heads"], "cls")
    scale = [1 / (ex["heads"]["row_norm"][e] * ts[e]) if e < k else torch.tensor(1 / ts[e]) for e in range(k + 1)]
    sp = ex["splits"]["test"]
    q = torch.stack([(sp["scores"][:32, e] if e < k else sp["logits"][:32]) * scale[e] for e in range(k + 1)],
                    1).softmax(-1)
    pol = P.Policy(name, median_tau(name, q))
    if name == "static":  # exit set skip1, its second exit: only exit 2 runs a head
        plan = L.Plan([2], pol, [scale[2]], static=True)
    else:
        plan = L.Plan(list(range(k + 1)), pol, scale)
    xs, y = L.samples(fake().test, 32)
    dy = L.dynamic(m, xs, plan, CPU)
    off_exit, off_pred = L.offline(ex, plan, 32)
    assert torch.equal(dy["exit"], off_exit) and torch.equal(dy["pred"], off_pred), name
    assert name == "static" or dy["exit"].unique().numel() > 1, f"{name}: every sample took the same exit"
    assert len(dy["t_dyn"]) == 32 and min(dy["t_dyn"]) > 0
    assert (y == sp["y"][:32]).all()


def test_choose_plans_the_sweep_pick(rd: Path) -> None:
    ex = torch.load(rd / "exits.pt", weights_only=False)
    plan, pick = L.choose(ex, 0.01, "any", 5)
    assert plan.checked[-1] <= len(ex["heads"]["dim"]) and len(plan.scale) == len(plan.checked)
    assert 0 < pick["test_mac_frac"] <= 1 and pick["mac_speedup"] >= 1 - 1e-9


def test_cli_end_to_end(rd: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(data, "load", lambda *_, **__: fake())
    argv = ["--run-dir", str(rd), "--n", "16", "--batches", "1", "4", "--cpu-batches", "1", "4", "--devices", "cpu1",
            "--reps", "3", "--budget", "0.05", "--n-thresh", "5", "--no-wandb", "--log", "WARNING"]
    L.main(argv)
    res = json.loads((rd / "latency.json").read_text())
    c = res["configs"]["cpu1"]
    assert c["threads"] == 1 and len(c["static"]) == 4 and {"b1_ms", "b4_ms"} <= set(c["static"][0])
    assert c["dynamic"]["agree"] == 1.0 and c["dynamic"]["acc"] == c["dynamic"]["offline_acc"]
    assert sum(c["dynamic"]["exit_hist"]) == 16 and res["device"]["cpu_count"]
    assert "cpu1" in capsys.readouterr().out
    assert L.run_one(rd, L.args(argv)) == "exists"


def test_missing_files_exit_1(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        L.main(["--run-dir", str(tmp_path), "--no-wandb", "--log", "WARNING"])
