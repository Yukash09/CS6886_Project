import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import lls_exit.dump as D
from lls_granular import ckpt, data, run
from lls_granular.train import Best
from lls_granular.wrapper import GranularLLS

from .helpers import batches

CPU = torch.device("cpu")


def loader(n: int, seed: int) -> DataLoader:
    xs, ys = zip(*batches(n, seed))
    return DataLoader(TensorDataset(torch.cat(xs), torch.cat(ys)), batch_size=8)


@pytest.fixture(scope="module")
def model() -> tuple[GranularLLS, object]:
    a = run.args(["--partition", "2,2,3"])
    m, _ = run.model(a, 10)
    for x, y in batches(2):  # a couple of steps so eval and train weights differ
        m.step(x, y)
    return m, a  # type: ignore[return-value]


def test_rebuild_uses_blocks_and_ignores_extras() -> None:
    a = D.rebuild({"arch": "mobilenet", "partition": "aligned", "blocks": [2, 2, 2, 6, 2], "temp": 0.3, "gpu": "x"})
    assert (a.arch, a.partition, a.temp) == ("mobilenet", "2,2,2,6,2", 0.3) and not hasattr(a, "gpu")


def test_balanced() -> None:
    y = [i % 10 for i in range(1000)]
    idx = D.balanced(y, list(range(500)), 100, 10)
    assert len(idx) == 100 and all(i < 500 for i in idx)
    assert torch.bincount(torch.tensor(y)[idx]).tolist() == [10] * 10
    assert idx == D.balanced(y, list(range(500)), 100, 10)


def test_dump_matches_forward_all(model: tuple[GranularLLS, object]) -> None:
    m, a = model
    lds = {"test": loader(2, 5), "val": loader(1, 6)}
    ex, fe = D.dump(m, a, lds, batches(2), CPU, True, {"run": "t"})  # type: ignore[arg-type]
    t = ex["splits"]["test"]
    assert t["scores"].shape == (16, 3, 10) and t["logits"].shape == (16, 10)
    assert torch.equal(t["idx"], torch.arange(16))
    assert ex["meta"]["names"] == ["conv1+conv2", "conv3+conv4", "conv5+conv6+fc1", "cls"]
    assert ex["costs"]["backbone"].shape == (4,) and ex["heads"]["row_norm"].shape == (3, 10)

    m.opt_mode(False)  # same eval weights + refreshed BN that the dump used
    m.eval()
    with torch.no_grad():
        logits, scores = m.forward_all(torch.cat([x for x, _ in lds["test"]]))
    m.opt_mode(True)
    assert torch.allclose(t["logits"], logits, atol=1e-5)
    assert torch.allclose(t["scores"], torch.stack(scores, 1), atol=1e-5)

    f = fe["splits"]["test"]["feats"]
    assert [v.shape[1] for v in f] == ex["heads"]["dim"] and f[0].dtype == torch.float16
    assert torch.allclose(f[2].float().norm(dim=1), torch.ones(16), atol=1e-2)


def test_main_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = run.args(["--partition", "2,2,3"])
    m, _ = run.model(a, 10)
    ckpt.save(tmp_path / "ckpt.pt", m, 4, Best(0.5, 3, 0.4))
    (tmp_path / "config.json").write_text(json.dumps({**vars(a), "blocks": [2, 2, 3], "git": "abc"}))
    fake = data.Data(train=loader(2, 1), val=loader(1, 2), test=loader(1, 3), n_cls=10)
    monkeypatch.setattr(D.data, "load", lambda *_, **__: fake)

    D.main(["--run-dir", str(tmp_path), "--n-train", "0", "--no-feats", "--log", "WARNING"])
    ex = torch.load(tmp_path / "exits.pt", weights_only=False)
    assert set(ex["splits"]) == {"test", "val"} and not (tmp_path / "feats.pt").exists()
    assert ex["meta"]["epochs_done"] == 5 and ex["meta"]["git_train"] == "abc"


def test_targets_skip_bp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLS_OUT", str(tmp_path))
    rds = D.targets(["mobilenet300"], None)
    assert len(rds) == 8 and all(r.parent == tmp_path and "_bp_" not in r.name for r in rds)
    assert tmp_path / "mobilenet_cifar10aug_p1.1.1.1.1.1.1.1.1.1.1.1.1.1_e300_s0" in rds


def test_exp_skips_done_and_flags_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("LLS_OUT", str(tmp_path))
    done = D.targets(["main"], None)[0]
    done.mkdir()
    (done / "exits.pt").touch()
    with pytest.raises(SystemExit):
        D.main(["--exp", "main", "--log", "WARNING"])
    out = capsys.readouterr().out
    assert out.count("no ckpt") == 5 and out.count("exists") == 1
