import json
from pathlib import Path

import pytest
import torch

import lls_exit.train_exits as X
from lls_exit.exitnet import ExitNet, parts
from lls_exit.sweep import sweep_run
from lls_granular import ckpt, data, run
from lls_granular.train import Best, bn_refresh
from lls_granular.units import vgg8

from .helpers import batches
from .test_exit_dump import loader

CPU = torch.device("cpu")


def state(mod: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in mod.state_dict().items()}


def test_joint_step_trains_everything() -> None:
    torch.manual_seed(0)
    m = ExitNet(vgg8(), (2, 2, 3), 10)
    assert [m.units[e].name for e in m.ends] == ["conv2", "conv4", "fc1"]
    before = state(m)
    x, y = batches(1)[0]
    out = m.step(x, y)
    assert out.losses.shape == (4,) and out.hits.shape == (3,) and out.logits.shape == (8, 10)
    after = m.state_dict()
    assert not torch.equal(before["units.0.layer.conv.weight"], after["units.0.layer.conv.weight"])
    assert not torch.equal(before["heads.0.fc.weight"], after["heads.0.fc.weight"])
    _, scores = m.forward_all(x)
    assert [s.shape for s in scores] == [(8, 10)] * 3


def test_frozen_step_only_trains_heads() -> None:
    torch.manual_seed(0)
    m = ExitNet(vgg8(), (1,) * 7, 10, frozen=True)
    assert all(not u.training for u in m.units) and m.heads.training
    before = state(m)
    for x, y in batches(2):
        m.step(x, y)
    bn_refresh(m, batches(2), CPU)  # type: ignore[arg-type]
    after = m.state_dict()
    for k, v in before.items():
        if k.startswith(("units.", "cls.")):  # weights and BN running stats
            assert torch.equal(v, after[k]), k
    assert not torch.equal(before["heads.6.fc.weight"], after["heads.6.fc.weight"])


@pytest.mark.parametrize("spec", ["bp", "2,2,3"])
def test_frozen_on_source_reproduces_it(spec: str) -> None:
    m, part = run.model(run.args(["--partition", spec]), 10)
    for x, y in batches(2):
        m.step(x, y)
    m.eval()
    units, cls, p = parts(m)
    assert p == part and len(units) == 7
    e = ExitNet(units, p or (1,) * 7, 10, frozen=True, cls=cls)
    x = batches(1, seed=9)[0][0]
    with torch.no_grad():
        assert torch.allclose(e(x), m(x), atol=1e-6)


def test_run_names() -> None:
    j = X.ExitRunId("vgg8", "cifar10aug", (1,) * 7, 100, 0, variant="joint")
    assert j.name == "vgg8_cifar10aug_xjoint-p1.1.1.1.1.1.1_e100_s0" and j.tag == "xjoint"
    f = X.ExitRunId("vgg8", "cifar10aug", (2, 2, 3), 30, 0, variant="frozen", source="vgg8_cifar10aug_p2.2.3_e300_s0",
                    src_kind="lls")
    assert f.name == "vgg8_cifar10aug_p2.2.3_e300_s0_xfrozen_e30" and f.tag == "xprobe"
    assert f.fields()["kind"] == "xprobe" and "xprobe" in f.labels()


def fake_data(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = data.Data(train=loader(2, 1), val=loader(1, 2), test=loader(1, 3), n_cls=10)
    monkeypatch.setattr(X.data, "load", lambda *_, **__: fake)


def check_dump(rd: Path, n_exits: int) -> None:
    ex = torch.load(rd / "exits.pt", weights_only=False)
    assert ex["splits"]["test"]["scores"].shape == (8, n_exits, 10) and ex["meta"]["temp"] == 1.0
    assert ex["costs"]["backbone"].shape == (n_exits + 1,) and (rd / "exit_summary.json").exists()
    df, _ = sweep_run(ex, n=3)  # the policy sweep runs on it unchanged
    assert {"margin", "oracle"} <= set(df.policy)


def test_joint_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_data(monkeypatch)
    argv = ["--variant", "joint", "--partition", "2,2,3", "--epochs", "1", "--out", str(tmp_path), "--no-wandb",
            "--log", "WARNING"]
    X.main(argv)
    rd = tmp_path / "vgg8_cifar10aug_xjoint-p2.2.3_e1_s0"
    check_dump(rd, 3)
    assert json.loads((rd / "config.json").read_text())["kind"] == "xjoint"
    assert X.run_one(X.args(argv)) == "exists"  # done: skipped, nothing rewritten


def test_frozen_end_to_end_and_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_data(monkeypatch)
    src = tmp_path / "vgg8_cifar10aug_bp_e300_s0"
    src.mkdir()
    a = run.args(["--partition", "bp", "--epochs", "300"])
    m, _ = run.model(a, 10)
    ckpt.save(src / "ckpt.pt", m, 299, Best(0.5, 3, 0.4))
    (src / "config.json").write_text(json.dumps(vars(a)))
    X.main(["--variant", "frozen", "--source", str(src), "--epochs", "1", "--out", str(tmp_path), "--no-wandb",
            "--log", "WARNING"])
    check_dump(tmp_path / "vgg8_cifar10aug_bp_e300_s0_xfrozen_e1", 7)

    monkeypatch.setenv("LLS_OUT", str(tmp_path))  # plan: the main BP source exists, the LLS ones don't
    with pytest.raises(SystemExit):
        X.main(["--plan", "vgg8_frozen", "--epochs", "1", "--no-wandb", "--log", "WARNING"])
    assert (tmp_path / "vgg8_cifar10aug_bp_e300_s0_xfrozen_e1" / "exits.pt").exists()


def test_frozen_needs_source() -> None:
    with pytest.raises(SystemExit):
        X.args(["--variant", "frozen"])
