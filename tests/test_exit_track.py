import json
from pathlib import Path
from typing import Any

import pytest
import torch

import lls_exit.track as T


def fake_exits() -> dict[str, Any]:
    """2 heads + cls over 4 samples: head 0 right on 0-1, head 1 right on 0-2, cls right on 3 only."""
    y = torch.tensor([0, 1, 2, 3])
    sc = torch.zeros(4, 2, 4)
    sc[torch.arange(4), 0, torch.tensor([0, 1, 0, 0])] = 1
    sc[torch.arange(4), 1, torch.tensor([0, 1, 2, 0])] = 1
    lg = torch.zeros(4, 4)
    lg[torch.arange(4), torch.tensor([1, 0, 0, 3])] = 1
    sp = {"scores": sc, "logits": lg, "y": y, "idx": torch.arange(4)}
    return {"meta": {"epochs_done": 300, "names": ["a", "b", "cls"]},
            "costs": {"backbone": torch.tensor([10.0, 60.0, 60.0]), "head": torch.tensor([1.0, 1.0, 2.0]),
                      "full": 62.0},
            "heads": {"dim": [8, 4], "coherence": [0.1, 0.2]},
            "splits": {"test": sp, "val": sp}}


def test_exit_summary() -> None:
    s = T.exit_summary(fake_exits())
    assert s["exit/n_heads"] == 2 and s["exit/epochs"] == 300
    assert s["exit/0/test_acc"] == 0.5 and s["exit/1/test_acc"] == 0.75
    assert s["exit/test_cls_acc"] == 0.25 and s["exit/test_oracle_acc"] == 1.0
    assert s["exit/0/mac_frac"] == pytest.approx(10 / 62) and s["exit/heads_mac_frac"] == pytest.approx(2 / 62)
    assert "exit/1/val_acc" in s and s["exit/1/dim"] == 4
    json.dumps(s)  # local backup must be JSON-serialisable


class FakeRun:
    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}
        self.arts: list[Any] = []
        self.done = False

    def log_artifact(self, a: Any) -> None:
        self.arts.append(a)

    def finish(self) -> None:
        self.done = True


class FakeArt:
    def __init__(self, name: str, type: str, metadata: dict[str, Any]) -> None:
        self.name, self.type, self.metadata, self.files = name, type, metadata, []

    def add_file(self, p: str) -> None:
        self.files.append(p)


def test_publish_reopens_training_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wandb = pytest.importorskip("wandb")
    (tmp_path / "wandb_id").write_text("abc123\n")
    (tmp_path / "exits.pt").write_bytes(b"x")
    run, calls = FakeRun(), []
    monkeypatch.setattr(wandb, "init", lambda **k: calls.append(k) or run)
    monkeypatch.setattr(wandb, "Artifact", FakeArt)
    s = T.exit_summary(fake_exits())
    assert T.publish(tmp_path, tmp_path / "exits.pt", s, "proj", None)
    assert calls[0]["id"] == "abc123" and calls[0]["resume"] == "must" and calls[0]["project"] == "proj"
    assert run.summary["exit/1/test_acc"] == 0.75 and run.done
    a = run.arts[0]
    assert a.name == f"exits-{tmp_path.name}" and a.type == "exits" and a.files == [str(tmp_path / "exits.pt")]
    assert "exit/test_cls_acc" in a.metadata and "exit/0/test_acc" not in a.metadata


def test_publish_without_id_or_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wandb = pytest.importorskip("wandb")
    assert not T.publish(tmp_path, tmp_path / "exits.pt", {}, "proj", None)  # no wandb_id
    (tmp_path / "wandb_id").write_text("abc")

    def boom(**_: Any) -> None:
        raise RuntimeError("offline")

    monkeypatch.setattr(wandb, "init", boom)
    assert not T.publish(tmp_path, tmp_path / "exits.pt", {}, "proj", None)
