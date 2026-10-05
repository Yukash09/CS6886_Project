import threading
from pathlib import Path

import pytest

from lls_granular.exps import EXPS, Job, jobs
from lls_granular.launch import restore, run_dir, schedule
from lls_granular.run import args, model


@pytest.mark.parametrize("name", sorted(EXPS))
def test_every_job_parses(name: str) -> None:
    for j in EXPS[name]:
        a = args(j.argv())
        if a.arch == "vgg8" and a.dataset != "cifar100aug":
            model(a, 10)


def test_sweep_covers_all_64_partitions() -> None:
    names = {j.partition for j in EXPS["sweep"]}
    assert len(names) == 64 and "1,1,1,1,1,1,1" in names and "7" in names


def test_seeds_expand_seed_major() -> None:
    js = jobs("timing", [0, 1])
    assert [(j.seed, j.partition) for j in js] == [(0, "uniform:1"), (0, "bp"), (1, "uniform:1"), (1, "bp")]


def test_run_dir_matches_run_naming(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLS_OUT", str(tmp_path))
    assert run_dir(Job("G-P", "2,2,3", 300)).name == "vgg8_cifar10aug_p2.2.3_e300_s0"
    assert run_dir(Job("G-BP", "bp", 300, seed=1)).name == "vgg8_cifar10aug_bp_e300_s1"


def test_schedule_runs_each_job_once_one_per_device() -> None:
    js = [Job("x", f"uniform:{k}", 1) for k in (1, 2, 3, 7)]
    busy: set[str] = set()
    seen: list[tuple[str, str]] = []
    lock = threading.Lock()

    def runner(j: Job, dev: str) -> int:
        with lock:
            assert dev not in busy, "two jobs on one device"
            busy.add(dev)
            seen.append((j.name, dev))
        with lock:
            busy.discard(dev)
        return 0 if j.partition != "uniform:7" else 3

    codes = schedule(js, ["0", "1"], runner)
    assert sorted(codes) == sorted(j.name for j in js)
    assert codes[js[-1].name] == 3 and len(seen) == 4


def test_restore_copies_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prev = tmp_path / "input" / "nb" / "runs" / "r1"
    prev.mkdir(parents=True)
    (prev / "ckpt.pt").write_text("x")
    out = tmp_path / "working" / "runs"
    monkeypatch.setenv("LLS_OUT", str(out))
    restore(str(tmp_path / "input" / "*" / "runs"))
    assert (out / "r1" / "ckpt.pt").exists()
    (out / "r1" / "ckpt.pt").write_text("newer")
    restore(str(tmp_path / "input" / "*" / "runs"))
    assert (out / "r1" / "ckpt.pt").read_text() == "newer"
