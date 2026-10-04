from collections import Counter

import pytest
import torch
from PIL import Image

from lls_granular.data import load, root, split, tfms

TARGETS = [i % 10 for i in range(1000)]


def test_split_stratified_disjoint_complete() -> None:
    tr, va = split(TARGETS, 100, 10)
    assert len(va) == 100 and len(tr) == 900
    assert not set(tr) & set(va)
    assert sorted(tr + va) == list(range(1000))
    assert Counter(TARGETS[i] for i in va) == {c: 10 for c in range(10)}


def test_split_fixed_by_seed() -> None:
    assert split(TARGETS, 100, 10) == split(TARGETS, 100, 10)
    assert split(TARGETS, 100, 10, seed=1) != split(TARGETS, 100, 10)


def test_split_rejects_uneven() -> None:
    with pytest.raises(ValueError):
        split(TARGETS, 105, 10)


@pytest.mark.parametrize("aug", [True, False])
def test_tfms(aug: bool) -> None:
    img = Image.new("RGB", (32, 32), (128, 64, 32))
    train, ev = tfms(aug)
    assert train(img).shape == ev(img).shape == (3, 32, 32)
    assert torch.equal(ev(img), ev(img))


def test_root_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLS_DATA", "/env/dir")
    assert str(root("/arg/dir")) == "/arg/dir"
    assert str(root(None)) == "/env/dir"
    monkeypatch.delenv("LLS_DATA")
    assert str(root(None)) == "data"


@pytest.mark.skipif(not (root(None) / "cifar-10-batches-py").exists(), reason="CIFAR10 not present")
def test_load_cifar10aug() -> None:
    d = load("cifar10aug", bs=16, workers=0, limit=64)
    x, y = next(iter(d.train))
    assert x.shape == (16, 3, 32, 32) and y.shape == (16,)
    assert d.n_cls == 10 and d.val is not None
    assert len(d.test.dataset) == 10000
