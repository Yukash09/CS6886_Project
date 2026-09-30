import pytest

from lls_granular.partitions import (aligned, check, enumerate_all, from_cuts, label, parse,
                                     spans, uniform)
from lls_granular.units import vgg8

SHAPES = [u.shape for u in vgg8()]


def test_uniform() -> None:
    assert uniform(7, 1) == (1,) * 7
    assert uniform(7, 2) == (2, 2, 2, 1)
    assert uniform(7, 3) == (3, 3, 1)
    assert uniform(7, 7) == (7,)


def test_aligned_vgg8() -> None:
    assert aligned(SHAPES) == (2, 2, 1, 1, 1)


def test_enumerate_all() -> None:
    parts = enumerate_all(7)
    assert len(parts) == 64
    assert len(set(parts)) == 64
    assert parts[0] == (7,) and parts[-1] == (1,) * 7
    for p in parts:
        check(p, 7)


def test_spans() -> None:
    assert spans((2, 2, 3)) == [range(0, 2), range(2, 4), range(4, 7)]


def test_from_cuts() -> None:
    assert from_cuts([2, 4], 7) == (2, 2, 3)
    assert from_cuts([], 7) == (7,)


def test_parse() -> None:
    assert parse("uniform:2", SHAPES) == (2, 2, 2, 1)
    assert parse("aligned", SHAPES) == (2, 2, 1, 1, 1)
    assert parse("2,2,3", SHAPES) == (2, 2, 3)
    assert label((2, 2, 3)) == "2-2-3"


@pytest.mark.parametrize("p", [(), (2, 2, 2), (0, 7), (3, -1, 5)])
def test_check_rejects(p: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        check(p, 7)
