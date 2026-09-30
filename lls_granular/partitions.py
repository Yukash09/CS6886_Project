"""A partition is a tuple of block sizes over the units, e.g. (2, 2, 1, 1, 1)."""
import logging
from itertools import accumulate
from typing import Sequence

log = logging.getLogger(__name__)

Partition = tuple[int, ...]
Shape = tuple[int, ...]


def check(p: Partition, n: int) -> None:
    if not p or any(s < 1 for s in p) or sum(p) != n:
        raise ValueError(f"invalid partition {p} for {n} units")


def from_cuts(cuts: Sequence[int], n: int) -> Partition:
    """Cut positions (unit indices that start a new block) -> block sizes."""
    edges = [0, *sorted(set(cuts)), n]
    return tuple(b - a for a, b in zip(edges, edges[1:]))


def spans(p: Partition) -> list[range]:
    """Block sizes -> unit index range of each block."""
    ends = list(accumulate(p))
    return [range(e - s, e) for s, e in zip(p, ends)]


def uniform(n: int, k: int) -> Partition:
    """Blocks of k units; the remainder forms the last block."""
    return from_cuts(range(k, n, k), n)


def aligned(shapes: Sequence[Shape]) -> Partition:
    """Cut wherever the spatial resolution changes between consecutive units."""
    cuts = [i for i in range(1, len(shapes)) if shapes[i][1:] != shapes[i - 1][1:]]
    return from_cuts(cuts, len(shapes))


def enumerate_all(n: int) -> list[Partition]:
    """All 2^(n-1) contiguous partitions, in a fixed order (bit i set = cut before unit i+1)."""
    gaps = range(1, n)
    log.debug("enumerating %d partitions of %d units", 1 << (n - 1), n)
    return [from_cuts([g for g in gaps if m >> (g - 1) & 1], n) for m in range(1 << (n - 1))]


def parse(spec: str, shapes: Sequence[Shape]) -> Partition:
    """'uniform:k' | 'aligned' | explicit sizes like '2,2,1,1,1'."""
    n = len(shapes)
    if spec.startswith("uniform:"):
        p = uniform(n, int(spec.split(":")[1]))
    elif spec == "aligned":
        p = aligned(shapes)
    else:
        p = tuple(int(s) for s in spec.split(","))
    check(p, n)
    log.debug("partition %r -> %s, blocks %s", spec, p, [list(r) for r in spans(p)])
    return p


def label(p: Partition) -> str:
    return "-".join(map(str, p))
