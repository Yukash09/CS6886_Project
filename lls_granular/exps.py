"""Named experiments: each is a list of runs. Edit here, not in notebooks."""
from dataclasses import dataclass, field, replace

from .partitions import enumerate_all, label


@dataclass(frozen=True)
class Job:
    exp: str
    partition: str
    epochs: int
    seed: int = 0
    arch: str = "vgg8"
    dataset: str = "cifar10aug"
    sched: str = "full"
    extra: tuple[str, ...] = field(default_factory=tuple)

    @property
    def name(self) -> str:
        p = self.partition.replace(":", "").replace(",", ".")
        return f"{self.exp}_{self.arch}_{self.dataset}_{p}_e{self.epochs}_s{self.seed}"

    def argv(self) -> list[str]:
        return ["--exp", self.exp, "--partition", self.partition, "--epochs", str(self.epochs),
                "--seed", str(self.seed), "--arch", self.arch, "--dataset", self.dataset,
                "--sched", self.sched, *self.extra]


MAIN = [  # LLS and BP first: the two reference points finish earliest
    ("G-U", "uniform:1"), ("G-BP", "bp"), ("G-U", "uniform:2"), ("G-U", "uniform:3"),
    ("G-P", "aligned"), ("G-P", "2,2,3"), ("G-U", "uniform:7"),
]
C100 = [("G-C100", "uniform:1"), ("G-C100", "aligned"), ("G-C100", "bp"),
        ("G-C100", "1,1,5"), ("G-C100", "2,5")]  # the two best VGG8 splits from the sweep (H3)
MOBILE = ["uniform:1", "bp", "uniform:2", "aligned", "uniform:14"]
MOBILE_MERGE = [  # where to merge on a flat memory profile (H7): late vs early vs coarse at resolution changes
    ",".join(["1"] * 12 + ["2"]), ",".join(["1"] * 10 + ["4"]), ",".join(["2"] + ["1"] * 12), "4,2,6,2",
]
MOBILE_GAP = [  # fill the front between k=2 (62 MB) and coarse 4,2,6,2 (144 MB): is merging the first layers enough?
    ",".join(["3"] + ["1"] * 11), ",".join(["4"] + ["1"] * 10), ",".join(["4", "2"] + ["1"] * 8),
    "4,2,2,2,2,2", "4,2,6,1,1",
]
SEED_SPLITS = ["uniform:1", "1,1,5", "2,5", "uniform:7", "bp"]  # key VGG8 splits for error bars
FULL = 100  # the 300-epoch VGG8 runs reach >=99% of their best within 61 epochs (D14)


SWEEP = 60  # every main split is within ~1% of its best by epoch 60; epoch 10 of these runs is the cheap proxy (H5)


def _sweep() -> list[Job]:
    return [Job("G-ALL", label(p).replace("-", ","), SWEEP) for p in enumerate_all(7)]


def _seeds() -> list[Job]:
    """Seeds 1-2 at the sweep's epochs (seed 0 comes from the sweep), plus BP seed 0, which the sweep lacks."""
    return [Job("G-SEED", "bp", SWEEP)] + [Job("G-SEED", p, SWEEP, seed=s) for s in (1, 2) for p in SEED_SPLITS]


EXPS: dict[str, list[Job]] = {
    "smoke": [Job("smoke", "aligned", 2, extra=("--limit", "1024"))],
    "timing": [Job("timing", "uniform:1", 3), Job("timing", "bp", 3)],
    "main": [Job(e, p, 300) for e, p in MAIN],
    "sweep": _sweep(),
    "cifar100": [Job(e, p, FULL, dataset="cifar100aug") for e, p in C100],
    "mobilenet": [Job("G-MN", p, FULL, arch="mobilenet") for p in MOBILE],
    # LLS and k=2 were still improving at epoch 100, so every MobileNet split gets the same 300 epochs
    "mobilenet300": [Job("G-MN", p, 300, arch="mobilenet") for p in MOBILE + MOBILE_MERGE],
    "mobilenet_gap": [Job("G-MN", p, 300, arch="mobilenet") for p in MOBILE_GAP],
    "seeds": _seeds(),
}


def jobs(name: str, seeds: list[int] | None = None) -> list[Job]:
    """The runs of an experiment, optionally repeated over seeds (seed-major order)."""
    base = EXPS[name]
    return [replace(j, seed=s) for s in seeds for j in base] if seeds else list(base)
