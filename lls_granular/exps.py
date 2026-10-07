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
C100 = [("G-C100", "uniform:1"), ("G-C100", "aligned"), ("G-C100", "bp")]
MOBILE = ["uniform:1", "bp", "uniform:2", "aligned", "uniform:14"]
FULL = 100  # the 300-epoch VGG8 runs reach >=99% of their best within 61 epochs (D14)


def _sweep() -> list[Job]:
    return [Job("G-ALL", label(p).replace("-", ","), 10, sched="proxy") for p in enumerate_all(7)]


EXPS: dict[str, list[Job]] = {
    "smoke": [Job("smoke", "aligned", 2, extra=("--limit", "1024"))],
    "timing": [Job("timing", "uniform:1", 3), Job("timing", "bp", 3)],
    "main": [Job(e, p, 300) for e, p in MAIN],
    "sweep": _sweep(),
    "cifar100": [Job(e, p, FULL, dataset="cifar100aug") for e, p in C100],
    "mobilenet": [Job("G-MN", p, FULL, arch="mobilenet") for p in MOBILE],
}


def jobs(name: str, seeds: list[int] | None = None) -> list[Job]:
    """The runs of an experiment, optionally repeated over seeds (seed-major order)."""
    base = EXPS[name]
    return [replace(j, seed=s) for s in seeds for j in base] if seeds else list(base)
