from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Job:
    exp: str
    partition: str
    epochs: int
    seed: int = 0
    arch: str = "resnet20"
    dataset: str = "cifar10aug"
    sched: str = "full"
    lls_mode: str = "lls"
    extra: tuple[str, ...] = field(default_factory=tuple)

    @property
    def name(self) -> str:
        partition = self.partition.replace(":", "").replace(",", ".")
        variant = f"_{self.lls_mode}" if self.partition != "bp" and self.lls_mode != "lls" else ""
        return f"{self.exp}_{self.arch}_{self.dataset}_{partition}{variant}_e{self.epochs}_s{self.seed}"

    def argv(self) -> list[str]:
        return ["--exp", self.exp, "--partition", self.partition,
                "--epochs", str(self.epochs), "--seed", str(self.seed),
                "--arch", self.arch, "--dataset", self.dataset,
                "--sched", self.sched, "--lls-mode", self.lls_mode, *self.extra]


METHODS = ("bp", "uniform:1", "uniform:2", "aligned")
DEPTHS = ("resnet20", "resnet32", "resnet56")

EXPS: dict[str, list[Job]] = {
    "smoke": [
        Job("S-SMOKE", "bp", 1, extra=("--limit", "128")),
        Job("S-SMOKE", "uniform:1", 1, extra=("--limit", "128")),
        Job("S-SMOKE", "aligned", 1, extra=("--limit", "128")),
    ],
    "resnet20": [Job("S-R20", method, 300) for method in METHODS],
    "depth": [Job("S-DEPTH", method, 300, arch=arch) for arch in DEPTHS for method in METHODS],
    "cifar100": [Job("S-C100", method, 300, dataset="cifar100aug") for method in METHODS],
    "variants": [Job("S-VAR", "uniform:1", 100, lls_mode=mode,
                     extra=("--patience", "10")) for mode in ("lls", "lls-m", "lls-mxm")],
}


def jobs(name: str, seeds: list[int] | None = None) -> list[Job]:
    base = EXPS[name]
    return [replace(job, seed=seed) for seed in seeds for job in base] if seeds else list(base)
