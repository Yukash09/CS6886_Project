from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Job:
    exp: str
    partition: str
    epochs: int
    seed: int = 0
    arch: str = "vit_tiny"
    dataset: str = "cifar10aug"
    lls_mode: str = "lls-mxm"
    readout: str = "linear"
    extra: tuple[str, ...] = field(default_factory=tuple)

    @property
    def name(self) -> str:
        part = self.partition.replace(":", "").replace(",", ".")
        mode = f"_{self.lls_mode}" if self.partition != "bp" else ""
        readout = "_lls-readout" if self.readout == "lls" else ""
        return f"{self.exp}_{self.arch}_{self.dataset}_{part}{mode}{readout}_e{self.epochs}_s{self.seed}"

    def argv(self) -> list[str]:
        mode = "lls" if self.partition == "bp" else self.lls_mode
        readout = "linear" if self.partition == "bp" else self.readout
        return ["--exp", self.exp, "--partition", self.partition, "--epochs", str(self.epochs),
                "--seed", str(self.seed), "--arch", self.arch, "--dataset", self.dataset,
                "--lls-mode", mode, "--readout", readout, *self.extra]


SECTIONS = ("bp", "uniform:1", "uniform:2", "uniform:3", "uniform:4", "uniform:12")

EXPS: dict[str, list[Job]] = {
    "smoke": [
        Job("V-SMOKE", "bp", 1, arch="vit_micro", extra=("--limit", "128", "--no-wandb")),
        Job("V-SMOKE", "uniform:1", 1, arch="vit_micro", extra=("--limit", "128", "--no-wandb")),
    ],
    "cifar10": [Job("V-C10", part, 300) for part in SECTIONS],
    "cifar100": [Job("V-C100", part, 300, dataset="cifar100aug") for part in SECTIONS],
    "variants": [Job("V-VAR", "uniform:2", 300, lls_mode=mode)
                 for mode in ("lls", "lls-m", "lls-mxm")],
    "readout": [Job("V-READ", "uniform:2", 300, readout=readout)
                for readout in ("linear", "lls")],
}


def jobs(name: str, seeds: list[int] | None = None) -> list[Job]:
    base = EXPS[name]
    return [replace(job, seed=seed) for seed in seeds for job in base] if seeds else list(base)

