import argparse
import os
from pathlib import Path
from typing import Any

import torch

from lls_granular.bp import BPNet
from lls_granular.data import SETS
from lls_granular.measure import save, to_wandb
from lls_granular.metrics import Costs, gpu_peak
from lls_granular.partitions import Partition, uniform
from lls_granular.track import kind, method
from lls_granular.wrapper import GranularLLS

from .architectures import ARCHS


def candidates(arch: str) -> list[Partition | None]:
    n = len(ARCHS[arch]())
    sizes = (1, 2, 3, 4, 6, n)
    return list(dict.fromkeys([None, *(uniform(n, size) for size in sizes if size <= n)]))


def rows(arch: str, n_cls: int, bs: int, dev: torch.device, gpu: bool) -> list[dict[str, Any]]:
    torch.manual_seed(0)
    costs = Costs(ARCHS[arch](), n_cls, bs, dev)
    result: list[dict[str, Any]] = []
    for part in candidates(arch):
        row: dict[str, Any] = {
            "method": method(part), "kind": kind(part),
            "n_blocks": len(part) if part else None,
            "max_block": max(part) if part else None,
            **costs.cost(part),
        }
        if gpu:
            torch.manual_seed(0)
            units = ARCHS[arch]()
            network = BPNet(units, n_cls) if part is None else GranularLLS(
                units, part, n_cls, mode="lls-mxm")
            row |= gpu_peak(network, bs, n_cls, dev)
            del network
            torch.cuda.empty_cache()
        result.append(row)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cost of ViT LLS partitions.")
    parser.add_argument("--arch", default="vit_tiny", choices=sorted(ARCHS))
    parser.add_argument("--dataset", default="cifar10aug", choices=sorted(SETS))
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "lls-vit"))
    args = parser.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = dev.type == "cuda" and not args.no_gpu
    result = rows(args.arch, SETS[args.dataset][1], args.bs, dev, use_gpu)
    output = os.environ.get("LLS_VIT_OUT") or os.environ.get("LLS_OUT") or "./runs"
    name = f"cost_{args.arch}_{args.dataset}_bs{args.bs}"
    save(result, Path(output).expanduser() / f"{name}.csv")
    if args.wandb:
        gpu_name = torch.cuda.get_device_name(0) if use_gpu else "cpu"
        to_wandb(result, name, {**vars(args), "n_cls": SETS[args.dataset][1], "gpu": gpu_name}, args.project)


if __name__ == "__main__":
    main()

