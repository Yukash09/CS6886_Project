import torch

BS = 8


def batches(n: int, seed: int = 1, n_cls: int = 10) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """n random CIFAR-shaped batches from a private generator (leaves the global RNG untouched)."""
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(BS, 3, 32, 32, generator=g), torch.randint(0, n_cls, (BS,), generator=g))
            for _ in range(n)]
