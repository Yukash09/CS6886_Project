import logging
from collections.abc import Callable

import torch
from torch import Tensor, nn

from lls_granular.units import Unit, describe

log = logging.getLogger(__name__)

MAX_HEAD_DIM = 2048


class PatchEmbedding(nn.Module):
    def __init__(self, in_ch: int, img: int, patch: int, dim: int, dropout: float) -> None:
        super().__init__()
        if img % patch:
            raise ValueError(f"image side {img} must be divisible by patch size {patch}")
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.n_patches = (img // patch) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        patches = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        return self.dropout(torch.cat((cls, patches), dim=1) + self.pos_embed)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"embedding dimension {dim} must be divisible by {heads} heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        weights = self.attn_drop((q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1))
        out = (weights @ v).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(out))

    def extra_macs(self, out: Tensor) -> int:
        tokens, dim = out.shape[1], out.shape[2]
        return 2 * tokens * tokens * dim


class MLP(nn.Sequential):
    def __init__(self, dim: int, ratio: float, dropout: float) -> None:
        hidden = int(dim * ratio)
        super().__init__(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class FinalEncoderBlock(nn.Module):
    def __init__(self, block: EncoderBlock, dim: int) -> None:
        super().__init__()
        self.block = block
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.block(x))


class StemBlock(nn.Module):
    def __init__(self, stem: PatchEmbedding, block: nn.Module) -> None:
        super().__init__()
        self.stem = stem
        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(self.stem(x))


def _initialise(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def cifar_vit(*, depth: int, dim: int, heads: int, patch: int = 4, mlp_ratio: float = 4.0,
              dropout: float = 0.0, in_ch: int = 3, img: int = 32) -> list[Unit]:
    if depth < 1:
        raise ValueError(f"depth must be positive, got {depth}")
    stem = PatchEmbedding(in_ch, img, patch, dim, dropout)
    blocks = [EncoderBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
    encoder: list[nn.Module] = [*blocks[:-1], FinalEncoderBlock(blocks[-1], dim)]
    first = StemBlock(stem, encoder[0])
    first.apply(_initialise)
    nn.init.trunc_normal_(stem.cls_token, std=0.02)
    nn.init.trunc_normal_(stem.pos_embed, std=0.02)
    for block in encoder[1:]:
        block.apply(_initialise)

    tokens = stem.n_patches + 1
    head_dim = min(2 * dim, MAX_HEAD_DIM)
    units = [Unit("enc1", first, head_dim, (tokens, dim))]
    units += [Unit(f"enc{i}", block, head_dim, (tokens, dim))
              for i, block in enumerate(encoder[1:], start=2)]
    log.debug("vit depth=%d dim=%d heads=%d patch=%d tokens=%d", depth, dim, heads, patch, tokens)
    describe(units)
    return units


def vit_micro(in_ch: int = 3, img: int = 32) -> list[Unit]:
    return cifar_vit(depth=2, dim=64, heads=2, patch=4, in_ch=in_ch, img=img)


def vit_tiny(in_ch: int = 3, img: int = 32) -> list[Unit]:
    return cifar_vit(depth=12, dim=192, heads=3, patch=4, in_ch=in_ch, img=img)


ARCHS: dict[str, Callable[..., list[Unit]]] = {"vit_micro": vit_micro, "vit_tiny": vit_tiny}

