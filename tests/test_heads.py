import pytest
import torch

from lls_granular.heads import LLSHead, basis, coherence, pool_flat
from lls_granular.upstream import compute_LLS

CASES = [  # (activation shape, pool): conv1-like, conv6-like, fc1-like
    ((8, 128, 32, 32), 4),
    ((8, 512, 4, 4), 2),
    ((8, 1024), 1024),
]


@pytest.mark.usefixtures("cpu_cuda")
@pytest.mark.parametrize("shape,pool", CASES)
@pytest.mark.parametrize("wave", ["square", "cosine"])
def test_matches_upstream(shape: tuple[int, ...], pool: int, wave: str) -> None:
    torch.manual_seed(0)
    h = torch.randn(*shape)
    y = torch.randint(0, 10, (shape[0],))
    dim = pool_flat(h, pool).shape[1]

    ours, _ = LLSHead(10, dim, pool, wave=wave, temp=0.5)(h, y)
    ref = compute_LLS(h, y, temperature=0.5, act_size=pool, n_classes=10, waveform=wave)
    assert torch.allclose(ours, ref, atol=1e-6)


def test_scores_and_grad() -> None:
    head = LLSHead(10, 2048, 4)
    h = torch.randn(8, 128, 32, 32, requires_grad=True)
    loss, s = head(h, torch.randint(0, 10, (8,)))
    assert s.shape == (8, 10)
    assert (s.abs() <= head.B.norm(dim=1) + 1e-4).all()  # Cauchy-Schwarz: |<z, b>| <= |b|
    loss.backward()
    assert h.grad is not None and h.grad.abs().sum() > 0


def test_no_params_and_buffer_moves() -> None:
    head = LLSHead(10, 1024, 2)
    assert sum(p.numel() for p in head.parameters()) == 0
    assert "B" not in head.state_dict()
    assert head.double().B.dtype == torch.float64


def test_basis() -> None:
    b = basis(10, 1024)
    assert b.shape == (10, 1024)
    assert set(b.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert 0 <= coherence(b) <= 1
    with pytest.raises(ValueError):
        basis(10, 1024, wave="triangle")


def test_dim_mismatch() -> None:
    with pytest.raises(ValueError):
        LLSHead(10, 1024, 4).scores(torch.randn(2, 128, 32, 32))
