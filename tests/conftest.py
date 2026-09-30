import pytest
import torch


@pytest.fixture
def cpu_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make upstream's hard-coded `.cuda()` a no-op so it runs on CPU."""
    monkeypatch.setattr(torch.Tensor, "cuda", lambda t, *_, **__: t)
