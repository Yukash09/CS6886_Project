import pytest
import torch
import torch.nn.functional as F

import lls_exit.policies as P
from lls_granular.heads import LLSHead


def onehots(rows: list[list[int]], c: int = 3, hot: float = 0.9) -> torch.Tensor:
    """(N, E, C) near-one-hot probabilities from argmax indices."""
    a = torch.tensor(rows)
    p = torch.full((*a.shape, c), (1 - hot) / (c - 1))
    return p.scatter(-1, a[..., None], hot)


def test_cosines_undo_row_norms() -> None:
    torch.manual_seed(0)
    h = LLSHead(10, 64, 64)  # 2D input pooled to length 64 = identity
    z = torch.randn(5, 64)
    s = h.scores(z)
    cos = P.cosines(s[:, None], h.B.norm(dim=1)[None])[:, 0]
    assert torch.allclose(cos, F.cosine_similarity(z[:, None], h.B[None], dim=-1), atol=1e-5)


def test_fit_temp_recovers_true_temperature() -> None:
    g = torch.Generator().manual_seed(0)
    z = torch.randn(20000, 5, generator=g)
    y = torch.multinomial(F.softmax(z / 0.2, -1), 1, generator=g)[:, 0]
    assert P.fit_temp(z, y) == pytest.approx(0.2, rel=0.1)


def test_probs_variants() -> None:
    torch.manual_seed(0)
    sp = {"scores": torch.randn(6, 3, 4), "logits": torch.randn(6, 4), "y": torch.randint(0, 4, (6,))}
    heads = {"row_norm": torch.full((3, 4), 2.0)}
    p = P.probs(sp, heads, "head")
    assert p.shape == (6, 3, 4) and torch.allclose(p, F.softmax(sp["scores"] / 0.5, -1))
    p = P.probs(sp, heads, "cls")
    assert p.shape == (6, 4, 4) and torch.allclose(p[:, -1], F.softmax(sp["logits"], -1))
    q = P.probs(sp, heads, "cls", temps=[1.0, 1.0, 1.0, 2.0])
    assert torch.allclose(q[:, 0], F.softmax(sp["scores"][:, 0] / 2.0, -1))  # cosine = score / 2
    assert torch.allclose(q[:, -1], F.softmax(sp["logits"] / 2.0, -1))
    assert len(P.fit_temps(sp, heads, "cls")) == 4
    with pytest.raises(ValueError):
        P.probs(sp, heads, "head", temps=[1.0])
    with pytest.raises(ValueError):
        P.probs(sp, heads, "last")


def test_signals() -> None:
    p = torch.tensor([[[0.7, 0.2, 0.1], [1 / 3, 1 / 3, 1 / 3]]])
    assert torch.allclose(P.confidence(p), torch.tensor([[0.7, 1 / 3]]))
    assert torch.allclose(P.margin(p), torch.tensor([[0.5, 0.0]]), atol=1e-6)
    assert P.neg_entropy(p)[0, 1] == pytest.approx(-torch.log(torch.tensor(3.0)).item(), abs=1e-5)
    assert P.streak(onehots([[0, 0, 1, 1, 1, 0]])).tolist() == [[1, 2, 1, 2, 3, 1]]


def test_sync() -> None:
    same = onehots([[1, 1, 1]])
    s = P.sync(same, 2)
    assert s[0, 0] == float("-inf") and torch.allclose(s[0, 1:], torch.ones(2))
    hard = torch.eye(3)[None]  # three exits, orthogonal one-hots
    assert torch.allclose(P.sync(hard, 3)[0, 2], torch.tensor(0.0))
    mixed = onehots([[0, 0, 1]], hot=1.0)
    assert P.sync(mixed, 3)[0, 2] == pytest.approx(1 / 3)  # pairs: (0,0)=1, (0,1)=0, (0,1)=0


def test_fire_and_first_exit() -> None:
    p = torch.tensor([[[0.5, 0.5], [0.95, 0.05], [0.6, 0.4]],    # confident at exit 1
                      [[0.5, 0.5], [0.55, 0.45], [0.6, 0.4]]])  # never confident -> final
    ex = P.first_exit(P.fire(p, P.Policy("confidence", {"tau": 0.9})))
    assert ex.tolist() == [1, 2]
    assert P.first_exit(P.fire(p, P.Policy("static", {"e": 0}))).tolist() == [0, 0]
    assert P.first_exit(P.fire(p, P.Policy("patience", {"n": 2}))).tolist() == [1, 1]
    for pol in [P.Policy("entropy", {"tau": 0.5}), P.Policy("margin", {"tau": 0.5}),
                P.Policy("sync", {"w": 2, "tau": 0.9}), P.Policy("sync_margin", {"w": 2, "tau": 0.9, "m": 0.5})]:
        assert P.fire(p, pol).shape == (2, 3)
    with pytest.raises(ValueError):
        P.fire(p, P.Policy("magic"))
    assert P.Policy("sync", {"w": 3, "tau": 0.9}).label() == "sync w=3 tau=0.9"


def test_oracle_exit() -> None:
    p = onehots([[2, 0, 0], [1, 1, 2], [0, 0, 0]])
    assert P.oracle_exit(p, torch.tensor([0, 2, 1])).tolist() == [1, 2, 2]  # last: never right -> final


def test_decision_ops() -> None:
    c = 10
    ops = {n: P.decision_ops(P.Policy(n, {"w": 3}), c) for n in ["static", "confidence", "margin", "sync"]}
    assert ops["static"] == 0 and ops["confidence"] < ops["margin"] < ops["sync"]
    with pytest.raises(ValueError):
        P.decision_ops(P.Policy("magic"), c)
