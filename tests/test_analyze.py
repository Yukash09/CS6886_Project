import pandas as pd
import pytest

from lls_granular.analyze import acc_within, budget, dedup, kendall, pareto, parse_method


def test_pareto() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 2, 4], "y": [0.5, 0.7, 0.9, 0.6, 0.9]})
    assert pareto(df, "x", "y").tolist() == [True, True, True, False, False]


def test_kendall() -> None:
    assert kendall([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert kendall([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert kendall([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(2 / 3)


def test_budget_picks_best_within_budget() -> None:
    df = pd.DataFrame({"method": ["a", "b", "c"], "act_mb": [10, 20, 30], "val": [0.5, 0.8, 0.9],
                       "test": [0.4, 0.7, 0.85]})
    t = budget(df, [5, 15, 25, 35], "val")
    assert t["method"].tolist() == ["a", "b", "c"]
    assert t["budget_mb"].tolist() == [15, 25, 35]


def test_acc_within_uses_only_early_epochs() -> None:
    h = pd.DataFrame({"epoch": [0, 1, 2, 3], "val/acc": [0.1, 0.5, 0.4, 0.9], "test/acc": [0.1, 0.45, 0.4, 0.88]})
    assert acc_within(h, 3) == (0.5, 0.45)


def test_dedup_keeps_latest() -> None:
    df = pd.DataFrame({"name": ["r", "r", "s"], "created": ["2026-01-01", "2026-02-01", "2026-01-01"], "v": [1, 2, 3]})
    assert dedup(df).set_index("name")["v"].to_dict() == {"r": 2, "s": 3}


def test_parse_method() -> None:
    assert parse_method("bp") is None
    assert parse_method("p2.2.3") == (2, 2, 3)


def test_table_md_formats_percentages() -> None:
    from lls_granular.analyze import table_md
    md = table_md(pd.DataFrame({"method": ["p7"], "test": [0.9298], "act_mb": [964.87]}), ["method", "test", "act_mb"])
    assert "| p7 | 92.98 | 964.9 |" in md


def test_pretty() -> None:
    from lls_granular.analyze import pretty
    assert [pretty(m) for m in ("bp", "p1.1.1", "p7", "p2.2.3")] == ["BP", "LLS (per layer)", "one block", "blocks 2-2-3"]


def test_seed_stats_needs_two_seeds() -> None:
    from lls_granular.analyze import seed_stats
    df = pd.DataFrame({"method": ["a", "a", "a", "b"], "test_at_best_val": [0.90, 0.92, 0.91, 0.80]})
    st = seed_stats(df)
    assert st["method"].tolist() == ["a"]
    assert st["test"].iloc[0] == pytest.approx(0.91) and st["std"].iloc[0] == pytest.approx(0.01)


def test_short_compresses_runs_of_ones() -> None:
    from lls_granular.analyze import short
    assert [short(m) for m in ("p1.1.5", "p4.2.1.1.1.1", "p2.2.2")] == ["1,1,5", "4,2,1×4", "2,2,2"]
