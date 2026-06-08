"""Regression tests for the upgraded backtest engine interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from run_ultimate_backtest import (
    BacktestResult,
    ExecutionConfig,
    StrategyPolicy,
    run_real_backtest,
    validate_backtest_dataset,
)


class AlwaysHomeModel:
    """Test model that records call order and never reads future data."""

    def __init__(self, events: list[tuple[str, Any]]) -> None:
        self.events = events

    def predict(
        self,
        home: str,
        away: str,
        date: Any = None,
        row: Any = None,
        **kwargs: Any,
    ) -> tuple[float, float, float]:
        self.events.append(("predict", tuple(row.index if row is not None else ())))
        return (0.75, 0.15, 0.10)

    def update(
        self,
        home: str,
        away: str,
        hg: int,
        ag: int,
        date: Any = None,
        row: Any = None,
        **kwargs: Any,
    ) -> None:
        self.events.append(("update", f"{home} v {away}"))


class AlwaysAwayModel:
    """Test model that always selects away at a positive EV."""

    def predict(
        self,
        home: str,
        away: str,
        date: Any = None,
        row: Any = None,
        **kwargs: Any,
    ) -> tuple[float, float, float]:
        return (0.10, 0.10, 0.80)

    def update(
        self,
        home: str,
        away: str,
        hg: int,
        ag: int,
        date: Any = None,
        row: Any = None,
        **kwargs: Any,
    ) -> None:
        return None


def _write_sample_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,"
                "B365CH,B365CD,B365CA,HS,HST,AS,AST",
                "01/08/2022,Alpha,Beta,2,0,H,2.10,3.30,3.40,2.00,3.20,3.60,10,5,8,3",
                "02/08/2022,Gamma,Delta,0,1,A,2.20,3.10,3.30,2.05,3.00,3.50,9,2,11,6",
            ]
        ),
        encoding="utf-8",
    )


def _run_sample(
    path: Path,
    execution_config: ExecutionConfig | None = None,
    strategy_policy: StrategyPolicy | None = None,
) -> BacktestResult:
    events: list[tuple[str, Any]] = []

    def factory() -> AlwaysHomeModel:
        return AlwaysHomeModel(events)

    result = run_real_backtest(
        csv_paths=(str(path),),
        model_factory=factory,
        ev_threshold=1.01,
        initial_capital=1000.0,
        odds_mode="opening",
        verbose=False,
        return_result=True,
        execution_config=execution_config,
        strategy_policy=strategy_policy,
    )
    assert isinstance(result, BacktestResult)
    result.metrics["events"] = events
    return result


def test_backtest_result_keeps_legacy_dict_compatibility(tmp_path: Path) -> None:
    """Typed results should still expose legacy bet record aliases."""
    csv_path = tmp_path / "sample.csv"
    _write_sample_csv(csv_path)

    result = _run_sample(csv_path)
    payload = result.to_dict()

    assert result.metrics["trades"] == 2
    assert len(result.trade_records) == 2
    assert payload["trades"] == 2
    assert payload["bet_records"][0]["odds_taken"] == pytest.approx(2.10)
    assert payload["bet_records"][0]["odds_close"] == pytest.approx(2.00)
    assert payload["bet_records"][0]["clv_pct"] == pytest.approx(0.05)
    assert payload["execution_summary"]["mode"] == "legacy"


def test_predict_happens_before_update_and_future_columns_are_sanitized(
    tmp_path: Path,
) -> None:
    """Prediction rows must not expose settlement or closing-line fields."""
    csv_path = tmp_path / "sample.csv"
    _write_sample_csv(csv_path)

    result = _run_sample(csv_path)
    events = result.metrics["events"]

    assert [event[0] for event in events] == ["predict", "update", "predict", "update"]
    predict_columns = [set(event[1]) for event in events if event[0] == "predict"]
    for columns in predict_columns:
        assert "FTR" not in columns
        assert "FTHG" not in columns
        assert "FTAG" not in columns
        assert "B365CH" not in columns
        assert "B365CD" not in columns
        assert "B365CA" not in columns
        assert "HS" not in columns
        assert "HST" not in columns


def test_realistic_execution_models_partial_fill_and_odds_friction(
    tmp_path: Path,
) -> None:
    """Realistic execution should reduce fill size and quoted odds explicitly."""
    csv_path = tmp_path / "sample.csv"
    _write_sample_csv(csv_path)

    result = _run_sample(
        csv_path,
        ExecutionConfig(
            mode="realistic",
            spread_pct=0.05,
            slippage_pct=0.02,
            partial_fill_rate=0.50,
            seed=7,
        ),
    )
    record = result.trade_records[0]

    assert record.execution_status == "partial_fill"
    assert record.stake == pytest.approx(record.requested_stake * 0.50)
    assert record.odds == pytest.approx(2.10 * 0.93)
    assert result.execution_summary["counts"]["partial_fill"] == 2


def test_realistic_execution_rejection_does_not_count_as_trade(tmp_path: Path) -> None:
    """Rejected simulated orders must not be counted as filled trades."""
    csv_path = tmp_path / "sample.csv"
    _write_sample_csv(csv_path)

    result = _run_sample(
        csv_path,
        ExecutionConfig(mode="realistic", rejection_rate=1.0, seed=7),
    )

    assert result.metrics["trades"] == 0
    assert len(result.trade_records) == 2
    assert {record.execution_status for record in result.trade_records} == {"rejected"}
    assert result.execution_summary["counts"]["rejected"] == 2


def test_validate_backtest_dataset_fails_closed_on_missing_columns(
    tmp_path: Path,
) -> None:
    """Unsafe CSV inputs should be reported and rejected by the main runner."""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A",
                "01/08/2022,Alpha,Beta,2,0,H,2.10,3.30,3.40",
            ]
        ),
        encoding="utf-8",
    )

    diagnostics = validate_backtest_dataset([str(csv_path)])

    failed = diagnostics[diagnostics["status"] == "failed"]
    assert "required_columns" in set(failed["check"])
    with pytest.raises(ValueError, match="Unsafe backtest dataset"):
        run_real_backtest(
            csv_paths=(str(csv_path),),
            model_factory=lambda: AlwaysHomeModel([]),
            verbose=False,
        )


def test_opening_mode_requires_opening_odds(tmp_path: Path) -> None:
    """Opening-line backtests must not silently fall back to closing odds."""
    csv_path = tmp_path / "closing_only.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365CH,B365CD,B365CA",
                "01/08/2022,Alpha,Beta,2,0,H,2.00,3.20,3.60",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing opening odds"):
        run_real_backtest(
            csv_paths=(str(csv_path),),
            model_factory=lambda: AlwaysHomeModel([]),
            odds_mode="opening",
            verbose=False,
        )


def test_backtest_metrics_include_statistical_robustness_fields(
    tmp_path: Path,
) -> None:
    """Every backtest should expose sample size and robustness diagnostics."""
    csv_path = tmp_path / "sample.csv"
    _write_sample_csv(csv_path)

    result = _run_sample(csv_path)

    assert result.metrics["sample_size"] == result.metrics["trades"]
    assert "p_value" in result.metrics
    assert "adjusted_p_value" in result.metrics
    assert "deflated_sharpe" in result.metrics
    assert result.metrics["robustness_status"] in {
        "observe_only",
        "statistically_positive",
    }


def test_empty_strategy_policy_preserves_legacy_sample_results(tmp_path: Path) -> None:
    """A no-op StrategyPolicy must not change legacy behavior."""
    csv_path = tmp_path / "sample.csv"
    _write_sample_csv(csv_path)

    baseline = _run_sample(csv_path)
    no_op = _run_sample(csv_path, strategy_policy=StrategyPolicy(name="noop"))

    assert no_op.metrics["trades"] == baseline.metrics["trades"]
    assert no_op.metrics["final_capital"] == pytest.approx(baseline.metrics["final_capital"])
    assert [item.pnl for item in no_op.trade_records] == pytest.approx(
        [item.pnl for item in baseline.trade_records]
    )


def test_strategy_policy_blocks_away_selection(tmp_path: Path) -> None:
    """Policy should be able to exclude structurally weak away bets."""
    csv_path = tmp_path / "sample.csv"
    _write_sample_csv(csv_path)

    result = run_real_backtest(
        csv_paths=(str(csv_path),),
        model_factory=AlwaysAwayModel,
        ev_threshold=1.01,
        initial_capital=1000.0,
        odds_mode="opening",
        verbose=False,
        return_result=True,
        strategy_policy=StrategyPolicy(name="no_away", allowed_selections=("H", "D")),
    )

    assert isinstance(result, BacktestResult)
    assert result.metrics["trades"] == 0
    assert len(result.trade_records) == 2
    assert {item.execution_status for item in result.trade_records} == {"policy_blocked"}
    assert result.risk_summary["risk_blocks"]["strategy_policy"] == 2


def test_strategy_policy_blocks_high_odds_and_ev_band(tmp_path: Path) -> None:
    """Policy should filter by odds and EV ranges before staking."""
    csv_path = tmp_path / "sample.csv"
    _write_sample_csv(csv_path)

    away_blocked = run_real_backtest(
        csv_paths=(str(csv_path),),
        model_factory=AlwaysAwayModel,
        ev_threshold=1.01,
        initial_capital=1000.0,
        odds_mode="opening",
        verbose=False,
        return_result=True,
        strategy_policy=StrategyPolicy(name="odds_cap", max_odds=3.0),
    )
    home_blocked = _run_sample(
        csv_path,
        strategy_policy=StrategyPolicy(name="ev_band", min_ev=1.10, max_ev=1.20),
    )

    assert isinstance(away_blocked, BacktestResult)
    assert away_blocked.metrics["trades"] == 0
    assert home_blocked.metrics["trades"] == 0
    assert all(item.execution_status == "policy_blocked" for item in home_blocked.trade_records)


def test_strategy_policy_can_reduce_high_odds_stake(tmp_path: Path) -> None:
    """High-odds stake multipliers should reduce requested order size."""
    csv_path = tmp_path / "sample.csv"
    _write_sample_csv(csv_path)

    unrestricted = run_real_backtest(
        csv_paths=(str(csv_path),),
        model_factory=AlwaysAwayModel,
        ev_threshold=1.01,
        initial_capital=1000.0,
        odds_mode="opening",
        verbose=False,
        return_result=True,
    )
    reduced = run_real_backtest(
        csv_paths=(str(csv_path),),
        model_factory=AlwaysAwayModel,
        ev_threshold=1.01,
        initial_capital=1000.0,
        odds_mode="opening",
        verbose=False,
        return_result=True,
        strategy_policy=StrategyPolicy(
            name="high_odds_reduce",
            high_odds_threshold=3.0,
            high_odds_stake_multiplier=0.25,
        ),
    )

    assert isinstance(unrestricted, BacktestResult)
    assert isinstance(reduced, BacktestResult)
    assert reduced.trade_records[0].requested_stake == pytest.approx(
        unrestricted.trade_records[0].requested_stake * 0.25
    )
