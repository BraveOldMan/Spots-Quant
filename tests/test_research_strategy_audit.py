"""Regression tests for the strategy research audit gates."""

import pandas as pd

import research_strategy_audit as rsa
from run_ultimate_backtest import TradeRecord
from research_strategy_audit import (
    _market_consensus_probs,
    _attach_season_robustness,
    build_strategy_segment_audit,
    build_alpha_ablation_cases,
    build_audit_cases,
    build_calibration_research_cases,
    default_candidate_gate_fail_reasons,
    fetch_research_api_snapshot,
    full_history_csv_paths,
    gate_fail_reasons,
    passes_research_gate,
    SegmentCalibratedXGBModel,
)


def test_passes_research_gate_requires_all_conditions() -> None:
    """Candidate status requires positive CLV, beat-close rate, and ROI CI."""
    assert passes_research_gate(
        {"clv_mean": 0.01, "beat_close": 0.51, "roi_ci": (0.001, 0.02)}
    )


def test_research_gate_blocks_non_positive_clv() -> None:
    """Non-positive CLV should remain observe-only."""
    assert not passes_research_gate(
        {"clv_mean": 0.0, "beat_close": 0.60, "roi_ci": (0.001, 0.02)}
    )


def test_research_gate_blocks_weak_beat_close() -> None:
    """A strategy must beat close more than half the time."""
    assert not passes_research_gate(
        {"clv_mean": 0.01, "beat_close": 0.50, "roi_ci": (0.001, 0.02)}
    )


def test_research_gate_blocks_non_significant_roi() -> None:
    """ROI confidence interval lower bound must be positive."""
    assert not passes_research_gate(
        {"clv_mean": 0.01, "beat_close": 0.60, "roi_ci": (0.0, 0.02)}
    )


def test_audit_cases_use_opening_odds_for_clv() -> None:
    """All research cases use opening odds so CLV is measured correctly."""
    cases = build_audit_cases()

    assert len(cases) == 60
    assert {case.odds_mode for case in cases} == {"opening"}


def test_gate_fail_reasons_cover_all_gate_types() -> None:
    """Failure reasons should explain every failed main gate."""
    reasons = gate_fail_reasons(
        {"clv_mean": -0.01, "beat_close": 0.49, "roi_ci": (-0.1, 0.2)}
    )

    assert reasons == ["clv_mean<=0", "beat_close<=0.50", "roi_ci_low<=0"]


def test_season_failure_keeps_overall_observe_only() -> None:
    """Any failed season prevents an overall candidate upgrade."""
    overall = pd.DataFrame(
        [
            {
                "model": "m",
                "ev_threshold": 1.05,
                "commission_on_win": 0.0,
                "gate_status": "candidate",
                "gate_fail_reasons": "",
            }
        ]
    )
    by_season = pd.DataFrame(
        [
            {
                "model": "m",
                "ev_threshold": 1.05,
                "commission_on_win": 0.0,
                "season": "s1",
                "clv_mean": 0.01,
                "roi_ci_low": 0.01,
                "beat_close": 0.51,
                "gate_status": "candidate",
            },
            {
                "model": "m",
                "ev_threshold": 1.05,
                "commission_on_win": 0.0,
                "season": "s2",
                "clv_mean": -0.01,
                "roi_ci_low": -0.01,
                "beat_close": 0.40,
                "gate_status": "observe_only",
            },
        ]
    )

    merged = _attach_season_robustness(overall, by_season)

    assert merged.loc[0, "gate_status"] == "observe_only"
    assert merged.loc[0, "season_count"] == 2
    assert "season_gate_failed" in merged.loc[0, "gate_fail_reasons"]


def test_alpha_ablation_cases_use_expected_features_and_opening_odds() -> None:
    """Alpha ablation cases should cover feature groups without closing CLV."""
    cases = build_alpha_ablation_cases()
    names = {case.model_name for case in cases}

    assert names == {
        "alpha_full",
        "alpha_no_rest_congestion",
        "alpha_no_dc",
        "alpha_elo_momentum_only",
        "alpha_full_calibrated",
    }
    assert {case.odds_mode for case in cases} == {"opening"}
    assert all(case.feature_set for case in cases)


def test_api_snapshot_missing_key_fails_closed(monkeypatch, tmp_path) -> None:
    """Missing API credentials should return empty data and write evidence."""

    class MissingKeyClient:
        def __init__(self, db_path: str) -> None:
            raise ValueError("missing key")

    monkeypatch.setattr(rsa, "FootballAPIClient", MissingKeyClient)

    snapshot = fetch_research_api_snapshot(str(tmp_path))

    assert snapshot.empty
    assert (tmp_path / "research_api_snapshot_summary.csv").exists()
    assert "missing key" in (tmp_path / "research_api_snapshot_failure.md").read_text(
        encoding="utf-8"
    )


def test_market_consensus_probs_use_pre_match_columns_only() -> None:
    """Market consensus features must ignore closing-line columns."""
    row = pd.Series(
        {
            "AvgH": 2.0,
            "AvgD": 4.0,
            "AvgA": 4.0,
            "AvgCH": 100.0,
            "AvgCD": 1.01,
            "AvgCA": 1.01,
            "FTR": "A",
        }
    )

    probs = _market_consensus_probs(row)

    assert probs is not None
    assert probs[0] == 0.5
    assert probs[1] == 0.25
    assert probs[2] == 0.25


def test_segment_audit_groups_trade_ledger() -> None:
    """Segment audit should surface structural PnL by season and bet attributes."""
    records = [
        TradeRecord(
            season="s1",
            date="2024-01-01",
            match="A v B",
            home="A",
            away="B",
            model="m",
            odds_mode="opening",
            selection="A",
            odds=4.0,
            closing_odds=3.8,
            ev=1.30,
            p_model=0.325,
            requested_stake=10.0,
            stake=10.0,
            pnl=-10.0,
            capital_before=100.0,
            capital_after=90.0,
            clv=0.05,
            won=False,
            risk_action="accepted",
            execution_status="filled",
        ),
        TradeRecord(
            season="s1",
            date="2024-01-02",
            match="C v D",
            home="C",
            away="D",
            model="m",
            odds_mode="opening",
            selection="H",
            odds=2.0,
            closing_odds=2.1,
            ev=1.15,
            p_model=0.575,
            requested_stake=20.0,
            stake=20.0,
            pnl=20.0,
            capital_before=90.0,
            capital_after=110.0,
            clv=-0.047,
            won=True,
            risk_action="accepted",
            execution_status="filled",
        ),
    ]

    audit = build_strategy_segment_audit(records)

    assert {"season", "selection", "odds_bin", "ev_bin", "p_bin"}.issubset(
        set(audit["group_type"])
    )
    away_row = audit[(audit["group_type"] == "selection") & (audit["group"] == "A")].iloc[0]
    assert away_row["trades"] == 1
    assert away_row["pnl"] == -10.0
    assert away_row["roi_on_stake"] == -1.0


def test_full_history_paths_are_fixed_standardized_six_seasons() -> None:
    """Full-history research must use the standardized six-season data only."""
    paths = full_history_csv_paths()

    assert paths == (
        "data_standardized/api_backtest/data_seasons/E0_1920.csv",
        "data_standardized/api_backtest/data_seasons/E0_2021.csv",
        "data_standardized/api_backtest/data_seasons/E0_2122.csv",
        "data_standardized/api_backtest/data_seasons/E0_2223.csv",
        "data_standardized/api_backtest/data_seasons/E0_2324.csv",
        "data_standardized/api_backtest/data_seasons/E0_2425.csv",
    )


def test_calibration_cases_cover_expected_modes() -> None:
    """Calibration research should keep all variants on opening-line audit."""
    cases = build_calibration_research_cases()

    assert [case.feature_set for case in cases] == [
        "raw",
        "global",
        "selection_bin",
        "odds_bin",
        "selection_odds_bin",
    ]
    assert {case.odds_mode for case in cases} == {"opening"}


def test_segment_calibrator_predict_strips_future_columns() -> None:
    """Calibration predict must not expose result or closing columns to the base model."""

    class GuardBase:
        def predict(self, home, away, date=None, row=None, **kwargs):
            assert row is not None
            assert "FTR" not in row
            assert "B365CH" not in row
            return (0.50, 0.25, 0.25)

        def update(self, home, away, hg, ag, date=None, row=None, **kwargs):
            return None

    model = SegmentCalibratedXGBModel(
        mode="global",
        base_factory=GuardBase,
        min_calibration=1,
    )
    row = pd.Series(
        {
            "B365H": 2.0,
            "B365D": 3.5,
            "B365A": 4.0,
            "B365CH": 1.9,
            "FTR": "H",
        }
    )

    assert model.predict("A", "B", date="2024-01-01", row=row) == (0.5, 0.25, 0.25)
    model.update("A", "B", 1, 0, date="2024-01-01", row=row)
    assert model._history_y["global"] == [0]


def test_default_candidate_gate_blocks_any_main_or_subgate_failure() -> None:
    """Default gate should fail closed on main, season, segment, or stress failure."""
    passing_row = {"clv_mean": 0.01, "beat_close": 0.55, "roi_ci_low": 0.01}
    segment_df = pd.DataFrame(
        [
            {
                "group_type": "selection",
                "group": "A",
                "trades": 30,
                "roi_on_stake": -0.10,
            }
        ]
    )
    by_season = pd.DataFrame(
        [
            {"gate_status": "candidate"},
            {"gate_status": "observe_only"},
        ]
    )

    assert default_candidate_gate_fail_reasons(
        {"clv_mean": -0.01, "beat_close": 0.55, "roi_ci_low": 0.01}
    ) == ["clv_mean<=0"]
    reasons = default_candidate_gate_fail_reasons(
        passing_row,
        segment_df=segment_df,
        by_season_df=by_season,
        stress_metrics={"stress_return_delta": -0.10},
    )

    assert "season_core_gate_failed" in reasons
    assert "segment_loss:selection=A" in reasons
    assert "realistic_execution_degraded" in reasons
