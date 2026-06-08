"""Strategy research audits with CLV, robustness, ablation, and API snapshots."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import quant_core as qc
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from alpha_model import AlphaXGBModel
from api_client import FootballAPIClient
from run_ultimate_backtest import (
    BacktestResult,
    ExecutionConfig,
    StrategyPolicy,
    TradeRecord,
    WalkForwardPoissonModel,
    WalkForwardXGBoostModel,
    run_real_backtest,
)
from wf_models import DixonColesWFModel

CSV_PATHS = (
    "data_seasons/E0_2223.csv",
    "data_seasons/E0_2324.csv",
    "data_seasons/E0_2425.csv",
)
FULL_HISTORY_CSV_PATHS = (
    "data_standardized/api_backtest/data_seasons/E0_1920.csv",
    "data_standardized/api_backtest/data_seasons/E0_2021.csv",
    "data_standardized/api_backtest/data_seasons/E0_2122.csv",
    "data_standardized/api_backtest/data_seasons/E0_2223.csv",
    "data_standardized/api_backtest/data_seasons/E0_2324.csv",
    "data_standardized/api_backtest/data_seasons/E0_2425.csv",
)
EV_THRESHOLDS = (1.02, 1.05, 1.08, 1.10, 1.15)
COMMISSIONS_ON_WIN = (0.0, 0.01, 0.02)
ABLATION_EV_THRESHOLDS = (1.05, 1.15)
ABLATION_COMMISSIONS_ON_WIN = (0.0,)
KELLY_MULT = 0.05
ODDS_MODE = "opening"
RESULT_TO_CLASS = {"A": 0, "D": 1, "H": 2}
CALIBRATION_RESEARCH_MODES = (
    "raw",
    "global",
    "selection_bin",
    "odds_bin",
    "selection_odds_bin",
)
CALIBRATION_MIN_SAMPLES = 40
CALIBRATION_EV_THRESHOLD = 1.05
CALIBRATION_COMMISSION_ON_WIN = 0.0
DEFAULT_GATE_STRESS_CONFIG = ExecutionConfig(
    mode="realistic",
    spread_pct=0.005,
    slippage_pct=0.005,
    price_delay_pct=0.0025,
    rejection_rate=0.02,
    partial_fill_rate=0.80,
    max_stake_fraction=0.02,
    seed=42,
)
MAX_STRESS_RETURN_DEGRADATION = 0.05
CALIBRATION_FORBIDDEN_COLUMNS = {
    "FTHG",
    "FTAG",
    "FTR",
    "HTHG",
    "HTAG",
    "HTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
    "B365CH",
    "B365CD",
    "B365CA",
}

ModelFactory = Callable[..., object]


@dataclass(frozen=True)
class AuditCase:
    """Single research case with fixed opening-line CLV semantics."""

    model_name: str
    model_factory: ModelFactory
    ev_threshold: float
    commission_on_win: float
    kelly_mult: float = KELLY_MULT
    odds_mode: str = ODDS_MODE
    research_group: str = "overall"
    feature_set: str = ""
    strategy_policy: StrategyPolicy | None = None


@dataclass(frozen=True)
class AlphaAblationSpec:
    """In-memory AlphaXGB ablation setup."""

    name: str
    feature_order: tuple[str, ...]
    calibrated: bool = False


@dataclass(frozen=True)
class UpgradePolicySpec:
    """Research-only strategy policy candidate."""

    name: str
    policy: StrategyPolicy


MODEL_FACTORIES: tuple[tuple[str, ModelFactory], ...] = (
    ("walkforward_xgboost", WalkForwardXGBoostModel),
    ("walkforward_poisson", WalkForwardPoissonModel),
    ("dixon_coles_wf", DixonColesWFModel),
    ("alpha_xgb", AlphaXGBModel),
)

ALPHA_ABLATION_SPECS = (
    AlphaAblationSpec(
        "alpha_full",
        ("elo_diff", "mom_diff", "rest_diff", "cong_diff", "dc_p_home", "dc_p_draw"),
    ),
    AlphaAblationSpec(
        "alpha_no_rest_congestion",
        ("elo_diff", "mom_diff", "dc_p_home", "dc_p_draw"),
    ),
    AlphaAblationSpec(
        "alpha_no_dc",
        ("elo_diff", "mom_diff", "rest_diff", "cong_diff"),
    ),
    AlphaAblationSpec("alpha_elo_momentum_only", ("elo_diff", "mom_diff")),
    AlphaAblationSpec(
        "alpha_full_calibrated",
        ("elo_diff", "mom_diff", "rest_diff", "cong_diff", "dc_p_home", "dc_p_draw"),
        calibrated=True,
    ),
)

UPGRADE_POLICY_SPECS = (
    UpgradePolicySpec(
        "no_away",
        StrategyPolicy(name="no_away", allowed_selections=("H", "D")),
    ),
    UpgradePolicySpec(
        "no_away_odds_le_5",
        StrategyPolicy(
            name="no_away_odds_le_5",
            allowed_selections=("H", "D"),
            max_odds=5.0,
        ),
    ),
    UpgradePolicySpec(
        "odds_le_3",
        StrategyPolicy(name="odds_le_3", max_odds=3.0),
    ),
    UpgradePolicySpec(
        "ev_110_120",
        StrategyPolicy(name="ev_110_120", min_ev=1.10, max_ev=1.20),
    ),
    UpgradePolicySpec(
        "high_odds_penalty",
        StrategyPolicy(
            name="high_odds_penalty",
            high_odds_threshold=3.0,
            high_odds_min_ev=1.50,
            high_odds_stake_multiplier=0.35,
        ),
    ),
    UpgradePolicySpec(
        "away_high_bar",
        StrategyPolicy(
            name="away_high_bar",
            selection_min_ev={"A": 1.60},
            selection_stake_multiplier={"A": 0.35},
        ),
    ),
    UpgradePolicySpec(
        "rolling_segment_guard",
        StrategyPolicy(
            name="rolling_segment_guard",
            segment_lookback=30,
            segment_min_trades=10,
            segment_min_roi=0.0,
            segment_stake_multiplier=0.0,
        ),
    ),
)


class AlphaAblationModel(AlphaXGBModel):
    """AlphaXGB variant that keeps all research state in memory."""

    def __init__(
        self,
        feature_order: Sequence[str],
        calibrated: bool = False,
        refit_every: int = 15,
        min_train: int = 80,
        min_calibration: int = 80,
    ) -> None:
        super().__init__(refit_every=refit_every, min_train=min_train)
        self.feature_order = tuple(feature_order)
        self.calibrated = calibrated
        self.min_calibration = min_calibration
        self._calibration_x: list[tuple[float, float, float]] = []
        self._calibration_y: list[int] = []
        self._calibrators: list[IsotonicRegression | None] = [None, None, None]

    def _raw_predict(self, home: str, away: str, date: Any = None) -> tuple[float, float, float] | None:
        if self.model is None:
            return None
        feats = self._make_features(home, away, date)
        x = np.array([[feats[key] for key in self.feature_order]], dtype=float)
        proba = self.model.predict_proba(x)[0]
        cls = list(self.model.classes_)
        p = {c: float(proba[i]) for i, c in enumerate(cls)}
        return (p.get(2, 0.0), p.get(1, 0.0), p.get(0, 0.0))

    def _apply_calibration(
        self, raw: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        if not self.calibrated or not any(self._calibrators):
            return raw
        adjusted = []
        for idx, prob in enumerate(raw):
            calibrator = self._calibrators[idx]
            if calibrator is None:
                adjusted.append(prob)
            else:
                adjusted.append(float(calibrator.predict([prob])[0]))
        arr = np.clip(np.asarray(adjusted, dtype=float), 0.01, 0.99)
        total = float(arr.sum())
        if total <= 0:
            return raw
        return tuple(float(value / total) for value in arr)

    def _fit_calibrators(self) -> None:
        if len(self._calibration_y) < self.min_calibration:
            return
        x = np.asarray(self._calibration_x, dtype=float)
        y = np.asarray(self._calibration_y, dtype=int)
        calibrators: list[IsotonicRegression | None] = []
        for class_idx in (2, 1, 0):
            target = (y == class_idx).astype(int)
            if len(np.unique(target)) < 2:
                calibrators.append(None)
                continue
            model = IsotonicRegression(out_of_bounds="clip")
            source_col = {2: 0, 1: 1, 0: 2}[class_idx]
            model.fit(x[:, source_col], target)
            calibrators.append(model)
        self._calibrators = calibrators

    def predict(
        self, home: str, away: str, date: Any = None, row: Any = None, **kwargs: Any
    ) -> tuple[float, float, float] | None:
        """Predict 1X2 probabilities using the selected in-memory feature set."""
        raw = self._raw_predict(home, away, date)
        if raw is None:
            return None
        return self._apply_calibration(raw)

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
        """Update walk-forward state after settlement without using future data."""
        raw_before_update = self._raw_predict(home, away, date) if self.calibrated else None
        feats = self._make_features(home, away, date)
        ftr = "H" if hg > ag else ("D" if hg == ag else "A")
        self._X.append([feats[key] for key in self.feature_order])
        self._y.append(RESULT_TO_CLASS[ftr])
        self._since_fit += 1

        if len(self._y) >= self.min_train and self._since_fit >= self.refit_every:
            self._fit()
            self._since_fit = 0

        self.dc.update(home, away, hg, ag)
        exp_h = qc.elo_expected(self.elo[home], self.elo[away])
        s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        self.elo[home] = qc.elo_update(self.elo[home], s_h, exp_h, 40.0)
        self.elo[away] = qc.elo_update(self.elo[away], 1 - s_h, 1 - exp_h, 40.0)
        if row is not None:
            try:
                self.pxg[home].append(qc.proxy_xg(float(row["HS"]), float(row["HST"])))
                self.pxg[away].append(qc.proxy_xg(float(row["AS"]), float(row["AST"])))
            except (KeyError, TypeError, ValueError):
                pass
        if date is not None:
            self.last_dates[home].append(date)
            self.last_dates[away].append(date)

        if raw_before_update is not None:
            self._calibration_x.append(raw_before_update)
            self._calibration_y.append(RESULT_TO_CLASS[ftr])
            self._fit_calibrators()

    def _fit(self) -> None:
        x = np.asarray(self._X, dtype=float)
        y = np.asarray(self._y, dtype=int)
        if len(np.unique(y)) < 2:
            return
        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            learning_rate=0.05,
            max_depth=3,
            n_estimators=120,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        self.model.fit(x, y)


class CalibratedConsensusModel:
    """Walk-forward probability calibrator with pre-match market consensus blend."""

    def __init__(
        self,
        base_factory: ModelFactory = WalkForwardXGBoostModel,
        min_calibration: int = 80,
        market_blend: float = 0.25,
    ) -> None:
        self.base = base_factory()
        self.min_calibration = min_calibration
        self.market_blend = market_blend
        self._calibration_x: list[tuple[float, float, float]] = []
        self._calibration_y: list[int] = []
        self._calibrators: list[IsotonicRegression | None] = [None, None, None]
        self._pending_raw: dict[tuple[str, str, str], tuple[float, float, float]] = {}

    def predict(
        self,
        home: str,
        away: str,
        date: Any = None,
        row: Any = None,
        **kwargs: Any,
    ) -> tuple[float, float, float] | None:
        """Predict from base model, historical calibration, and pre-match consensus."""
        raw = self.base.predict(home, away, date=date, row=row, **kwargs)
        if raw is None:
            return None
        raw_tuple = tuple(float(value) for value in raw)
        self._pending_raw[_match_key(home, away, date)] = raw_tuple
        calibrated = self._apply_calibration(raw_tuple)
        consensus = _market_consensus_probs(row)
        if consensus is None or self.market_blend <= 0:
            return calibrated
        blended = (
            (1.0 - self.market_blend) * np.asarray(calibrated, dtype=float)
            + self.market_blend * np.asarray(consensus, dtype=float)
        )
        return _normalize_probs(blended)

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
        """Update calibration after settlement, then update the wrapped model."""
        raw = self._pending_raw.pop(_match_key(home, away, date), None)
        if raw is not None:
            outcome = 0 if hg > ag else (1 if hg == ag else 2)
            self._calibration_x.append(raw)
            self._calibration_y.append(outcome)
            self._fit_calibrators()
        self.base.update(home, away, hg, ag, date=date, row=row, **kwargs)

    def _apply_calibration(
        self, raw: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        if not any(self._calibrators):
            return raw
        adjusted = []
        for idx, prob in enumerate(raw):
            calibrator = self._calibrators[idx]
            adjusted.append(prob if calibrator is None else float(calibrator.predict([prob])[0]))
        return _normalize_probs(np.asarray(adjusted, dtype=float))

    def _fit_calibrators(self) -> None:
        if len(self._calibration_y) < self.min_calibration:
            return
        x = np.asarray(self._calibration_x, dtype=float)
        y = np.asarray(self._calibration_y, dtype=int)
        calibrators: list[IsotonicRegression | None] = []
        for class_idx in (0, 1, 2):
            target = (y == class_idx).astype(int)
            if len(np.unique(target)) < 2:
                calibrators.append(None)
                continue
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(x[:, class_idx], target)
            calibrators.append(model)
        self._calibrators = calibrators


class SegmentCalibratedXGBModel:
    """Research-only walk-forward XGB probability calibrator by market segment."""

    def __init__(
        self,
        mode: str = "raw",
        base_factory: ModelFactory = WalkForwardXGBoostModel,
        min_calibration: int = CALIBRATION_MIN_SAMPLES,
    ) -> None:
        if mode not in CALIBRATION_RESEARCH_MODES:
            raise ValueError(f"Unknown calibration mode: {mode}")
        self.mode = mode
        self.base = base_factory()
        self.min_calibration = min_calibration
        self._pending: dict[tuple[str, str, str], tuple[tuple[float, float, float], str]] = {}
        self._history_x: dict[str, list[tuple[float, float, float]]] = {}
        self._history_y: dict[str, list[int]] = {}
        self._calibrators: dict[str, list[IsotonicRegression | None]] = {}

    def predict(
        self,
        home: str,
        away: str,
        date: Any = None,
        row: Any = None,
        **kwargs: Any,
    ) -> tuple[float, float, float] | None:
        """Predict with calibration fitted only on previously settled matches."""
        safe_row = _strip_calibration_future_columns(row)
        raw = self.base.predict(home, away, date=date, row=safe_row, **kwargs)
        if raw is None:
            return None
        raw_tuple = _normalize_probs(np.asarray(raw, dtype=float))
        segment_key = self._segment_key(raw_tuple, safe_row)
        self._pending[_match_key(home, away, date)] = (raw_tuple, segment_key)
        if self.mode == "raw":
            return raw_tuple
        return self._apply_segment_calibration(raw_tuple, segment_key)

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
        """Add the previous prediction to calibration history after settlement."""
        pending = self._pending.pop(_match_key(home, away, date), None)
        if pending is not None and self.mode != "raw":
            raw, segment_key = pending
            outcome = _backtest_outcome_index(hg, ag)
            self._history_x.setdefault(segment_key, []).append(raw)
            self._history_y.setdefault(segment_key, []).append(outcome)
            self._fit_segment(segment_key)
        self.base.update(home, away, hg, ag, date=date, row=row, **kwargs)

    def _segment_key(
        self,
        raw: tuple[float, float, float],
        row: Any,
    ) -> str:
        selection = ("H", "D", "A")[int(np.argmax(raw))]
        odds = _decision_selection_odds(row, selection)
        odds_bin = _segment_odds_bin(odds if odds is not None else float("nan"))
        if self.mode == "global":
            return "global"
        if self.mode == "selection_bin":
            return f"selection={selection}"
        if self.mode == "odds_bin":
            return f"odds={odds_bin}"
        if self.mode == "selection_odds_bin":
            return f"selection={selection}|odds={odds_bin}"
        return "raw"

    def _fit_segment(self, segment_key: str) -> None:
        x_rows = self._history_x.get(segment_key, [])
        y_rows = self._history_y.get(segment_key, [])
        if len(y_rows) < self.min_calibration:
            return
        x = np.asarray(x_rows, dtype=float)
        y = np.asarray(y_rows, dtype=int)
        calibrators: list[IsotonicRegression | None] = []
        for class_idx in (0, 1, 2):
            target = (y == class_idx).astype(int)
            if len(np.unique(target)) < 2:
                calibrators.append(None)
                continue
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(x[:, class_idx], target)
            calibrators.append(model)
        self._calibrators[segment_key] = calibrators

    def _apply_segment_calibration(
        self,
        raw: tuple[float, float, float],
        segment_key: str,
    ) -> tuple[float, float, float]:
        calibrators = self._calibrators.get(segment_key)
        if not calibrators or not any(calibrators):
            return raw
        adjusted = []
        for idx, prob in enumerate(raw):
            calibrator = calibrators[idx]
            adjusted.append(prob if calibrator is None else float(calibrator.predict([prob])[0]))
        return _normalize_probs(np.asarray(adjusted, dtype=float))


def _match_key(home: str, away: str, date: Any) -> tuple[str, str, str]:
    return (home, away, str(date))


def _strip_calibration_future_columns(row: Any) -> Any:
    if row is None or not hasattr(row, "drop"):
        return row
    return row.drop(labels=[col for col in CALIBRATION_FORBIDDEN_COLUMNS if col in row.index])


def _decision_selection_odds(row: Any, selection: str) -> float | None:
    if row is None:
        return None
    column = {"H": "B365H", "D": "B365D", "A": "B365A"}[selection]
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value > 1.0 else None


def _backtest_outcome_index(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def _normalize_probs(values: np.ndarray) -> tuple[float, float, float]:
    clipped = np.clip(values.astype(float), 0.01, 0.99)
    total = float(clipped.sum())
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    normalized = clipped / total
    return (float(normalized[0]), float(normalized[1]), float(normalized[2]))


def _market_consensus_probs(row: Any) -> tuple[float, float, float] | None:
    """Build H/D/A consensus probabilities from pre-match bookmaker columns only."""
    if row is None:
        return None
    book_sets = (
        ("AvgH", "AvgD", "AvgA"),
        ("MaxH", "MaxD", "MaxA"),
        ("PSH", "PSD", "PSA"),
        ("B365H", "B365D", "B365A"),
    )
    probs = []
    for columns in book_sets:
        odds = []
        for col in columns:
            try:
                value = float(row[col])
            except (KeyError, TypeError, ValueError):
                odds = []
                break
            if not np.isfinite(value) or value <= 1.0:
                odds = []
                break
            odds.append(value)
        if len(odds) == 3:
            probs.append(qc.remove_margin(odds))
    if not probs:
        return None
    return _normalize_probs(np.asarray(probs, dtype=float).mean(axis=0))


def build_audit_cases() -> list[AuditCase]:
    """Build the full model/threshold/cost grid for sample-out research."""
    return [
        AuditCase(model_name, factory, threshold, commission)
        for model_name, factory in MODEL_FACTORIES
        for threshold in EV_THRESHOLDS
        for commission in COMMISSIONS_ON_WIN
    ]


def _make_alpha_ablation_factory(spec: AlphaAblationSpec) -> ModelFactory:
    def factory() -> AlphaAblationModel:
        return AlphaAblationModel(
            feature_order=spec.feature_order,
            calibrated=spec.calibrated,
        )

    return factory


def build_alpha_ablation_cases() -> list[AuditCase]:
    """Build representative AlphaXGB ablation cases using opening-line CLV."""
    cases = []
    for spec in ALPHA_ABLATION_SPECS:
        for threshold in ABLATION_EV_THRESHOLDS:
            for commission in ABLATION_COMMISSIONS_ON_WIN:
                cases.append(
                    AuditCase(
                        model_name=spec.name,
                        model_factory=_make_alpha_ablation_factory(spec),
                        ev_threshold=threshold,
                        commission_on_win=commission,
                        research_group="alpha_ablation",
                        feature_set=",".join(spec.feature_order),
                    )
                )
    return cases


def _make_consensus_calibrated_factory() -> CalibratedConsensusModel:
    return CalibratedConsensusModel(
        base_factory=WalkForwardXGBoostModel,
        min_calibration=80,
        market_blend=0.25,
    )


def build_strategy_upgrade_cases() -> list[AuditCase]:
    """Build research-only strategy policy cases without changing defaults."""
    cases: list[AuditCase] = []
    model_specs: tuple[tuple[str, ModelFactory], ...] = (
        ("walkforward_xgboost", WalkForwardXGBoostModel),
        ("xgb_calibrated_consensus", _make_consensus_calibrated_factory),
    )
    for model_name, factory in model_specs:
        for spec in UPGRADE_POLICY_SPECS:
            cases.append(
                AuditCase(
                    model_name=model_name,
                    model_factory=factory,
                    ev_threshold=1.05,
                    commission_on_win=0.0,
                    odds_mode=ODDS_MODE,
                    research_group="strategy_upgrade",
                    feature_set="policy_research",
                    strategy_policy=spec.policy,
                )
            )
    return cases


def full_history_csv_paths() -> tuple[str, ...]:
    """Return the fixed six-season standardized research sample paths."""
    return FULL_HISTORY_CSV_PATHS


def _make_segment_calibrated_factory(mode: str) -> ModelFactory:
    def factory() -> SegmentCalibratedXGBModel:
        return SegmentCalibratedXGBModel(
            mode=mode,
            base_factory=WalkForwardXGBoostModel,
            min_calibration=CALIBRATION_MIN_SAMPLES,
        )

    return factory


def build_calibration_research_cases() -> list[AuditCase]:
    """Build research-only probability calibration experiments."""
    cases: list[AuditCase] = []
    for mode in CALIBRATION_RESEARCH_MODES:
        factory: ModelFactory
        if mode == "raw":
            factory = WalkForwardXGBoostModel
        else:
            factory = _make_segment_calibrated_factory(mode)
        cases.append(
            AuditCase(
                model_name=f"xgb_calibration_{mode}",
                model_factory=factory,
                ev_threshold=CALIBRATION_EV_THRESHOLD,
                commission_on_win=CALIBRATION_COMMISSION_ON_WIN,
                odds_mode=ODDS_MODE,
                research_group="calibration_research",
                feature_set=mode,
            )
        )
    return cases


def gate_fail_reasons(metrics: dict[str, object]) -> list[str]:
    """Explain which main research gates failed."""
    reasons = []
    clv_mean = float(metrics.get("clv_mean", 0.0))
    beat_close = float(metrics.get("beat_close", 0.0))
    roi_ci_low = _roi_ci_low(metrics)
    if not np.isfinite(clv_mean) or clv_mean <= 0.0:
        reasons.append("clv_mean<=0")
    if not np.isfinite(beat_close) or beat_close <= 0.50:
        reasons.append("beat_close<=0.50")
    if not np.isfinite(roi_ci_low) or roi_ci_low <= 0.0:
        reasons.append("roi_ci_low<=0")
    return reasons


def passed_gate_count(metrics: dict[str, object]) -> int:
    """Count passed main gates out of CLV, beat-close, and ROI significance."""
    return 3 - len(gate_fail_reasons(metrics))


def passes_research_gate(metrics: dict[str, object]) -> bool:
    """Return True only when CLV and ROI significance gates all pass."""
    return len(gate_fail_reasons(metrics)) == 0


def _roi_ci_low(metrics: dict[str, object]) -> float:
    roi_ci = metrics.get("roi_ci")
    if isinstance(roi_ci, (tuple, list)) and len(roi_ci) >= 2:
        return float(roi_ci[0])
    return float(metrics.get("roi_ci_low", 0.0))


def _metrics_to_row(case: AuditCase, metrics: dict[str, Any]) -> dict[str, Any]:
    roi_ci = metrics.get("roi_ci", (float("nan"), float("nan")))
    roi_ci_low, roi_ci_high = roi_ci
    block_roi_ci = metrics.get("block_roi_ci", (float("nan"), float("nan")))
    block_roi_ci_low, block_roi_ci_high = block_roi_ci
    reasons = gate_fail_reasons(metrics)
    return {
        "model": case.model_name,
        "policy": case.strategy_policy.name if case.strategy_policy else "",
        "research_group": case.research_group,
        "feature_set": case.feature_set,
        "ev_threshold": case.ev_threshold,
        "commission_on_win": case.commission_on_win,
        "kelly_mult": case.kelly_mult,
        "odds_mode": case.odds_mode,
        "gate_status": "candidate" if not reasons else "observe_only",
        "gate_fail_reasons": ";".join(reasons),
        "passed_gate_count": 3 - len(reasons),
        "total_matches": metrics["total_matches"],
        "trades": metrics["trades"],
        "win_rate": metrics["win_rate"],
        "total_return": metrics["total_return"],
        "max_drawdown": metrics["max_drawdown"],
        "final_capital": metrics["final_capital"],
        "per_bet_sharpe": metrics["per_bet_sharpe"],
        "sortino": metrics["sortino"],
        "calmar": metrics["calmar"],
        "profit_factor": metrics["profit_factor"],
        "max_losing_streak": metrics["max_losing_streak"],
        "roi_mean": metrics["roi_mean"],
        "roi_ci_low": roi_ci_low,
        "roi_ci_high": roi_ci_high,
        "block_roi_mean": metrics.get("block_roi_mean", float("nan")),
        "block_roi_ci_low": block_roi_ci_low,
        "block_roi_ci_high": block_roi_ci_high,
        "brier": metrics["brier"],
        "brier_base": metrics["brier_base"],
        "log_loss": metrics["log_loss"],
        "clv_mean": metrics["clv_mean"],
        "beat_close": metrics["beat_close"],
        "sample_size": metrics.get("sample_size", metrics["trades"]),
        "season_count": metrics.get("season_count", 0),
        "p_value": metrics.get("p_value", float("nan")),
        "adjusted_p_value": metrics.get("adjusted_p_value", float("nan")),
        "deflated_sharpe": metrics.get("deflated_sharpe", float("nan")),
        "robustness_status": metrics.get("robustness_status", "observe_only"),
    }


def _run_case(case: AuditCase, csv_paths: Sequence[str]) -> dict[str, Any]:
    metrics = run_real_backtest(
        csv_paths=tuple(csv_paths),
        model_factory=case.model_factory,
        ev_threshold=case.ev_threshold,
        commission_on_win=case.commission_on_win,
        kelly_mult=case.kelly_mult,
        odds_mode=case.odds_mode,
        verbose=False,
        strategy_policy=case.strategy_policy,
    )
    return _metrics_to_row(case, metrics)


def _run_case_result(case: AuditCase, csv_paths: Sequence[str]) -> BacktestResult:
    """Run one research case and return the typed backtest result."""
    result = run_real_backtest(
        csv_paths=tuple(csv_paths),
        model_factory=case.model_factory,
        ev_threshold=case.ev_threshold,
        commission_on_win=case.commission_on_win,
        kelly_mult=case.kelly_mult,
        odds_mode=case.odds_mode,
        verbose=False,
        strategy_policy=case.strategy_policy,
        return_result=True,
    )
    if not isinstance(result, BacktestResult):
        raise TypeError("run_real_backtest(return_result=True) did not return BacktestResult")
    return result


def build_strategy_segment_audit(
    trade_records: Sequence[TradeRecord | dict[str, Any]],
) -> pd.DataFrame:
    """Aggregate settled trades by selection, odds, EV, probability, and season.

    Only filled or partially filled trades with positive stake are included; blocked
    attempts remain execution evidence but are not treated as realized segment PnL.
    """
    rows = [
        item if isinstance(item, dict) else item.__dict__
        for item in trade_records
    ]
    columns = [
        "group_type",
        "group",
        "trades",
        "wins",
        "stake",
        "pnl",
        "roi_on_stake",
        "clv_mean",
        "beat_close",
        "avg_odds",
        "avg_ev",
        "avg_p_model",
        "structural_loss",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    if "stake" not in df:
        return pd.DataFrame(columns=columns)
    df = df[pd.to_numeric(df["stake"], errors="coerce").fillna(0.0) > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    df.loc[:, "won_int"] = df["won"].astype(int)
    df.loc[:, "beat_close_int"] = pd.to_numeric(df["clv"], errors="coerce").fillna(0.0) > 0
    df.loc[:, "odds_bin"] = pd.to_numeric(df["odds"], errors="coerce").map(_segment_odds_bin)
    df.loc[:, "ev_bin"] = pd.to_numeric(df["ev"], errors="coerce").map(_segment_ev_bin)
    df.loc[:, "p_bin"] = pd.to_numeric(df["p_model"], errors="coerce").map(_segment_p_bin)

    audit_frames = []
    for group_type, group_col in (
        ("selection", "selection"),
        ("odds_bin", "odds_bin"),
        ("ev_bin", "ev_bin"),
        ("p_bin", "p_bin"),
        ("season", "season"),
    ):
        grouped = df.groupby(group_col, dropna=False).agg(
            trades=("stake", "count"),
            wins=("won_int", "sum"),
            stake=("stake", "sum"),
            pnl=("pnl", "sum"),
            clv_mean=("clv", "mean"),
            beat_close=("beat_close_int", "mean"),
            avg_odds=("odds", "mean"),
            avg_ev=("ev", "mean"),
            avg_p_model=("p_model", "mean"),
        )
        grouped = grouped.reset_index().rename(columns={group_col: "group"})
        grouped.insert(0, "group_type", group_type)
        audit_frames.append(grouped)

    audit = pd.concat(audit_frames, ignore_index=True)
    audit.loc[:, "roi_on_stake"] = audit["pnl"] / audit["stake"].replace(0.0, np.nan)
    audit.loc[:, "structural_loss"] = (
        (audit["trades"] >= 30) & (audit["roi_on_stake"] < 0.0)
    )
    ordered = audit[columns].sort_values(
        by=["structural_loss", "pnl", "trades"],
        ascending=[False, True, False],
    )
    return ordered.reset_index(drop=True)


def run_strategy_segment_audit(output_dir: str = "reports") -> pd.DataFrame:
    """Run the default opening-line backtest and write segment attribution reports."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case = AuditCase(
        "walkforward_xgboost",
        WalkForwardXGBoostModel,
        1.05,
        0.0,
        odds_mode=ODDS_MODE,
        research_group="segment_audit",
    )
    result = _run_case_result(case, CSV_PATHS)
    audit = build_strategy_segment_audit(result.trade_records)
    audit.to_csv(out_dir / "strategy_segment_audit.csv", index=False)
    _write_segment_markdown(audit, out_dir / "strategy_segment_audit.md")
    return audit


def run_strategy_upgrade_audit(output_dir: str = "reports") -> pd.DataFrame:
    """Evaluate research-only strategy policies without changing the default strategy."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    segment_rows: list[pd.DataFrame] = []

    for case in build_strategy_upgrade_cases():
        result = _run_case_result(case, CSV_PATHS)
        row = _metrics_to_row(case, result.metrics)
        segment_audit = build_strategy_segment_audit(result.trade_records)
        segment_reasons = _segment_gate_fail_reasons(segment_audit)
        if segment_reasons:
            row["gate_status"] = "observe_only"
            row["gate_fail_reasons"] = _append_reasons(
                row["gate_fail_reasons"], segment_reasons
            )
        row["segment_structural_loss_count"] = len(segment_reasons)
        row["candidate_after_segment_gate"] = row["gate_status"] == "candidate"
        rows.append(row)

        if not segment_audit.empty:
            segment_audit = segment_audit.copy()
            segment_audit.insert(0, "model", case.model_name)
            segment_audit.insert(1, "policy", case.strategy_policy.name if case.strategy_policy else "")
            segment_rows.append(segment_audit)

    df = _apply_statistical_adjustment(pd.DataFrame(rows))
    df.to_csv(out_dir / "strategy_upgrade_audit.csv", index=False)
    segments = pd.concat(segment_rows, ignore_index=True) if segment_rows else pd.DataFrame()
    segments.to_csv(out_dir / "strategy_upgrade_segments.csv", index=False)
    _write_upgrade_markdown(df, segments, out_dir / "strategy_upgrade_audit.md")
    return df


def run_full_history_research_audit(output_dir: str = "reports") -> pd.DataFrame:
    """Run the overall research grid on the fixed six-season standardized sample."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_run_case(case, FULL_HISTORY_CSV_PATHS) for case in build_audit_cases()]
    df = _apply_statistical_adjustment(pd.DataFrame(rows))
    df.to_csv(out_dir / "full_history_research_grid.csv", index=False)

    segment_case = AuditCase(
        "walkforward_xgboost",
        WalkForwardXGBoostModel,
        1.05,
        0.0,
        odds_mode=ODDS_MODE,
        research_group="full_history_segment_audit",
    )
    segment_result = _run_case_result(segment_case, FULL_HISTORY_CSV_PATHS)
    segment_df = build_strategy_segment_audit(segment_result.trade_records)
    segment_df.to_csv(out_dir / "full_history_segment_audit.csv", index=False)
    _write_full_history_markdown(df, segment_df, out_dir / "full_history_research_audit.md")
    return df


def run_calibration_research_audit(output_dir: str = "reports") -> pd.DataFrame:
    """Run research-only XGB probability calibration experiments."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    segment_frames: list[pd.DataFrame] = []
    for case in build_calibration_research_cases():
        result = _run_case_result(case, FULL_HISTORY_CSV_PATHS)
        row = _metrics_to_row(case, result.metrics)
        rows.append(row)
        segment_df = build_strategy_segment_audit(result.trade_records)
        if not segment_df.empty:
            segment_df = segment_df.copy()
            segment_df.insert(0, "model", case.model_name)
            segment_df.insert(1, "feature_set", case.feature_set)
            segment_frames.append(segment_df)
    df = _apply_statistical_adjustment(pd.DataFrame(rows))
    segments = pd.concat(segment_frames, ignore_index=True) if segment_frames else pd.DataFrame()
    df.to_csv(out_dir / "calibration_research_grid.csv", index=False)
    segments.to_csv(out_dir / "calibration_research_segments.csv", index=False)
    _write_calibration_markdown(df, segments, out_dir / "calibration_research_audit.md")
    return df


def run_default_candidate_gate(output_dir: str = "reports") -> pd.DataFrame:
    """Evaluate whether any research row is eligible to become a default candidate."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / "full_history_research_grid.csv"
    calibration_path = out_dir / "calibration_research_grid.csv"
    full_df = pd.read_csv(full_path) if full_path.exists() else run_full_history_research_audit(output_dir)
    calibration_df = (
        pd.read_csv(calibration_path)
        if calibration_path.exists()
        else run_calibration_research_audit(output_dir)
    )
    source_rows = []
    for source_name, df in (
        ("full_history", full_df),
        ("calibration", calibration_df),
    ):
        if df.empty:
            continue
        tagged = df.copy()
        tagged.insert(0, "candidate_source", source_name)
        source_rows.append(tagged)
    combined = pd.concat(source_rows, ignore_index=True) if source_rows else pd.DataFrame()
    case_map = _default_gate_case_map()
    gate_rows: list[dict[str, Any]] = []

    for _, row in combined.iterrows():
        payload = row.to_dict()
        stress_metrics: dict[str, Any] | None = None
        season_df: pd.DataFrame | None = None
        segment_df: pd.DataFrame | None = None
        reasons = default_candidate_gate_fail_reasons(payload)
        case = case_map.get(_case_key_from_row(payload))
        if not reasons and case is not None:
            season_df = pd.DataFrame([_run_case(case, (path,)) for path in FULL_HISTORY_CSV_PATHS])
            result = _run_case_result(case, FULL_HISTORY_CSV_PATHS)
            segment_df = build_strategy_segment_audit(result.trade_records)
            stress_metrics = _run_candidate_stress(case, float(payload["total_return"]))
            reasons = default_candidate_gate_fail_reasons(
                payload,
                segment_df=segment_df,
                by_season_df=season_df,
                stress_metrics=stress_metrics,
            )
        payload["default_gate_status"] = "candidate" if not reasons else "observe_only"
        payload["default_gate_fail_reasons"] = ";".join(reasons)
        payload["season_fail_count"] = _season_fail_count(season_df)
        payload["segment_structural_loss_count"] = (
            len(_segment_gate_fail_reasons(segment_df)) if segment_df is not None else 0
        )
        if stress_metrics:
            payload.update(stress_metrics)
        else:
            payload.update(
                {
                    "stress_total_return": float("nan"),
                    "stress_return_delta": float("nan"),
                    "stress_per_bet_sharpe": float("nan"),
                    "stress_status": "not_run",
                }
            )
        gate_rows.append(payload)

    gate_df = pd.DataFrame(gate_rows)
    if not gate_df.empty:
        gate_df = gate_df.sort_values(
            by=["default_gate_status", "passed_gate_count", "roi_ci_low", "clv_mean"],
            ascending=[True, False, False, False],
        )
    gate_df.to_csv(out_dir / "default_candidate_gate.csv", index=False)
    _write_default_candidate_gate_markdown(gate_df, out_dir / "default_candidate_gate.md")
    return gate_df


def default_candidate_gate_fail_reasons(
    row: dict[str, Any] | pd.Series,
    segment_df: pd.DataFrame | None = None,
    by_season_df: pd.DataFrame | None = None,
    stress_metrics: dict[str, Any] | None = None,
) -> list[str]:
    """Return all default-upgrade gate failures for one candidate row."""
    row_dict = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    reasons = gate_fail_reasons(row_dict)
    if by_season_df is not None and not by_season_df.empty:
        failed = by_season_df[by_season_df["gate_status"] != "candidate"]
        if len(by_season_df) < len(FULL_HISTORY_CSV_PATHS) or not failed.empty:
            reasons.append("season_core_gate_failed")
    if segment_df is not None:
        reasons.extend(_segment_gate_fail_reasons(segment_df))
    if stress_metrics is not None:
        delta = float(stress_metrics.get("stress_return_delta", float("nan")))
        if not np.isfinite(delta) or delta < -MAX_STRESS_RETURN_DEGRADATION:
            reasons.append("realistic_execution_degraded")
    return _unique_reasons(reasons)


def _default_gate_case_map() -> dict[tuple[str, str, str, float, float, str], AuditCase]:
    cases = build_audit_cases() + build_calibration_research_cases()
    return {_case_key_from_case(case): case for case in cases}


def _case_key_from_case(case: AuditCase) -> tuple[str, str, str, float, float, str]:
    return (
        case.model_name,
        case.research_group,
        case.feature_set,
        round(float(case.ev_threshold), 8),
        round(float(case.commission_on_win), 8),
        case.strategy_policy.name if case.strategy_policy else "",
    )


def _case_key_from_row(row: dict[str, Any]) -> tuple[str, str, str, float, float, str]:
    return (
        str(row.get("model", "")),
        str(row.get("research_group", "")),
        str(row.get("feature_set", "")),
        round(float(row.get("ev_threshold", 0.0)), 8),
        round(float(row.get("commission_on_win", 0.0)), 8),
        str(row.get("policy", "")),
    )


def _run_candidate_stress(case: AuditCase, base_return: float) -> dict[str, Any]:
    result = run_real_backtest(
        csv_paths=FULL_HISTORY_CSV_PATHS,
        model_factory=case.model_factory,
        ev_threshold=case.ev_threshold,
        commission_on_win=case.commission_on_win,
        kelly_mult=case.kelly_mult,
        odds_mode=case.odds_mode,
        verbose=False,
        strategy_policy=case.strategy_policy,
        execution_config=DEFAULT_GATE_STRESS_CONFIG,
        return_result=True,
    )
    if not isinstance(result, BacktestResult):
        raise TypeError("run_real_backtest(return_result=True) did not return BacktestResult")
    stress_return = float(result.metrics["total_return"])
    delta = stress_return - float(base_return)
    return {
        "stress_total_return": stress_return,
        "stress_return_delta": delta,
        "stress_per_bet_sharpe": float(result.metrics["per_bet_sharpe"]),
        "stress_status": (
            "passed"
            if np.isfinite(delta) and delta >= -MAX_STRESS_RETURN_DEGRADATION
            else "failed"
        ),
    }


def _season_fail_count(by_season_df: pd.DataFrame | None) -> int:
    if by_season_df is None or by_season_df.empty:
        return 0
    return int((by_season_df["gate_status"] != "candidate").sum())


def _unique_reasons(reasons: Sequence[str]) -> list[str]:
    unique = []
    for reason in reasons:
        if reason and reason not in unique:
            unique.append(reason)
    return unique


def _attach_season_robustness(
    overall_df: pd.DataFrame, by_season_df: pd.DataFrame
) -> pd.DataFrame:
    keys = ["model", "ev_threshold", "commission_on_win"]
    base_df = overall_df.drop(
        columns=[
            "season_count",
            "min_season_clv",
            "min_season_roi_ci_low",
            "min_season_beat_close",
            "season_candidate_count",
        ],
        errors="ignore",
    )
    grouped = by_season_df.groupby(keys, dropna=False).agg(
        season_count=("season", "nunique"),
        min_season_clv=("clv_mean", "min"),
        min_season_roi_ci_low=("roi_ci_low", "min"),
        min_season_beat_close=("beat_close", "min"),
        season_candidate_count=("gate_status", lambda values: int((values == "candidate").sum())),
    )
    merged = base_df.merge(grouped.reset_index(), on=keys, how="left")
    for col in [
        "season_count",
        "min_season_clv",
        "min_season_roi_ci_low",
        "min_season_beat_close",
        "season_candidate_count",
    ]:
        if col not in merged:
            merged[col] = 0
    season_failed = merged["season_candidate_count"] < merged["season_count"]
    merged.loc[season_failed, "gate_fail_reasons"] = merged.loc[
        season_failed, "gate_fail_reasons"
    ].map(lambda value: _append_reason(value, "season_gate_failed"))
    merged.loc[season_failed, "gate_status"] = "observe_only"
    return merged


def _append_reason(existing: str, reason: str) -> str:
    reasons = [item for item in str(existing).split(";") if item]
    if reason not in reasons:
        reasons.append(reason)
    return ";".join(reasons)


def _append_reasons(existing: str, new_reasons: Sequence[str]) -> str:
    result = str(existing)
    for reason in new_reasons:
        result = _append_reason(result, reason)
    return result


def _segment_gate_fail_reasons(segment_df: pd.DataFrame) -> list[str]:
    """Return segment-level structural-loss gate failures for material groups."""
    if segment_df.empty:
        return []
    failures = segment_df[
        (segment_df["group_type"].isin(["selection", "odds_bin", "ev_bin", "p_bin"]))
        & (pd.to_numeric(segment_df["trades"], errors="coerce") >= 30)
        & (pd.to_numeric(segment_df["roi_on_stake"], errors="coerce") < 0.0)
    ]
    return [
        f"segment_loss:{row.group_type}={row.group}"
        for row in failures.itertuples()
    ]


def _segment_odds_bin(odds: float) -> str:
    if not np.isfinite(odds):
        return "missing"
    if odds <= 1.5:
        return "1.00-1.50"
    if odds <= 2.0:
        return "1.50-2.00"
    if odds <= 3.0:
        return "2.00-3.00"
    if odds <= 5.0:
        return "3.00-5.00"
    if odds <= 10.0:
        return "5.00-10.00"
    return "10.00+"


def _segment_ev_bin(ev: float) -> str:
    if not np.isfinite(ev):
        return "missing"
    if ev <= 1.10:
        return "1.00-1.10"
    if ev <= 1.20:
        return "1.10-1.20"
    if ev <= 1.50:
        return "1.20-1.50"
    return "1.50+"


def _segment_p_bin(probability: float) -> str:
    if not np.isfinite(probability):
        return "missing"
    if probability <= 0.20:
        return "0.00-0.20"
    if probability <= 0.35:
        return "0.20-0.35"
    if probability <= 0.50:
        return "0.35-0.50"
    if probability <= 0.65:
        return "0.50-0.65"
    if probability <= 0.80:
        return "0.65-0.80"
    return "0.80-1.00"


def _apply_statistical_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """Apply grid-level multiple-testing adjustment without changing gates."""
    if df.empty or "p_value" not in df:
        return df
    adjusted = df.copy()
    n_trials = max(len(adjusted), 1)
    adjusted.loc[:, "adjusted_p_value"] = (
        pd.to_numeric(adjusted["p_value"], errors="coerce") * n_trials
    ).clip(upper=1.0)
    stat_pass = (
        (adjusted["gate_status"] == "candidate")
        & (pd.to_numeric(adjusted["adjusted_p_value"], errors="coerce") <= 0.05)
        & (pd.to_numeric(adjusted["deflated_sharpe"], errors="coerce") > 0.0)
    )
    adjusted.loc[:, "robustness_status"] = np.where(
        stat_pass,
        "statistically_robust",
        "observe_only",
    )
    return adjusted


def run_strategy_research_by_season(output_dir: str = "reports") -> pd.DataFrame:
    """Run each overall research case on each sample-out season separately."""
    rows = []
    for case in build_audit_cases():
        for csv_path in CSV_PATHS:
            row = _run_case(case, (csv_path,))
            row["season"] = Path(csv_path).stem
            row["season_path"] = csv_path
            rows.append(row)
    df = _apply_statistical_adjustment(pd.DataFrame(rows))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "strategy_research_by_season.csv", index=False)
    return df


def run_alpha_ablation_audit(output_dir: str = "reports") -> pd.DataFrame:
    """Run in-memory AlphaXGB feature and calibration ablations."""
    rows = [_run_case(case, CSV_PATHS) for case in build_alpha_ablation_cases()]
    df = _apply_statistical_adjustment(pd.DataFrame(rows))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "strategy_research_ablation.csv", index=False)
    return df


def fetch_research_api_snapshot(output_dir: str = "reports") -> pd.DataFrame:
    """Fetch a small API-Football research snapshot without changing default data."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failure_path = out_dir / "research_api_snapshot_failure.md"
    try:
        client = FootballAPIClient(db_path=":memory:")
    except Exception as exc:
        failure_path.write_text(f"API snapshot skipped: {exc}\n", encoding="utf-8")
        empty = pd.DataFrame(
            columns=["endpoint", "status", "response_count", "fixture_id", "reason"]
        )
        empty.to_csv(out_dir / "research_api_snapshot_summary.csv", index=False)
        _write_local_data_inventory(out_dir)
        return empty

    today = datetime.now().strftime("%Y-%m-%d")
    rows: list[dict[str, Any]] = []
    status = client.get("/status", bypass_cache=True)
    rows.append(_api_row("/status", status))
    fixtures = client.get("/fixtures", {"date": today}, bypass_cache=True)
    rows.append(_api_row("/fixtures", fixtures))

    fixture_ids = []
    if fixtures and isinstance(fixtures.get("response"), list):
        fixture_ids = [
            item.get("fixture", {}).get("id")
            for item in fixtures["response"][:3]
            if item.get("fixture", {}).get("id") is not None
        ]

    for fixture_id in fixture_ids:
        for endpoint, params in (
            ("/fixtures/lineups", {"fixture": fixture_id}),
            ("/injuries", {"fixture": fixture_id}),
            ("/fixtures/statistics", {"fixture": fixture_id}),
        ):
            response = client.get(endpoint, params, bypass_cache=True)
            row = _api_row(endpoint, response)
            row["fixture_id"] = fixture_id
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "research_api_snapshot_summary.csv", index=False)
    _write_api_markdown(df, out_dir / "research_api_snapshot_status.md")
    _write_local_data_inventory(out_dir)
    return df


def _api_row(endpoint: str, response: dict[str, Any] | None) -> dict[str, Any]:
    if not response:
        return {
            "endpoint": endpoint,
            "status": "failed",
            "response_count": 0,
            "fixture_id": "",
            "reason": "empty_or_network_failure",
        }
    errors = response.get("errors")
    if errors:
        return {
            "endpoint": endpoint,
            "status": "failed",
            "response_count": 0,
            "fixture_id": "",
            "reason": str(errors),
        }
    payload = response.get("response", [])
    count = len(payload) if isinstance(payload, list) else 1
    return {
        "endpoint": endpoint,
        "status": "ok",
        "response_count": count,
        "fixture_id": "",
        "reason": "",
    }


def _write_local_data_inventory(out_dir: Path) -> None:
    rows = []
    for pattern in (
        "data_seasons/E0_*.csv",
        "kaggle_dataset/*.gz",
        "betfair_closing_odds_full.csv",
    ):
        for path in sorted(Path(".").glob(pattern)):
            rows.append(
                {
                    "path": path.as_posix(),
                    "kind": path.suffix.lstrip("."),
                    "bytes": path.stat().st_size,
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "research_local_data_inventory.csv", index=False)


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _markdown_table(df: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    if df.empty:
        return "None."
    display = df[list(columns)].copy() if columns else df.copy()
    display = display.astype(object)
    for col in [
        "total_return",
        "max_drawdown",
        "clv_mean",
        "beat_close",
        "min_season_beat_close",
    ]:
        if col in display:
            display.loc[:, col] = display[col].map(_format_pct)
    for col in ["per_bet_sharpe", "roi_ci_low", "roi_ci_high", "min_season_roi_ci_low"]:
        if col in display:
            display.loc[:, col] = display[col].map(_format_float)
    for col in ["p_value", "adjusted_p_value", "deflated_sharpe"]:
        if col in display:
            display.loc[:, col] = display[col].map(_format_float)
    if "min_season_clv" in display:
        display.loc[:, "min_season_clv"] = display["min_season_clv"].map(_format_pct)
    return display.to_markdown(index=False)


def _write_segment_markdown(segment_df: pd.DataFrame, output_path: Path) -> None:
    """Write human-readable segment attribution for the default strategy."""
    structural = (
        segment_df[segment_df["structural_loss"]].copy()
        if not segment_df.empty and "structural_loss" in segment_df
        else pd.DataFrame()
    )
    top_losses = (
        segment_df.sort_values(by="pnl", ascending=True).head(15)
        if not segment_df.empty
        else pd.DataFrame()
    )
    columns = [
        "group_type",
        "group",
        "trades",
        "stake",
        "pnl",
        "roi_on_stake",
        "clv_mean",
        "beat_close",
        "structural_loss",
    ]
    content = [
        "# Strategy Segment Audit",
        "",
        "## Conclusion",
        "",
        (
            "Structural-loss segments exist; use this as research evidence only and do not "
            "upgrade the default strategy without a separate plan."
            if not structural.empty
            else "No material structural-loss segment was detected by this gate."
        ),
        "",
        "## Segment Gate",
        "",
        "- only filled or partially filled trades with positive stake are included",
        "- material core segments require at least 30 trades",
        "- negative ROI on stake in selection, odds, EV, or probability bins blocks candidate status",
        "",
        "## Structural-Loss Segments",
        "",
        _markdown_table(structural, columns),
        "",
        "## Largest Loss Segments",
        "",
        _markdown_table(top_losses, columns),
        "",
        "## Files",
        "",
        "- reports/strategy_segment_audit.csv",
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")


def _write_upgrade_markdown(
    upgrade_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write the research-only strategy policy audit report."""
    candidates = (
        upgrade_df[upgrade_df["gate_status"] == "candidate"].copy()
        if "gate_status" in upgrade_df
        else pd.DataFrame()
    )
    near_misses = (
        upgrade_df.sort_values(
            by=["passed_gate_count", "roi_ci_low", "beat_close", "clv_mean"],
            ascending=[False, False, False, False],
        ).head(15)
        if not upgrade_df.empty
        else pd.DataFrame()
    )
    structural = (
        segment_df[segment_df["structural_loss"]].copy()
        if not segment_df.empty and "structural_loss" in segment_df
        else pd.DataFrame()
    )
    core_columns = [
        "model",
        "policy",
        "trades",
        "total_return",
        "max_drawdown",
        "per_bet_sharpe",
        "clv_mean",
        "beat_close",
        "roi_ci_low",
        "segment_structural_loss_count",
        "gate_fail_reasons",
        "gate_status",
        "robustness_status",
    ]
    segment_columns = [
        "model",
        "policy",
        "group_type",
        "group",
        "trades",
        "pnl",
        "roi_on_stake",
        "clv_mean",
        "beat_close",
    ]
    content = [
        "# Strategy Upgrade Audit",
        "",
        "## Conclusion",
        "",
        (
            "Candidate rows exist in reports only; the default strategy was not changed."
            if not candidates.empty
            else "No candidate passed CLV + beat-close + ROI significance + segment-loss gates; do not upgrade the default strategy."
        ),
        "",
        "## Scope",
        "",
        "- fixed opening-line evaluation",
        "- no model files are retrained or overwritten",
        "- closing odds are used only for CLV and post-trade evaluation",
        "- StrategyPolicy is research-only unless explicitly passed by a caller",
        "",
        "## Near Miss Policies",
        "",
        _markdown_table(near_misses, core_columns),
        "",
        "## Candidate Policies",
        "",
        _markdown_table(candidates, core_columns),
        "",
        "## Structural-Loss Evidence",
        "",
        _markdown_table(structural.head(30), segment_columns),
        "",
        "## Files",
        "",
        "- reports/strategy_upgrade_audit.csv",
        "- reports/strategy_upgrade_segments.csv",
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")


def _write_full_history_markdown(
    df: pd.DataFrame,
    segment_df: pd.DataFrame,
    output_path: Path,
) -> None:
    candidates = df[df["gate_status"] == "candidate"].copy()
    near_misses = df.sort_values(
        by=["passed_gate_count", "roi_ci_low", "beat_close", "clv_mean"],
        ascending=[False, False, False, False],
    ).head(12)
    structural = (
        segment_df[segment_df["structural_loss"]].copy()
        if not segment_df.empty and "structural_loss" in segment_df
        else pd.DataFrame()
    )
    core_columns = [
        "model",
        "ev_threshold",
        "commission_on_win",
        "trades",
        "total_return",
        "max_drawdown",
        "per_bet_sharpe",
        "clv_mean",
        "beat_close",
        "roi_ci_low",
        "gate_fail_reasons",
        "gate_status",
        "robustness_status",
    ]
    segment_columns = [
        "group_type",
        "group",
        "trades",
        "pnl",
        "roi_on_stake",
        "clv_mean",
        "beat_close",
        "structural_loss",
    ]
    content = [
        "# Full History Research Audit",
        "",
        "## Conclusion",
        "",
        (
            "Candidate rows exist as research observations only; the default strategy was not changed."
            if not candidates.empty
            else "No candidate passed the full-history research gate; do not upgrade the default strategy."
        ),
        "",
        "## Scope",
        "",
        "- fixed six standardized data_seasons CSVs from E0_1920 through E0_2425",
        "- opening-line evaluation only",
        "- no model files are retrained or overwritten",
        "",
        "## Near Miss Rows",
        "",
        _markdown_table(near_misses, core_columns),
        "",
        "## Candidate Rows",
        "",
        _markdown_table(candidates, core_columns),
        "",
        "## Segment Structural Losses",
        "",
        _markdown_table(structural.head(30), segment_columns),
        "",
        "## Files",
        "",
        "- reports/full_history_research_grid.csv",
        "- reports/full_history_segment_audit.csv",
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")


def _write_calibration_markdown(
    df: pd.DataFrame,
    segments: pd.DataFrame,
    output_path: Path,
) -> None:
    candidates = df[df["gate_status"] == "candidate"].copy()
    structural = (
        segments[segments["structural_loss"]].copy()
        if not segments.empty and "structural_loss" in segments
        else pd.DataFrame()
    )
    core_columns = [
        "model",
        "feature_set",
        "trades",
        "total_return",
        "max_drawdown",
        "per_bet_sharpe",
        "clv_mean",
        "beat_close",
        "roi_ci_low",
        "gate_fail_reasons",
        "gate_status",
        "robustness_status",
    ]
    segment_columns = [
        "model",
        "feature_set",
        "group_type",
        "group",
        "trades",
        "pnl",
        "roi_on_stake",
        "structural_loss",
    ]
    content = [
        "# Calibration Research Audit",
        "",
        "## Conclusion",
        "",
        (
            "Candidate calibration rows exist in reports only; the default model was not changed."
            if not candidates.empty
            else "No calibration experiment passed the research gate; do not upgrade the default strategy."
        ),
        "",
        "## Scope",
        "",
        "- calibration uses only previously settled predictions and outcomes",
        "- closing odds are not used in calibration inputs",
        "- no XGBoost JSON or other model artifact is overwritten",
        "",
        "## Calibration Grid",
        "",
        _markdown_table(
            df.sort_values(
                by=["passed_gate_count", "roi_ci_low", "clv_mean"],
                ascending=[False, False, False],
            ),
            core_columns,
        ),
        "",
        "## Candidate Rows",
        "",
        _markdown_table(candidates, core_columns),
        "",
        "## Structural-Loss Segments",
        "",
        _markdown_table(structural.head(30), segment_columns),
        "",
        "## Files",
        "",
        "- reports/calibration_research_grid.csv",
        "- reports/calibration_research_segments.csv",
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")


def _write_default_candidate_gate_markdown(df: pd.DataFrame, output_path: Path) -> None:
    candidates = (
        df[df["default_gate_status"] == "candidate"].copy()
        if "default_gate_status" in df
        else pd.DataFrame()
    )
    display = (
        df.sort_values(
            by=["passed_gate_count", "roi_ci_low", "clv_mean"],
            ascending=[False, False, False],
        ).head(20)
        if not df.empty
        else pd.DataFrame()
    )
    columns = [
        "candidate_source",
        "model",
        "feature_set",
        "ev_threshold",
        "trades",
        "total_return",
        "max_drawdown",
        "per_bet_sharpe",
        "clv_mean",
        "beat_close",
        "roi_ci_low",
        "season_fail_count",
        "segment_structural_loss_count",
        "stress_status",
        "default_gate_fail_reasons",
        "default_gate_status",
    ]
    content = [
        "# Default Candidate Gate",
        "",
        "## Conclusion",
        "",
        (
            "Candidate rows exist in reports only; the default strategy still was not changed."
            if not candidates.empty
            else "No candidate passed the default-upgrade gate; do not upgrade the default strategy."
        ),
        "",
        "## Gate",
        "",
        "- `clv_mean > 0`",
        "- `beat_close > 0.50`",
        "- `roi_ci_low > 0`",
        "- all six seasons must pass the core gate",
        "- core selection, odds, EV, and probability segments must avoid structural loss",
        "- realistic execution stress must not degrade return by more than 5 percentage points",
        "",
        "## Reviewed Rows",
        "",
        _markdown_table(display, columns),
        "",
        "## Candidate Rows",
        "",
        _markdown_table(candidates, columns),
        "",
        "## Files",
        "",
        "- reports/default_candidate_gate.csv",
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")


def _write_markdown_report(
    df: pd.DataFrame,
    output_path: Path,
    by_season_df: pd.DataFrame | None = None,
    ablation_df: pd.DataFrame | None = None,
    api_df: pd.DataFrame | None = None,
) -> None:
    candidates = df[df["gate_status"] == "candidate"].copy()
    near_misses = df.sort_values(
        by=["passed_gate_count", "roi_ci_low", "beat_close", "clv_mean"],
        ascending=[False, False, False, False],
    )
    candidate_text = (
        "Candidate rows found; keep them in reports only until a separate upgrade plan."
        if not candidates.empty
        else "No candidate passed CLV + beat-close + ROI significance gates; do not upgrade the default strategy."
    )
    core_columns = [
        "model",
        "ev_threshold",
        "commission_on_win",
        "trades",
        "total_return",
        "max_drawdown",
        "per_bet_sharpe",
        "clv_mean",
        "beat_close",
        "roi_ci_low",
        "adjusted_p_value",
        "deflated_sharpe",
        "gate_fail_reasons",
        "gate_status",
        "robustness_status",
    ]
    robust_columns = [
        "model",
        "ev_threshold",
        "commission_on_win",
        "season_count",
        "min_season_clv",
        "min_season_beat_close",
        "min_season_roi_ci_low",
        "gate_fail_reasons",
    ]
    content = [
        "# Strategy Research Audit",
        "",
        "## Conclusion",
        "",
        candidate_text,
        "",
        "## Gate",
        "",
        "- `clv_mean > 0`",
        "- `beat_close > 0.50`",
        "- `roi_ci_low > 0`",
        "- season-by-season rows must also pass the same gate",
        "",
        "## Near Miss Rows",
        "",
        _markdown_table(near_misses.head(12), core_columns),
        "",
        "## Season Robustness",
        "",
        _markdown_table(near_misses.head(12), robust_columns),
        "",
        "## Candidate Rows",
        "",
        _markdown_table(candidates, core_columns),
        "",
    ]
    if by_season_df is not None:
        content.extend(
            [
                "## By Season Worst Rows",
                "",
                _markdown_table(
                    by_season_df.sort_values(
                        by=["passed_gate_count", "roi_ci_low"], ascending=[True, True]
                    ).head(12),
                    [
                        "season",
                        "model",
                        "ev_threshold",
                        "commission_on_win",
                        "trades",
                        "clv_mean",
                        "beat_close",
                        "roi_ci_low",
                        "adjusted_p_value",
                        "gate_fail_reasons",
                    ],
                ),
                "",
            ]
        )
    if ablation_df is not None:
        content.extend(
            [
                "## Alpha Ablation",
                "",
                _markdown_table(
                    ablation_df.sort_values(
                        by=["passed_gate_count", "roi_ci_low", "clv_mean"],
                        ascending=[False, False, False],
                    ),
                    [
                        "model",
                        "feature_set",
                        "ev_threshold",
                        "trades",
                        "total_return",
                        "clv_mean",
                        "beat_close",
                        "roi_ci_low",
                        "adjusted_p_value",
                        "deflated_sharpe",
                        "gate_fail_reasons",
                        "gate_status",
                        "robustness_status",
                    ],
                ),
                "",
            ]
        )
    if api_df is not None:
        content.extend(
            [
                "## API Snapshot",
                "",
                _markdown_table(api_df, ["endpoint", "status", "response_count", "fixture_id", "reason"]),
                "",
            ]
        )
    output_path.write_text("\n".join(content), encoding="utf-8")


def _write_api_markdown(df: pd.DataFrame, output_path: Path) -> None:
    output_path.write_text(
        "\n".join(["# Research API Snapshot", "", _markdown_table(df)]),
        encoding="utf-8",
    )


def run_strategy_research_audit(output_dir: str = "reports") -> pd.DataFrame:
    """Run the full research suite and write CSV/Markdown audit reports."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_season_df = run_strategy_research_by_season(output_dir)
    rows = [_run_case(case, CSV_PATHS) for case in build_audit_cases()]
    df = _apply_statistical_adjustment(
        _attach_season_robustness(pd.DataFrame(rows), by_season_df)
    )
    df.to_csv(out_dir / "strategy_research_grid.csv", index=False)
    ablation_df = run_alpha_ablation_audit(output_dir)
    api_df = fetch_research_api_snapshot(output_dir)
    _write_markdown_report(
        df,
        out_dir / "strategy_research_audit.md",
        by_season_df=by_season_df,
        ablation_df=ablation_df,
        api_df=api_df,
    )
    return df


if __name__ == "__main__":
    audit = run_strategy_research_audit()
    n_candidates = int((audit["gate_status"] == "candidate").sum())
    print(
        f"Completed {len(audit)} strategy research cases; "
        f"candidates={n_candidates}."
    )
