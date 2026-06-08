"""Research-grade walk-forward backtest engine for 1X2 football markets.

The default path intentionally preserves the historical Spots-Quant baseline:
model probabilities, market odds, and match results remain independent inputs;
`predict()` is called before `update()` for every fixture; and legacy fills use
the quoted entry odds without extra execution friction.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

import quant_core as qc
from asian_handicap_engine import PoissonPricer

OUTCOME_IDX = {"H": 0, "D": 1, "A": 2}
SELECTION_LABELS = "HDA"
DEFAULT_CSV_PATHS = (
    "data_seasons/E0_2223.csv",
    "data_seasons/E0_2324.csv",
    "data_seasons/E0_2425.csv",
)
REQUIRED_COLUMNS = (
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "B365CH",
    "B365CD",
    "B365CA",
)
OPENING_COLUMNS = ("B365H", "B365D", "B365A")
RESULT_COLUMNS = {
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
}
CLOSING_COLUMNS = {"B365CH", "B365CD", "B365CA"}


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution assumptions for backtest fills.

    `legacy` keeps the historical fill model. `realistic` applies conservative
    odds deterioration, rejection, partial fill, and stake/liquidity caps.
    """

    mode: str = "legacy"
    spread_pct: float = 0.0
    slippage_pct: float = 0.0
    price_delay_pct: float = 0.0
    rejection_rate: float = 0.0
    partial_fill_rate: float = 1.0
    max_stake_fraction: float | None = None
    match_liquidity: float | None = None
    seed: int = 42


@dataclass(frozen=True)
class StrategyPolicy:
    """Optional research-only filters and stake multipliers for bet selection.

    An empty policy is a no-op. Policy checks use only decision-time selection,
    odds, EV, probability, and rolling settled trade history.
    """

    name: str = "unrestricted"
    allowed_selections: tuple[str, ...] | None = None
    min_odds: float | None = None
    max_odds: float | None = None
    min_ev: float | None = None
    max_ev: float | None = None
    selection_min_ev: dict[str, float] | None = None
    selection_stake_multiplier: dict[str, float] | None = None
    high_odds_threshold: float | None = None
    high_odds_min_ev: float | None = None
    high_odds_stake_multiplier: float = 1.0
    segment_lookback: int = 0
    segment_min_trades: int = 0
    segment_min_roi: float | None = None
    segment_stake_multiplier: float = 0.0


@dataclass(frozen=True)
class CostConfig:
    """Backtest cost assumptions applied after a winning settlement."""

    commission_on_win: float = 0.0


@dataclass(frozen=True)
class RiskConfig:
    """Risk controls used by the walk-forward backtest loop."""

    max_drawdown_limit: float = 0.15
    kelly_mult: float = 0.05
    max_fraction: float = 0.03
    max_match_exposure: float | None = None


@dataclass(frozen=True)
class BacktestConfig:
    """Immutable description of a concrete backtest run."""

    csv_paths: tuple[str, ...]
    model_name: str
    ev_threshold: float
    initial_capital: float
    odds_mode: str


@dataclass(frozen=True)
class TradeRecord:
    """Auditable record for a filled or attempted trade.

    The record stores decision-time inputs, execution assumptions, settlement,
    CLV, and capital before/after the trade.
    """

    season: str
    date: str
    match: str
    home: str
    away: str
    model: str
    odds_mode: str
    selection: str
    odds: float
    closing_odds: float
    ev: float
    p_model: float
    requested_stake: float
    stake: float
    pnl: float
    capital_before: float
    capital_after: float
    clv: float
    won: bool
    risk_action: str
    execution_status: str
    execution_reason: str = ""


@dataclass(frozen=True)
class BacktestResult:
    """Typed result object for audit/reporting callers."""

    config: BacktestConfig
    execution_config: ExecutionConfig
    strategy_policy: StrategyPolicy | None
    cost_config: CostConfig
    risk_config: RiskConfig
    metrics: dict[str, Any]
    trade_records: list[TradeRecord]
    data_diagnostics: pd.DataFrame
    execution_summary: dict[str, Any]
    risk_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the legacy metrics dict plus audit-compatible additions."""
        payload = dict(self.metrics)
        payload["bet_records"] = [_trade_record_to_dict(item) for item in self.trade_records]
        payload["execution_summary"] = dict(self.execution_summary)
        payload["risk_summary"] = dict(self.risk_summary)
        payload["backtest_config"] = asdict(self.config)
        payload["execution_config"] = asdict(self.execution_config)
        payload["strategy_policy"] = (
            asdict(self.strategy_policy) if self.strategy_policy is not None else None
        )
        payload["cost_config"] = asdict(self.cost_config)
        payload["risk_config"] = asdict(self.risk_config)
        return payload


@dataclass(frozen=True)
class _ExecutionFill:
    status: str
    odds: float
    stake: float
    reason: str = ""


@dataclass(frozen=True)
class _PolicyDecision:
    allowed: bool
    stake_multiplier: float = 1.0
    reason: str = ""


class RiskManagementV9:
    """Fractional Kelly risk manager with drawdown and match exposure controls."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_drawdown_limit: float = 0.15,
        kelly_mult: float = 0.05,
        max_fraction: float = 0.03,
        max_match_exposure: float | None = None,
    ) -> None:
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_capital = initial_capital
        self.max_drawdown_limit = max_drawdown_limit
        self.kelly_mult = kelly_mult
        self.max_fraction = max_fraction
        self.max_match_exposure = max_match_exposure
        self._match_exposure: defaultdict[Any, float] = defaultdict(float)
        self.trading_frozen = False
        self.freeze_events = 0
        self.equity_curve = [initial_capital]
        self.trades = 0
        self.wins = 0

    def calculate_bet_size(
        self, p: float, b: float, match_id: Any | None = None
    ) -> float:
        """Calculate fractional Kelly stake using only decision-time inputs."""
        if self.trading_frozen:
            return 0.0
        base_fraction = qc.kelly_fraction(p, b, fraction=self.kelly_mult)
        penalty = 1.0 / math.sqrt(b) if b > 2.0 else 1.0
        final_fraction = min(max(base_fraction * penalty, 0.0), self.max_fraction)
        stake = self.current_capital * final_fraction

        if match_id is not None and self.max_match_exposure is not None:
            cap = self.current_capital * self.max_match_exposure
            remaining = max(cap - self._match_exposure[match_id], 0.0)
            stake = min(stake, remaining)
        return stake

    def register_exposure(self, match_id: Any | None, stake: float) -> None:
        """Register filled stake against a fixture-level exposure bucket."""
        if match_id is not None:
            self._match_exposure[match_id] += stake

    def update_capital(self, pnl: float) -> None:
        """Settle PnL and freeze further trading after the drawdown limit."""
        if self.trading_frozen:
            return
        self.current_capital += pnl
        self.equity_curve.append(self.current_capital)
        if self.current_capital > self.max_capital:
            self.max_capital = self.current_capital
        dd = (self.max_capital - self.current_capital) / self.max_capital
        if dd >= self.max_drawdown_limit:
            self.trading_frozen = True
            self.freeze_events += 1


class WalkForwardPoissonModel:
    """No-lookahead Poisson strength model updated only after settlement."""

    def __init__(self, min_history: int = 3) -> None:
        self.min_history = min_history
        self.pricer = PoissonPricer(max_goals=10)
        self.gf: defaultdict[str, float] = defaultdict(float)
        self.ga: defaultdict[str, float] = defaultdict(float)
        self.played: defaultdict[str, int] = defaultdict(int)
        self.total_home_goals = 0.0
        self.total_away_goals = 0.0
        self.n_matches = 0

    def predict(self, home: str, away: str, **kwargs: Any) -> tuple[float, float, float] | None:
        """Predict 1X2 probabilities from matches completed before this fixture."""
        if self.played[home] < self.min_history or self.played[away] < self.min_history:
            return None
        if self.n_matches < 20:
            return None
        league_home_avg = self.total_home_goals / self.n_matches
        league_away_avg = self.total_away_goals / self.n_matches
        league_overall_avg = (self.total_home_goals + self.total_away_goals) / (
            2 * self.n_matches
        )
        if league_overall_avg <= 0:
            return None
        atk_home = (self.gf[home] / self.played[home]) / league_overall_avg
        def_home = (self.ga[home] / self.played[home]) / league_overall_avg
        atk_away = (self.gf[away] / self.played[away]) / league_overall_avg
        def_away = (self.ga[away] / self.played[away]) / league_overall_avg
        lam_h = float(np.clip(atk_home * def_away * league_home_avg, 0.05, 8.0))
        lam_a = float(np.clip(atk_away * def_home * league_away_avg, 0.05, 8.0))
        return self.pricer.calculate_1x2_from_lambdas(lam_h, lam_a)

    def update(self, home: str, away: str, hg: int, ag: int, **kwargs: Any) -> None:
        """Update team scoring/conceding state after a completed match."""
        self.gf[home] += hg
        self.ga[home] += ag
        self.played[home] += 1
        self.gf[away] += ag
        self.ga[away] += hg
        self.played[away] += 1
        self.total_home_goals += hg
        self.total_away_goals += ag
        self.n_matches += 1


class WalkForwardXGBoostModel:
    """No-lookahead XGBoost inference wrapper using historical ELO/proxy-xG."""

    def __init__(self, min_history: int = 3) -> None:
        self.min_history = min_history
        self.elo: defaultdict[str, float] = defaultdict(lambda: 1500.0)
        self.pxg_hist: defaultdict[str, deque[float]] = defaultdict(lambda: deque(maxlen=5))

        self.xgb_h = xgb.XGBRegressor()
        self.xgb_d = xgb.XGBRegressor()
        self.xgb_a = xgb.XGBRegressor()
        self.xgb_h.load_model("xgb_h_distilled.json")
        self.xgb_d.load_model("xgb_d_distilled.json")
        self.xgb_a.load_model("xgb_a_distilled.json")

        self.played: defaultdict[str, int] = defaultdict(int)

    def _momentum(self, team: str) -> float:
        h = self.pxg_hist[team]
        return sum(h) / len(h) if h else 1.0

    def predict(self, home: str, away: str, **kwargs: Any) -> tuple[float, float, float] | None:
        """Predict 1X2 probabilities from pre-match walk-forward state."""
        if self.played[home] < self.min_history or self.played[away] < self.min_history:
            return None

        elo_diff = self.elo[home] - self.elo[away]
        mom_diff = self._momentum(home) - self._momentum(away)

        x_frame = pd.DataFrame(
            [
                {
                    "elo_diff": elo_diff,
                    "mom_diff": mom_diff,
                    "rating_diff": 0.0,
                    "mif_home": 0.0,
                    "mif_away": 0.0,
                }
            ]
        )

        p_h = float(self.xgb_h.predict(x_frame)[0])
        p_d = float(self.xgb_d.predict(x_frame)[0])
        p_a = float(self.xgb_a.predict(x_frame)[0])

        p_h = max(min(p_h, 0.99), 0.01)
        p_d = max(min(p_d, 0.99), 0.01)
        p_a = max(min(p_a, 0.99), 0.01)

        total = p_h + p_d + p_a
        return (p_h / total, p_d / total, p_a / total)

    def update(
        self,
        home: str,
        away: str,
        hg: int,
        ag: int,
        **kwargs: Any,
    ) -> None:
        """Update ELO/proxy-xG after settlement for future fixtures."""
        self.played[home] += 1
        self.played[away] += 1

        elo_diff = self.elo[home] - self.elo[away]
        exp_h = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
        s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)

        self.elo[home] += 40.0 * (s_h - exp_h)
        self.elo[away] += 40.0 * ((1.0 - s_h) - (1.0 - exp_h))

        row = kwargs.get("row")
        if row is not None:
            if pd.notna(row.get("HST")) and pd.notna(row.get("HS")):
                sog = max(float(row["HST"]), 0)
                soff = max(float(row["HS"]) - sog, 0)
                self.pxg_hist[home].append(sog * 0.25 + soff * 0.05)
            if pd.notna(row.get("AST")) and pd.notna(row.get("AS")):
                sog = max(float(row["AST"]), 0)
                soff = max(float(row["AS"]) - sog, 0)
                self.pxg_hist[away].append(sog * 0.25 + soff * 0.05)


def validate_backtest_dataset(paths: list[str]) -> pd.DataFrame:
    """Validate CSV inputs for safe chronological football backtests."""
    rows: list[dict[str, Any]] = []
    for csv_path in paths:
        path = Path(csv_path)
        if not path.exists():
            rows.append(_diagnostic_row(csv_path, "error", "path_exists", "failed", "file missing", 0))
            continue
        try:
            df = pd.read_csv(path)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            rows.append(_diagnostic_row(csv_path, "error", "read_csv", "failed", str(exc), 0))
            continue

        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        status = "failed" if missing else "ok"
        rows.append(
            _diagnostic_row(
                csv_path,
                "error" if missing else "info",
                "required_columns",
                status,
                ",".join(missing) if missing else "all required columns present",
                len(df),
            )
        )
        if missing:
            continue

        parsed_dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        bad_dates = int(parsed_dates.isna().sum())
        rows.append(
            _diagnostic_row(
                csv_path,
                "error" if bad_dates else "info",
                "date_parse",
                "failed" if bad_dates else "ok",
                f"{bad_dates} unparsable dates",
                bad_dates,
            )
        )
        sorted_ok = bool(parsed_dates.dropna().is_monotonic_increasing)
        rows.append(
            _diagnostic_row(
                csv_path,
                "warning" if not sorted_ok else "info",
                "date_order",
                "failed" if not sorted_ok else "ok",
                "rows are not sorted by parsed date" if not sorted_ok else "date order ok",
                len(df),
            )
        )

        duplicate_count = int(
            df.duplicated(subset=["Date", "HomeTeam", "AwayTeam"], keep=False).sum()
        )
        rows.append(
            _diagnostic_row(
                csv_path,
                "error" if duplicate_count else "info",
                "duplicate_fixture",
                "failed" if duplicate_count else "ok",
                f"{duplicate_count} duplicated fixture rows",
                duplicate_count,
            )
        )

        result_missing = int(df[["FTHG", "FTAG", "FTR"]].isna().any(axis=1).sum())
        rows.append(
            _diagnostic_row(
                csv_path,
                "error" if result_missing else "info",
                "result_missing",
                "failed" if result_missing else "ok",
                f"{result_missing} rows missing settlement result",
                result_missing,
            )
        )

        close_cols = ["B365CH", "B365CD", "B365CA"]
        close_numeric = df[close_cols].apply(pd.to_numeric, errors="coerce")
        bad_close = int((close_numeric.isna() | (close_numeric <= 1.0)).any(axis=1).sum())
        rows.append(
            _diagnostic_row(
                csv_path,
                "error" if bad_close else "info",
                "closing_odds_valid",
                "failed" if bad_close else "ok",
                f"{bad_close} rows with missing or invalid closing odds",
                bad_close,
            )
        )

        missing_opening = [col for col in OPENING_COLUMNS if col not in df.columns]
        if missing_opening:
            rows.append(
                _diagnostic_row(
                    csv_path,
                    "warning",
                    "opening_odds_available",
                    "failed",
                    ",".join(missing_opening),
                    len(df),
                )
            )
        else:
            open_numeric = df[list(OPENING_COLUMNS)].apply(pd.to_numeric, errors="coerce")
            bad_open = int((open_numeric.isna() | (open_numeric <= 1.0)).any(axis=1).sum())
            rows.append(
                _diagnostic_row(
                    csv_path,
                    "warning" if bad_open else "info",
                    "opening_odds_valid",
                    "failed" if bad_open else "ok",
                    f"{bad_open} rows with missing or invalid opening odds",
                    bad_open,
                )
            )

        future_cols = sorted((RESULT_COLUMNS | CLOSING_COLUMNS).intersection(df.columns))
        rows.append(
            _diagnostic_row(
                csv_path,
                "info",
                "future_columns_sanitized",
                "ok",
                ",".join(future_cols),
                len(future_cols),
            )
        )
    return pd.DataFrame(
        rows,
        columns=["path", "severity", "check", "status", "detail", "rows_affected"],
    )


def run_real_backtest(
    csv_paths: str | Sequence[str] = DEFAULT_CSV_PATHS,
    model_factory: Callable[[], Any] = WalkForwardXGBoostModel,
    ev_threshold: float = 1.05,
    initial_capital: float = 10000.0,
    max_drawdown_limit: float = 0.15,
    odds_mode: str = "closing",
    commission_on_win: float = 0.0,
    kelly_mult: float = 0.05,
    verbose: bool = True,
    return_result: bool = False,
    execution_config: ExecutionConfig | None = None,
    risk_config: RiskConfig | None = None,
    strategy_policy: StrategyPolicy | None = None,
) -> dict[str, Any] | BacktestResult:
    """Run a chronological walk-forward 1X2 backtest.

    Default arguments preserve the legacy baseline. Pass `return_result=True`
    for typed audit data, `ExecutionConfig(mode="realistic")` for explicit
    execution-friction stress testing, or `StrategyPolicy` for research filters.
    """
    if isinstance(csv_paths, str):
        csv_paths = (csv_paths,)
    csv_tuple = tuple(csv_paths)
    if odds_mode not in {"closing", "opening"}:
        raise ValueError("odds_mode must be 'closing' or 'opening'")

    execution = _normalize_execution_config(execution_config)
    costs = CostConfig(commission_on_win=commission_on_win)
    active_risk = risk_config or RiskConfig(
        max_drawdown_limit=max_drawdown_limit,
        kelly_mult=kelly_mult,
    )
    diagnostics = validate_backtest_dataset(list(csv_tuple))
    _raise_on_validation_errors(diagnostics)

    def log(*items: Any) -> None:
        if verbose:
            print(*items)

    log("==================================================")
    log("   Walk-forward backtest (CLV + calibration + costs)")
    log(
        f"   odds_mode={odds_mode} | ev_threshold={ev_threshold} | "
        f"commission_on_win={commission_on_win} | execution={execution.mode}"
    )
    log("==================================================")

    risk_mgr = RiskManagementV9(
        initial_capital,
        active_risk.max_drawdown_limit,
        kelly_mult=active_risk.kelly_mult,
        max_fraction=active_risk.max_fraction,
        max_match_exposure=active_risk.max_match_exposure,
    )

    per_bet_returns: list[float] = []
    bet_results: list[bool] = []
    clv_list: list[float] = []
    trade_dates: list[str] = []
    trade_seasons: list[str] = []
    loaded_seasons: set[str] = set()
    cal_probs: list[float] = []
    cal_outcomes: list[int] = []
    trade_records: list[TradeRecord] = []
    total_matches = 0
    risk_blocks: defaultdict[str, int] = defaultdict(int)
    execution_counts: defaultdict[str, int] = defaultdict(int)
    segment_history: defaultdict[str, deque[float]] = defaultdict(deque)
    rng = np.random.default_rng(execution.seed)
    model_name = _model_factory_name(model_factory)

    for csv_path in csv_tuple:
        df = _load_season(csv_path)
        total_matches += len(df)
        season = Path(csv_path).stem
        loaded_seasons.add(season)
        has_open = _has_opening(df)
        if odds_mode == "opening" and not has_open:
            raise ValueError(f"{csv_path} missing opening odds for odds_mode='opening'")
        model = model_factory()

        for row_idx, row in df.iterrows():
            home = str(row["HomeTeam"])
            away = str(row["AwayTeam"])
            hg = int(row["FTHG"])
            ag = int(row["FTAG"])
            ftr = str(row["FTR"])
            match_id = f"{season}:{row_idx}:{home}:{away}"
            match_label = f"{home} v {away}"
            date_label = _format_date(row["_date"])
            close_odds = [
                float(row["B365CH"]),
                float(row["B365CD"]),
                float(row["B365CA"]),
            ]
            entry_odds = (
                [float(row["B365H"]), float(row["B365D"]), float(row["B365A"])]
                if has_open
                else close_odds
            )
            use_odds = entry_odds if odds_mode == "opening" else close_odds

            decision_row = _decision_row(row, odds_mode)
            preds = model.predict(home, away, date=row["_date"], row=decision_row)
            if preds is not None:
                cal_probs.append(float(preds[0]))
                cal_outcomes.append(1 if ftr == "H" else 0)
                if risk_mgr.trading_frozen:
                    risk_blocks["drawdown_frozen"] += 1
                elif all(o > 1.0 for o in use_odds):
                    best = _select_best_bet(preds, use_odds, ev_threshold)
                    if best is not None:
                        ev, sel, p_sel, quoted_odds = best
                        selection = SELECTION_LABELS[sel]
                        policy_decision = _apply_strategy_policy(
                            strategy_policy,
                            selection,
                            ev,
                            quoted_odds,
                            p_sel,
                            segment_history,
                        )
                        won = OUTCOME_IDX[ftr] == sel
                        if not policy_decision.allowed:
                            risk_blocks["strategy_policy"] += 1
                            execution_counts["policy_blocked"] += 1
                            capital_before = risk_mgr.current_capital
                            trade_records.append(
                                TradeRecord(
                                    season=season,
                                    date=date_label,
                                    match=match_label,
                                    home=home,
                                    away=away,
                                    model=model_name,
                                    odds_mode=odds_mode,
                                    selection=selection,
                                    odds=quoted_odds,
                                    closing_odds=close_odds[sel],
                                    ev=ev,
                                    p_model=p_sel,
                                    requested_stake=0.0,
                                    stake=0.0,
                                    pnl=0.0,
                                    capital_before=capital_before,
                                    capital_after=capital_before,
                                    clv=qc.clv_pct(quoted_odds, close_odds[sel]),
                                    won=won,
                                    risk_action="blocked",
                                    execution_status="policy_blocked",
                                    execution_reason=policy_decision.reason,
                                )
                            )
                            model.update(home, away, hg, ag, date=row["_date"], row=row)
                            continue

                        requested_stake = risk_mgr.calculate_bet_size(
                            p_sel, quoted_odds, match_id=match_id
                        )
                        requested_stake *= policy_decision.stake_multiplier
                        if requested_stake <= 0:
                            risk_blocks["stake_zero_or_exposure_cap"] += 1
                        else:
                            capital_before = risk_mgr.current_capital
                            fill = _apply_execution(
                                requested_stake,
                                quoted_odds,
                                capital_before,
                                execution,
                                rng,
                            )
                            execution_counts[fill.status] += 1
                            pnl = 0.0
                            if fill.stake > 0:
                                pnl = _settle_pnl(
                                    fill.stake,
                                    fill.odds,
                                    won,
                                    costs.commission_on_win,
                                )
                                if won:
                                    risk_mgr.wins += 1
                                risk_mgr.update_capital(pnl)
                                risk_mgr.trades += 1
                                risk_mgr.register_exposure(match_id, fill.stake)
                                per_bet_returns.append(pnl / fill.stake)
                                bet_results.append(won)
                                clv_list.append(qc.clv_pct(fill.odds, close_odds[sel]))
                                trade_dates.append(date_label)
                                trade_seasons.append(season)
                                for segment_key in _policy_segment_keys(
                                    selection, fill.odds, ev, p_sel
                                ):
                                    _append_segment_return(
                                        segment_history,
                                        segment_key,
                                        pnl / fill.stake,
                                        strategy_policy,
                                    )

                            trade_records.append(
                                TradeRecord(
                                    season=season,
                                    date=date_label,
                                    match=match_label,
                                    home=home,
                                    away=away,
                                    model=model_name,
                                    odds_mode=odds_mode,
                                    selection=selection,
                                    odds=fill.odds,
                                    closing_odds=close_odds[sel],
                                    ev=ev,
                                    p_model=p_sel,
                                    requested_stake=requested_stake,
                                    stake=fill.stake,
                                    pnl=pnl,
                                    capital_before=capital_before,
                                    capital_after=risk_mgr.current_capital,
                                    clv=qc.clv_pct(fill.odds, close_odds[sel]),
                                    won=won,
                                    risk_action="accepted" if fill.stake > 0 else "blocked",
                                    execution_status=fill.status,
                                    execution_reason=_join_reasons(
                                        policy_decision.reason,
                                        fill.reason,
                                    ),
                                )
                            )

            model.update(home, away, hg, ag, date=row["_date"], row=row)

    risk_summary = {
        "trading_frozen": risk_mgr.trading_frozen,
        "freeze_events": risk_mgr.freeze_events,
        "risk_blocks": dict(risk_blocks),
        "max_match_exposure": active_risk.max_match_exposure,
        "max_fraction": active_risk.max_fraction,
    }
    execution_summary = {
        "mode": execution.mode,
        "counts": dict(execution_counts),
        "spread_pct": execution.spread_pct,
        "slippage_pct": execution.slippage_pct,
        "price_delay_pct": execution.price_delay_pct,
        "rejection_rate": execution.rejection_rate,
        "partial_fill_rate": execution.partial_fill_rate,
    }
    metrics = _summarize(
        risk_mgr,
        per_bet_returns,
        bet_results,
        clv_list,
        cal_probs,
        cal_outcomes,
        total_matches,
        initial_capital,
        odds_mode,
        log,
        trade_dates=trade_dates,
        loaded_seasons=loaded_seasons,
    )
    config = BacktestConfig(
        csv_paths=csv_tuple,
        model_name=model_name,
        ev_threshold=ev_threshold,
        initial_capital=initial_capital,
        odds_mode=odds_mode,
    )
    result = BacktestResult(
        config=config,
        execution_config=execution,
        strategy_policy=strategy_policy,
        cost_config=costs,
        risk_config=active_risk,
        metrics=metrics,
        trade_records=trade_records,
        data_diagnostics=diagnostics,
        execution_summary=execution_summary,
        risk_summary=risk_summary,
    )
    return result if return_result else result.to_dict()


def run_backtest_audit(output_dir: str = "reports") -> BacktestResult:
    """Run the default backtest and write audit ledger/diagnostic reports."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_real_backtest(verbose=False, return_result=True)
    result.data_diagnostics.to_csv(out_dir / "backtest_data_diagnostics.csv", index=False)
    pd.DataFrame(result.to_dict()["bet_records"]).to_csv(
        out_dir / "backtest_trade_ledger.csv", index=False
    )
    stress = run_execution_stress_test(output_dir)
    _write_backtest_audit_markdown(result, stress, out_dir / "backtest_audit.md")
    return result


def run_execution_stress_test(output_dir: str = "reports") -> pd.DataFrame:
    """Run explicit execution-friction stress scenarios and write CSV output."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = [
        ("legacy_baseline", ExecutionConfig()),
        (
            "realistic_mild",
            ExecutionConfig(
                mode="realistic",
                spread_pct=0.005,
                slippage_pct=0.005,
                price_delay_pct=0.0025,
                rejection_rate=0.02,
                partial_fill_rate=0.80,
                max_stake_fraction=0.02,
                seed=42,
            ),
        ),
        (
            "realistic_harsh",
            ExecutionConfig(
                mode="realistic",
                spread_pct=0.015,
                slippage_pct=0.010,
                price_delay_pct=0.005,
                rejection_rate=0.05,
                partial_fill_rate=0.60,
                max_stake_fraction=0.01,
                seed=42,
            ),
        ),
    ]
    rows: list[dict[str, Any]] = []
    baseline_return = None
    for name, execution in scenarios:
        result = run_real_backtest(
            verbose=False,
            return_result=True,
            execution_config=execution,
        )
        metrics = result.metrics
        if baseline_return is None:
            baseline_return = float(metrics["total_return"])
        rows.append(
            {
                "scenario": name,
                "execution_mode": execution.mode,
                "trades": metrics["trades"],
                "total_return": metrics["total_return"],
                "return_delta_vs_legacy": float(metrics["total_return"]) - baseline_return,
                "max_drawdown": metrics["max_drawdown"],
                "per_bet_sharpe": metrics["per_bet_sharpe"],
                "clv_mean": metrics["clv_mean"],
                "beat_close": metrics["beat_close"],
                "execution_counts": result.execution_summary["counts"],
                "default_strategy_changed": False,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "backtest_execution_stress.csv", index=False)
    return df


def _load_season(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    df = df.assign(_date=pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"))
    df = df.dropna(subset=["_date"])
    return df.sort_values("_date").reset_index(drop=True)


def _has_opening(df: pd.DataFrame) -> bool:
    return all(col in df.columns for col in OPENING_COLUMNS)


def _max_consecutive_losses(results: Sequence[bool]) -> int:
    longest = cur = 0
    for won in results:
        cur = 0 if won else cur + 1
        longest = max(longest, cur)
    return longest


def _bootstrap_roi_ci(
    per_bet_returns: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap per-bet ROI mean and confidence interval."""
    returns = np.asarray(per_bet_returns, dtype=float)
    if len(returns) < 5:
        return (
            float(returns.mean()) if len(returns) else 0.0,
            float("nan"),
            float("nan"),
        )
    rng = np.random.default_rng(seed)
    means = [rng.choice(returns, size=len(returns), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(returns.mean()), float(lo), float(hi)


def _block_bootstrap_roi_ci(
    per_bet_returns: Sequence[float],
    block_keys: Sequence[str],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap ROI by resampling date blocks instead of individual bets."""
    returns = np.asarray(per_bet_returns, dtype=float)
    if len(returns) < 5 or len(block_keys) != len(returns):
        return _bootstrap_roi_ci(returns, n_boot=n_boot, alpha=alpha, seed=seed)
    frame = pd.DataFrame({"key": list(block_keys), "ret": returns})
    blocks = [group["ret"].to_numpy(dtype=float) for _, group in frame.groupby("key")]
    if len(blocks) < 2:
        return _bootstrap_roi_ci(returns, n_boot=n_boot, alpha=alpha, seed=seed)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sampled = rng.choice(len(blocks), size=len(blocks), replace=True)
        values = np.concatenate([blocks[int(idx)] for idx in sampled])
        means.append(float(values.mean()))
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(returns.mean()), float(lo), float(hi)


def _sign_flip_p_value(
    per_bet_returns: Sequence[float],
    n_perm: int = 2000,
    seed: int = 42,
) -> float:
    """One-sided sign-flip permutation p-value for positive mean ROI."""
    returns = np.asarray(per_bet_returns, dtype=float)
    if len(returns) < 5:
        return float("nan")
    observed = float(returns.mean())
    if observed <= 0:
        return 1.0
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(returns), replace=True)
        draws.append(float((returns * signs).mean()))
    return float((np.asarray(draws) >= observed).mean())


def _deflated_sharpe(
    sharpe: float,
    sample_size: int,
    n_trials: int = 1,
) -> float:
    """Apply a simple multiple-trial penalty to per-bet Sharpe."""
    if sample_size <= 1 or not np.isfinite(sharpe):
        return 0.0
    trial_penalty = math.sqrt(2.0 * math.log(max(n_trials, 1))) / math.sqrt(sample_size)
    return float(sharpe - trial_penalty)


def _summarize(
    risk_mgr: RiskManagementV9,
    per_bet_returns: Sequence[float],
    bet_results: Sequence[bool],
    clv_list: Sequence[float],
    cal_probs: Sequence[float],
    cal_outcomes: Sequence[int],
    total_matches: int,
    initial_capital: float,
    odds_mode: str,
    log: Callable[..., None],
    trade_dates: Sequence[str],
    loaded_seasons: set[str],
) -> dict[str, Any]:
    final = risk_mgr.current_capital
    total_return = (final - initial_capital) / initial_capital
    win_rate = (risk_mgr.wins / risk_mgr.trades) if risk_mgr.trades else 0.0

    equity = pd.Series(risk_mgr.equity_curve)
    max_dd = (
        ((equity.cummax() - equity) / equity.cummax()).max() if len(equity) > 1 else 0.0
    )

    returns = np.asarray(per_bet_returns, dtype=float)
    per_bet_sharpe = (
        float(returns.mean() / returns.std())
        if len(returns) > 1 and returns.std() > 0
        else 0.0
    )
    eq_ret = equity.pct_change().dropna()
    eq_down = eq_ret[eq_ret < 0]
    sortino = (
        float(eq_ret.mean() / eq_down.std())
        if len(eq_down) > 1 and eq_down.std() > 0
        else 0.0
    )
    calmar = float(total_return / max_dd) if max_dd > 0 else 0.0
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    profit_factor = (
        float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0
    )
    max_losing_streak = _max_consecutive_losses(bet_results)
    roi_mean, roi_lo, roi_hi = _bootstrap_roi_ci(per_bet_returns)
    block_roi_mean, block_roi_lo, block_roi_hi = _block_bootstrap_roi_ci(
        per_bet_returns, trade_dates
    )

    brier = qc.brier_score(cal_probs, cal_outcomes)
    logloss = qc.log_loss(cal_probs, cal_outcomes)
    base_rate = float(np.mean(cal_outcomes)) if cal_outcomes else 0.0
    brier_base = (
        qc.brier_score([base_rate] * len(cal_outcomes), cal_outcomes)
        if cal_outcomes
        else float("nan")
    )

    clv_arr = np.asarray(clv_list, dtype=float)
    clv_mean = float(clv_arr.mean()) if len(clv_arr) else 0.0
    beat_close = float((clv_arr > 0).mean()) if len(clv_arr) else 0.0
    sample_size = int(len(returns))
    p_value = _sign_flip_p_value(per_bet_returns)
    adjusted_p_value = p_value
    deflated_sharpe = _deflated_sharpe(per_bet_sharpe, sample_size)
    robustness_status = (
        "statistically_positive"
        if np.isfinite(roi_lo) and roi_lo > 0 and deflated_sharpe > 0
        else "observe_only"
    )

    log("")
    log("==================================================")
    log("                BACKTEST SUMMARY")
    log("==================================================")
    log(
        f"matches={total_matches:,} | trades={risk_mgr.trades:,} | "
        f"win_rate={win_rate * 100:.2f}%"
    )
    log(
        f"capital=${initial_capital:,.0f} -> ${final:,.2f} | "
        f"return={total_return * 100:.2f}%"
    )
    log(
        f"max_drawdown={max_dd * 100:.2f}% | per_bet_sharpe={per_bet_sharpe:.3f} | "
        f"sortino={sortino:.3f}"
    )
    log(
        f"roi_ci={roi_mean * 100:.2f}% [{roi_lo * 100:.2f}%, {roi_hi * 100:.2f}%] | "
        f"block_ci=[{block_roi_lo * 100:.2f}%, {block_roi_hi * 100:.2f}%]"
    )
    log(f"brier={brier:.4f} (base={brier_base:.4f}) | log_loss={logloss:.4f}")
    log(
        f"odds_mode={odds_mode} | clv_mean={clv_mean * 100:.2f}% | "
        f"beat_close={beat_close * 100:.1f}%"
    )
    if odds_mode != "opening":
        log("note: CLV is economically meaningful when odds_mode='opening'.")
    if risk_mgr.trading_frozen:
        log("warning: drawdown freeze triggered; later bets were blocked.")
    log("==================================================")

    return {
        "total_matches": total_matches,
        "trades": risk_mgr.trades,
        "win_rate": win_rate,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "final_capital": final,
        "per_bet_sharpe": per_bet_sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "profit_factor": profit_factor,
        "max_losing_streak": max_losing_streak,
        "roi_mean": roi_mean,
        "roi_ci": (roi_lo, roi_hi),
        "block_roi_mean": block_roi_mean,
        "block_roi_ci": (block_roi_lo, block_roi_hi),
        "brier": brier,
        "brier_base": brier_base,
        "log_loss": logloss,
        "clv_mean": clv_mean,
        "beat_close": beat_close,
        "sample_size": sample_size,
        "season_count": len(loaded_seasons),
        "p_value": p_value,
        "adjusted_p_value": adjusted_p_value,
        "deflated_sharpe": deflated_sharpe,
        "robustness_status": robustness_status,
    }


def _diagnostic_row(
    path: str,
    severity: str,
    check: str,
    status: str,
    detail: str,
    rows_affected: int,
) -> dict[str, Any]:
    return {
        "path": path,
        "severity": severity,
        "check": check,
        "status": status,
        "detail": detail,
        "rows_affected": rows_affected,
    }


def _raise_on_validation_errors(diagnostics: pd.DataFrame) -> None:
    errors = diagnostics[
        (diagnostics["severity"] == "error") & (diagnostics["status"] == "failed")
    ]
    if not errors.empty:
        details = "; ".join(
            f"{row.path}:{row.check}:{row.detail}" for row in errors.itertuples()
        )
        raise ValueError(f"Unsafe backtest dataset: {details}")


def _normalize_execution_config(config: ExecutionConfig | None) -> ExecutionConfig:
    execution = config or ExecutionConfig()
    if execution.mode not in {"legacy", "realistic"}:
        raise ValueError("ExecutionConfig.mode must be 'legacy' or 'realistic'")
    for name in ("spread_pct", "slippage_pct", "price_delay_pct", "rejection_rate"):
        value = float(getattr(execution, name))
        if value < 0:
            raise ValueError(f"ExecutionConfig.{name} must be non-negative")
    if not 0.0 <= execution.partial_fill_rate <= 1.0:
        raise ValueError("ExecutionConfig.partial_fill_rate must be between 0 and 1")
    return execution


def _decision_row(row: pd.Series, odds_mode: str) -> pd.Series:
    forbidden = set(RESULT_COLUMNS) | CLOSING_COLUMNS
    return row.drop(labels=[col for col in forbidden if col in row.index])


def _select_best_bet(
    preds: Sequence[float],
    odds: Sequence[float],
    ev_threshold: float,
) -> tuple[float, int, float, float] | None:
    best = None
    for idx, prob in enumerate(preds):
        ev = qc.expected_value(float(prob), float(odds[idx]))
        if ev > ev_threshold and (best is None or ev > best[0]):
            best = (ev, idx, float(prob), float(odds[idx]))
    return best


def _apply_strategy_policy(
    policy: StrategyPolicy | None,
    selection: str,
    ev: float,
    odds: float,
    p_model: float,
    segment_history: dict[str, deque[float]],
) -> _PolicyDecision:
    """Apply optional research policy filters before staking."""
    if policy is None:
        return _PolicyDecision(True, 1.0, "")

    if policy.allowed_selections is not None and selection not in policy.allowed_selections:
        return _PolicyDecision(False, 0.0, f"selection_not_allowed:{selection}")
    if policy.min_odds is not None and odds < policy.min_odds:
        return _PolicyDecision(False, 0.0, f"odds_below_min:{odds:.3f}")
    if policy.max_odds is not None and odds > policy.max_odds:
        return _PolicyDecision(False, 0.0, f"odds_above_max:{odds:.3f}")
    if policy.min_ev is not None and ev < policy.min_ev:
        return _PolicyDecision(False, 0.0, f"ev_below_min:{ev:.3f}")
    if policy.max_ev is not None and ev > policy.max_ev:
        return _PolicyDecision(False, 0.0, f"ev_above_max:{ev:.3f}")

    min_ev = (policy.selection_min_ev or {}).get(selection)
    if min_ev is not None and ev < min_ev:
        return _PolicyDecision(False, 0.0, f"selection_ev_below_min:{selection}:{ev:.3f}")

    multiplier = (policy.selection_stake_multiplier or {}).get(selection, 1.0)
    reasons: list[str] = []
    if multiplier != 1.0:
        reasons.append(f"selection_stake_multiplier:{selection}:{multiplier:.3f}")

    if policy.high_odds_threshold is not None and odds > policy.high_odds_threshold:
        if policy.high_odds_min_ev is not None and ev < policy.high_odds_min_ev:
            return _PolicyDecision(False, 0.0, f"high_odds_ev_below_min:{ev:.3f}")
        multiplier *= policy.high_odds_stake_multiplier
        if policy.high_odds_stake_multiplier != 1.0:
            reasons.append(f"high_odds_stake_multiplier:{policy.high_odds_stake_multiplier:.3f}")

    rolling_decision = _rolling_segment_policy_decision(
        policy, selection, odds, ev, p_model, segment_history
    )
    if not rolling_decision.allowed:
        return rolling_decision
    multiplier *= rolling_decision.stake_multiplier
    if rolling_decision.reason:
        reasons.append(rolling_decision.reason)

    if multiplier <= 0:
        return _PolicyDecision(False, 0.0, "stake_multiplier_zero")
    return _PolicyDecision(True, multiplier, ";".join(reasons))


def _rolling_segment_policy_decision(
    policy: StrategyPolicy,
    selection: str,
    odds: float,
    ev: float,
    p_model: float,
    segment_history: dict[str, deque[float]],
) -> _PolicyDecision:
    """Reduce or block stake when recent segment ROI is below the policy floor."""
    if (
        policy.segment_lookback <= 0
        or policy.segment_min_trades <= 0
        or policy.segment_min_roi is None
    ):
        return _PolicyDecision(True, 1.0, "")
    for segment_key in _policy_segment_keys(selection, odds, ev, p_model):
        returns = segment_history.get(segment_key, deque())
        if len(returns) < policy.segment_min_trades:
            continue
        mean_roi = float(np.mean(returns))
        if mean_roi < policy.segment_min_roi:
            reason = f"rolling_segment_roi:{segment_key}:{mean_roi:.3f}"
            if policy.segment_stake_multiplier <= 0:
                return _PolicyDecision(False, 0.0, reason)
            return _PolicyDecision(True, policy.segment_stake_multiplier, reason)
    return _PolicyDecision(True, 1.0, "")


def _policy_segment_keys(selection: str, odds: float, ev: float, p_model: float) -> tuple[str, ...]:
    """Return stable segment keys for rolling policy risk controls."""
    return (
        f"selection={selection}",
        f"odds_bin={_odds_bin(odds)}",
        f"ev_bin={_ev_bin(ev)}",
        f"p_bin={_p_bin(p_model)}",
    )


def _append_segment_return(
    segment_history: dict[str, deque[float]],
    segment_key: str,
    value: float,
    policy: StrategyPolicy | None,
) -> None:
    """Append a settled return and cap rolling history length when configured."""
    history = segment_history[segment_key]
    history.append(value)
    max_len = policy.segment_lookback if policy is not None else 0
    if max_len > 0:
        while len(history) > max_len:
            history.popleft()


def _odds_bin(odds: float) -> str:
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


def _ev_bin(ev: float) -> str:
    if ev <= 1.10:
        return "1.00-1.10"
    if ev <= 1.20:
        return "1.10-1.20"
    if ev <= 1.50:
        return "1.20-1.50"
    return "1.50+"


def _p_bin(probability: float) -> str:
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


def _join_reasons(*reasons: str) -> str:
    return ";".join(reason for reason in reasons if reason)


def _apply_execution(
    requested_stake: float,
    quoted_odds: float,
    capital_before: float,
    execution: ExecutionConfig,
    rng: np.random.Generator,
) -> _ExecutionFill:
    if execution.mode == "legacy":
        return _ExecutionFill("filled", quoted_odds, requested_stake)

    if rng.random() < execution.rejection_rate:
        return _ExecutionFill("rejected", quoted_odds, 0.0, "random_rejection")

    stake = requested_stake
    if execution.max_stake_fraction is not None:
        stake = min(stake, max(capital_before * execution.max_stake_fraction, 0.0))
    if execution.match_liquidity is not None:
        stake = min(stake, max(execution.match_liquidity, 0.0))
    stake *= execution.partial_fill_rate
    if stake <= 0:
        return _ExecutionFill("rejected", quoted_odds, 0.0, "no_liquidity")

    odds_penalty = execution.spread_pct + execution.slippage_pct + execution.price_delay_pct
    executed_odds = max(1.01, quoted_odds * (1.0 - odds_penalty))
    status = "partial_fill" if stake < requested_stake - 1e-9 else "filled"
    reason = "execution_friction" if odds_penalty > 0 or status == "partial_fill" else ""
    return _ExecutionFill(status, executed_odds, stake, reason)


def _settle_pnl(
    stake: float,
    odds: float,
    won: bool,
    commission_on_win: float,
) -> float:
    if won:
        gross = stake * (odds - 1.0)
        return gross * (1.0 - commission_on_win)
    return -stake


def _trade_record_to_dict(record: TradeRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload["odds_taken"] = record.odds
    payload["odds_close"] = record.closing_odds
    payload["clv_pct"] = record.clv
    return payload


def _model_factory_name(model_factory: Callable[[], Any]) -> str:
    return getattr(model_factory, "__name__", model_factory.__class__.__name__)


def _format_date(value: Any) -> str:
    if pd.isna(value):
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _write_backtest_audit_markdown(
    result: BacktestResult,
    stress_df: pd.DataFrame,
    output_path: Path,
) -> None:
    metrics = result.metrics
    candidate_text = (
        "Default strategy was not changed. Realistic execution is reported as a stress test only."
    )
    lines = [
        "# Backtest Audit",
        "",
        "## Conclusion",
        "",
        candidate_text,
        "",
        "## Default Baseline",
        "",
        f"- trades: {metrics['trades']}",
        f"- total_return: {metrics['total_return']:.12f}",
        f"- max_drawdown: {metrics['max_drawdown']:.12f}",
        f"- per_bet_sharpe: {metrics['per_bet_sharpe']:.15f}",
        f"- brier: {metrics['brier']:.12f}",
        f"- execution_mode: {result.execution_config.mode}",
        "",
        "## Statistical Robustness",
        "",
        f"- roi_ci: {metrics['roi_ci']}",
        f"- block_roi_ci: {metrics['block_roi_ci']}",
        f"- p_value: {metrics['p_value']}",
        f"- adjusted_p_value: {metrics['adjusted_p_value']}",
        f"- deflated_sharpe: {metrics['deflated_sharpe']}",
        f"- robustness_status: {metrics['robustness_status']}",
        "",
        "## Execution Stress",
        "",
        stress_df.to_markdown(index=False),
        "",
        "## Risk Summary",
        "",
        f"- trading_frozen: {result.risk_summary['trading_frozen']}",
        f"- freeze_events: {result.risk_summary['freeze_events']}",
        f"- risk_blocks: {result.risk_summary['risk_blocks']}",
        "",
        "## Files",
        "",
        "- reports/backtest_trade_ledger.csv",
        "- reports/backtest_data_diagnostics.csv",
        "- reports/backtest_execution_stress.csv",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_real_backtest()
