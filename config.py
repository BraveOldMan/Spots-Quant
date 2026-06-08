"""Central configuration for Spots-Quant.

The module keeps legacy constant names for existing imports while exposing typed
settings for new code. Defaults preserve the current backtest and live-risk
baseline; optional environment variables can override them at process start.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EV_THRESHOLD = 1.05
DEFAULT_MIN_HISTORY = 3
DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_KELLY_MULT = 0.05
DEFAULT_MAX_BET_FRACTION = 0.03
DEFAULT_MAX_MATCH_EXPOSURE = 0.05
DEFAULT_MAX_DRAWDOWN_LIMIT = 0.15

DEFAULT_ELO_START = 1500.0
DEFAULT_ELO_K_DEFAULT = 20.0
DEFAULT_ELO_K_FACTORS = {1: 60.0, 4: 50.0, 6: 40.0, 7: 50.0, 9: 50.0, 22: 50.0, 10: 10.0}
DEFAULT_ELO_HOME_ADV = 80.0
DEFAULT_HALF_LIFE_DAYS = 365.0
DEFAULT_NEUTRAL_LEAGUE_IDS = {10}
DEFAULT_DC_RHO = -0.05
DEFAULT_GOALS_BLEND = 0.6
DEFAULT_PXG_BLEND = 0.4
DEFAULT_PXG_W_ON = 0.25
DEFAULT_PXG_W_OFF = 0.05
DEFAULT_SHRINK_K = 5.0
DEFAULT_DISTILL_FEATURES = ("elo_diff", "mom_diff", "rating_diff", "mif_home", "mif_away")

DEFAULT_CACHE_TTL_ODDS = 300
DEFAULT_CACHE_TTL_FIXTURES = 3600
DEFAULT_CACHE_TTL_HISTORICAL: int | None = None
DEFAULT_API_CACHE_DB = "api_cache.db"
DEFAULT_REPORT_DIR = "reports"
DEFAULT_BACKTEST_CSV_PATHS = (
    "data_seasons/E0_2223.csv",
    "data_seasons/E0_2324.csv",
    "data_seasons/E0_2425.csv",
)
DEFAULT_ODDS_MODE = "closing"
DEFAULT_COMMISSION_ON_WIN = 0.0

DEFAULT_POLYMARKET_HOST = "https://clob.polymarket.com"
DEFAULT_POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"
DEFAULT_POLYMARKET_DATA_URL = "https://data-api.polymarket.com"
DEFAULT_POLYMARKET_CHAIN_ID = 137
DEFAULT_POLYMARKET_SIGNATURE_TYPE = 1
DEFAULT_POLYMARKET_RELAYER_URL = "https://relayer-v2.polymarket.com"
DEFAULT_POLYMARKET_RATE_LIMIT_ENABLED = True
ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")


@dataclass(frozen=True)
class StrategySettings:
    """Signal threshold settings available before a betting decision."""

    ev_threshold: float = DEFAULT_EV_THRESHOLD
    min_history: int = DEFAULT_MIN_HISTORY


@dataclass(frozen=True)
class RiskSettings:
    """Capital, Kelly, exposure, and drawdown guardrails."""

    initial_capital: float = DEFAULT_INITIAL_CAPITAL
    kelly_mult: float = DEFAULT_KELLY_MULT
    max_bet_fraction: float = DEFAULT_MAX_BET_FRACTION
    max_match_exposure: float | None = DEFAULT_MAX_MATCH_EXPOSURE
    max_drawdown_limit: float = DEFAULT_MAX_DRAWDOWN_LIMIT


@dataclass(frozen=True)
class ModelSettings:
    """Model feature and rating parameters."""

    elo_start: float = DEFAULT_ELO_START
    elo_k_default: float = DEFAULT_ELO_K_DEFAULT
    elo_k_factors: dict[int, float] | None = None
    elo_home_adv: float = DEFAULT_ELO_HOME_ADV
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS
    neutral_league_ids: set[int] | None = None
    dc_rho: float = DEFAULT_DC_RHO
    goals_blend: float = DEFAULT_GOALS_BLEND
    pxg_blend: float = DEFAULT_PXG_BLEND
    pxg_w_on: float = DEFAULT_PXG_W_ON
    pxg_w_off: float = DEFAULT_PXG_W_OFF
    shrink_k: float = DEFAULT_SHRINK_K
    distill_features: tuple[str, ...] = DEFAULT_DISTILL_FEATURES

    def __post_init__(self) -> None:
        """Fill mutable defaults without sharing state between instances."""
        if self.elo_k_factors is None:
            object.__setattr__(self, "elo_k_factors", dict(DEFAULT_ELO_K_FACTORS))
        if self.neutral_league_ids is None:
            object.__setattr__(self, "neutral_league_ids", set(DEFAULT_NEUTRAL_LEAGUE_IDS))


@dataclass(frozen=True)
class CacheSettings:
    """API cache time-to-live settings in seconds."""

    odds_seconds: int = DEFAULT_CACHE_TTL_ODDS
    fixtures_seconds: int = DEFAULT_CACHE_TTL_FIXTURES
    historical_seconds: int | None = DEFAULT_CACHE_TTL_HISTORICAL
    db_path: str = DEFAULT_API_CACHE_DB


@dataclass(frozen=True)
class BacktestSettings:
    """Default offline backtest inputs and settlement assumptions."""

    csv_paths: tuple[str, ...] = DEFAULT_BACKTEST_CSV_PATHS
    odds_mode: str = DEFAULT_ODDS_MODE
    commission_on_win: float = DEFAULT_COMMISSION_ON_WIN
    report_dir: str = DEFAULT_REPORT_DIR


@dataclass(frozen=True)
class ApiSettings:
    """Read-only external API credentials and endpoints."""

    api_football_key: str | None = None
    the_odds_api_key: str | None = None


@dataclass(frozen=True)
class PolymarketSettings:
    """Polymarket connector settings; dry-run is the default."""

    allow_real_money: bool = False
    private_key: str | None = None
    funder_address: str | None = None
    host: str = DEFAULT_POLYMARKET_HOST
    gamma_url: str = DEFAULT_POLYMARKET_GAMMA_URL
    data_url: str = DEFAULT_POLYMARKET_DATA_URL
    chain_id: int = DEFAULT_POLYMARKET_CHAIN_ID
    signature_type: int = DEFAULT_POLYMARKET_SIGNATURE_TYPE
    clob_api_key: str | None = None
    clob_api_secret: str | None = None
    clob_api_passphrase: str | None = None
    relayer_url: str = DEFAULT_POLYMARKET_RELAYER_URL
    api_address: str | None = None
    relayer_api_key: str | None = None
    relayer_api_key_address: str | None = None
    rate_limit_enabled: bool = DEFAULT_POLYMARKET_RATE_LIMIT_ENABLED


@dataclass(frozen=True)
class Settings:
    """Complete application configuration snapshot."""

    strategy: StrategySettings
    risk: RiskSettings
    model: ModelSettings
    cache: CacheSettings
    backtest: BacktestSettings
    api: ApiSettings
    polymarket: PolymarketSettings


def load_env_file(path: str | Path = ".env", override: bool = False) -> None:
    """Load key-value pairs from an env file without logging secret values."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value


def get_settings(env_file: str | Path = ".env", load_env: bool = True) -> Settings:
    """Return a typed settings snapshot using defaults plus environment overrides."""
    if load_env:
        load_env_file(env_file)
    return Settings(
        strategy=StrategySettings(
            ev_threshold=_env_float("SPOTS_EV_THRESHOLD", DEFAULT_EV_THRESHOLD),
            min_history=_env_int("SPOTS_MIN_HISTORY", DEFAULT_MIN_HISTORY),
        ),
        risk=RiskSettings(
            initial_capital=_env_float("SPOTS_INITIAL_CAPITAL", DEFAULT_INITIAL_CAPITAL),
            kelly_mult=_env_float("SPOTS_KELLY_MULT", DEFAULT_KELLY_MULT),
            max_bet_fraction=_env_float("SPOTS_MAX_BET_FRACTION", DEFAULT_MAX_BET_FRACTION),
            max_match_exposure=_env_optional_float(
                "SPOTS_MAX_MATCH_EXPOSURE", DEFAULT_MAX_MATCH_EXPOSURE
            ),
            max_drawdown_limit=_env_float(
                "SPOTS_MAX_DRAWDOWN_LIMIT", DEFAULT_MAX_DRAWDOWN_LIMIT
            ),
        ),
        model=ModelSettings(
            elo_start=_env_float("SPOTS_ELO_START", DEFAULT_ELO_START),
            elo_k_default=_env_float("SPOTS_ELO_K_DEFAULT", DEFAULT_ELO_K_DEFAULT),
            elo_home_adv=_env_float("SPOTS_ELO_HOME_ADV", DEFAULT_ELO_HOME_ADV),
            half_life_days=_env_float("SPOTS_HALF_LIFE_DAYS", DEFAULT_HALF_LIFE_DAYS),
            dc_rho=_env_float("SPOTS_DC_RHO", DEFAULT_DC_RHO),
            goals_blend=_env_float("SPOTS_GOALS_BLEND", DEFAULT_GOALS_BLEND),
            pxg_blend=_env_float("SPOTS_PXG_BLEND", DEFAULT_PXG_BLEND),
            pxg_w_on=_env_float("SPOTS_PXG_W_ON", DEFAULT_PXG_W_ON),
            pxg_w_off=_env_float("SPOTS_PXG_W_OFF", DEFAULT_PXG_W_OFF),
            shrink_k=_env_float("SPOTS_SHRINK_K", DEFAULT_SHRINK_K),
            distill_features=_env_csv_tuple("SPOTS_DISTILL_FEATURES", DEFAULT_DISTILL_FEATURES),
        ),
        cache=CacheSettings(
            odds_seconds=_env_int("SPOTS_CACHE_TTL_ODDS", DEFAULT_CACHE_TTL_ODDS),
            fixtures_seconds=_env_int(
                "SPOTS_CACHE_TTL_FIXTURES", DEFAULT_CACHE_TTL_FIXTURES
            ),
            historical_seconds=_env_optional_int(
                "SPOTS_CACHE_TTL_HISTORICAL", DEFAULT_CACHE_TTL_HISTORICAL
            ),
            db_path=os.environ.get("SPOTS_API_CACHE_DB", DEFAULT_API_CACHE_DB),
        ),
        backtest=BacktestSettings(
            csv_paths=_env_path_tuple("SPOTS_BACKTEST_CSV_PATHS", DEFAULT_BACKTEST_CSV_PATHS),
            odds_mode=os.environ.get("SPOTS_ODDS_MODE", DEFAULT_ODDS_MODE),
            commission_on_win=_env_float(
                "SPOTS_COMMISSION_ON_WIN", DEFAULT_COMMISSION_ON_WIN
            ),
            report_dir=os.environ.get("SPOTS_REPORT_DIR", DEFAULT_REPORT_DIR),
        ),
        api=ApiSettings(
            api_football_key=_env_optional_str("API_FOOTBALL_KEY"),
            the_odds_api_key=_env_optional_str("THE_ODDS_API_KEY"),
        ),
        polymarket=PolymarketSettings(
            allow_real_money=_env_bool("ALLOW_REAL_MONEY", False),
            private_key=_env_optional_str("POLYGON_PRIVATE_KEY"),
            funder_address=_env_optional_address("FUNDER_ADDRESS"),
            host=os.environ.get("POLYMARKET_HOST", DEFAULT_POLYMARKET_HOST),
            gamma_url=os.environ.get(
                "POLYMARKET_GAMMA_URL", DEFAULT_POLYMARKET_GAMMA_URL
            ),
            data_url=os.environ.get(
                "POLYMARKET_DATA_URL", DEFAULT_POLYMARKET_DATA_URL
            ),
            chain_id=_env_int("POLYMARKET_CHAIN_ID", DEFAULT_POLYMARKET_CHAIN_ID),
            signature_type=_env_int(
                "POLYMARKET_SIGNATURE_TYPE", DEFAULT_POLYMARKET_SIGNATURE_TYPE
            ),
            clob_api_key=_env_first_str("POLYMARKET_CLOB_API_KEY", "CLOB_API_KEY"),
            clob_api_secret=_env_first_str("POLYMARKET_CLOB_API_SECRET", "CLOB_SECRET"),
            clob_api_passphrase=_env_first_str(
                "POLYMARKET_CLOB_API_PASSPHRASE", "CLOB_PASS_PHRASE"
            ),
            relayer_url=os.environ.get(
                "POLYMARKET_RELAYER_URL", DEFAULT_POLYMARKET_RELAYER_URL
            ),
            api_address=_env_optional_address("POLYMARKET_API_ADDRESS"),
            relayer_api_key=_env_first_str("POLYMARKET_RELAYER_API_KEY", "RELAYER_API_KEY"),
            relayer_api_key_address=_env_first_address(
                "POLYMARKET_RELAYER_API_KEY_ADDRESS", "RELAYER_API_KEY_ADDRESS"
            ),
            rate_limit_enabled=_env_bool(
                "POLYMARKET_RATE_LIMIT_ENABLED",
                DEFAULT_POLYMARKET_RATE_LIMIT_ENABLED,
            ),
        ),
    )


def redacted_settings(settings: Settings | None = None) -> dict[str, object]:
    """Return settings safe for logs or reports, with secrets masked."""
    active = settings or SETTINGS
    return {
        "strategy": active.strategy,
        "risk": active.risk,
        "model": active.model,
        "cache": active.cache,
        "backtest": active.backtest,
        "api": {
            "api_football_key": _mask_secret(active.api.api_football_key),
            "the_odds_api_key": _mask_secret(active.api.the_odds_api_key),
        },
        "polymarket": {
            "allow_real_money": active.polymarket.allow_real_money,
            "private_key": _mask_secret(active.polymarket.private_key),
            "funder_address": _mask_secret(active.polymarket.funder_address),
            "host": active.polymarket.host,
            "gamma_url": active.polymarket.gamma_url,
            "data_url": active.polymarket.data_url,
            "chain_id": active.polymarket.chain_id,
            "signature_type": active.polymarket.signature_type,
            "clob_api_key": _mask_secret(active.polymarket.clob_api_key),
            "clob_api_secret": _mask_secret(active.polymarket.clob_api_secret),
            "clob_api_passphrase": _mask_secret(active.polymarket.clob_api_passphrase),
            "relayer_url": active.polymarket.relayer_url,
            "api_address": _mask_secret(active.polymarket.api_address),
            "relayer_api_key": _mask_secret(active.polymarket.relayer_api_key),
            "relayer_api_key_address": _mask_secret(
                active.polymarket.relayer_api_key_address
            ),
            "rate_limit_enabled": active.polymarket.rate_limit_enabled,
        },
    }


def _env_optional_str(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _env_first_str(*names: str) -> str | None:
    for name in names:
        value = _env_optional_str(name)
        if value is not None:
            return value
    return None


def _env_optional_address(name: str) -> str | None:
    """Read an optional EVM address and fail closed on malformed values."""
    value = _env_optional_str(name)
    if value is None:
        return None
    if not ADDRESS_RE.match(value):
        raise ValueError(f"{name} must be a 0x-prefixed 40-byte address.")
    return value


def _env_first_address(*names: str) -> str | None:
    """Return the first configured EVM address after strict format validation."""
    for name in names:
        value = _env_optional_address(name)
        if value is not None:
            return value
    return None


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float.") from exc


def _env_optional_float(name: str, default: float | None) -> float | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    if value.strip().lower() in {"none", "null"}:
        return None
    return _env_float(name, default if default is not None else 0.0)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _env_optional_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    if value.strip().lower() in {"none", "null"}:
        return None
    return _env_int(name, default if default is not None else 0)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean: 1/0, true/false, yes/no, on/off.")


def _env_csv_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _env_path_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    separator = ";" if ";" in value else ","
    return tuple(item.strip() for item in value.split(separator) if item.strip())


def _mask_secret(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


SETTINGS = get_settings()

# Legacy constant exports. Keep these names stable for existing modules.
EV_THRESHOLD = SETTINGS.strategy.ev_threshold
MIN_HISTORY = SETTINGS.strategy.min_history
INITIAL_CAPITAL = SETTINGS.risk.initial_capital
KELLY_MULT = SETTINGS.risk.kelly_mult
MAX_BET_FRACTION = SETTINGS.risk.max_bet_fraction
MAX_MATCH_EXPOSURE = SETTINGS.risk.max_match_exposure
MAX_DRAWDOWN_LIMIT = SETTINGS.risk.max_drawdown_limit

ELO_START = SETTINGS.model.elo_start
ELO_K_DEFAULT = SETTINGS.model.elo_k_default
ELO_K_FACTORS = SETTINGS.model.elo_k_factors
ELO_HOME_ADV = SETTINGS.model.elo_home_adv
HALF_LIFE_DAYS = SETTINGS.model.half_life_days
NEUTRAL_LEAGUE_IDS = SETTINGS.model.neutral_league_ids
DC_RHO = SETTINGS.model.dc_rho
GOALS_BLEND = SETTINGS.model.goals_blend
PXG_BLEND = SETTINGS.model.pxg_blend
PXG_W_ON = SETTINGS.model.pxg_w_on
PXG_W_OFF = SETTINGS.model.pxg_w_off
SHRINK_K = SETTINGS.model.shrink_k
DISTILL_FEATURES = list(SETTINGS.model.distill_features)

CACHE_TTL_ODDS = SETTINGS.cache.odds_seconds
CACHE_TTL_FIXTURES = SETTINGS.cache.fixtures_seconds
CACHE_TTL_HISTORICAL = SETTINGS.cache.historical_seconds
