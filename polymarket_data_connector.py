"""Read-only connector for Polymarket Data API portfolio and activity data."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from config import get_settings, load_env_file
from polymarket_public_client import (
    JSONPayload,
    PolymarketPublicClient,
    QueryValue,
    validate_address,
    validate_hash64,
    validate_limit,
)
from polymarket_rate_limiter import PolymarketRateLimiter

DEFAULT_TIMEOUT_SECONDS = 15


class PolymarketDataConnector:
    """Public Data API client for user positions, trades, activity, and analytics."""

    def __init__(
        self,
        data_url: str,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        rate_limiter: PolymarketRateLimiter | None = None,
        rate_limit_enabled: bool = True,
    ) -> None:
        self.client = PolymarketPublicClient(
            data_url,
            timeout_seconds=timeout_seconds,
            rate_limiter=rate_limiter,
            rate_limit_enabled=rate_limit_enabled,
        )

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "PolymarketDataConnector":
        """Create a Data connector from `.env` without requiring API keys."""
        load_env_file(env_file)
        settings = get_settings(load_env=False)
        return cls(
            data_url=settings.polymarket.data_url,
            rate_limit_enabled=settings.polymarket.rate_limit_enabled,
        )

    def get_positions(
        self,
        user: str,
        limit: int = 100,
        offset: int = 0,
        markets: Sequence[str] | None = None,
        event_ids: Sequence[int] | None = None,
        **filters: QueryValue,
    ) -> JSONPayload:
        """Return current positions for a public wallet address."""
        params = {
            "user": validate_address(user, "user"),
            "limit": validate_limit(limit, 500),
            "offset": validate_limit(offset, 10_000, "offset"),
            "market": _hash_list(markets),
            "eventId": event_ids,
            **filters,
        }
        return self.client.get("/positions", params=params, limit_name="data_positions")

    def get_closed_positions(
        self,
        user: str,
        limit: int = 10,
        offset: int = 0,
        markets: Sequence[str] | None = None,
        event_ids: Sequence[int] | None = None,
        **filters: QueryValue,
    ) -> JSONPayload:
        """Return closed positions for a public wallet address."""
        params = {
            "user": validate_address(user, "user"),
            "limit": validate_limit(limit, 50),
            "offset": validate_limit(offset, 100_000, "offset"),
            "market": _hash_list(markets),
            "eventId": event_ids,
            **filters,
        }
        return self.client.get(
            "/closed-positions",
            params=params,
            limit_name="data_closed_positions",
        )

    def get_trades(
        self,
        user: str | None = None,
        limit: int = 100,
        offset: int = 0,
        markets: Sequence[str] | None = None,
        event_ids: Sequence[int] | None = None,
        side: str | None = None,
        **filters: QueryValue,
    ) -> JSONPayload:
        """Return public trades for a user and/or market filters."""
        params = {
            "user": validate_address(user, "user") if user else None,
            "limit": validate_limit(limit, 10_000),
            "offset": validate_limit(offset, 10_000, "offset"),
            "market": _hash_list(markets),
            "eventId": event_ids,
            "side": _side(side) if side else None,
            **filters,
        }
        return self.client.get("/trades", params=params, limit_name="data_trades")

    def get_activity(
        self,
        user: str,
        limit: int = 100,
        offset: int = 0,
        markets: Sequence[str] | None = None,
        event_ids: Sequence[int] | None = None,
        **filters: QueryValue,
    ) -> JSONPayload:
        """Return public activity rows for a wallet address."""
        params = {
            "user": validate_address(user, "user"),
            "limit": validate_limit(limit, 500),
            "offset": validate_limit(offset, 10_000, "offset"),
            "market": _hash_list(markets),
            "eventId": event_ids,
            **filters,
        }
        return self.client.get("/activity", params=params, limit_name="data_general")

    def get_holders(
        self,
        markets: Sequence[str],
        limit: int = 20,
        min_balance: int = 1,
    ) -> JSONPayload:
        """Return top holders for one or more condition IDs."""
        params = {
            "market": _required_hash_list(markets, "markets"),
            "limit": validate_limit(limit, 20),
            "minBalance": validate_limit(min_balance, 999_999, "min_balance"),
        }
        return self.client.get("/holders", params=params, limit_name="data_general")

    def get_total_value(
        self,
        user: str,
        markets: Sequence[str] | None = None,
    ) -> JSONPayload:
        """Return the total current value of a user's positions."""
        params = {
            "user": validate_address(user, "user"),
            "market": _hash_list(markets),
        }
        return self.client.get("/value", params=params, limit_name="data_general")

    def get_open_interest(self, markets: Sequence[str] | None = None) -> JSONPayload:
        """Return open interest for optional condition IDs."""
        return self.client.get(
            "/oi",
            params={"market": _hash_list(markets)},
            limit_name="data_general",
        )

    def get_total_markets_traded(self, user: str) -> dict[str, Any]:
        """Return total markets traded by a public wallet address."""
        payload = self.client.get(
            "/traded",
            params={"user": validate_address(user, "user")},
            limit_name="data_general",
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Expected traded payload to be an object.")
        return payload

    def get_market_positions(
        self,
        market: str,
        user: str | None = None,
        limit: int = 50,
        offset: int = 0,
        **filters: QueryValue,
    ) -> JSONPayload:
        """Return Data API positions for a condition ID, optionally by user."""
        params = {
            "market": validate_hash64(market, "market"),
            "user": validate_address(user, "user") if user else None,
            "limit": validate_limit(limit, 500),
            "offset": validate_limit(offset, 10_000, "offset"),
            **filters,
        }
        return self.client.get(
            "/v1/market-positions",
            params=params,
            limit_name="data_positions",
        )

    def get_builder_leaderboard(
        self,
        time_period: str = "DAY",
        limit: int = 25,
        offset: int = 0,
    ) -> JSONPayload:
        """Return aggregated builder leaderboard rows."""
        params = {
            "timePeriod": _time_period(time_period),
            "limit": validate_limit(limit, 50),
            "offset": validate_limit(offset, 1000, "offset"),
        }
        return self.client.get(
            "/v1/builders/leaderboard",
            params=params,
            limit_name="data_general",
        )


def _side(value: str) -> str:
    normalized = value.upper()
    if normalized not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL.")
    return normalized


def _time_period(value: str) -> str:
    normalized = value.upper()
    if normalized not in {"DAY", "WEEK", "MONTH", "ALL"}:
        raise ValueError("time_period must be DAY, WEEK, MONTH, or ALL.")
    return normalized


def _hash_list(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    return [validate_hash64(value, "market") for value in values]


def _required_hash_list(values: Sequence[str], name: str) -> list[str]:
    if not values:
        raise ValueError(f"{name} must not be empty.")
    return [validate_hash64(value, "market") for value in values]
