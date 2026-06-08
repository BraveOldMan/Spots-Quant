"""Read-only public REST connector for Polymarket CLOB market data."""

from __future__ import annotations

from typing import Any

from config import get_settings, load_env_file
from polymarket_public_client import (
    JSONPayload,
    PolymarketPublicClient,
    QueryValue,
    validate_token_id,
)
from polymarket_rate_limiter import PolymarketRateLimiter

DEFAULT_TIMEOUT_SECONDS = 15
PRICE_HISTORY_INTERVALS = {"max", "all", "1m", "1w", "1d", "6h", "1h"}


class PolymarketClobConnector:
    """Public CLOB client for order books, prices, spreads, and price history."""

    def __init__(
        self,
        clob_url: str,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        rate_limiter: PolymarketRateLimiter | None = None,
        rate_limit_enabled: bool = True,
    ) -> None:
        self.client = PolymarketPublicClient(
            clob_url,
            timeout_seconds=timeout_seconds,
            rate_limiter=rate_limiter,
            rate_limit_enabled=rate_limit_enabled,
        )

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "PolymarketClobConnector":
        """Create a public CLOB connector from `.env` without requiring credentials."""
        load_env_file(env_file)
        settings = get_settings(load_env=False)
        return cls(
            clob_url=settings.polymarket.host,
            rate_limit_enabled=settings.polymarket.rate_limit_enabled,
        )

    def get_order_book(self, token_id: str) -> dict[str, Any]:
        """Return public order-book summary for one outcome token."""
        payload = self.client.get(
            "/book",
            params={"token_id": validate_token_id(token_id)},
            limit_name="clob_book",
        )
        return _expect_dict(payload, "order book")

    def get_price(self, token_id: str, side: str) -> dict[str, Any]:
        """Return best bid for BUY or best ask for SELL."""
        payload = self.client.get(
            "/price",
            params={"token_id": validate_token_id(token_id), "side": _side(side)},
            limit_name="clob_price",
        )
        return _expect_dict(payload, "price")

    def get_midpoint(self, token_id: str) -> dict[str, Any]:
        """Return midpoint price for one outcome token."""
        payload = self.client.get(
            "/midpoint",
            params={"token_id": validate_token_id(token_id)},
            limit_name="clob_midpoint",
        )
        return _expect_dict(payload, "midpoint")

    def get_spread(self, token_id: str) -> dict[str, Any]:
        """Return bid-ask spread for one outcome token."""
        payload = self.client.get(
            "/spread",
            params={"token_id": validate_token_id(token_id)},
            limit_name="clob_price",
        )
        return _expect_dict(payload, "spread")

    def get_prices_history(
        self,
        token_id: str,
        interval: str | None = None,
        start_ts: int | float | None = None,
        end_ts: int | float | None = None,
        fidelity: int | None = None,
    ) -> dict[str, Any]:
        """Return historical CLOB prices for one outcome token."""
        params: dict[str, QueryValue] = {
            "market": validate_token_id(token_id, "market"),
            "startTs": start_ts,
            "endTs": end_ts,
            "interval": _interval(interval) if interval else None,
            "fidelity": fidelity,
        }
        payload = self.client.get(
            "/prices-history",
            params=params,
            limit_name="clob_prices_history",
        )
        return _expect_dict(payload, "prices history")

    def get_market_by_token(self, token_id: str) -> dict[str, Any]:
        """Resolve parent condition and paired tokens from an outcome token ID."""
        payload = self.client.get(
            f"/markets-by-token/{validate_token_id(token_id)}",
            limit_name="clob_general",
        )
        return _expect_dict(payload, "market by token")

    def get_server_time(self) -> int:
        """Return Polymarket CLOB server Unix timestamp."""
        payload = self.client.get("/time", limit_name="clob_general")
        if not isinstance(payload, int):
            raise RuntimeError("Expected server time payload to be an integer.")
        return payload


def _side(value: str) -> str:
    normalized = value.upper()
    if normalized not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL.")
    return normalized


def _interval(value: str) -> str:
    if value not in PRICE_HISTORY_INTERVALS:
        raise ValueError("interval must be max, all, 1m, 1w, 1d, 6h, or 1h.")
    return value


def _expect_dict(payload: JSONPayload, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected {name} payload to be an object.")
    return payload
