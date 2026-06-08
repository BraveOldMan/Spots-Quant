"""Regression tests for Polymarket live-mode safety gates."""

import pytest

import polymarket_connector
from polymarket_connector import PolymarketConnector


class FakeRateLimiter:
    """Collect rate-limit buckets requested by a connector."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def acquire(self, limit_name: str) -> float:
        self.calls.append(limit_name)
        return 0.0


def _live_connector() -> PolymarketConnector:
    connector = PolymarketConnector.__new__(PolymarketConnector)
    connector.is_live = True
    connector.client = object()
    connector.rate_limiter = FakeRateLimiter()
    return connector


def test_live_fetch_market_odds_requires_tokens() -> None:
    """Live odds reads fail closed instead of returning simulated prices."""
    connector = _live_connector()

    assert connector.fetch_market_odds("Home vs Away") == {}


def test_live_fetch_market_odds_requires_complete_order_book(monkeypatch) -> None:
    """Incomplete live books should block prices rather than random fallback."""
    connector = _live_connector()
    monkeypatch.setattr(connector, "get_order_book", lambda token: {"asks": []})

    assert connector.fetch_market_odds(
        "Home vs Away", {"home_price": "token-home"}
    ) == {}


def test_live_place_order_is_blocked() -> None:
    """Live order submission is false until real network posting is enabled."""
    connector = _live_connector()

    assert connector.place_order("token-home", "BUY", 1.0, 0.5) is False


def test_live_order_book_read_uses_clob_book_rate_limit() -> None:
    """Live order-book reads should respect Polymarket's /book quota."""

    class FakeClient:
        @staticmethod
        def get_order_book(token: str) -> dict[str, list[dict[str, str]]]:
            assert token == "token-home"
            return {"asks": [{"price": "0.50"}]}

    connector = _live_connector()
    connector.client = FakeClient()

    assert connector.get_order_book("token-home") == {"asks": [{"price": "0.50"}]}
    assert connector.rate_limiter.calls == ["clob_book"]


def test_polymarket_v2_sdk_is_available() -> None:
    """The connector should import the official Polymarket CLOB V2 SDK."""
    assert polymarket_connector.HAS_CLOB_CLIENT


def test_allow_real_money_missing_credentials_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live mode must not silently downgrade to random dry-run prices."""
    monkeypatch.setattr(polymarket_connector, "load_dotenv", lambda: None)
    monkeypatch.setenv("ALLOW_REAL_MONEY", "1")
    monkeypatch.delenv("POLYGON_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("FUNDER_ADDRESS", raising=False)

    with pytest.raises(RuntimeError, match="required Polymarket credentials"):
        PolymarketConnector()


def test_live_invalid_funder_address_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live connector should reject malformed funder addresses before client setup."""
    monkeypatch.setattr(polymarket_connector, "load_dotenv", lambda: None)
    monkeypatch.setenv("ALLOW_REAL_MONEY", "1")
    monkeypatch.setenv("POLYGON_PRIVATE_KEY", "0x" + "1" * 64)
    monkeypatch.setenv("FUNDER_ADDRESS", "not-an-address")

    with pytest.raises((RuntimeError, ValueError), match="FUNDER_ADDRESS"):
        PolymarketConnector()


def test_partial_clob_api_credentials_fail_closed() -> None:
    """Partial L2 API credentials should not create an ambiguous live client."""
    with pytest.raises(RuntimeError, match="Incomplete Polymarket CLOB API credentials"):
        PolymarketConnector._api_creds("key", None, "passphrase")
