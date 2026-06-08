"""Tests for read-only Polymarket Gamma, Data, and CLOB connectors."""

from __future__ import annotations

from typing import Any

import pytest

import polymarket_public_client
from polymarket_clob_connector import PolymarketClobConnector
from polymarket_data_connector import PolymarketDataConnector
from polymarket_gamma_connector import PolymarketGammaConnector
from polymarket_public_client import PolymarketPublicClient

ADDRESS = "0x1111111111111111111111111111111111111111"
MARKET_A = "0x" + "a" * 64
MARKET_B = "0x" + "b" * 64


class FakeRateLimiter:
    """Collect Polymarket rate-limit bucket names."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def acquire(self, limit_name: str) -> float:
        self.calls.append(limit_name)
        return 0.0


class FakeResponse:
    """Minimal requests response used by connector tests."""

    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        return self._payload


def test_public_client_rejects_non_https_url() -> None:
    """Polymarket connectors should never be configured with plain HTTP."""
    with pytest.raises(ValueError, match="https"):
        PolymarketPublicClient("http://gamma-api.polymarket.com")


def test_gamma_connector_uses_public_endpoints_and_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gamma connector should route market discovery through Gamma API."""
    calls: list[dict[str, Any]] = []

    def fake_request(
        method: str,
        url: str,
        params: dict[str, Any],
        json: dict[str, Any] | None,
        timeout: int,
    ) -> FakeResponse:
        calls.append(
            {
                "method": method,
                "url": url,
                "params": params,
                "json": json,
                "timeout": timeout,
            }
        )
        return FakeResponse({"ok": True})

    monkeypatch.setattr(polymarket_public_client.requests, "request", fake_request)
    limiter = FakeRateLimiter()
    connector = PolymarketGammaConnector(
        "https://gamma-api.polymarket.com",
        rate_limiter=limiter,
    )

    assert connector.list_markets(limit=2, active=True) == {"ok": True}
    assert connector.search("epl", limit_per_type=3) == {"ok": True}

    assert calls[0]["url"] == "https://gamma-api.polymarket.com/markets"
    assert calls[0]["params"] == {"limit": 2, "offset": 0, "active": "true"}
    assert calls[1]["url"] == "https://gamma-api.polymarket.com/public-search"
    assert calls[1]["params"] == {"q": "epl", "limit_per_type": 3}
    assert limiter.calls == ["gamma_markets", "gamma_search"]


def test_gamma_connector_removes_stale_ready_false_for_open_order_books(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gamma rows with live order books should not be blocked by stale ready flags."""

    def fake_request(
        method: str,
        url: str,
        params: dict[str, Any],
        json: dict[str, Any] | None,
        timeout: int,
    ) -> FakeResponse:
        return FakeResponse(
            {
                "events": [
                    {
                        "title": "Armenia vs. Moldova",
                        "markets": [
                            {
                                "question": "Will Armenia win?",
                                "ready": False,
                                "closed": False,
                                "acceptingOrders": True,
                                "enableOrderBook": True,
                            },
                            {
                                "question": "Unavailable market",
                                "ready": False,
                                "closed": False,
                                "acceptingOrders": True,
                                "enableOrderBook": False,
                            },
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr(polymarket_public_client.requests, "request", fake_request)
    connector = PolymarketGammaConnector("https://gamma-api.polymarket.com")

    payload = connector.search("armenia moldova")
    markets = payload["events"][0]["markets"]

    assert "ready" not in markets[0]
    assert markets[1]["ready"] is False


def test_data_connector_validates_addresses_and_condition_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Data connector should validate public wallet and market identifiers."""
    calls: list[dict[str, Any]] = []

    def fake_request(
        method: str,
        url: str,
        params: dict[str, Any],
        json: dict[str, Any] | None,
        timeout: int,
    ) -> FakeResponse:
        calls.append({"method": method, "url": url, "params": params})
        return FakeResponse([{"ok": True}])

    monkeypatch.setattr(polymarket_public_client.requests, "request", fake_request)
    limiter = FakeRateLimiter()
    connector = PolymarketDataConnector(
        "https://data-api.polymarket.com",
        rate_limiter=limiter,
    )

    assert connector.get_positions(ADDRESS, markets=[MARKET_A], limit=1) == [
        {"ok": True}
    ]
    assert connector.get_trades(markets=[MARKET_A, MARKET_B], side="buy") == [
        {"ok": True}
    ]

    assert calls[0]["url"] == "https://data-api.polymarket.com/positions"
    assert calls[0]["params"]["user"] == ADDRESS
    assert calls[0]["params"]["market"] == MARKET_A
    assert calls[1]["url"] == "https://data-api.polymarket.com/trades"
    assert calls[1]["params"]["market"] == f"{MARKET_A},{MARKET_B}"
    assert calls[1]["params"]["side"] == "BUY"
    assert limiter.calls == ["data_positions", "data_trades"]

    with pytest.raises(ValueError, match="user"):
        connector.get_positions("bad-address")
    with pytest.raises(ValueError, match="market"):
        connector.get_holders(["bad-market"])


def test_clob_connector_uses_public_read_only_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLOB public connector should expose market data without trading calls."""
    calls: list[dict[str, Any]] = []

    def fake_request(
        method: str,
        url: str,
        params: dict[str, Any],
        json: dict[str, Any] | None,
        timeout: int,
    ) -> FakeResponse:
        calls.append({"method": method, "url": url, "params": params})
        if url.endswith("/time"):
            return FakeResponse(1234567890)
        return FakeResponse({"ok": True})

    monkeypatch.setattr(polymarket_public_client.requests, "request", fake_request)
    limiter = FakeRateLimiter()
    connector = PolymarketClobConnector(
        "https://clob.polymarket.com",
        rate_limiter=limiter,
    )

    assert connector.get_order_book("123") == {"ok": True}
    assert connector.get_price("123", "sell") == {"ok": True}
    assert connector.get_spread("123") == {"ok": True}
    assert connector.get_prices_history("123", interval="1h") == {"ok": True}
    assert connector.get_market_by_token("123") == {"ok": True}
    assert connector.get_server_time() == 1234567890

    assert calls[0]["url"] == "https://clob.polymarket.com/book"
    assert calls[1]["url"] == "https://clob.polymarket.com/price"
    assert calls[1]["params"] == {"token_id": "123", "side": "SELL"}
    assert calls[2]["url"] == "https://clob.polymarket.com/spread"
    assert calls[3]["url"] == "https://clob.polymarket.com/prices-history"
    assert calls[3]["params"] == {"market": "123", "interval": "1h"}
    assert calls[4]["url"] == "https://clob.polymarket.com/markets-by-token/123"
    assert calls[5]["url"] == "https://clob.polymarket.com/time"
    assert limiter.calls == [
        "clob_book",
        "clob_price",
        "clob_price",
        "clob_prices_history",
        "clob_general",
        "clob_general",
    ]

    with pytest.raises(ValueError, match="token_id"):
        connector.get_order_book("not-a-token")
    with pytest.raises(ValueError, match="side"):
        connector.get_price("123", "hold")
