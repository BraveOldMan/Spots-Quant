"""Tests for read-only Polymarket Relayer API configuration."""

from __future__ import annotations

import pytest

from polymarket_relayer_connector import PolymarketRelayerConnector, RelayerCredentials


class FakeRateLimiter:
    """Collect rate-limit buckets requested by a connector."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def acquire(self, limit_name: str) -> float:
        self.calls.append(limit_name)
        return 0.0


def test_relayer_headers_use_official_names() -> None:
    """Relayer auth headers should match the official API reference."""
    connector = PolymarketRelayerConnector(
        "https://relayer-v2.polymarket.com/",
        RelayerCredentials(
            api_key="key",
            api_key_address="0x1111111111111111111111111111111111111111",
        ),
    )

    assert connector.headers() == {
        "RELAYER_API_KEY": "key",
        "RELAYER_API_KEY_ADDRESS": "0x1111111111111111111111111111111111111111",
    }
    assert connector.relayer_url == "https://relayer-v2.polymarket.com"


def test_relayer_connector_rejects_bad_address() -> None:
    """Misconfigured addresses should fail before any network call."""
    with pytest.raises(ValueError, match="0x-prefixed address"):
        PolymarketRelayerConnector(
            "https://relayer-v2.polymarket.com",
            RelayerCredentials(api_key="key", api_key_address="not-an-address"),
        )


def test_relayer_connector_lists_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read-only API key listing should parse a list payload."""

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> list[dict[str, str]]:
            return [{"apiKey": "redacted", "address": "0xabc"}]

    calls = []

    def fake_get(url: str, headers: dict[str, str], timeout: int) -> FakeResponse:
        calls.append((url, headers, timeout))
        return FakeResponse()

    monkeypatch.setattr("polymarket_relayer_connector.requests.get", fake_get)
    limiter = FakeRateLimiter()
    connector = PolymarketRelayerConnector(
        "https://relayer-v2.polymarket.com",
        RelayerCredentials(
            api_key="key",
            api_key_address="0x1111111111111111111111111111111111111111",
        ),
        rate_limiter=limiter,
    )

    payload = connector.list_api_keys()

    assert payload == [{"apiKey": "redacted", "address": "0xabc"}]
    assert calls[0][0] == "https://relayer-v2.polymarket.com/relayer/api/keys"
    assert calls[0][1]["RELAYER_API_KEY"] == "key"
    assert limiter.calls == ["relayer_api_key"]
