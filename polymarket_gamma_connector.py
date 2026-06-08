"""Read-only connector for Polymarket Gamma API market discovery."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from config import get_settings, load_env_file
from polymarket_public_client import (
    JSONPayload,
    PolymarketPublicClient,
    QueryValue,
    validate_limit,
)
from polymarket_rate_limiter import PolymarketRateLimiter

DEFAULT_TIMEOUT_SECONDS = 15


class PolymarketGammaConnector:
    """Public Gamma API client for markets, events, tags, search, and sports metadata."""

    def __init__(
        self,
        gamma_url: str,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        rate_limiter: PolymarketRateLimiter | None = None,
        rate_limit_enabled: bool = True,
    ) -> None:
        self.client = PolymarketPublicClient(
            gamma_url,
            timeout_seconds=timeout_seconds,
            rate_limiter=rate_limiter,
            rate_limit_enabled=rate_limit_enabled,
        )

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "PolymarketGammaConnector":
        """Create a Gamma connector from `.env` without requiring credentials."""
        load_env_file(env_file)
        settings = get_settings(load_env=False)
        return cls(
            gamma_url=settings.polymarket.gamma_url,
            rate_limit_enabled=settings.polymarket.rate_limit_enabled,
        )

    def list_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        **filters: QueryValue,
    ) -> JSONPayload:
        """List Gamma markets using public pre-trade metadata only."""
        params = {"limit": validate_limit(limit, 1000), "offset": offset, **filters}
        payload = self.client.get("/markets", params=params, limit_name="gamma_markets")
        return _normalize_gamma_ready_flags(payload)

    def get_market(self, market_id: int | str) -> dict[str, Any]:
        """Return a single Gamma market by numeric ID."""
        payload = self.client.get(f"/markets/{market_id}", limit_name="gamma_markets")
        return _expect_dict(_normalize_gamma_ready_flags(payload), "market")

    def get_market_by_slug(self, slug: str) -> dict[str, Any]:
        """Return a single Gamma market by slug."""
        payload = self.client.get(f"/markets/slug/{slug}", limit_name="gamma_markets")
        return _expect_dict(_normalize_gamma_ready_flags(payload), "market")

    def list_events(
        self,
        limit: int = 100,
        offset: int = 0,
        **filters: QueryValue,
    ) -> JSONPayload:
        """List Gamma events with optional public filters."""
        params = {"limit": validate_limit(limit, 1000), "offset": offset, **filters}
        payload = self.client.get("/events", params=params, limit_name="gamma_events")
        return _normalize_gamma_ready_flags(payload)

    def get_event(self, event_id: int | str) -> dict[str, Any]:
        """Return a single Gamma event by numeric ID."""
        payload = self.client.get(f"/events/{event_id}", limit_name="gamma_events")
        return _expect_dict(_normalize_gamma_ready_flags(payload), "event")

    def get_event_by_slug(self, slug: str) -> dict[str, Any]:
        """Return a single Gamma event by slug."""
        payload = self.client.get(f"/events/slug/{slug}", limit_name="gamma_events")
        return _expect_dict(_normalize_gamma_ready_flags(payload), "event")

    def search(
        self,
        query: str,
        limit_per_type: int = 10,
        **filters: QueryValue,
    ) -> dict[str, Any]:
        """Search public markets, events, and profiles without authentication."""
        if not query.strip():
            raise ValueError("query is required.")
        params = {
            "q": query,
            "limit_per_type": validate_limit(limit_per_type, 100),
            **filters,
        }
        payload = self.client.get(
            "/public-search", params=params, limit_name="gamma_search"
        )
        return _expect_dict(_normalize_gamma_ready_flags(payload), "search")

    def list_tags(self, **filters: QueryValue) -> JSONPayload:
        """List public Gamma tags."""
        return self.client.get("/tags", params=filters, limit_name="gamma_tags")

    def get_tag(self, tag_id: int | str) -> dict[str, Any]:
        """Return a Gamma tag by ID."""
        payload = self.client.get(f"/tags/{tag_id}", limit_name="gamma_tags")
        return _expect_dict(payload, "tag")

    def list_series(self, **filters: QueryValue) -> JSONPayload:
        """List Gamma series metadata."""
        return self.client.get("/series", params=filters, limit_name="gamma_general")

    def get_sports_metadata(self) -> dict[str, Any]:
        """Return Gamma sports metadata used for sports market discovery."""
        payload = self.client.get("/sports", limit_name="gamma_general")
        return _expect_dict(payload, "sports metadata")


def _expect_dict(payload: JSONPayload, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected {name} payload to be an object.")
    return payload


def _normalize_gamma_ready_flags(payload: JSONPayload) -> JSONPayload:
    """Remove stale Gamma readiness flags when public order books are enabled."""
    if isinstance(payload, list):
        return [_normalize_gamma_ready_flags(item) for item in payload]
    if not isinstance(payload, Mapping):
        return payload
    normalized = {
        key: _normalize_gamma_ready_flags(value)
        if isinstance(value, (Mapping, list))
        else value
        for key, value in payload.items()
    }
    if (
        normalized.get("ready") is False
        and normalized.get("closed") is False
        and normalized.get("acceptingOrders") is True
        and normalized.get("enableOrderBook") is True
    ):
        normalized.pop("ready", None)
    return normalized
