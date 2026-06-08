"""Read-only Polymarket Relayer API connector.

The connector only builds Relayer API-key headers and supports safe read-only
status checks. It intentionally does not expose transaction execution helpers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import requests

from config import get_settings, load_env_file
from polymarket_rate_limiter import PolymarketRateLimiter

ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
DEFAULT_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class RelayerCredentials:
    """Relayer API-key auth headers for read-only API checks."""

    api_key: str
    api_key_address: str


class PolymarketRelayerConnector:
    """Small read-only client for Polymarket Relayer API-key validation."""

    def __init__(
        self,
        relayer_url: str,
        credentials: RelayerCredentials,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        rate_limiter: PolymarketRateLimiter | None = None,
        rate_limit_enabled: bool = True,
    ) -> None:
        self.relayer_url = relayer_url.rstrip("/")
        self.credentials = credentials
        self.timeout_seconds = timeout_seconds
        self.rate_limiter = rate_limiter or PolymarketRateLimiter(
            enabled=rate_limit_enabled
        )
        self._validate()

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "PolymarketRelayerConnector":
        """Create a connector from `.env`/environment without logging secrets."""
        load_env_file(env_file)
        settings = get_settings(load_env=False)
        key = settings.polymarket.relayer_api_key
        address = settings.polymarket.relayer_api_key_address
        if not key or not address:
            raise RuntimeError("Polymarket Relayer API key or key address is missing.")
        return cls(
            relayer_url=settings.polymarket.relayer_url,
            credentials=RelayerCredentials(api_key=key, api_key_address=address),
            rate_limit_enabled=settings.polymarket.rate_limit_enabled,
        )

    def headers(self) -> dict[str, str]:
        """Return official Relayer API-key auth headers."""
        return {
            "RELAYER_API_KEY": self.credentials.api_key,
            "RELAYER_API_KEY_ADDRESS": self.credentials.api_key_address,
        }

    def list_api_keys(self) -> list[dict[str, Any]]:
        """Return Relayer API keys visible to this credential."""
        self.rate_limiter.acquire("relayer_api_key")
        response = requests.get(
            f"{self.relayer_url}/relayer/api/keys",
            headers=self.headers(),
            timeout=self.timeout_seconds,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Relayer API key check failed: HTTP {response.status_code}")
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError("Relayer API key check returned a non-list payload.")
        return payload

    def _validate(self) -> None:
        if not self.relayer_url.startswith("https://"):
            raise ValueError("Polymarket relayer URL must use https.")
        if not self.credentials.api_key.strip():
            raise ValueError("Relayer API key is empty.")
        if not ADDRESS_RE.match(self.credentials.api_key_address):
            raise ValueError("Relayer API key address must be a 0x-prefixed address.")
