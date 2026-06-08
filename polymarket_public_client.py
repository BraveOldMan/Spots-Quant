"""Shared read-only HTTP helpers for Polymarket public APIs."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import requests

from polymarket_rate_limiter import PolymarketRateLimiter

ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
HASH64_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")
DEFAULT_TIMEOUT_SECONDS = 15
JSONPayload = dict[str, Any] | list[Any] | str | int | float | bool | None
QueryValue = str | int | float | bool | Sequence[str | int] | None


class PolymarketApiError(RuntimeError):
    """Raised when a Polymarket API request fails or returns invalid JSON."""


class PolymarketPublicClient:
    """Small requests-based client with HTTPS, rate-limit, and JSON guards."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        rate_limiter: PolymarketRateLimiter | None = None,
        rate_limit_enabled: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.rate_limiter = rate_limiter or PolymarketRateLimiter(
            enabled=rate_limit_enabled
        )
        self._validate()

    def get(
        self,
        path: str,
        params: Mapping[str, QueryValue] | None = None,
        limit_name: str = "general",
    ) -> JSONPayload:
        """Issue a GET request and return decoded JSON."""
        return self._request("GET", path, params=params, limit_name=limit_name)

    def post_json(
        self,
        path: str,
        payload: Mapping[str, Any],
        limit_name: str = "general",
    ) -> JSONPayload:
        """Issue a POST request with a JSON body and return decoded JSON."""
        return self._request("POST", path, json_payload=payload, limit_name=limit_name)

    def _request(
        self,
        method: str,
        path: str,
        params: Mapping[str, QueryValue] | None = None,
        json_payload: Mapping[str, Any] | None = None,
        limit_name: str = "general",
    ) -> JSONPayload:
        if not path.startswith("/"):
            raise ValueError("Polymarket API path must start with '/'.")
        self.rate_limiter.acquire(limit_name)
        try:
            response = requests.request(
                method,
                f"{self.base_url}{path}",
                params=_clean_params(params or {}),
                json=dict(json_payload) if json_payload is not None else None,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise PolymarketApiError(
                f"Polymarket API request failed: {method} {path}"
            ) from exc
        if not 200 <= response.status_code < 300:
            raise PolymarketApiError(
                f"Polymarket API {method} {path} failed: HTTP {response.status_code}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise PolymarketApiError(
                f"Polymarket API {method} {path} returned invalid JSON."
            ) from exc
        if not isinstance(payload, (dict, list, str, int, float, bool)) and payload is not None:
            raise PolymarketApiError(
                f"Polymarket API {method} {path} returned unsupported JSON."
            )
        return payload

    def _validate(self) -> None:
        if not self.base_url.startswith("https://"):
            raise ValueError("Polymarket API URL must use https.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")


def validate_address(address: str, name: str = "address") -> str:
    """Validate and return a 0x-prefixed EVM address."""
    value = address.strip()
    if not ADDRESS_RE.match(value):
        raise ValueError(f"{name} must be a 0x-prefixed address.")
    return value


def validate_hash64(value: str, name: str = "hash") -> str:
    """Validate and return a 0x-prefixed 64-byte hash string."""
    candidate = value.strip()
    if not HASH64_RE.match(candidate):
        raise ValueError(f"{name} must be a 0x-prefixed 64-byte hash.")
    return candidate


def validate_token_id(token_id: str, name: str = "token_id") -> str:
    """Validate and return a decimal Polymarket outcome token id."""
    value = str(token_id).strip()
    if not value.isdigit():
        raise ValueError(f"{name} must be a decimal token id.")
    return value


def validate_limit(value: int, maximum: int, name: str = "limit") -> int:
    """Validate an API pagination limit against documented endpoint bounds."""
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must be between 0 and {maximum}.")
    return value


def _clean_params(params: Mapping[str, QueryValue]) -> dict[str, str | int | float]:
    cleaned: dict[str, str | int | float] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            cleaned[key] = "true" if value else "false"
            continue
        if isinstance(value, str):
            cleaned[key] = value
            continue
        if isinstance(value, Sequence):
            cleaned[key] = ",".join(str(item) for item in value)
            continue
        cleaned[key] = value
    return cleaned
