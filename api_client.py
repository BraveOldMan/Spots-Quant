"""API-Football client with SQLite cache, request budget, and fail-closed errors."""

from __future__ import annotations

import json
import os
import sqlite3
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

API_BASE_URL = "https://v3.football.api-sports.io"
DEFAULT_CACHE_DB = "api_cache.db"
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_REQUEST_DELAY_SECONDS = 0.25


JSONDict = dict[str, Any]


@dataclass
class FootballAPIStats:
    """Runtime API usage counters safe to write into reports."""

    cache_hits: int = 0
    cache_misses: int = 0
    api_requests: int = 0
    api_errors: int = 0
    blocked_requests: int = 0
    remaining_requests: int | None = None
    budget_exhausted: bool = False
    quota_exhausted: bool = False
    last_error: str = ""
    status_checked: bool = False
    status_ok: bool = False
    status_detail: str = ""

    def as_dict(self) -> dict[str, object]:
        """Return a report-safe copy of the counters."""
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "api_requests": self.api_requests,
            "api_errors": self.api_errors,
            "blocked_requests": self.blocked_requests,
            "remaining_requests": self.remaining_requests,
            "budget_exhausted": self.budget_exhausted,
            "quota_exhausted": self.quota_exhausted,
            "last_error": self.last_error,
            "status_checked": self.status_checked,
            "status_ok": self.status_ok,
            "status_detail": self.status_detail,
        }


def load_env(file_path: str = ".env") -> None:
    """Load `.env` values, tolerating UTF-8 BOM without logging secrets."""
    env_path = Path(file_path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip().lstrip("\ufeff")] = value.strip().strip('"').strip("'")


class FootballAPIClient:
    """Cache-first API-Football client used by historical backtest enrichment."""

    def __init__(
        self,
        db_path: str = DEFAULT_CACHE_DB,
        max_api_requests: int | None = None,
        cache_only: bool = False,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
    ) -> None:
        load_env()
        self.api_key = os.environ.get("API_FOOTBALL_KEY")
        if not self.api_key and not cache_only:
            raise ValueError("Environment variable API_FOOTBALL_KEY is missing.")
        self.base_url = API_BASE_URL
        self.headers = {"x-apisports-key": self.api_key or ""}
        self.db_path = db_path
        self.max_api_requests = max_api_requests
        self.cache_only = cache_only
        self.timeout_seconds = timeout_seconds
        self.request_delay_seconds = request_delay_seconds
        self.stats = FootballAPIStats()
        self._init_db()

    def _init_db(self) -> None:
        """Create the API response cache table if needed."""
        with _connect_cache_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS api_cache (
                    endpoint TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
                """
            )
            conn.commit()

    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        bypass_cache: bool = False,
        max_age: float | None = None,
    ) -> JSONDict | None:
        """Return a cached or live API response without fabricating missing data."""
        full_endpoint = self._full_endpoint(endpoint, params)
        cached = self._get_cached(full_endpoint, bypass_cache=bypass_cache, max_age=max_age)
        if cached is not None:
            return cached

        if self.cache_only:
            return self._client_error("cache_only_miss", full_endpoint)
        if self.stats.quota_exhausted:
            return self._client_error("api_quota_exhausted", full_endpoint)
        if self._request_budget_exhausted():
            self.stats.budget_exhausted = True
            return self._client_error("api_request_budget_exhausted", full_endpoint)

        return self._request_live(full_endpoint)

    def preflight_status(self, max_age: float = 300) -> bool:
        """Check API-Football status before a large enrichment run."""
        response = self.get("/status", max_age=max_age)
        self.stats.status_checked = True
        if response is None:
            self.stats.status_detail = "empty_or_network_failure"
            return False
        errors = response.get("errors")
        if errors:
            detail = str(errors)
            self.stats.status_detail = detail
            if _quota_error(detail):
                self.stats.quota_exhausted = True
            return False
        remaining = _remaining_requests(response)
        if remaining is not None and remaining <= 0:
            self.stats.remaining_requests = remaining
            self.stats.quota_exhausted = True
            self.stats.status_detail = "remaining_requests=0"
            return False
        self.stats.remaining_requests = remaining
        self.stats.status_ok = True
        self.stats.status_detail = (
            f"remaining_requests={remaining}" if remaining is not None else "ok"
        )
        return True

    def _get_cached(
        self,
        full_endpoint: str,
        bypass_cache: bool,
        max_age: float | None,
    ) -> JSONDict | None:
        if bypass_cache:
            self.stats.cache_misses += 1
            return None
        with _connect_cache_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT response, timestamp FROM api_cache WHERE endpoint = ?",
                (full_endpoint,),
            )
            row = cursor.fetchone()
        if row is None:
            self.stats.cache_misses += 1
            return None

        cached_resp, cached_ts = row[0], row[1]
        is_fresh = max_age is None or (time.time() - float(cached_ts)) <= max_age
        if not is_fresh:
            self.stats.cache_misses += 1
            return None
        self.stats.cache_hits += 1
        return json.loads(str(cached_resp))

    def _request_live(self, full_endpoint: str) -> JSONDict | None:
        self.stats.api_requests += 1
        url = f"{self.base_url}{full_endpoint}"
        req = urllib.request.Request(url, headers=self.headers)
        try:
            time.sleep(self.request_delay_seconds)
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                resp_text = response.read().decode("utf-8")
            resp_json: JSONDict = json.loads(resp_text)
        except HTTPError as exc:
            return self._client_error(
                f"http_{exc.code}:{exc.reason}", full_endpoint, api_error=True
            )
        except URLError as exc:
            return self._client_error(f"network:{exc.reason}", full_endpoint, api_error=True)
        except Exception as exc:
            return self._client_error(f"unknown:{exc}", full_endpoint, api_error=True)

        errors = resp_json.get("errors")
        if errors:
            self.stats.api_errors += 1
            detail = str(errors)
            self.stats.last_error = detail
            if _quota_error(detail):
                self.stats.quota_exhausted = True
            return resp_json

        with _connect_cache_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO api_cache (endpoint, response, timestamp) "
                "VALUES (?, ?, ?)",
                (full_endpoint, resp_text, time.time()),
            )
            conn.commit()
        return resp_json

    def _request_budget_exhausted(self) -> bool:
        return (
            self.max_api_requests is not None
            and self.max_api_requests >= 0
            and self.stats.api_requests >= self.max_api_requests
        )

    def _client_error(
        self,
        reason: str,
        full_endpoint: str,
        api_error: bool = False,
    ) -> JSONDict:
        if api_error:
            self.stats.api_errors += 1
        else:
            self.stats.blocked_requests += 1
        self.stats.last_error = reason
        return {
            "errors": {"client": reason},
            "response": [],
            "endpoint": full_endpoint,
        }

    @staticmethod
    def _full_endpoint(endpoint: str, params: dict[str, Any] | None) -> str:
        if not params:
            return endpoint
        return f"{endpoint}?{urlencode(params)}"


def _quota_error(detail: str) -> bool:
    lowered = detail.lower()
    return "request limit" in lowered or "quota" in lowered or "rate limit" in lowered


def _connect_cache_db(db_path: str) -> sqlite3.Connection:
    """Open the local API cache with Windows-safe SQLite journal settings."""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _remaining_requests(response: JSONDict) -> int | None:
    payload = response.get("response")
    if not isinstance(payload, dict):
        return None
    requests = payload.get("requests")
    if not isinstance(requests, dict):
        return None
    for key in ("remaining", "remaining_day", "limit_remaining"):
        value = requests.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    current = requests.get("current")
    limit_day = requests.get("limit_day")
    try:
        return int(limit_day) - int(current)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    client = FootballAPIClient(max_api_requests=1)
    ok = client.preflight_status()
    print({"status_ok": ok, **client.stats.as_dict()})
