"""Persistent API-Football today-data sync daemon.

The module is read-only with respect to trading: it only fetches fixtures,
odds, and match-enrichment endpoints from API-Football and persists them into
the local `spots_quant.db` SQLite database for research and monitoring.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import sqlite3
import sys
import time
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType
from typing import Any, Callable
from zoneinfo import ZoneInfo

from api_client import DEFAULT_CACHE_DB, FootballAPIClient, load_env

JSONDict = dict[str, Any]
ClientFactory = Callable[[int, "APIFootballDaemonConfig"], FootballAPIClient]

DEFAULT_DB_PATH = "spots_quant.db"
DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_INTERVAL_MINUTES = 15
DEFAULT_DAILY_BUDGET = 7000
DEFAULT_MAX_API_REQUESTS_PER_RUN = 250
DEFAULT_FIXTURES_TTL_SECONDS = 60
DEFAULT_ODDS_TTL_SECONDS = 180
DEFAULT_LIVE_TTL_SECONDS = 300
DEFAULT_PRE_MATCH_TTL_SECONDS = 900
DEFAULT_FINISHED_TTL_SECONDS = 86_400
DEFAULT_ENRICHMENT_WINDOW_HOURS = 6.0
DEFAULT_LOCK_STALE_SECONDS = 600
DEFAULT_REQUEST_DELAY_SECONDS = 0.22

FINISHED_STATUSES = {"FT", "AET", "PEN"}
LIVE_STATUSES = {"1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE"}


@dataclass(frozen=True)
class APIFootballDaemonConfig:
    """Runtime settings for today's API-Football sync loop."""

    db_path: str = DEFAULT_DB_PATH
    cache_db_path: str = DEFAULT_CACHE_DB
    timezone_name: str = DEFAULT_TIMEZONE
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES
    daily_budget: int = DEFAULT_DAILY_BUDGET
    max_api_requests_per_run: int = DEFAULT_MAX_API_REQUESTS_PER_RUN
    request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS
    cache_only: bool = False
    lock_path: str = "runtime/api_football_today_daemon.lock"
    stop_path: str = "runtime/api_football_today_daemon.stop"
    log_path: str = "logs/api_football_today_daemon.log"
    fixtures_ttl_seconds: int = DEFAULT_FIXTURES_TTL_SECONDS
    odds_ttl_seconds: int = DEFAULT_ODDS_TTL_SECONDS
    live_ttl_seconds: int = DEFAULT_LIVE_TTL_SECONDS
    pre_match_ttl_seconds: int = DEFAULT_PRE_MATCH_TTL_SECONDS
    finished_ttl_seconds: int = DEFAULT_FINISHED_TTL_SECONDS
    enrichment_window_hours: float = DEFAULT_ENRICHMENT_WINDOW_HOURS
    lock_stale_seconds: int = DEFAULT_LOCK_STALE_SECONDS

    @classmethod
    def from_env(cls) -> "APIFootballDaemonConfig":
        """Build daemon config from `.env` and process environment values."""
        load_env()
        return cls(
            db_path=os.environ.get("SPOTS_QUANT_DB", DEFAULT_DB_PATH),
            cache_db_path=os.environ.get("SPOTS_API_CACHE_DB", DEFAULT_CACHE_DB),
            timezone_name=os.environ.get("SPOTS_API_FOOTBALL_TIMEZONE", DEFAULT_TIMEZONE),
            interval_minutes=_env_int(
                "SPOTS_API_FOOTBALL_DAEMON_INTERVAL_MINUTES",
                DEFAULT_INTERVAL_MINUTES,
            ),
            daily_budget=_env_int(
                "SPOTS_API_FOOTBALL_DAILY_BUDGET",
                DEFAULT_DAILY_BUDGET,
            ),
            max_api_requests_per_run=_env_int(
                "SPOTS_API_FOOTBALL_MAX_REQUESTS_PER_RUN",
                DEFAULT_MAX_API_REQUESTS_PER_RUN,
            ),
            request_delay_seconds=_env_float(
                "SPOTS_API_FOOTBALL_REQUEST_DELAY_SECONDS",
                DEFAULT_REQUEST_DELAY_SECONDS,
            ),
            cache_only=_env_bool("SPOTS_API_FOOTBALL_CACHE_ONLY", False),
            lock_path=os.environ.get(
                "SPOTS_API_FOOTBALL_DAEMON_LOCK",
                "runtime/api_football_today_daemon.lock",
            ),
            stop_path=os.environ.get(
                "SPOTS_API_FOOTBALL_DAEMON_STOP",
                "runtime/api_football_today_daemon.stop",
            ),
            log_path=os.environ.get(
                "SPOTS_API_FOOTBALL_DAEMON_LOG",
                "logs/api_football_today_daemon.log",
            ),
            fixtures_ttl_seconds=_env_int(
                "SPOTS_API_FOOTBALL_FIXTURES_TTL_SECONDS",
                DEFAULT_FIXTURES_TTL_SECONDS,
            ),
            odds_ttl_seconds=_env_int(
                "SPOTS_API_FOOTBALL_ODDS_TTL_SECONDS",
                DEFAULT_ODDS_TTL_SECONDS,
            ),
            live_ttl_seconds=_env_int(
                "SPOTS_API_FOOTBALL_LIVE_TTL_SECONDS",
                DEFAULT_LIVE_TTL_SECONDS,
            ),
            pre_match_ttl_seconds=_env_int(
                "SPOTS_API_FOOTBALL_PRE_MATCH_TTL_SECONDS",
                DEFAULT_PRE_MATCH_TTL_SECONDS,
            ),
            finished_ttl_seconds=_env_int(
                "SPOTS_API_FOOTBALL_FINISHED_TTL_SECONDS",
                DEFAULT_FINISHED_TTL_SECONDS,
            ),
            enrichment_window_hours=_env_float(
                "SPOTS_API_FOOTBALL_ENRICHMENT_WINDOW_HOURS",
                DEFAULT_ENRICHMENT_WINDOW_HOURS,
            ),
        )


@dataclass(frozen=True)
class SyncSummary:
    """Summary of one API-Football sync run safe to print or persist."""

    run_id: str
    status: str
    target_date: str
    started_at: str
    finished_at: str
    fixtures_count: int = 0
    odds_items_count: int = 0
    normalized_odds_count: int = 0
    enrichment_items_count: int = 0
    diagnostics_count: int = 0
    api_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    api_errors: int = 0
    blocked_requests: int = 0
    remaining_requests: int | None = None
    status_detail: str = ""
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "target_date": self.target_date,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "fixtures_count": self.fixtures_count,
            "odds_items_count": self.odds_items_count,
            "normalized_odds_count": self.normalized_odds_count,
            "enrichment_items_count": self.enrichment_items_count,
            "diagnostics_count": self.diagnostics_count,
            "api_requests": self.api_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "api_errors": self.api_errors,
            "blocked_requests": self.blocked_requests,
            "remaining_requests": self.remaining_requests,
            "status_detail": self.status_detail,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EnrichmentSpec:
    """One fixture-scoped API-Football enrichment endpoint request."""

    endpoint: str
    params: dict[str, object]
    ttl_seconds: int


class InstanceLock(AbstractContextManager["InstanceLock"]):
    """Filesystem lock preventing duplicate daemon instances."""

    def __init__(self, path: str | Path, stale_seconds: int = DEFAULT_LOCK_STALE_SECONDS) -> None:
        self.path = Path(path)
        self.stale_seconds = stale_seconds
        self.acquired = False

    def __enter__(self) -> "InstanceLock":
        """Acquire the lock or raise if another live process owns it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._remove_stale_lock():
                    continue
                raise RuntimeError(f"API-Football daemon already running: {self.path}") from None
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "pid": os.getpid(),
                        "created_at": _utc_now().isoformat(),
                    },
                    handle,
                )
            self.acquired = True
            return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """Release the lock on normal daemon shutdown."""
        if self.acquired:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        return None

    def _remove_stale_lock(self) -> bool:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        pid = _safe_int(payload.get("pid"))
        if pid is not None and _pid_is_running(pid):
            return False
        try:
            age = time.time() - self.path.stat().st_mtime
        except OSError:
            age = self.stale_seconds + 1
        if pid is None or age >= self.stale_seconds:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            return True
        return False


def init_sync_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Create or migrate SQLite tables used by the API-Football daemon."""
    db_parent = Path(db_path).parent
    if db_parent != Path("."):
        db_parent.mkdir(parents=True, exist_ok=True)
    with _connect_db(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS api_football_sync_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                target_date TEXT NOT NULL,
                status TEXT NOT NULL,
                cache_only INTEGER NOT NULL,
                max_api_requests INTEGER NOT NULL,
                daily_budget INTEGER NOT NULL,
                fixtures_count INTEGER NOT NULL,
                odds_items_count INTEGER NOT NULL,
                normalized_odds_count INTEGER NOT NULL,
                enrichment_items_count INTEGER NOT NULL,
                diagnostics_count INTEGER NOT NULL,
                api_requests INTEGER NOT NULL,
                cache_hits INTEGER NOT NULL,
                cache_misses INTEGER NOT NULL,
                api_errors INTEGER NOT NULL,
                blocked_requests INTEGER NOT NULL,
                remaining_requests INTEGER,
                status_detail TEXT NOT NULL,
                reason TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS api_football_daemon_heartbeat (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                pid INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                target_date TEXT NOT NULL,
                last_run_id TEXT NOT NULL,
                last_status TEXT NOT NULL,
                next_run_at TEXT NOT NULL,
                message TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS api_football_daily_usage (
                usage_date TEXT PRIMARY KEY,
                api_requests INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS api_football_fixtures (
                fixture_id INTEGER PRIMARY KEY,
                target_date TEXT NOT NULL,
                kickoff TEXT NOT NULL,
                status_short TEXT NOT NULL,
                status_long TEXT NOT NULL,
                elapsed INTEGER,
                league_id INTEGER,
                league_name TEXT NOT NULL,
                country TEXT NOT NULL,
                home_team_id INTEGER,
                away_team_id INTEGER,
                home_team_name TEXT NOT NULL,
                away_team_name TEXT NOT NULL,
                home_goals INTEGER,
                away_goals INTEGER,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS api_football_fixture_raw (
                fixture_id INTEGER PRIMARY KEY,
                target_date TEXT NOT NULL,
                raw_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS api_football_odds_raw (
                fixture_id INTEGER NOT NULL,
                bookmaker_id INTEGER NOT NULL,
                bookmaker_name TEXT NOT NULL,
                bet_id INTEGER NOT NULL,
                bet_name TEXT NOT NULL,
                update_time TEXT NOT NULL,
                target_date TEXT NOT NULL,
                raw_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (fixture_id, bookmaker_id, bet_id, update_time)
            );

            CREATE TABLE IF NOT EXISTS api_football_odds_1x2 (
                fixture_id INTEGER NOT NULL,
                bookmaker_id INTEGER NOT NULL,
                bookmaker_name TEXT NOT NULL,
                update_time TEXT NOT NULL,
                target_date TEXT NOT NULL,
                home_odds REAL NOT NULL,
                draw_odds REAL NOT NULL,
                away_odds REAL NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (fixture_id, bookmaker_id, update_time)
            );

            CREATE TABLE IF NOT EXISTS api_football_enrichment_raw (
                fixture_id INTEGER NOT NULL,
                endpoint TEXT NOT NULL,
                params_json TEXT NOT NULL,
                status TEXT NOT NULL,
                response_count INTEGER NOT NULL,
                raw_json TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (fixture_id, endpoint, params_json)
            );

            CREATE TABLE IF NOT EXISTS api_football_diagnostics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                target_date TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL,
                fixture_id INTEGER,
                endpoint TEXT NOT NULL,
                detail TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


def _connect_db(db_path: str) -> sqlite3.Connection:
    """Open SQLite with daemon-safe pragmas for local Windows persistence."""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def sync_api_football_today(
    config: APIFootballDaemonConfig | None = None,
    target_date: date | None = None,
    client: FootballAPIClient | None = None,
    now: datetime | None = None,
) -> SyncSummary:
    """Run one cache-first sync for today's fixtures, odds, and enrichment data."""
    active_config = config or APIFootballDaemonConfig.from_env()
    current_utc = _as_utc(now or _utc_now())
    target = target_date or current_utc.astimezone(ZoneInfo(active_config.timezone_name)).date()
    target_date_text = target.isoformat()
    run_id = uuid.uuid4().hex
    started_at = current_utc.isoformat()
    init_sync_db(active_config.db_path)

    budget_remaining = _daily_budget_remaining(active_config, target_date_text)
    if not active_config.cache_only and budget_remaining <= 0:
        _insert_diagnostic(
            active_config.db_path,
            run_id,
            target_date_text,
            "error",
            "skipped",
            "api_daily_budget_exhausted",
            None,
            "/status",
            f"daily_budget={active_config.daily_budget}",
        )
        summary = _summary_without_client(
            run_id,
            "blocked",
            target_date_text,
            started_at,
            "api_daily_budget_exhausted",
        )
        _write_run(active_config, summary)
        return summary

    max_requests = min(active_config.max_api_requests_per_run, max(budget_remaining, 0))
    try:
        api_client = client or _default_client_factory(max_requests, active_config)
    except Exception as exc:
        _insert_diagnostic(
            active_config.db_path,
            run_id,
            target_date_text,
            "error",
            "skipped",
            "api_client_unavailable",
            None,
            "/status",
            str(exc),
        )
        summary = _summary_without_client(
            run_id,
            "blocked",
            target_date_text,
            started_at,
            "api_client_unavailable",
        )
        _write_run(active_config, summary)
        return summary

    if not active_config.cache_only and not api_client.preflight_status(max_age=60):
        _insert_diagnostic(
            active_config.db_path,
            run_id,
            target_date_text,
            "error",
            "skipped",
            "api_preflight_failed",
            None,
            "/status",
            _client_status_detail(api_client),
        )
        summary = _summary_from_client(
            run_id,
            "blocked",
            target_date_text,
            started_at,
            api_client,
            "api_preflight_failed",
        )
        _record_usage(active_config.db_path, target_date_text, api_client.stats.api_requests)
        _write_run(active_config, summary)
        return summary

    counters = {"fixtures": 0, "odds_items": 0, "normalized_odds": 0, "enrichment": 0}
    fixtures_response = api_client.get(
        "/fixtures",
        {"date": target_date_text, "timezone": active_config.timezone_name},
        max_age=active_config.fixtures_ttl_seconds,
    )
    fixtures = _response_items(fixtures_response)
    if not _api_response_ok(fixtures_response):
        _insert_diagnostic(
            active_config.db_path,
            run_id,
            target_date_text,
            "error",
            "skipped",
            "fixtures_fetch_failed",
            None,
            "/fixtures",
            _api_error_detail(fixtures_response),
        )
    else:
        counters["fixtures"] = _upsert_fixtures(active_config.db_path, target_date_text, fixtures)

    odds_items = _fetch_odds_for_date(api_client, target_date_text, active_config, run_id)
    counters["odds_items"], counters["normalized_odds"] = _upsert_odds(
        active_config.db_path,
        target_date_text,
        odds_items,
    )

    counters["enrichment"] = _sync_enrichment(
        active_config,
        api_client,
        target_date_text,
        run_id,
        fixtures,
        current_utc,
    )
    status = "ok" if _api_response_ok(fixtures_response) else "failed"
    reason = "" if status == "ok" else "fixtures_fetch_failed"
    _record_usage(active_config.db_path, target_date_text, api_client.stats.api_requests)
    summary = _summary_from_client(
        run_id,
        status,
        target_date_text,
        started_at,
        api_client,
        reason,
        fixtures_count=counters["fixtures"],
        odds_items_count=counters["odds_items"],
        normalized_odds_count=counters["normalized_odds"],
        enrichment_items_count=counters["enrichment"],
        diagnostics_count=_diagnostic_count(active_config.db_path, run_id),
    )
    _write_run(active_config, summary)
    return summary


def run_daemon(
    config: APIFootballDaemonConfig | None = None,
    max_iterations: int | None = None,
) -> None:
    """Run the persistent sync loop until a stop file or KeyboardInterrupt appears."""
    active_config = config or APIFootballDaemonConfig.from_env()
    logger = _setup_logger(active_config.log_path)
    _remove_stop_file(active_config.stop_path)
    init_sync_db(active_config.db_path)
    iteration = 0
    with InstanceLock(active_config.lock_path, active_config.lock_stale_seconds):
        logger.info("api-football daemon started")
        while True:
            if _stop_requested(active_config.stop_path):
                _remove_stop_file(active_config.stop_path)
                _write_heartbeat(active_config, "", "stopped", "", "stop requested")
                logger.info("api-football daemon stopped by stop file")
                return
            try:
                summary = sync_api_football_today(active_config)
                logger.info("sync summary: %s", json.dumps(summary.as_dict(), sort_keys=True))
            except Exception as exc:
                summary = SyncSummary(
                    run_id=uuid.uuid4().hex,
                    status="failed",
                    target_date=_utc_now().date().isoformat(),
                    started_at=_utc_now().isoformat(),
                    finished_at=_utc_now().isoformat(),
                    reason=f"daemon_exception:{exc}",
                )
                logger.exception("sync failed")
            iteration += 1
            next_run = _utc_now() + timedelta(minutes=active_config.interval_minutes)
            _write_heartbeat(
                active_config,
                summary.run_id,
                summary.status,
                next_run.isoformat(),
                summary.reason or "sleeping",
            )
            if max_iterations is not None and iteration >= max_iterations:
                return
            _sleep_until_next_run(active_config.stop_path, next_run)


def read_daemon_status(db_path: str = DEFAULT_DB_PATH) -> dict[str, object]:
    """Read the latest daemon heartbeat and sync-run row from SQLite."""
    init_sync_db(db_path)
    with _connect_db(db_path) as conn:
        conn.row_factory = sqlite3.Row
        heartbeat = conn.execute(
            "SELECT * FROM api_football_daemon_heartbeat WHERE id = 1"
        ).fetchone()
        latest_run = conn.execute(
            "SELECT * FROM api_football_sync_runs ORDER BY finished_at DESC LIMIT 1"
        ).fetchone()
    return {
        "heartbeat": dict(heartbeat) if heartbeat is not None else {},
        "latest_run": dict(latest_run) if latest_run is not None else {},
    }


def _default_client_factory(
    max_requests: int,
    config: APIFootballDaemonConfig,
) -> FootballAPIClient:
    return FootballAPIClient(
        db_path=config.cache_db_path,
        max_api_requests=max_requests,
        cache_only=config.cache_only,
        request_delay_seconds=config.request_delay_seconds,
    )


def _fetch_odds_for_date(
    client: FootballAPIClient,
    target_date_text: str,
    config: APIFootballDaemonConfig,
    run_id: str,
) -> list[JSONDict]:
    first_page = client.get(
        "/odds",
        {"date": target_date_text, "page": 1},
        max_age=config.odds_ttl_seconds,
    )
    if not _api_response_ok(first_page):
        _insert_diagnostic(
            config.db_path,
            run_id,
            target_date_text,
            "warning",
            "skipped",
            "odds_fetch_failed",
            None,
            "/odds",
            _api_error_detail(first_page),
        )
        return []
    rows = list(_response_items(first_page))
    total_pages = _total_pages(first_page)
    for page in range(2, total_pages + 1):
        if _client_blocked(client):
            _insert_diagnostic(
                config.db_path,
                run_id,
                target_date_text,
                "warning",
                "skipped",
                "api_request_blocked",
                None,
                "/odds",
                _client_status_detail(client),
            )
            break
        page_response = client.get(
            "/odds",
            {"date": target_date_text, "page": page},
            max_age=config.odds_ttl_seconds,
        )
        if not _api_response_ok(page_response):
            _insert_diagnostic(
                config.db_path,
                run_id,
                target_date_text,
                "warning",
                "skipped",
                "odds_page_failed",
                None,
                "/odds",
                f"page={page};{_api_error_detail(page_response)}",
            )
            continue
        rows.extend(_response_items(page_response))
    return rows


def _sync_enrichment(
    config: APIFootballDaemonConfig,
    client: FootballAPIClient,
    target_date_text: str,
    run_id: str,
    fixtures: list[JSONDict],
    now: datetime,
) -> int:
    saved = 0
    for item in fixtures:
        fixture_id = _safe_int(item.get("fixture", {}).get("id"))
        if fixture_id is None:
            continue
        specs = _enrichment_specs_for_fixture(item, config, now)
        for spec in specs:
            if _client_blocked(client):
                _insert_diagnostic(
                    config.db_path,
                    run_id,
                    target_date_text,
                    "warning",
                    "skipped",
                    "api_request_blocked",
                    fixture_id,
                    spec.endpoint,
                    _client_status_detail(client),
                )
                return saved
            if _enrichment_is_fresh(config.db_path, fixture_id, spec):
                continue
            response = client.get(spec.endpoint, spec.params, max_age=spec.ttl_seconds)
            status = "ok" if _api_response_ok(response) else "failed"
            saved += _upsert_enrichment(config.db_path, fixture_id, spec, response, status)
            if status != "ok":
                _insert_diagnostic(
                    config.db_path,
                    run_id,
                    target_date_text,
                    "warning",
                    "skipped",
                    "enrichment_fetch_failed",
                    fixture_id,
                    spec.endpoint,
                    _api_error_detail(response),
                )
    return saved


def _enrichment_specs_for_fixture(
    item: JSONDict,
    config: APIFootballDaemonConfig,
    now: datetime,
) -> list[EnrichmentSpec]:
    fixture = item.get("fixture", {})
    fixture_id = _safe_int(fixture.get("id"))
    if fixture_id is None:
        return []
    status = str(fixture.get("status", {}).get("short") or "").upper()
    kickoff = _parse_api_datetime(fixture.get("date"))
    is_finished = status in FINISHED_STATUSES
    is_live = status in LIVE_STATUSES
    is_near = False
    if kickoff is not None:
        hours_to_kickoff = (kickoff - _as_utc(now)).total_seconds() / 3600
        is_near = -2 <= hours_to_kickoff <= config.enrichment_window_hours

    if is_finished:
        ttl = config.finished_ttl_seconds
    elif is_live:
        ttl = config.live_ttl_seconds
    else:
        ttl = config.pre_match_ttl_seconds

    specs: list[EnrichmentSpec] = []
    if is_near or is_live or is_finished:
        specs.extend(
            [
                EnrichmentSpec("/injuries", {"fixture": fixture_id}, ttl),
                EnrichmentSpec("/predictions", {"fixture": fixture_id}, ttl),
                EnrichmentSpec("/fixtures/lineups", {"fixture": fixture_id}, ttl),
            ]
        )
    if is_live or is_finished:
        specs.extend(
            [
                EnrichmentSpec("/fixtures/events", {"fixture": fixture_id}, ttl),
                EnrichmentSpec("/fixtures/statistics", {"fixture": fixture_id}, ttl),
                EnrichmentSpec("/fixtures/players", {"fixture": fixture_id}, ttl),
            ]
        )
    return specs


def _upsert_fixtures(db_path: str, target_date_text: str, fixtures: list[JSONDict]) -> int:
    now_text = _utc_now().isoformat()
    saved = 0
    with _connect_db(db_path) as conn:
        for item in fixtures:
            fixture = item.get("fixture", {})
            fixture_id = _safe_int(fixture.get("id"))
            if fixture_id is None:
                continue
            league = item.get("league", {})
            teams = item.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            goals = item.get("goals", {})
            status = fixture.get("status", {})
            conn.execute(
                """
                INSERT OR REPLACE INTO api_football_fixtures (
                    fixture_id, target_date, kickoff, status_short, status_long, elapsed,
                    league_id, league_name, country, home_team_id, away_team_id,
                    home_team_name, away_team_name, home_goals, away_goals, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fixture_id,
                    target_date_text,
                    str(fixture.get("date") or ""),
                    str(status.get("short") or ""),
                    str(status.get("long") or ""),
                    _safe_int(status.get("elapsed")),
                    _safe_int(league.get("id")),
                    str(league.get("name") or ""),
                    str(league.get("country") or ""),
                    _safe_int(home.get("id")),
                    _safe_int(away.get("id")),
                    str(home.get("name") or ""),
                    str(away.get("name") or ""),
                    _safe_int(goals.get("home")),
                    _safe_int(goals.get("away")),
                    now_text,
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO api_football_fixture_raw (
                    fixture_id, target_date, raw_json, updated_at
                ) VALUES (?, ?, ?, ?)
                """,
                (fixture_id, target_date_text, json.dumps(item, ensure_ascii=False), now_text),
            )
            saved += 1
    return saved


def _upsert_odds(
    db_path: str,
    target_date_text: str,
    odds_items: list[JSONDict],
) -> tuple[int, int]:
    now_text = _utc_now().isoformat()
    raw_count = 0
    normalized_count = 0
    with _connect_db(db_path) as conn:
        for item in odds_items:
            fixture_id = _safe_int(item.get("fixture", {}).get("id"))
            if fixture_id is None:
                continue
            update_time = str(item.get("update") or "")
            for bookmaker in item.get("bookmakers", []):
                bookmaker_id = _safe_int(bookmaker.get("id")) or 0
                bookmaker_name = str(bookmaker.get("name") or "")
                for bet in bookmaker.get("bets", []):
                    bet_id = _safe_int(bet.get("id")) or 0
                    bet_name = str(bet.get("name") or "")
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO api_football_odds_raw (
                            fixture_id, bookmaker_id, bookmaker_name, bet_id, bet_name,
                            update_time, target_date, raw_json, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            fixture_id,
                            bookmaker_id,
                            bookmaker_name,
                            bet_id,
                            bet_name,
                            update_time,
                            target_date_text,
                            json.dumps(bet, ensure_ascii=False),
                            now_text,
                        ),
                    )
                    raw_count += 1
                odds = _match_winner_odds(bookmaker)
                if odds is None:
                    continue
                conn.execute(
                    """
                    INSERT OR REPLACE INTO api_football_odds_1x2 (
                        fixture_id, bookmaker_id, bookmaker_name, update_time,
                        target_date, home_odds, draw_odds, away_odds, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fixture_id,
                        bookmaker_id,
                        bookmaker_name,
                        update_time,
                        target_date_text,
                        odds[0],
                        odds[1],
                        odds[2],
                        now_text,
                    ),
                )
                normalized_count += 1
    return raw_count, normalized_count


def _upsert_enrichment(
    db_path: str,
    fixture_id: int,
    spec: EnrichmentSpec,
    response: JSONDict | None,
    status: str,
) -> int:
    response_payload = response or {"errors": {"client": "empty_response"}, "response": []}
    response_count = len(_response_items(response_payload)) if _api_response_ok(response_payload) else 0
    with _connect_db(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO api_football_enrichment_raw (
                fixture_id, endpoint, params_json, status, response_count, raw_json, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fixture_id,
                spec.endpoint,
                _params_key(spec.params),
                status,
                response_count,
                json.dumps(response_payload, ensure_ascii=False),
                _utc_now().isoformat(),
            ),
        )
    return 1


def _enrichment_is_fresh(db_path: str, fixture_id: int, spec: EnrichmentSpec) -> bool:
    with _connect_db(db_path) as conn:
        row = conn.execute(
            """
            SELECT fetched_at FROM api_football_enrichment_raw
            WHERE fixture_id = ? AND endpoint = ? AND params_json = ?
            """,
            (fixture_id, spec.endpoint, _params_key(spec.params)),
        ).fetchone()
    if row is None:
        return False
    fetched_at = _parse_api_datetime(row[0])
    if fetched_at is None:
        return False
    return (_utc_now() - fetched_at).total_seconds() <= spec.ttl_seconds


def _match_winner_odds(bookmaker: JSONDict) -> tuple[float, float, float] | None:
    for bet in bookmaker.get("bets", []):
        if str(bet.get("name") or "").strip().lower() != "match winner":
            continue
        values: dict[str, float] = {}
        for item in bet.get("values", []):
            label = str(item.get("value") or "").strip().lower()
            if label not in {"home", "draw", "away"}:
                continue
            try:
                values[label] = float(item.get("odd"))
            except (TypeError, ValueError):
                return None
        odds = (values.get("home"), values.get("draw"), values.get("away"))
        if all(_valid_decimal_odds(value) for value in odds):
            return float(odds[0]), float(odds[1]), float(odds[2])
    return None


def _insert_diagnostic(
    db_path: str,
    run_id: str,
    target_date_text: str,
    severity: str,
    status: str,
    reason: str,
    fixture_id: int | None,
    endpoint: str,
    detail: str,
) -> None:
    with _connect_db(db_path) as conn:
        conn.execute(
            """
            INSERT INTO api_football_diagnostics (
                run_id, target_date, severity, status, reason,
                fixture_id, endpoint, detail, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                target_date_text,
                severity,
                status,
                reason,
                fixture_id,
                endpoint,
                detail,
                _utc_now().isoformat(),
            ),
        )


def _write_run(config: APIFootballDaemonConfig, summary: SyncSummary) -> None:
    with _connect_db(config.db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO api_football_sync_runs (
                run_id, started_at, finished_at, target_date, status, cache_only,
                max_api_requests, daily_budget, fixtures_count, odds_items_count,
                normalized_odds_count, enrichment_items_count, diagnostics_count,
                api_requests, cache_hits, cache_misses, api_errors, blocked_requests,
                remaining_requests, status_detail, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary.run_id,
                summary.started_at,
                summary.finished_at,
                summary.target_date,
                summary.status,
                int(config.cache_only),
                config.max_api_requests_per_run,
                config.daily_budget,
                summary.fixtures_count,
                summary.odds_items_count,
                summary.normalized_odds_count,
                summary.enrichment_items_count,
                summary.diagnostics_count,
                summary.api_requests,
                summary.cache_hits,
                summary.cache_misses,
                summary.api_errors,
                summary.blocked_requests,
                summary.remaining_requests,
                summary.status_detail,
                summary.reason,
            ),
        )


def _write_heartbeat(
    config: APIFootballDaemonConfig,
    last_run_id: str,
    last_status: str,
    next_run_at: str,
    message: str,
) -> None:
    init_sync_db(config.db_path)
    target_date_text = _utc_now().astimezone(ZoneInfo(config.timezone_name)).date().isoformat()
    with _connect_db(config.db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO api_football_daemon_heartbeat (
                id, pid, updated_at, target_date, last_run_id, last_status, next_run_at, message
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                os.getpid(),
                _utc_now().isoformat(),
                target_date_text,
                last_run_id,
                last_status,
                next_run_at,
                message,
            ),
        )


def _daily_budget_remaining(config: APIFootballDaemonConfig, target_date_text: str) -> int:
    if config.cache_only:
        return config.max_api_requests_per_run
    init_sync_db(config.db_path)
    with _connect_db(config.db_path) as conn:
        row = conn.execute(
            "SELECT api_requests FROM api_football_daily_usage WHERE usage_date = ?",
            (target_date_text,),
        ).fetchone()
    used = int(row[0]) if row else 0
    return max(config.daily_budget - used, 0)


def _record_usage(db_path: str, target_date_text: str, api_requests: int) -> None:
    if api_requests <= 0:
        return
    with _connect_db(db_path) as conn:
        conn.execute(
            """
            INSERT INTO api_football_daily_usage (usage_date, api_requests, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(usage_date) DO UPDATE SET
                api_requests = api_requests + excluded.api_requests,
                updated_at = excluded.updated_at
            """,
            (target_date_text, api_requests, _utc_now().isoformat()),
        )


def _summary_without_client(
    run_id: str,
    status: str,
    target_date_text: str,
    started_at: str,
    reason: str,
) -> SyncSummary:
    return SyncSummary(
        run_id=run_id,
        status=status,
        target_date=target_date_text,
        started_at=started_at,
        finished_at=_utc_now().isoformat(),
        diagnostics_count=1,
        reason=reason,
    )


def _summary_from_client(
    run_id: str,
    status: str,
    target_date_text: str,
    started_at: str,
    client: FootballAPIClient,
    reason: str,
    fixtures_count: int = 0,
    odds_items_count: int = 0,
    normalized_odds_count: int = 0,
    enrichment_items_count: int = 0,
    diagnostics_count: int = 0,
) -> SyncSummary:
    stats = client.stats
    return SyncSummary(
        run_id=run_id,
        status=status,
        target_date=target_date_text,
        started_at=started_at,
        finished_at=_utc_now().isoformat(),
        fixtures_count=fixtures_count,
        odds_items_count=odds_items_count,
        normalized_odds_count=normalized_odds_count,
        enrichment_items_count=enrichment_items_count,
        diagnostics_count=diagnostics_count,
        api_requests=int(getattr(stats, "api_requests", 0)),
        cache_hits=int(getattr(stats, "cache_hits", 0)),
        cache_misses=int(getattr(stats, "cache_misses", 0)),
        api_errors=int(getattr(stats, "api_errors", 0)),
        blocked_requests=int(getattr(stats, "blocked_requests", 0)),
        remaining_requests=getattr(stats, "remaining_requests", None),
        status_detail=_client_status_detail(client),
        reason=reason,
    )


def _diagnostic_count(db_path: str, run_id: str) -> int:
    with _connect_db(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM api_football_diagnostics WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    return int(row[0]) if row else 0


def _api_response_ok(response: JSONDict | None) -> bool:
    if not response or response.get("errors"):
        return False
    return isinstance(response.get("response"), list)


def _response_items(response: JSONDict | None) -> list[JSONDict]:
    if not response:
        return []
    payload = response.get("response")
    return payload if isinstance(payload, list) else []


def _api_error_detail(response: JSONDict | None) -> str:
    if not response:
        return "empty_response"
    errors = response.get("errors")
    return str(errors) if errors else "missing_response_list"


def _total_pages(response: JSONDict | None) -> int:
    if not response:
        return 1
    paging = response.get("paging")
    if not isinstance(paging, dict):
        return 1
    try:
        return max(int(paging.get("total", 1) or 1), 1)
    except (TypeError, ValueError):
        return 1


def _client_blocked(client: FootballAPIClient) -> bool:
    stats = client.stats
    return bool(getattr(stats, "quota_exhausted", False) or getattr(stats, "budget_exhausted", False))


def _client_status_detail(client: FootballAPIClient) -> str:
    stats = client.stats
    return str(getattr(stats, "status_detail", "") or getattr(stats, "last_error", ""))


def _params_key(params: dict[str, object]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _setup_logger(log_path: str) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("api_football_today_daemon")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=5)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)
    return logger


def _sleep_until_next_run(stop_path: str, next_run: datetime) -> None:
    while _utc_now() < next_run:
        if _stop_requested(stop_path):
            return
        time.sleep(5)


def _stop_requested(stop_path: str) -> bool:
    return Path(stop_path).exists()


def _remove_stop_file(stop_path: str) -> None:
    try:
        Path(stop_path).unlink()
    except FileNotFoundError:
        pass


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _parse_api_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _as_utc(parsed)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _valid_decimal_odds(value: object) -> bool:
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return False
    return 1.0 < odds <= 100.0


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float.") from exc


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync today's API-Football data to SQLite.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run one sync iteration and exit.")
    mode.add_argument("--daemon", action="store_true", help="Run as a persistent polling daemon.")
    mode.add_argument("--status", action="store_true", help="Print latest daemon status JSON.")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--cache-db-path", default=None)
    parser.add_argument("--interval-minutes", type=int, default=None)
    parser.add_argument("--daily-budget", type=int, default=None)
    parser.add_argument("--max-api-requests", type=int, default=None)
    parser.add_argument("--cache-only", action="store_true")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> APIFootballDaemonConfig:
    config = APIFootballDaemonConfig.from_env()
    values = config.__dict__.copy()
    if args.db_path is not None:
        values["db_path"] = args.db_path
    if args.cache_db_path is not None:
        values["cache_db_path"] = args.cache_db_path
    if args.interval_minutes is not None:
        values["interval_minutes"] = args.interval_minutes
    if args.daily_budget is not None:
        values["daily_budget"] = args.daily_budget
    if args.max_api_requests is not None:
        values["max_api_requests_per_run"] = args.max_api_requests
    if args.cache_only:
        values["cache_only"] = True
    return APIFootballDaemonConfig(**values)


def main() -> int:
    """CLI entry point for one-shot sync, daemon mode, and status output."""
    args = _parse_args()
    config = _config_from_args(args)
    if args.status:
        print(json.dumps(read_daemon_status(config.db_path), ensure_ascii=False, indent=2))
        return 0
    if args.daemon:
        run_daemon(config)
        return 0
    summary = sync_api_football_today(config)
    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
    return 0 if summary.status in {"ok", "blocked"} else 1


if __name__ == "__main__":
    sys.exit(main())
