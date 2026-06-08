"""Tests for the API-Football daily sync daemon."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

import api_football_today_daemon as daemon


class FakeStats:
    """Minimal API stats object matching FootballAPIClient.stats."""

    def __init__(self) -> None:
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_requests = 0
        self.api_errors = 0
        self.blocked_requests = 0
        self.remaining_requests = 100
        self.budget_exhausted = False
        self.quota_exhausted = False
        self.last_error = ""
        self.status_checked = False
        self.status_ok = False
        self.status_detail = ""

    def as_dict(self) -> dict[str, object]:
        """Return report-safe counters."""
        return self.__dict__.copy()


class FakeFootballClient:
    """Response-map fake for daemon sync tests."""

    def __init__(
        self,
        responses: dict[tuple[str, tuple[tuple[str, Any], ...]], dict[str, Any]],
        preflight_ok: bool = True,
    ) -> None:
        self.responses = responses
        self.preflight_ok = preflight_ok
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.stats = FakeStats()
        self.max_api_requests = 100

    def preflight_status(self, max_age: float = 300) -> bool:
        """Pretend to validate API quota."""
        self.stats.status_checked = True
        self.stats.status_ok = self.preflight_ok
        self.stats.status_detail = "ok" if self.preflight_ok else "remaining_requests=0"
        self.stats.quota_exhausted = not self.preflight_ok
        return self.preflight_ok

    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        bypass_cache: bool = False,
        max_age: float | None = None,
    ) -> dict[str, Any] | None:
        """Return a mapped response and count it as a live request."""
        active = dict(params or {})
        self.calls.append((endpoint, active))
        self.stats.api_requests += 1
        key = (endpoint, tuple(sorted(active.items())))
        return self.responses.get(key, {"errors": [], "response": []})


def _fixture_item(
    fixture_id: int = 10,
    status: str = "NS",
    fixture_date: str = "2026-06-07T08:00:00+00:00",
) -> dict[str, Any]:
    return {
        "fixture": {
            "id": fixture_id,
            "date": fixture_date,
            "status": {"short": status, "long": "Not Started", "elapsed": None},
        },
        "league": {"id": 39, "name": "Premier League", "country": "England"},
        "teams": {
            "home": {"id": 1, "name": "Home FC"},
            "away": {"id": 2, "name": "Away FC"},
        },
        "goals": {"home": None, "away": None},
    }


def _fixture_response(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"errors": [], "response": items}


def _odds_response(page: int = 1, total: int = 1) -> dict[str, Any]:
    return {
        "errors": [],
        "paging": {"current": page, "total": total},
        "response": [
            {
                "fixture": {"id": 10},
                "update": "2026-06-07T07:30:00+00:00",
                "bookmakers": [
                    {
                        "id": 8,
                        "name": "Bet365",
                        "bets": [
                            {
                                "id": 1,
                                "name": "Match Winner",
                                "values": [
                                    {"value": "Home", "odd": "2.10"},
                                    {"value": "Draw", "odd": "3.20"},
                                    {"value": "Away", "odd": "3.70"},
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def _config(tmp_path: Path) -> daemon.APIFootballDaemonConfig:
    return daemon.APIFootballDaemonConfig(
        db_path=str(tmp_path / "spots_quant.db"),
        cache_db_path=str(tmp_path / "api_cache.db"),
        lock_path=str(tmp_path / "runtime" / "daemon.lock"),
        stop_path=str(tmp_path / "runtime" / "daemon.stop"),
        log_path=str(tmp_path / "logs" / "daemon.log"),
        daily_budget=50,
        max_api_requests_per_run=20,
        enrichment_window_hours=0.0,
    )


def test_sync_once_writes_fixtures_raw_odds_and_1x2(tmp_path: Path) -> None:
    """A safe one-shot sync should persist core fixtures and 1X2 odds."""
    target = date(2026, 6, 7)
    client = FakeFootballClient(
        {
            ("/fixtures", (("date", "2026-06-07"), ("timezone", "Asia/Shanghai"))): (
                _fixture_response([_fixture_item()])
            ),
            ("/odds", (("date", "2026-06-07"), ("page", 1))): _odds_response(),
        }
    )

    summary = daemon.sync_api_football_today(
        _config(tmp_path),
        target_date=target,
        client=client,
        now=datetime(2026, 6, 7, 0, 0, tzinfo=timezone.utc),
    )

    assert summary.status == "ok"
    assert summary.fixtures_count == 1
    assert summary.normalized_odds_count == 1
    with sqlite3.connect(tmp_path / "spots_quant.db") as conn:
        fixture_count = conn.execute("SELECT COUNT(*) FROM api_football_fixtures").fetchone()[0]
        raw_count = conn.execute("SELECT COUNT(*) FROM api_football_fixture_raw").fetchone()[0]
        odds = conn.execute(
            "SELECT home_odds, draw_odds, away_odds FROM api_football_odds_1x2"
        ).fetchone()
    assert fixture_count == 1
    assert raw_count == 1
    assert odds == (2.1, 3.2, 3.7)


def test_sync_odds_pagination_is_followed(tmp_path: Path) -> None:
    """Odds pagination should request all advertised API-Football pages."""
    client = FakeFootballClient(
        {
            ("/fixtures", (("date", "2026-06-07"), ("timezone", "Asia/Shanghai"))): (
                _fixture_response([_fixture_item()])
            ),
            ("/odds", (("date", "2026-06-07"), ("page", 1))): _odds_response(1, 2),
            ("/odds", (("date", "2026-06-07"), ("page", 2))): _odds_response(2, 2),
        }
    )

    daemon.sync_api_football_today(_config(tmp_path), date(2026, 6, 7), client=client)

    assert ("/odds", {"date": "2026-06-07", "page": 2}) in client.calls


def test_daily_budget_exhaustion_blocks_live_requests(tmp_path: Path) -> None:
    """A depleted daily budget should write evidence and avoid API requests."""
    cfg = _config(tmp_path)
    cfg = daemon.APIFootballDaemonConfig(
        **{**cfg.__dict__, "daily_budget": 0, "max_api_requests_per_run": 10}
    )

    summary = daemon.sync_api_football_today(cfg, target_date=date(2026, 6, 7))

    assert summary.status == "blocked"
    with sqlite3.connect(tmp_path / "spots_quant.db") as conn:
        reason = conn.execute(
            "SELECT reason FROM api_football_diagnostics ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
    assert reason == "api_daily_budget_exhausted"


def test_preflight_failure_fails_closed_and_writes_diagnostics(tmp_path: Path) -> None:
    """Quota or status failures should stop the run before fixture reads."""
    client = FakeFootballClient({}, preflight_ok=False)

    summary = daemon.sync_api_football_today(
        _config(tmp_path),
        target_date=date(2026, 6, 7),
        client=client,
    )

    assert summary.status == "blocked"
    assert client.calls == []
    with sqlite3.connect(tmp_path / "spots_quant.db") as conn:
        reason = conn.execute(
            "SELECT reason FROM api_football_diagnostics ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
    assert reason == "api_preflight_failed"


def test_singleton_lock_blocks_second_running_instance(tmp_path: Path) -> None:
    """The lock file should prevent two daemon instances in one workspace."""
    lock_path = tmp_path / "runtime" / "daemon.lock"

    with daemon.InstanceLock(lock_path):
        with pytest.raises(RuntimeError, match="already running"):
            with daemon.InstanceLock(lock_path):
                pass


def test_status_reads_latest_heartbeat_and_run(tmp_path: Path) -> None:
    """Status should expose the latest heartbeat without printing secrets."""
    cfg = _config(tmp_path)
    daemon.init_sync_db(cfg.db_path)
    with sqlite3.connect(cfg.db_path) as conn:
        conn.execute(
            """
            INSERT INTO api_football_sync_runs (
                run_id, started_at, finished_at, target_date, status, cache_only,
                max_api_requests, daily_budget, fixtures_count, odds_items_count,
                normalized_odds_count, enrichment_items_count, diagnostics_count,
                api_requests, cache_hits, cache_misses, api_errors, blocked_requests,
                remaining_requests, status_detail, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run-1",
                "2026-06-07T00:00:00+00:00",
                "2026-06-07T00:00:01+00:00",
                "2026-06-07",
                "ok",
                0,
                20,
                50,
                1,
                2,
                1,
                0,
                0,
                3,
                0,
                3,
                0,
                0,
                100,
                "ok",
                "",
            ),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO api_football_daemon_heartbeat (
                id, pid, updated_at, target_date, last_run_id, last_status, next_run_at, message
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            """,
            (os.getpid(), "now", "2026-06-07", "run-1", "ok", "next", "healthy"),
        )

    status = daemon.read_daemon_status(cfg.db_path)

    assert status["heartbeat"]["last_run_id"] == "run-1"
    assert status["latest_run"]["status"] == "ok"
    json.dumps(status)
