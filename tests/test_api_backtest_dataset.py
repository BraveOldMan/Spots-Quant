"""Tests for API-enriched backtest dataset standardization."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import api_client
import api_backtest_dataset as abd
from run_ultimate_backtest import validate_backtest_dataset


class FakeFootballClient:
    """Small response map for API-Football client tests."""

    def __init__(self, responses: dict[tuple[str, tuple[tuple[str, Any], ...]], dict[str, Any]]):
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        bypass_cache: bool = False,
        max_age: float | None = None,
    ) -> dict[str, Any] | None:
        active = params or {}
        self.calls.append((endpoint, active))
        key = (endpoint, tuple(active.items()))
        return self.responses.get(key)


def _api_fixture(
    fixture_id: int = 100,
    home: str = "Manchester United",
    away: str = "Chelsea",
    date: str = "2023-08-01T15:00:00+00:00",
    hg: int = 2,
    ag: int = 1,
) -> dict[str, Any]:
    return {
        "fixture": {
            "id": fixture_id,
            "date": date,
            "status": {"short": "FT"},
        },
        "league": {"season": 2023},
        "teams": {
            "home": {"id": 1, "name": home},
            "away": {"id": 2, "name": away},
        },
        "goals": {"home": hg, "away": ag},
    }


def _fixtures_response(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"errors": [], "response": items}


def _odds_response(update: str, fixture_id: int = 100) -> dict[str, Any]:
    return {
        "errors": [],
        "paging": {"current": 1, "total": 1},
        "response": [
            {
                "fixture": {"id": fixture_id, "date": "2023-08-01T15:00:00+00:00"},
                "update": update,
                "bookmakers": [
                    {
                        "name": "Pinnacle",
                        "bets": [
                            {
                                "name": "Match Winner",
                                "values": [
                                    {"value": "Home", "odd": "2.10"},
                                    {"value": "Draw", "odd": "3.20"},
                                    {"value": "Away", "odd": "3.60"},
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def test_extract_api_entry_odds_requires_pre_kickoff_match_winner() -> None:
    """Entry odds must come from a pre-kickoff Match Winner snapshot."""
    client = FakeFootballClient(
        {
            (
                "/odds",
                (("fixture", 100),),
            ): _odds_response("2023-08-01T10:00:00+00:00"),
        }
    )

    odds = abd.extract_api_entry_odds(client, 100, pd.Timestamp("2023-08-01T15:00:00Z"))

    assert odds is not None
    assert odds["B365H"] == 2.10
    assert odds["entry_bookmaker"] == "Pinnacle"


def test_extract_api_entry_odds_ignores_after_kickoff_and_uses_date_fallback() -> None:
    """Post-kickoff odds are invalid; date fallback may provide a safe entry line."""
    client = FakeFootballClient(
        {
            (
                "/odds",
                (("fixture", 100),),
            ): _odds_response("2023-08-01T16:00:00+00:00"),
            (
                "/odds",
                (("date", "2023-08-01"), ("page", 1)),
            ): _odds_response("2023-08-01T09:00:00+00:00"),
        }
    )

    odds = abd.extract_api_entry_odds(client, 100, pd.Timestamp("2023-08-01T15:00:00Z"))

    assert odds is not None
    assert odds["entry_snapshot_time"].startswith("2023-08-01T09:00:00")


def test_fixture_matching_skips_low_score_and_ambiguous_candidates() -> None:
    """Low-confidence or tied fuzzy matches must fail closed."""
    fixtures = pd.DataFrame(
        [
            {
                "fixture_id": 1,
                "kickoff": pd.Timestamp("2023-08-01T15:00:00Z"),
                "home": "Arsenal",
                "away": "Chelsea",
                "home_id": 1,
                "away_id": 2,
                "hg": 1,
                "ag": 0,
                "ftr": "H",
            },
            {
                "fixture_id": 2,
                "kickoff": pd.Timestamp("2023-08-01T15:00:00Z"),
                "home": "Arsenal",
                "away": "Chelsea",
                "home_id": 3,
                "away_id": 4,
                "hg": 1,
                "ag": 0,
                "ftr": "H",
            },
        ]
    )
    diagnostics: list[dict[str, Any]] = []
    local = {"Date": "01/08/2023", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea"}

    match = abd._match_fixture(local, fixtures, "unit", "1", diagnostics)

    assert match is None
    assert diagnostics[-1]["reason"] == "ambiguous_match"


def test_betfair_probs_filter_extreme_or_invalid_odds() -> None:
    """Betfair probability rows should not produce extreme decimal odds."""
    assert abd._betfair_probs_to_odds([0.50, 0.25, 0.25]) == (2.0, 4.0, 4.0)
    assert abd._betfair_probs_to_odds([0.001, 0.499, 0.500]) is None
    assert abd._betfair_probs_to_odds([0.30, 0.30, 0.30]) is None


def test_kaggle_standardization_maps_closing_and_skips_api_result_conflict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Kaggle rows should use local close odds but reject API score conflicts."""
    kaggle_dir = tmp_path / "kaggle_dataset"
    kaggle_dir.mkdir()
    pd.DataFrame(
        [
            {
                "match_id": 10,
                "league": "England: Premier League",
                "match_date": "2023-08-01",
                "home_team": "Manchester United",
                "home_score": 2,
                "away_team": "Chelsea",
                "away_score": 1,
                "avg_odds_home_win": 2.00,
                "avg_odds_draw": 3.30,
                "avg_odds_away_win": 4.00,
            }
        ]
    ).to_csv(kaggle_dir / "closing_odds.csv.gz", index=False)
    monkeypatch.setattr(abd, "KAGGLE_DIR", kaggle_dir)
    client = FakeFootballClient(
        {
            (
                "/fixtures",
                (("league", abd.EPL_LEAGUE_ID), ("season", 2023)),
            ): _fixtures_response([_api_fixture(hg=0, ag=0)]),
        }
    )
    diagnostics: list[dict[str, Any]] = []

    manifest = abd._standardize_kaggle(tmp_path / "out", client, diagnostics)
    closing_row = next(row for row in manifest if row["source_name"] == "kaggle_closing")

    assert closing_row["rows"] == 0
    assert any(item["reason"] == "api_result_conflict" for item in diagnostics)


def test_kaggle_odds_series_selects_opening_and_closing_suffixes() -> None:
    """Kaggle odds suffixes use max as opening and min as latest close."""
    row = pd.Series(
        {
            "home_b26_0": 1.90,
            "draw_b26_0": 3.40,
            "away_b26_0": 4.20,
            "home_b26_3": 2.10,
            "draw_b26_3": 3.20,
            "away_b26_3": 3.80,
            "home_b3_5": 9.00,
            "draw_b3_5": 9.00,
            "away_b3_5": 9.00,
        }
    )

    selected = abd._select_kaggle_odds_snapshots(row)

    assert selected is not None
    assert selected["bookmaker"] == 26
    assert selected["opening_suffix"] == 3
    assert selected["opening"] == (2.10, 3.20, 3.80)
    assert selected["closing_suffix"] == 0
    assert selected["closing"] == (1.90, 3.40, 4.20)


def test_kaggle_odds_series_filters_epl_and_writes_backtest_csv(
    tmp_path: Path,
) -> None:
    """Local Kaggle odds-series rows should become valid opening backtest CSVs."""
    matches_path = tmp_path / "odds_series_matches.csv.gz"
    odds_path = tmp_path / "odds_series.csv.gz"
    pd.DataFrame(
        [
            {
                "match_id": 1,
                "league": "England: Premier League",
                "match_datetime": "2016-01-01T15:00:00+00:00",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "score": "2:1",
            },
            {
                "match_id": 2,
                "league": "Spain: LaLiga",
                "match_datetime": "2016-01-01T15:00:00+00:00",
                "home_team": "A",
                "away_team": "B",
                "score": "0:0",
            },
        ]
    ).to_csv(matches_path, index=False)
    pd.DataFrame(
        [
            {
                "match_id": 1,
                "score_home": 2,
                "score_away": 1,
                "home_b26_0": 1.90,
                "draw_b26_0": 3.40,
                "away_b26_0": 4.20,
                "home_b26_4": 2.10,
                "draw_b26_4": 3.20,
                "away_b26_4": 3.80,
            },
            {
                "match_id": 2,
                "score_home": 0,
                "score_away": 0,
                "home_b26_0": 2.00,
                "draw_b26_0": 3.00,
                "away_b26_0": 4.00,
            },
        ]
    ).to_csv(odds_path, index=False)
    diagnostics: list[dict[str, Any]] = []

    manifest = abd._standardize_kaggle_odds_series(
        "odds_series",
        odds_path,
        matches_path,
        tmp_path / "out.csv",
        diagnostics,
    )
    output = pd.read_csv(tmp_path / "out.csv")

    assert manifest["rows"] == 1
    assert output.loc[0, "HomeTeam"] == "Arsenal"
    assert output.loc[0, "B365H"] == 2.10
    assert output.loc[0, "B365CH"] == 1.90
    assert output.loc[0, "entry_bookmaker"] == "kaggle_b26"
    failures = validate_backtest_dataset([str(tmp_path / "out.csv")])
    assert failures[failures["status"] == "failed"].empty


def test_kaggle_odds_series_falls_back_to_next_complete_bookmaker() -> None:
    """Bookmaker priority should skip incomplete three-way snapshots."""
    row = pd.Series(
        {
            "home_b26_0": 1.90,
            "draw_b26_0": None,
            "away_b26_0": 4.20,
            "home_b3_0": 1.80,
            "draw_b3_0": 3.30,
            "away_b3_0": 4.30,
            "home_b3_2": 2.05,
            "draw_b3_2": 3.10,
            "away_b3_2": 3.90,
        }
    )

    selected = abd._select_kaggle_odds_snapshots(row)

    assert selected is not None
    assert selected["bookmaker"] == 3
    assert selected["opening"] == (2.05, 3.10, 3.90)


def test_betfair_quality_audit_flags_extreme_and_duplicate_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Betfair quality audit should keep bad rows isolated from backtests."""
    source_path = tmp_path / "betfair.csv"
    pd.DataFrame(
        [
            {
                "match_name": "A v B",
                "market_time": "2024-01-01T15:00:00Z",
                "home_prob": 0.50,
                "draw_prob": 0.25,
                "away_prob": 0.25,
                "raw_odds": "{}",
            },
            {
                "match_name": "A v B",
                "market_time": "2024-01-01T15:00:00Z",
                "home_prob": 0.50,
                "draw_prob": 0.25,
                "away_prob": 0.25,
                "raw_odds": "{}",
            },
            {
                "match_name": "BadName",
                "market_time": "2024-01-02T15:00:00Z",
                "home_prob": 0.001,
                "draw_prob": 0.499,
                "away_prob": 0.500,
                "raw_odds": "{}",
            },
        ]
    ).to_csv(source_path, index=False)
    monkeypatch.setattr(abd, "BETFAIR_CLOSING_PATH", source_path)

    audit = abd.run_betfair_quality_audit(str(tmp_path))

    assert len(audit) == 3
    assert "duplicate_market" in audit.loc[0, "reason"]
    assert "bad_implied_odds" in audit.loc[2, "reason"]
    assert "bad_match_name" in audit.loc[2, "reason"]
    assert (tmp_path / "betfair_quality_audit.md").exists()


def test_build_handles_missing_api_key_and_writes_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing API credentials should not fabricate API-enriched rows."""

    class MissingKeyClient:
        def __init__(self) -> None:
            raise ValueError("missing key")

    data_dir = tmp_path / "data_seasons"
    data_dir.mkdir()
    pd.DataFrame(
        [
            {
                "Date": "01/08/2023",
                "HomeTeam": "A",
                "AwayTeam": "B",
                "FTHG": 1,
                "FTAG": 0,
                "FTR": "H",
                "B365H": 2.0,
                "B365D": 3.2,
                "B365A": 4.0,
                "B365CH": 1.9,
                "B365CD": 3.1,
                "B365CA": 4.2,
            }
        ]
    ).to_csv(data_dir / "E0_2324.csv", index=False)
    monkeypatch.setattr(abd, "FootballAPIClient", MissingKeyClient)
    monkeypatch.setattr(abd, "DATA_SEASONS_DIR", data_dir)
    monkeypatch.setattr(abd, "KAGGLE_DIR", tmp_path / "missing_kaggle")
    monkeypatch.setattr(abd, "BETFAIR_CLOSING_PATH", tmp_path / "missing_betfair.csv")

    manifest = abd.build_api_enriched_backtest_datasets(
        str(tmp_path / "out"), str(tmp_path / "reports")
    )

    assert (tmp_path / "out" / "diagnostics.csv").exists()
    assert "api_client_unavailable" in (tmp_path / "out" / "diagnostics.csv").read_text(
        encoding="utf-8"
    )
    data_rows = manifest[manifest["source_name"] == "data_seasons"]
    assert int(data_rows.iloc[0]["rows"]) == 1
    diagnostics = validate_backtest_dataset([str(tmp_path / "out" / "data_seasons" / "E0_2324.csv")])
    assert diagnostics[diagnostics["status"] == "failed"].empty


def test_api_env_loader_accepts_utf8_bom(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A UTF-8 BOM before API_FOOTBALL_KEY should not hide the configured key."""
    env_path = tmp_path / ".env"
    env_path.write_text("\ufeffAPI_FOOTBALL_KEY=abc123\n", encoding="utf-8")
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)

    api_client.load_env(str(env_path))

    assert os.environ["API_FOOTBALL_KEY"] == "abc123"


def test_cache_only_client_can_read_cache_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cache-only replay must not require a live API key or network request."""
    db_path = tmp_path / "api_cache.db"
    endpoint = "/fixtures?league=39&season=2024"
    payload = {"errors": [], "response": [{"fixture": {"id": 1}}]}
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE api_cache (endpoint TEXT PRIMARY KEY, response TEXT, timestamp REAL)"
        )
        conn.execute(
            "INSERT INTO api_cache VALUES (?, ?, ?)",
            (endpoint, json.dumps(payload), 1_900_000_000),
        )
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)

    client = api_client.FootballAPIClient(str(db_path), cache_only=True)
    response = client.get("/fixtures", {"league": 39, "season": 2024})

    assert response == payload
    assert client.stats.cache_hits == 1
    assert client.stats.api_requests == 0


def test_live_request_budget_fails_closed_without_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Budget exhaustion should return an API-shaped error instead of requesting."""
    monkeypatch.setenv("API_FOOTBALL_KEY", "abc123")
    client = api_client.FootballAPIClient(str(tmp_path / "cache.db"), max_api_requests=0)

    response = client.get("/fixtures", {"league": 39, "season": 2024})

    assert response is not None
    assert response["errors"]["client"] == "api_request_budget_exhausted"
    assert client.stats.budget_exhausted
    assert client.stats.api_requests == 0


def test_build_preflight_failure_writes_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Quota or status failure should stop API enrichment and write evidence."""

    class FakeStats:
        status_detail = "remaining_requests=0"
        last_error = ""

        def as_dict(self) -> dict[str, object]:
            return {
                "cache_hits": 0,
                "cache_misses": 0,
                "api_requests": 1,
                "api_errors": 0,
                "budget_exhausted": False,
                "quota_exhausted": True,
                "last_error": "",
                "status_checked": True,
                "status_ok": False,
                "status_detail": self.status_detail,
            }

    class PreflightFailClient:
        def __init__(self, max_api_requests: int, cache_only: bool) -> None:
            self.max_api_requests = max_api_requests
            self.cache_only = cache_only
            self.stats = FakeStats()

        def preflight_status(self) -> bool:
            return False

    monkeypatch.setattr(abd, "FootballAPIClient", PreflightFailClient)
    monkeypatch.setattr(abd, "DATA_SEASONS_DIR", tmp_path / "missing_data")
    monkeypatch.setattr(abd, "KAGGLE_DIR", tmp_path / "missing_kaggle")
    monkeypatch.setattr(abd, "BETFAIR_CLOSING_PATH", tmp_path / "missing_betfair.csv")

    abd.build_api_enriched_backtest_datasets(str(tmp_path / "out"), str(tmp_path / "reports"))
    diagnostics = (tmp_path / "out" / "diagnostics.csv").read_text(encoding="utf-8")
    report = (tmp_path / "reports" / "api_enrichment_audit.md").read_text(encoding="utf-8")

    assert "api_preflight_failed" in diagnostics
    assert "remaining_requests=0" in diagnostics
    assert "quota_exhausted" in report
