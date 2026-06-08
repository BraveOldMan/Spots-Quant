"""Build API-enriched, backtest-ready football datasets.

The module standardizes local `data_seasons`, Kaggle, and Betfair sources into
the minimum CSV contract accepted by `run_real_backtest`. API-Football is used
only through the existing `FootballAPIClient`; when required data is missing the
row is skipped and diagnostics are written instead of inventing values.
"""

from __future__ import annotations

import argparse
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from api_client import FootballAPIClient
from config import get_settings
from run_ultimate_backtest import validate_backtest_dataset

EPL_LEAGUE_ID = 39
MATCH_WINDOW_HOURS = 36
MATCH_SCORE_THRESHOLD = 0.86
MATCH_TIE_MARGIN = 0.02
MAX_REASONABLE_ODDS = 100.0

DATA_SEASONS_DIR = Path("data_seasons")
KAGGLE_DIR = Path("kaggle_dataset")
BETFAIR_CLOSING_PATH = Path("betfair_closing_odds_full.csv")
KAGGLE_BOOKMAKER_PRIORITY = (26, 3, 1)
KAGGLE_ODDS_SERIES_SOURCES = (
    ("odds_series", "odds_series.csv.gz", "odds_series_matches.csv.gz"),
    ("odds_series_b", "odds_series_b.csv.gz", "odds_series_b_matches.csv.gz"),
)

STANDARD_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "B365H",
    "B365D",
    "B365A",
    "B365CH",
    "B365CD",
    "B365CA",
]
AUDIT_COLUMNS = [
    "source_name",
    "source_match_id",
    "api_fixture_id",
    "api_home_id",
    "api_away_id",
    "match_score",
    "entry_bookmaker",
    "entry_snapshot_time",
    "closing_source",
    "api_status",
]
OUTPUT_COLUMNS = STANDARD_COLUMNS + AUDIT_COLUMNS


@dataclass(frozen=True)
class FixtureMatch:
    """Accepted API-Football fixture match for one local source row."""

    fixture_id: int
    home_id: int | None
    away_id: int | None
    home: str
    away: str
    kickoff: pd.Timestamp
    hg: int | None
    ag: int | None
    ftr: str | None
    score: float


def build_api_enriched_backtest_datasets(
    output_dir: str = "data_standardized/api_backtest",
    reports_dir: str = "reports",
    max_api_requests: int = 250,
    cache_only: bool = False,
) -> pd.DataFrame:
    """Standardize all local source families and write manifest plus diagnostics."""
    out_dir = Path(output_dir)
    report_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    diagnostics: list[dict[str, Any]] = []
    client, api_stats = _make_client(
        diagnostics,
        max_api_requests=max_api_requests,
        cache_only=cache_only,
    )
    manifest_rows: list[dict[str, Any]] = []

    for source_name in ("data_seasons", "kaggle", "betfair"):
        result = _standardize_source_dataset(source_name, out_dir, client, diagnostics)
        manifest_rows.extend(result)

    manifest = pd.DataFrame(
        manifest_rows,
        columns=[
            "source_name",
            "output_path",
            "rows",
            "status",
            "validation_status",
            "reason",
        ],
    )
    diagnostics_df = pd.DataFrame(
        diagnostics,
        columns=["source_name", "source_match_id", "severity", "status", "reason", "detail"],
    )
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    diagnostics_df.to_csv(out_dir / "diagnostics.csv", index=False)
    api_summary = _api_connection_summary(client, max_api_requests, cache_only, api_stats)
    _write_audit_markdown(
        manifest,
        diagnostics_df,
        report_dir / "api_enrichment_audit.md",
        api_summary,
    )
    return manifest


def load_api_fixtures(
    client: FootballAPIClient,
    league_id: int,
    seasons: Sequence[int],
) -> pd.DataFrame:
    """Load API-Football fixtures for league seasons into a normalized table."""
    rows: list[dict[str, Any]] = []
    api_errors: list[dict[str, Any]] = []
    for season in sorted(set(int(item) for item in seasons)):
        response = client.get("/fixtures", {"league": league_id, "season": season})
        if not _api_response_ok(response):
            api_errors.append({"season": season, "detail": _api_error_detail(response)})
            continue
        for item in response.get("response", []):
            fixture = item.get("fixture", {})
            teams = item.get("teams", {})
            goals = item.get("goals", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            kickoff = pd.to_datetime(fixture.get("date"), utc=True, errors="coerce")
            if pd.isna(kickoff):
                continue
            hg = _safe_int(goals.get("home"))
            ag = _safe_int(goals.get("away"))
            rows.append(
                {
                    "fixture_id": _safe_int(fixture.get("id")),
                    "season": season,
                    "kickoff": kickoff,
                    "home": str(home.get("name", "")),
                    "away": str(away.get("name", "")),
                    "home_id": _safe_int(home.get("id")),
                    "away_id": _safe_int(away.get("id")),
                    "status": fixture.get("status", {}).get("short", ""),
                    "hg": hg,
                    "ag": ag,
                    "ftr": _ftr_from_goals(hg, ag),
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["api_errors"] = api_errors
    return frame


def extract_api_entry_odds(
    client: FootballAPIClient,
    fixture_id: int,
    kickoff: pd.Timestamp,
) -> dict[str, object] | None:
    """Return the earliest valid pre-kickoff Match Winner odds for one fixture."""
    kickoff_ts = _utc_timestamp(kickoff)
    candidates: list[dict[str, object]] = []

    fixture_response = client.get("/odds", {"fixture": int(fixture_id)})
    candidates.extend(_odds_candidates_from_response(fixture_response, fixture_id, kickoff_ts))

    if not candidates:
        date_str = kickoff_ts.strftime("%Y-%m-%d")
        first_page = client.get("/odds", {"date": date_str, "page": 1})
        candidates.extend(_odds_candidates_from_response(first_page, fixture_id, kickoff_ts))
        total_pages = _total_pages(first_page)
        for page in range(2, total_pages + 1):
            page_response = client.get("/odds", {"date": date_str, "page": page})
            candidates.extend(_odds_candidates_from_response(page_response, fixture_id, kickoff_ts))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item["snapshot_time"], item["bookmaker_rank"]))
    selected = candidates[0]
    return {
        "B365H": selected["B365H"],
        "B365D": selected["B365D"],
        "B365A": selected["B365A"],
        "entry_bookmaker": selected["entry_bookmaker"],
        "entry_snapshot_time": selected["snapshot_time"].isoformat(),
        "api_status": "api_odds_matched",
    }


def standardize_source_dataset(source_name: str, output_dir: str) -> pd.DataFrame:
    """Standardize one source family into backtest-ready CSV files."""
    diagnostics: list[dict[str, Any]] = []
    client, _ = _make_client(diagnostics, max_api_requests=250, cache_only=False)
    out_dir = Path(output_dir)
    manifest_rows = _standardize_source_dataset(source_name, out_dir, client, diagnostics)
    frames = []
    for row in manifest_rows:
        output_path = row.get("output_path")
        if output_path and Path(str(output_path)).exists():
            frames.append(pd.read_csv(str(output_path)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)


def run_betfair_quality_audit(output_dir: str = "reports") -> pd.DataFrame:
    """Audit Betfair closing data quality without promoting it to backtest input."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if BETFAIR_CLOSING_PATH.exists():
        source = pd.read_csv(BETFAIR_CLOSING_PATH)
        duplicate_mask = source.duplicated(subset=["match_name", "market_time"], keep=False)
        for idx, row in source.iterrows():
            probs = [row.get("home_prob"), row.get("draw_prob"), row.get("away_prob")]
            prob_sum = _safe_prob_sum(probs)
            closing_odds = _betfair_probs_to_odds(probs)
            parsed_names = _split_match_name(str(row.get("match_name", "")))
            rows.append(
                {
                    "row_id": idx,
                    "match_name": row.get("match_name", ""),
                    "market_time": row.get("market_time", ""),
                    "prob_sum": prob_sum,
                    "valid_prob_sum": np.isfinite(prob_sum) and 0.95 <= prob_sum <= 1.05,
                    "valid_odds": closing_odds is not None,
                    "duplicate_market": bool(duplicate_mask.iloc[idx]),
                    "name_parse_ok": parsed_names is not None,
                    "would_enter_backtest": False,
                    "reason": _betfair_quality_reason(prob_sum, closing_odds, bool(duplicate_mask.iloc[idx]), parsed_names),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "betfair_quality_audit.csv", index=False)
    _write_betfair_quality_markdown(df, out_dir / "betfair_quality_audit.md")
    return df


def _standardize_source_dataset(
    source_name: str,
    output_dir: Path,
    client: FootballAPIClient | None,
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if source_name == "data_seasons":
        return _standardize_data_seasons(output_dir, client, diagnostics)
    if source_name == "kaggle":
        return _standardize_kaggle(output_dir, client, diagnostics)
    if source_name == "betfair":
        return _standardize_betfair(output_dir, client, diagnostics)
    raise ValueError(f"Unknown source_name: {source_name}")


def _standardize_data_seasons(
    output_dir: Path,
    client: FootballAPIClient | None,
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_dir = output_dir / "data_seasons"
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(DATA_SEASONS_DIR.glob("E0_*.csv"))
    seasons = sorted({_epl_season_from_path(path) for path in paths})
    fixtures = _load_fixtures_or_empty(client, seasons, diagnostics, "data_seasons")
    manifest: list[dict[str, Any]] = []

    for path in paths:
        source = pd.read_csv(path)
        rows = []
        for idx, row in source.iterrows():
            source_id = f"{path.stem}:{idx}"
            base = _local_standard_row(row, "data_seasons", source_id, "football_data_close")
            if base is None:
                _diagnostic(diagnostics, "data_seasons", source_id, "warning", "skipped", "bad_row", "")
                continue
            match = _match_fixture(base, fixtures, "data_seasons", source_id, diagnostics)
            if match is not None:
                base.update(_api_audit_fields(match, "api_fixture_matched"))
                if _local_result_conflicts(base, match):
                    _diagnostic(
                        diagnostics,
                        "data_seasons",
                        source_id,
                        "warning",
                        "kept",
                        "api_result_conflict",
                        f"local={base['FTHG']}-{base['FTAG']} api={match.hg}-{match.ag}",
                    )
            rows.append(base)
        out_path = target_dir / path.name
        manifest.append(_write_standardized_frame(pd.DataFrame(rows), out_path, "data_seasons"))
    return manifest


def _standardize_kaggle(
    output_dir: Path,
    client: FootballAPIClient | None,
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_dir = output_dir / "kaggle"
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for label, odds_file, matches_file in KAGGLE_ODDS_SERIES_SOURCES:
        manifest.append(
            _standardize_kaggle_odds_series(
                label,
                KAGGLE_DIR / odds_file,
                KAGGLE_DIR / matches_file,
                target_dir / f"{label}_standardized.csv",
                diagnostics,
            )
        )

    closing_path = KAGGLE_DIR / "closing_odds.csv.gz"
    if not closing_path.exists():
        manifest.append(
            _empty_manifest("kaggle_closing", str(target_dir / "epl_closing_standardized.csv"), "missing_source")
        )
        return manifest
    if client is None:
        _diagnostic(diagnostics, "kaggle", "", "error", "skipped", "api_unavailable", "")
        manifest.append(
            _empty_manifest("kaggle_closing", str(target_dir / "epl_closing_standardized.csv"), "api_unavailable")
        )
        return manifest

    closing = pd.read_csv(closing_path, encoding="latin-1")
    league_mask = closing["league"].astype(str).str.strip() == "England: Premier League"
    closing = closing.loc[league_mask].copy()
    closing.loc[:, "_date"] = pd.to_datetime(closing["match_date"], errors="coerce", utc=True)
    seasons = sorted({_epl_season_from_timestamp(value) for value in closing["_date"].dropna()})
    fixtures = _load_fixtures_or_empty(client, seasons, diagnostics, "kaggle")

    rows = []
    for _, row in closing.iterrows():
        source_id = str(row["match_id"])
        if _api_calls_blocked(client):
            _diagnostic(
                diagnostics,
                "kaggle",
                source_id,
                "warning",
                "skipped",
                "api_request_blocked",
                _api_block_reason(client),
            )
            break
        local = {
            "Date": _format_date(row["_date"]),
            "HomeTeam": str(row["home_team"]),
            "AwayTeam": str(row["away_team"]),
            "FTHG": _safe_int(row["home_score"]),
            "FTAG": _safe_int(row["away_score"]),
            "FTR": _ftr_from_goals(_safe_int(row["home_score"]), _safe_int(row["away_score"])),
            "B365CH": float(row["avg_odds_home_win"]),
            "B365CD": float(row["avg_odds_draw"]),
            "B365CA": float(row["avg_odds_away_win"]),
            "source_name": "kaggle",
            "source_match_id": source_id,
            "api_fixture_id": "",
            "api_home_id": "",
            "api_away_id": "",
            "match_score": 0.0,
            "entry_bookmaker": "",
            "entry_snapshot_time": "",
            "closing_source": "kaggle_avg_close",
            "api_status": "pending",
        }
        if not _standard_row_valid(local, require_opening=False):
            _diagnostic(diagnostics, "kaggle", source_id, "warning", "skipped", "bad_local_row", "")
            continue
        match = _match_fixture(local, fixtures, "kaggle", source_id, diagnostics)
        if match is None:
            continue
        if _local_result_conflicts(local, match):
            _diagnostic(diagnostics, "kaggle", source_id, "warning", "skipped", "api_result_conflict", "")
            continue
        odds = extract_api_entry_odds(client, match.fixture_id, match.kickoff)
        if odds is None:
            _diagnostic(diagnostics, "kaggle", source_id, "warning", "skipped", "entry_odds_missing", "")
            continue
        local.update(_api_audit_fields(match, "api_fixture_matched"))
        local.update(odds)
        rows.append(local)

    out_path = target_dir / "epl_closing_standardized.csv"
    manifest.append(_write_standardized_frame(pd.DataFrame(rows), out_path, "kaggle_closing"))
    return manifest


def _standardize_kaggle_odds_series(
    label: str,
    odds_path: Path,
    matches_path: Path,
    output_path: Path,
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Standardize Kaggle odds-series rows using only local pre-match odds."""
    source_name = f"kaggle_{label}"
    if not odds_path.exists() or not matches_path.exists():
        _diagnostic(diagnostics, source_name, "", "warning", "skipped", "missing_source", "")
        return _empty_manifest(source_name, str(output_path), "missing_source")

    matches = pd.read_csv(matches_path, encoding="latin-1")
    matches.columns = [str(col).strip() for col in matches.columns]
    matches = matches[matches["league"].astype(str).str.strip() == "England: Premier League"].copy()
    matches = matches.drop_duplicates(subset=["match_id"], keep="first")
    if matches.empty:
        return _write_standardized_frame(pd.DataFrame(), output_path, source_name)

    usecols = _kaggle_odds_usecols(odds_path)
    odds = pd.read_csv(odds_path, encoding="latin-1", usecols=lambda col: col in usecols)
    odds = odds[odds["match_id"].isin(set(matches["match_id"]))].copy()
    merged = matches.merge(odds, on="match_id", how="inner", suffixes=("_match", "_odds"))

    rows = []
    seen_ids: set[int] = set()
    for _, row in merged.iterrows():
        source_id = str(row["match_id"])
        if int(row["match_id"]) in seen_ids:
            _diagnostic(diagnostics, source_name, source_id, "warning", "skipped", "duplicate_match_id", "")
            continue
        seen_ids.add(int(row["match_id"]))
        selected = _select_kaggle_odds_snapshots(row)
        if selected is None:
            _diagnostic(diagnostics, source_name, source_id, "warning", "skipped", "no_complete_odds_series", "")
            continue
        hg = _safe_int(row.get("score_home", row.get("score_home_odds")))
        ag = _safe_int(row.get("score_away", row.get("score_away_odds")))
        if hg is None or ag is None:
            hg, ag = _parse_score(str(row.get("score", "")))
        ftr = _ftr_from_goals(hg, ag)
        kickoff = pd.to_datetime(row.get("match_datetime"), errors="coerce", utc=True)
        if pd.isna(kickoff) or hg is None or ag is None or ftr is None:
            _diagnostic(diagnostics, source_name, source_id, "warning", "skipped", "bad_match_result", "")
            continue
        local = {
            "Date": _format_date(kickoff),
            "HomeTeam": str(row["home_team"]).strip(),
            "AwayTeam": str(row["away_team"]).strip(),
            "FTHG": hg,
            "FTAG": ag,
            "FTR": ftr,
            "B365H": selected["opening"][0],
            "B365D": selected["opening"][1],
            "B365A": selected["opening"][2],
            "B365CH": selected["closing"][0],
            "B365CD": selected["closing"][1],
            "B365CA": selected["closing"][2],
            "source_name": source_name,
            "source_match_id": source_id,
            "api_fixture_id": "",
            "api_home_id": "",
            "api_away_id": "",
            "match_score": 1.0,
            "entry_bookmaker": f"kaggle_b{selected['bookmaker']}",
            "entry_snapshot_time": f"suffix_{selected['opening_suffix']}",
            "closing_source": f"kaggle_b{selected['bookmaker']}_suffix_{selected['closing_suffix']}",
            "api_status": "local_kaggle_odds_series",
        }
        if _standard_row_valid(local, require_opening=True):
            rows.append(local)
        else:
            _diagnostic(diagnostics, source_name, source_id, "warning", "skipped", "bad_standardized_row", "")

    return _write_standardized_frame(pd.DataFrame(rows), output_path, source_name)


def _standardize_betfair(
    output_dir: Path,
    client: FootballAPIClient | None,
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_dir = output_dir / "betfair"
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "betfair_standardized.csv"
    if not BETFAIR_CLOSING_PATH.exists():
        return [_empty_manifest("betfair", str(out_path), "missing_source")]
    if client is None:
        _diagnostic(diagnostics, "betfair", "", "error", "skipped", "api_unavailable", "")
        return [_empty_manifest("betfair", str(out_path), "api_unavailable")]

    betfair = pd.read_csv(BETFAIR_CLOSING_PATH)
    betfair = betfair.copy()
    betfair.loc[:, "_date"] = pd.to_datetime(betfair["market_time"], errors="coerce", utc=True)
    betfair = betfair.dropna(subset=["_date"]).copy()
    seasons = sorted({_epl_season_from_timestamp(value) for value in betfair["_date"]})
    fixtures = _load_fixtures_or_empty(client, seasons, diagnostics, "betfair")

    seen: set[tuple[str, str, str]] = set()
    rows = []
    for idx, row in betfair.iterrows():
        source_id = str(idx)
        if _api_calls_blocked(client):
            _diagnostic(
                diagnostics,
                "betfair",
                source_id,
                "warning",
                "skipped",
                "api_request_blocked",
                _api_block_reason(client),
            )
            break
        probs = [row.get("home_prob"), row.get("draw_prob"), row.get("away_prob")]
        closing_odds = _betfair_probs_to_odds(probs)
        if closing_odds is None:
            _diagnostic(diagnostics, "betfair", source_id, "warning", "skipped", "bad_closing_probs", "")
            continue
        parsed_names = _split_match_name(str(row.get("match_name", "")))
        if parsed_names is None:
            _diagnostic(diagnostics, "betfair", source_id, "warning", "skipped", "bad_match_name", "")
            continue
        home, away = parsed_names
        local = {
            "Date": _format_date(row["_date"]),
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": np.nan,
            "FTAG": np.nan,
            "FTR": "",
            "B365CH": closing_odds[0],
            "B365CD": closing_odds[1],
            "B365CA": closing_odds[2],
            "source_name": "betfair",
            "source_match_id": source_id,
            "api_fixture_id": "",
            "api_home_id": "",
            "api_away_id": "",
            "match_score": 0.0,
            "entry_bookmaker": "",
            "entry_snapshot_time": "",
            "closing_source": "betfair_prob_close",
            "api_status": "pending",
        }
        match = _match_fixture(local, fixtures, "betfair", source_id, diagnostics)
        if match is None or match.hg is None or match.ag is None or match.ftr is None:
            _diagnostic(diagnostics, "betfair", source_id, "warning", "skipped", "api_result_missing", "")
            continue
        duplicate_key = (_format_date(match.kickoff), match.home, match.away)
        if duplicate_key in seen:
            _diagnostic(diagnostics, "betfair", source_id, "warning", "skipped", "duplicate_fixture", "")
            continue
        odds = extract_api_entry_odds(client, match.fixture_id, match.kickoff)
        if odds is None:
            _diagnostic(diagnostics, "betfair", source_id, "warning", "skipped", "entry_odds_missing", "")
            continue
        seen.add(duplicate_key)
        local.update(
            {
                "Date": _format_date(match.kickoff),
                "HomeTeam": match.home,
                "AwayTeam": match.away,
                "FTHG": match.hg,
                "FTAG": match.ag,
                "FTR": match.ftr,
            }
        )
        local.update(_api_audit_fields(match, "api_fixture_matched"))
        local.update(odds)
        rows.append(local)

    return [_write_standardized_frame(pd.DataFrame(rows), out_path, "betfair")]


def _make_client(
    diagnostics: list[dict[str, Any]],
    max_api_requests: int,
    cache_only: bool,
) -> tuple[FootballAPIClient | None, dict[str, object]]:
    try:
        client = FootballAPIClient(
            max_api_requests=max_api_requests,
            cache_only=cache_only,
        )
    except Exception as exc:
        _diagnostic(diagnostics, "api", "", "error", "skipped", "api_client_unavailable", str(exc))
        return None, {}
    if cache_only:
        _diagnostic(diagnostics, "api", "", "info", "kept", "api_cache_only", "")
        return client, client.stats.as_dict()
    if not client.preflight_status():
        stats = client.stats.as_dict()
        _diagnostic(
            diagnostics,
            "api",
            "",
            "error",
            "skipped",
            "api_preflight_failed",
            client.stats.status_detail or client.stats.last_error,
        )
        return None, stats
    if client.max_api_requests is not None and client.stats.remaining_requests is not None:
        client.max_api_requests = min(
            client.max_api_requests,
            max(client.stats.remaining_requests, 0),
        )
    _diagnostic(
        diagnostics,
        "api",
        "",
        "info",
        "kept",
        "api_preflight_ok",
        client.stats.status_detail,
    )
    return client, client.stats.as_dict()


def _api_calls_blocked(client: FootballAPIClient) -> bool:
    stats = getattr(client, "stats", None)
    return bool(
        getattr(stats, "budget_exhausted", False)
        or getattr(stats, "quota_exhausted", False)
    )


def _api_block_reason(client: FootballAPIClient) -> str:
    stats = getattr(client, "stats", None)
    if getattr(stats, "quota_exhausted", False):
        return "api_quota_exhausted"
    if getattr(stats, "budget_exhausted", False):
        return "api_request_budget_exhausted"
    return str(getattr(stats, "last_error", "api_blocked"))


def _load_fixtures_or_empty(
    client: FootballAPIClient | None,
    seasons: Sequence[int],
    diagnostics: list[dict[str, Any]],
    source_name: str,
) -> pd.DataFrame:
    if client is None or not seasons:
        return pd.DataFrame()
    fixtures = load_api_fixtures(client, EPL_LEAGUE_ID, seasons)
    for item in fixtures.attrs.get("api_errors", []):
        _diagnostic(
            diagnostics,
            source_name,
            str(item.get("season", "")),
            "warning",
            "skipped",
            "api_fixtures_error",
            str(item.get("detail", "")),
        )
    if fixtures.empty:
        _diagnostic(
            diagnostics,
            source_name,
            "",
            "warning",
            "skipped",
            "api_fixtures_empty",
            ",".join(str(item) for item in seasons),
        )
    return fixtures


def _match_fixture(
    local: dict[str, Any],
    fixtures: pd.DataFrame,
    source_name: str,
    source_id: str,
    diagnostics: list[dict[str, Any]],
) -> FixtureMatch | None:
    if fixtures.empty:
        _diagnostic(diagnostics, source_name, source_id, "warning", "skipped", "no_api_fixtures", "")
        return None
    local_date = pd.to_datetime(local.get("Date"), dayfirst=True, errors="coerce", utc=True)
    if pd.isna(local_date):
        _diagnostic(diagnostics, source_name, source_id, "warning", "skipped", "bad_date", "")
        return None
    window = pd.Timedelta(hours=MATCH_WINDOW_HOURS)
    candidates = fixtures[
        (fixtures["kickoff"] >= local_date - window)
        & (fixtures["kickoff"] <= local_date + window)
    ].copy()
    if candidates.empty:
        _diagnostic(diagnostics, source_name, source_id, "warning", "skipped", "no_time_match", "")
        return None

    scored = []
    for candidate in candidates.itertuples(index=False):
        score = (
            _name_score(str(local["HomeTeam"]), str(candidate.home))
            + _name_score(str(local["AwayTeam"]), str(candidate.away))
        ) / 2.0
        scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -1.0
    if best_score < MATCH_SCORE_THRESHOLD:
        _diagnostic(
            diagnostics,
            source_name,
            source_id,
            "warning",
            "skipped",
            "low_match_score",
            f"{best_score:.3f}",
        )
        return None
    if second_score >= best_score - MATCH_TIE_MARGIN:
        _diagnostic(
            diagnostics,
            source_name,
            source_id,
            "warning",
            "skipped",
            "ambiguous_match",
            f"best={best_score:.3f};second={second_score:.3f}",
        )
        return None
    return FixtureMatch(
        fixture_id=int(best.fixture_id),
        home_id=_safe_int(best.home_id),
        away_id=_safe_int(best.away_id),
        home=str(best.home),
        away=str(best.away),
        kickoff=best.kickoff,
        hg=_safe_int(best.hg),
        ag=_safe_int(best.ag),
        ftr=best.ftr if isinstance(best.ftr, str) else None,
        score=float(best_score),
    )


def _api_audit_fields(match: FixtureMatch, status: str) -> dict[str, Any]:
    return {
        "api_fixture_id": match.fixture_id,
        "api_home_id": match.home_id if match.home_id is not None else "",
        "api_away_id": match.away_id if match.away_id is not None else "",
        "match_score": round(match.score, 6),
        "api_status": status,
    }


def _local_standard_row(
    row: pd.Series,
    source_name: str,
    source_id: str,
    closing_source: str,
) -> dict[str, Any] | None:
    payload = {column: row.get(column) for column in STANDARD_COLUMNS}
    payload.update(
        {
            "source_name": source_name,
            "source_match_id": source_id,
            "api_fixture_id": "",
            "api_home_id": "",
            "api_away_id": "",
            "match_score": 0.0,
            "entry_bookmaker": "local",
            "entry_snapshot_time": "",
            "closing_source": closing_source,
            "api_status": "local_only",
        }
    )
    return payload if _standard_row_valid(payload, require_opening=True) else None


def _standard_row_valid(row: dict[str, Any], require_opening: bool) -> bool:
    required = STANDARD_COLUMNS if require_opening else [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "B365CH",
        "B365CD",
        "B365CA",
    ]
    if any(pd.isna(row.get(column)) or str(row.get(column)).strip() == "" for column in required):
        return False
    odds_columns = ["B365CH", "B365CD", "B365CA"]
    if require_opening:
        odds_columns.extend(["B365H", "B365D", "B365A"])
    return all(_valid_decimal_odds(row.get(column)) for column in odds_columns)


def _write_standardized_frame(
    frame: pd.DataFrame,
    output_path: Path,
    source_name: str,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        frame = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        frame = frame.copy()
        for column in OUTPUT_COLUMNS:
            if column not in frame:
                frame[column] = ""
        frame = frame[OUTPUT_COLUMNS]
        frame = frame.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="first")
    frame.to_csv(output_path, index=False)
    validation_status = "empty"
    reason = ""
    if not frame.empty:
        diagnostics = validate_backtest_dataset([str(output_path)])
        failures = diagnostics[
            (diagnostics["severity"] == "error") & (diagnostics["status"] == "failed")
        ]
        validation_status = "failed" if not failures.empty else "ok"
        reason = ";".join(failures["check"].astype(str).tolist())
    return {
        "source_name": source_name,
        "output_path": str(output_path),
        "rows": len(frame),
        "status": "ok" if len(frame) else "empty",
        "validation_status": validation_status,
        "reason": reason,
    }


def _empty_manifest(source_name: str, output_path: str, reason: str) -> dict[str, Any]:
    return {
        "source_name": source_name,
        "output_path": output_path,
        "rows": 0,
        "status": "empty",
        "validation_status": "empty",
        "reason": reason,
    }


def _kaggle_odds_usecols(path: Path) -> set[str]:
    header = pd.read_csv(path, encoding="latin-1", nrows=0)
    keep = {"match_id", "match_date", "match_time", "score_home", "score_away"}
    pattern = re.compile(r"^(home|draw|away)_b(\d+)_(\d+)$")
    for column in header.columns:
        match = pattern.match(str(column))
        if match and int(match.group(2)) in KAGGLE_BOOKMAKER_PRIORITY:
            keep.add(str(column))
    return keep


def _select_kaggle_odds_snapshots(row: pd.Series) -> dict[str, Any] | None:
    """Select same-bookmaker opening and closing snapshots from Kaggle columns."""
    for bookmaker in KAGGLE_BOOKMAKER_PRIORITY:
        snapshots: list[tuple[int, tuple[float, float, float]]] = []
        suffixes = _kaggle_suffixes_for_bookmaker(row.index, bookmaker)
        for suffix in suffixes:
            odds = (
                row.get(f"home_b{bookmaker}_{suffix}"),
                row.get(f"draw_b{bookmaker}_{suffix}"),
                row.get(f"away_b{bookmaker}_{suffix}"),
            )
            if all(_valid_decimal_odds(value) for value in odds):
                snapshots.append((suffix, (float(odds[0]), float(odds[1]), float(odds[2]))))
        if snapshots:
            snapshots.sort(key=lambda item: item[0])
            return {
                "bookmaker": bookmaker,
                "closing_suffix": snapshots[0][0],
                "closing": snapshots[0][1],
                "opening_suffix": snapshots[-1][0],
                "opening": snapshots[-1][1],
            }
    return None


def _kaggle_suffixes_for_bookmaker(columns: Sequence[Any], bookmaker: int) -> list[int]:
    pattern = re.compile(rf"^home_b{bookmaker}_(\d+)$")
    suffixes = []
    column_set = {str(column) for column in columns}
    for column in column_set:
        match = pattern.match(column)
        if not match:
            continue
        suffix = int(match.group(1))
        if (
            f"draw_b{bookmaker}_{suffix}" in column_set
            and f"away_b{bookmaker}_{suffix}" in column_set
        ):
            suffixes.append(suffix)
    return sorted(suffixes)


def _parse_score(value: str) -> tuple[int | None, int | None]:
    if ":" not in value:
        return None, None
    left, right = value.split(":", 1)
    return _safe_int(left.strip()), _safe_int(right.strip())


def _safe_prob_sum(values: Sequence[Any]) -> float:
    try:
        probs = [float(value) for value in values]
    except (TypeError, ValueError):
        return float("nan")
    if any(not np.isfinite(value) for value in probs):
        return float("nan")
    return float(sum(probs))


def _betfair_quality_reason(
    prob_sum: float,
    closing_odds: tuple[float, float, float] | None,
    duplicate_market: bool,
    parsed_names: tuple[str, str] | None,
) -> str:
    reasons = []
    if not np.isfinite(prob_sum) or not 0.95 <= prob_sum <= 1.05:
        reasons.append("bad_probability_sum")
    if closing_odds is None:
        reasons.append("bad_implied_odds")
    if duplicate_market:
        reasons.append("duplicate_market")
    if parsed_names is None:
        reasons.append("bad_match_name")
    if not reasons:
        reasons.append("needs_api_fixture_and_entry_odds")
    return ";".join(reasons)


def _write_betfair_quality_markdown(df: pd.DataFrame, output_path: Path) -> None:
    valid_rows = int(df["reason"].eq("needs_api_fixture_and_entry_odds").sum()) if not df.empty else 0
    reason_summary = (
        df.groupby("reason").size().reset_index(name="count").sort_values("count", ascending=False)
        if not df.empty
        else pd.DataFrame(columns=["reason", "count"])
    )
    content = [
        "# Betfair Quality Audit",
        "",
        "## Conclusion",
        "",
        (
            "Betfair remains isolated from default backtests. Rows need API fixture "
            "matching and safe opening odds before promotion."
        ),
        "",
        "## Gate",
        "",
        "- probability sum must be in [0.95, 1.05]",
        "- implied decimal odds must be in (1, 100]",
        "- duplicated `match_name + market_time` rows are blocked",
        "- parsed fixture names, API fixture, result, and opening odds are still required",
        "",
        "## Summary",
        "",
        f"- rows_audited: {len(df)}",
        f"- rows_passing_local_quality_only: {valid_rows}",
        "- rows_entering_backtest: 0",
        "",
        "## Reason Counts",
        "",
        reason_summary.to_markdown(index=False) if not reason_summary.empty else "None.",
        "",
        "## Files",
        "",
        "- reports/betfair_quality_audit.csv",
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")


def _odds_candidates_from_response(
    response: dict[str, Any] | None,
    fixture_id: int,
    kickoff: pd.Timestamp,
) -> list[dict[str, object]]:
    if not _api_response_ok(response):
        return []
    candidates = []
    for item in response.get("response", []):
        fixture = item.get("fixture", {})
        if _safe_int(fixture.get("id")) != int(fixture_id):
            continue
        update_time = pd.to_datetime(item.get("update"), utc=True, errors="coerce")
        if pd.isna(update_time) or update_time >= kickoff:
            continue
        for bookmaker in item.get("bookmakers", []):
            odds = _match_winner_odds(bookmaker)
            if odds is None:
                continue
            candidates.append(
                {
                    "B365H": odds[0],
                    "B365D": odds[1],
                    "B365A": odds[2],
                    "entry_bookmaker": str(bookmaker.get("name", "")),
                    "snapshot_time": update_time,
                    "bookmaker_rank": _bookmaker_rank(str(bookmaker.get("name", ""))),
                }
            )
    return candidates


def _match_winner_odds(bookmaker: dict[str, Any]) -> tuple[float, float, float] | None:
    for bet in bookmaker.get("bets", []):
        if str(bet.get("name", "")).strip().lower() != "match winner":
            continue
        values: dict[str, float] = {}
        for item in bet.get("values", []):
            label = str(item.get("value", "")).strip().lower()
            if label in {"home", "draw", "away"}:
                try:
                    values[label] = float(item.get("odd"))
                except (TypeError, ValueError):
                    return None
        odds = (values.get("home"), values.get("draw"), values.get("away"))
        if all(_valid_decimal_odds(value) for value in odds):
            return float(odds[0]), float(odds[1]), float(odds[2])
    return None


def _bookmaker_rank(name: str) -> int:
    normalized = _normalize_name(name)
    if normalized == "bet365":
        return 0
    if normalized == "pinnacle":
        return 1
    return 2


def _total_pages(response: dict[str, Any] | None) -> int:
    if not response:
        return 1
    paging = response.get("paging")
    if not isinstance(paging, dict):
        return 1
    return max(int(paging.get("total", 1) or 1), 1)


def _api_response_ok(response: dict[str, Any] | None) -> bool:
    if not response:
        return False
    errors = response.get("errors")
    if errors:
        return False
    return isinstance(response.get("response"), list)


def _api_error_detail(response: dict[str, Any] | None) -> str:
    if not response:
        return "empty_or_network_failure"
    errors = response.get("errors")
    if errors:
        return str(errors)
    return "missing_response_list"


def _local_result_conflicts(local: dict[str, Any], match: FixtureMatch) -> bool:
    if match.hg is None or match.ag is None:
        return False
    return int(local["FTHG"]) != match.hg or int(local["FTAG"]) != match.ag


def _betfair_probs_to_odds(values: Sequence[Any]) -> tuple[float, float, float] | None:
    try:
        probs = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if any(not np.isfinite(value) or value <= 0 for value in probs):
        return None
    prob_sum = sum(probs)
    if not 0.95 <= prob_sum <= 1.05:
        return None
    odds = tuple(round(1.0 / value, 4) for value in probs)
    if not all(_valid_decimal_odds(value) for value in odds):
        return None
    return odds


def _split_match_name(value: str) -> tuple[str, str] | None:
    if " v " not in value:
        return None
    home, away = value.split(" v ", 1)
    home = home.strip()
    away = away.strip()
    if not home or not away:
        return None
    return home, away


def _valid_decimal_odds(value: Any) -> bool:
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(odds) and 1.0 < odds <= MAX_REASONABLE_ODDS)


def _name_score(left: str, right: str) -> float:
    left_norm = _normalize_name(left)
    right_norm = _normalize_name(right)
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def _normalize_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    replacements = {
        "nott m forest": "nottingham forest",
        "nottm forest": "nottingham forest",
        "man utd": "manchester united",
        "man united": "manchester united",
        "man city": "manchester city",
        "spurs": "tottenham",
        "wolves": "wolverhampton",
    }
    return replacements.get(normalized, normalized)


def _epl_season_from_path(path: Path) -> int:
    digits = re.findall(r"\d+", path.stem)
    if not digits:
        return 0
    token = digits[-1]
    if len(token) == 4:
        return int(f"20{token[:2]}")
    return int(token)


def _epl_season_from_timestamp(value: Any) -> int:
    timestamp = _utc_timestamp(value)
    return int(timestamp.year if timestamp.month >= 7 else timestamp.year - 1)


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"Invalid timestamp: {value}")
    return timestamp


def _format_date(value: Any) -> str:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return ""
    return timestamp.strftime("%d/%m/%Y")


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ftr_from_goals(home_goals: int | None, away_goals: int | None) -> str | None:
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _diagnostic(
    diagnostics: list[dict[str, Any]],
    source_name: str,
    source_match_id: str,
    severity: str,
    status: str,
    reason: str,
    detail: str,
) -> None:
    diagnostics.append(
        {
            "source_name": source_name,
            "source_match_id": source_match_id,
            "severity": severity,
            "status": status,
            "reason": reason,
            "detail": detail,
        }
    )


def _api_connection_summary(
    client: FootballAPIClient | None,
    max_api_requests: int,
    cache_only: bool,
    fallback_stats: dict[str, object] | None = None,
) -> dict[str, object]:
    settings = get_settings()
    stats = client.stats.as_dict() if client is not None else (fallback_stats or {})
    return {
        "api_key_configured": bool(settings.api.api_football_key),
        "cache_only": cache_only,
        "max_api_requests": max_api_requests,
        "effective_max_api_requests": (
            getattr(client, "max_api_requests", None) if client is not None else max_api_requests
        ),
        "client_available": client is not None,
        **stats,
    }


def _write_audit_markdown(
    manifest: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_path: Path,
    api_summary: dict[str, object],
) -> None:
    candidate_text = (
        "Standardized datasets were generated as explicit research inputs only; "
        "the default strategy and default backtest inputs were not changed."
    )
    api_table = pd.DataFrame(
        [{"field": key, "value": value} for key, value in api_summary.items()]
    )
    content = [
        "# API Enrichment Audit",
        "",
        "## Conclusion",
        "",
        candidate_text,
        "",
        "## API-Football Connection",
        "",
        api_table.to_markdown(index=False),
        "",
        "## Manifest",
        "",
        manifest.to_markdown(index=False) if not manifest.empty else "None.",
        "",
        "## Diagnostics Summary",
        "",
        (
            diagnostics.groupby(["source_name", "reason"]).size().reset_index(name="count").to_markdown(
                index=False
            )
            if not diagnostics.empty
            else "None."
        ),
        "",
        "## Files",
        "",
        "- data_standardized/api_backtest/manifest.csv",
        "- data_standardized/api_backtest/diagnostics.csv",
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build API-enriched backtest CSV datasets.")
    parser.add_argument("--output-dir", default="data_standardized/api_backtest")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument(
        "--max-api-requests",
        type=int,
        default=250,
        help="Maximum live API-Football requests for this run; cache hits do not count.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Use only api_cache.db and do not make live API-Football requests.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    manifest_df = build_api_enriched_backtest_datasets(
        args.output_dir,
        args.reports_dir,
        max_api_requests=args.max_api_requests,
        cache_only=args.cache_only,
    )
    print(manifest_df.to_string(index=False))
