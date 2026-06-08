"""Generate a read-only Polymarket football betting research report.

The script combines Polymarket public market data, API-Football fixtures and
odds, and The Odds API prices. It writes local artifacts and can send a Feishu
Markdown document plus a concise chat card after all data and content gates pass.
It never places orders or submits signed transactions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api_football_today_daemon import APIFootballDaemonConfig, sync_api_football_today  # noqa: E402
from config import load_env_file  # noqa: E402
from polymarket_clob_connector import PolymarketClobConnector  # noqa: E402
from polymarket_gamma_connector import PolymarketGammaConnector  # noqa: E402

DEFAULT_CHAT_ENV = "FEISHU_POLYMARKET_FOOTBALL_CHAT_ID"
DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_MIN_EV = 1.05
DEFAULT_MAX_SPREAD = 0.08
DEFAULT_MATCH_THRESHOLD = 0.86
DEFAULT_TIE_MARGIN = 0.02
DEFAULT_OUTPUT_DIR = "reports"
DEFAULT_API_DB = "spots_quant.db"
DEFAULT_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SELECTIONS = ("H", "D", "A")
SELECTION_LABELS = {"H": "主胜", "D": "平局", "A": "客胜"}
FINISHED_OR_LIVE_STATUSES = {"1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE", "FT", "AET", "PEN"}
SOCCER_SEARCH_TERMS = ("soccer", "fifa", "uefa")
SOCCER_INCLUDE_TERMS = (
    "soccer",
    "fifa",
    "uefa",
    "premier league",
    "champions league",
    "europa league",
    "la liga",
    "serie a",
    "bundesliga",
    "ligue 1",
    "mls",
    "world cup",
    "club world cup",
    "nations league",
    "copa",
    "concacaf",
)
NON_SOCCER_TERMS = (
    "american football",
    "college football",
    "pro-football",
    "nfl",
    "super bowl",
    "draft",
    "drafted",
    "overall pick",
)


@dataclass(frozen=True)
class Fixture:
    """API-Football fixture row available before a betting decision."""

    fixture_id: int
    kickoff: datetime
    status_short: str
    home_team: str
    away_team: str
    api_probs: dict[str, float]


@dataclass(frozen=True)
class PolymarketSelection:
    """One Polymarket outcome quote for a 1X2 football selection."""

    selection: str
    token_id: str
    market_slug: str
    question: str
    bid: float
    ask: float
    mid: float
    spread: float
    liquidity: float | None = None


@dataclass(frozen=True)
class PolymarketMatchMarket:
    """Grouped Polymarket quotes for one football match."""

    title: str
    slug: str
    kickoff: datetime | None
    home_team: str
    away_team: str
    selections: dict[str, PolymarketSelection]


@dataclass(frozen=True)
class OddsAPIMatch:
    """The Odds API football H2H prices for one match."""

    commence_time: datetime
    home_team: str
    away_team: str
    bookmaker: str
    probs: dict[str, float]


@dataclass(frozen=True)
class MatchResult:
    """One matched Polymarket/API-Football football market with advice status."""

    match_name: str
    kickoff: str
    status: str
    selection: str
    fair_prob: float
    api_prob: float
    odds_api_prob: float
    polymarket_ask: float
    polymarket_mid: float
    spread: float
    ev: float
    max_price: float
    confidence: str
    stake_units: float
    market_slug: str
    token_id: str
    gate_status: str
    reasons: list[str]


@dataclass(frozen=True)
class ReportPayload:
    """Serializable daily report payload used for Markdown and Feishu card rendering."""

    target_date: str
    generated_at: str
    data_gate: str
    send_gate: str
    candidates: list[MatchResult]
    rejected: list[MatchResult]
    coverage: dict[str, int]
    diagnostics: list[str]
    artifacts: dict[str, str]
    feishu: dict[str, object]


def parse_target_date(value: str, timezone_name: str = DEFAULT_TIMEZONE) -> date:
    """Parse `today` or an ISO date using the configured report timezone."""
    if value.lower() == "today":
        return datetime.now(ZoneInfo(timezone_name)).date()
    return date.fromisoformat(value)


def normalize_name(value: str) -> str:
    """Normalize football team names for fuzzy matching across data vendors."""
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    replacements = {
        "man utd": "manchester united",
        "man united": "manchester united",
        "man city": "manchester city",
        "nottm forest": "nottingham forest",
        "nott m forest": "nottingham forest",
        "wolves": "wolverhampton",
        "spurs": "tottenham",
        "fc imabari": "imabari",
    }
    return replacements.get(normalized, normalized)


def name_score(left: str, right: str) -> float:
    """Return a deterministic fuzzy score between two team names."""
    left_norm = normalize_name(left)
    right_norm = normalize_name(right)
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def no_vig_probabilities(odds: Mapping[str, float]) -> dict[str, float]:
    """Convert three decimal odds into normalized no-vig probabilities."""
    implied = {}
    for selection in SELECTIONS:
        value = float(odds[selection])
        if not math.isfinite(value) or value <= 1.0 or value > 100.0:
            raise ValueError("decimal odds must be in (1, 100].")
        implied[selection] = 1.0 / value
    total = sum(implied.values())
    if total <= 0:
        raise ValueError("implied probability sum must be positive.")
    return {selection: implied[selection] / total for selection in SELECTIONS}


def evaluate_selection(
    selection: str,
    fair_prob: float,
    api_prob: float,
    odds_api_prob: float,
    quote: PolymarketSelection,
    min_ev: float = DEFAULT_MIN_EV,
    max_spread: float = DEFAULT_MAX_SPREAD,
) -> tuple[str, list[str], float, float, str, float]:
    """Evaluate one Polymarket YES share against external fair probability."""
    reasons: list[str] = []
    if quote.ask <= 0 or quote.ask >= 1:
        reasons.append("bad_polymarket_ask")
    if quote.spread > max_spread:
        reasons.append("spread_too_wide")
    if abs(api_prob - odds_api_prob) > 0.12:
        reasons.append("external_probability_disagreement")
    ev = fair_prob / quote.ask if quote.ask > 0 else 0.0
    max_price = fair_prob / min_ev if min_ev > 0 else fair_prob
    if ev < min_ev:
        reasons.append("ev_below_threshold")
    confidence = confidence_label(ev, quote.spread, abs(api_prob - odds_api_prob))
    stake = stake_units(ev, quote.spread, confidence) if not reasons else 0.0
    return ("candidate" if not reasons else "observe_only", reasons, ev, max_price, confidence, stake)


def confidence_label(ev: float, spread: float, disagreement: float) -> str:
    """Map edge, liquidity friction, and source disagreement to a confidence label."""
    if ev >= 1.15 and spread <= 0.04 and disagreement <= 0.05:
        return "high"
    if ev >= 1.08 and spread <= 0.06 and disagreement <= 0.08:
        return "medium"
    return "low"


def stake_units(ev: float, spread: float, confidence: str) -> float:
    """Return conservative paper stake units; this is not an order instruction."""
    if confidence == "high":
        return 0.75
    if confidence == "medium":
        return 0.50
    if ev >= 1.05 and spread <= DEFAULT_MAX_SPREAD:
        return 0.25
    return 0.0


def match_polymarket_to_fixture(
    market: PolymarketMatchMarket,
    fixtures: Sequence[Fixture],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    tie_margin: float = DEFAULT_TIE_MARGIN,
) -> tuple[Fixture | None, float, str]:
    """Match a Polymarket market to one API-Football fixture, failing on ambiguity."""
    scored = []
    for fixture in fixtures:
        if market.kickoff is not None:
            delta_hours = abs((market.kickoff - fixture.kickoff).total_seconds()) / 3600
            if delta_hours > 36:
                continue
        direct = (
            name_score(market.home_team, fixture.home_team)
            + name_score(market.away_team, fixture.away_team)
        ) / 2.0
        scored.append((direct, fixture))
    if not scored:
        return None, 0.0, "no_time_or_name_match"
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_fixture = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -1.0
    if best_score < threshold:
        return None, best_score, "low_match_score"
    if second_score >= best_score - tie_margin:
        return None, best_score, "ambiguous_match"
    return best_fixture, best_score, ""


def run_daily_report(
    target_date: date,
    output_dir: Path,
    chat_id: str | None,
    send: bool,
    dry_run: bool,
    max_api_requests: int,
    min_ev: float,
    max_spread: float,
) -> ReportPayload:
    """Build, optionally send, and return the daily football report payload."""
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics: list[str] = []
    fixtures = load_api_football_fixtures(target_date, max_api_requests, diagnostics)
    odds_api_matches = fetch_the_odds_api_matches(diagnostics, max_requests=max_api_requests)
    polymarket_markets = discover_polymarket_markets(target_date, diagnostics, fixtures)
    odds_lookup = build_odds_api_lookup(odds_api_matches)
    candidates: list[MatchResult] = []
    rejected: list[MatchResult] = []

    for market in polymarket_markets:
        match_result_rows = analyze_market(
            market,
            fixtures,
            odds_lookup,
            min_ev=min_ev,
            max_spread=max_spread,
        )
        for row in match_result_rows:
            if row.gate_status == "candidate":
                candidates.append(row)
            else:
                rejected.append(row)

    data_gate = "ok"
    if not fixtures:
        data_gate = "blocked:no_api_football_fixtures"
    elif not odds_api_matches:
        data_gate = "blocked:no_odds_api_prices"
    elif not polymarket_markets:
        data_gate = "blocked:no_polymarket_football_markets"
    elif not candidates and not rejected:
        data_gate = "blocked:no_matched_markets"

    payload = ReportPayload(
        target_date=target_date.isoformat(),
        generated_at=datetime.now(timezone.utc).isoformat(),
        data_gate=data_gate,
        send_gate="pending",
        candidates=sorted(candidates, key=lambda item: item.ev, reverse=True),
        rejected=rejected,
        coverage={
            "api_football_fixtures": len(fixtures),
            "odds_api_matches": len(odds_api_matches),
            "polymarket_markets": len(polymarket_markets),
            "matched_rows": len(candidates) + len(rejected),
            "candidates": len(candidates),
        },
        diagnostics=diagnostics,
        artifacts={},
        feishu={},
    )
    artifacts = write_artifacts(payload, output_dir)
    payload = replace_payload(payload, artifacts=artifacts)
    if send and not dry_run:
        payload = send_to_feishu(payload, Path(artifacts["report_md"]), Path(artifacts["card_md"]), chat_id)
    elif send and dry_run:
        payload = write_receipt(payload, "dry_run")
    else:
        payload = write_receipt(payload, "skipped")
    return payload


def analyze_market(
    market: PolymarketMatchMarket,
    fixtures: Sequence[Fixture],
    odds_lookup: Mapping[str, OddsAPIMatch],
    min_ev: float,
    max_spread: float,
) -> list[MatchResult]:
    """Analyze one Polymarket market against API-Football and The Odds API."""
    fixture, _, reason = match_polymarket_to_fixture(market, fixtures)
    if fixture is None:
        return [blocked_result(market, "H", reason)]
    if fixture.status_short.upper() in FINISHED_OR_LIVE_STATUSES or fixture.kickoff <= datetime.now(timezone.utc):
        return [blocked_result(market, "H", "fixture_in_play_or_closed", fixture)]
    odds_api = find_odds_api_match(fixture, odds_lookup)
    if odds_api is None:
        return [blocked_result(market, "H", "odds_api_match_missing", fixture)]
    rows = []
    for selection in SELECTIONS:
        quote = market.selections.get(selection)
        if quote is None:
            rows.append(blocked_result(market, selection, "polymarket_selection_missing", fixture))
            continue
        api_prob = fixture.api_probs.get(selection)
        odds_api_prob = odds_api.probs.get(selection)
        if api_prob is None or odds_api_prob is None:
            rows.append(blocked_result(market, selection, "external_probability_missing", fixture))
            continue
        fair_prob = (api_prob + odds_api_prob) / 2.0
        gate, reasons, ev, max_price, confidence, stake = evaluate_selection(
            selection,
            fair_prob,
            api_prob,
            odds_api_prob,
            quote,
            min_ev=min_ev,
            max_spread=max_spread,
        )
        rows.append(
            MatchResult(
                match_name=f"{fixture.home_team} vs {fixture.away_team}",
                kickoff=fixture.kickoff.isoformat(),
                status=fixture.status_short,
                selection=selection,
                fair_prob=round(fair_prob, 6),
                api_prob=round(api_prob, 6),
                odds_api_prob=round(odds_api_prob, 6),
                polymarket_ask=round(quote.ask, 6),
                polymarket_mid=round(quote.mid, 6),
                spread=round(quote.spread, 6),
                ev=round(ev, 6),
                max_price=round(max_price, 6),
                confidence=confidence,
                stake_units=stake,
                market_slug=quote.market_slug,
                token_id=quote.token_id,
                gate_status=gate,
                reasons=reasons,
            )
        )
    return rows


def blocked_result(
    market: PolymarketMatchMarket,
    selection: str,
    reason: str,
    fixture: Fixture | None = None,
) -> MatchResult:
    """Build a blocked analysis row with explicit reason evidence."""
    match_name = (
        f"{fixture.home_team} vs {fixture.away_team}"
        if fixture is not None
        else f"{market.home_team} vs {market.away_team}"
    )
    kickoff = fixture.kickoff.isoformat() if fixture is not None else ""
    status = fixture.status_short if fixture is not None else ""
    return MatchResult(
        match_name=match_name,
        kickoff=kickoff,
        status=status,
        selection=selection,
        fair_prob=0.0,
        api_prob=0.0,
        odds_api_prob=0.0,
        polymarket_ask=0.0,
        polymarket_mid=0.0,
        spread=0.0,
        ev=0.0,
        max_price=0.0,
        confidence="blocked",
        stake_units=0.0,
        market_slug=market.slug,
        token_id="",
        gate_status="blocked",
        reasons=[reason],
    )


def load_api_football_fixtures(
    target_date: date,
    max_api_requests: int,
    diagnostics: list[str],
    db_path: str = DEFAULT_API_DB,
) -> list[Fixture]:
    """Load API-Football fixtures and 1X2 odds from the local daemon database."""
    if not Path(db_path).exists():
        diagnostics.append("api_football_db_missing")
        sync_api_football_today(
            APIFootballDaemonConfig(max_api_requests_per_run=max_api_requests),
            target_date=target_date,
        )
    rows = read_api_football_rows(db_path, target_date)
    if not rows:
        diagnostics.append("api_football_rows_missing;running_sync_once")
        sync_api_football_today(
            APIFootballDaemonConfig(max_api_requests_per_run=max_api_requests),
            target_date=target_date,
        )
        rows = read_api_football_rows(db_path, target_date)
    return rows


def read_api_football_rows(db_path: str, target_date: date) -> list[Fixture]:
    """Read fixture rows plus latest complete 1X2 API-Football odds from SQLite."""
    query = """
    SELECT
        f.fixture_id,
        f.kickoff,
        f.status_short,
        f.home_team_name,
        f.away_team_name,
        o.home_odds,
        o.draw_odds,
        o.away_odds
    FROM api_football_fixtures f
    JOIN (
        SELECT fixture_id, home_odds, draw_odds, away_odds, MAX(update_time) AS update_time
        FROM api_football_odds_1x2
        WHERE target_date = ?
        GROUP BY fixture_id
    ) o ON o.fixture_id = f.fixture_id
    WHERE f.target_date = ?
    """
    if not Path(db_path).exists():
        return []
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(query, (target_date.isoformat(), target_date.isoformat())).fetchall()
    except sqlite3.Error:
        return []
    fixtures = []
    for row in rows:
        kickoff = parse_datetime(row[1])
        if kickoff is None:
            continue
        try:
            probs = no_vig_probabilities({"H": row[5], "D": row[6], "A": row[7]})
        except (TypeError, ValueError):
            continue
        fixtures.append(
            Fixture(
                fixture_id=int(row[0]),
                kickoff=kickoff,
                status_short=str(row[2] or ""),
                home_team=str(row[3] or ""),
                away_team=str(row[4] or ""),
                api_probs=probs,
            )
        )
    return fixtures


def fetch_the_odds_api_matches(
    diagnostics: list[str],
    max_requests: int = 25,
) -> list[OddsAPIMatch]:
    """Fetch soccer H2H odds from The Odds API without exposing the API key."""
    api_key = os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        diagnostics.append("the_odds_api_key_missing")
        return []
    if max_requests <= 0:
        diagnostics.append("the_odds_api_budget_exhausted")
        return []
    sports_url = f"{DEFAULT_ODDS_API_BASE}/sports/?{urlencode({'apiKey': api_key})}"
    sports_payload = fetch_the_odds_api_json(sports_url, "sports", diagnostics)
    if not isinstance(sports_payload, list):
        return []
    sport_keys = sorted(
        {
            str(item.get("key") or "")
            for item in sports_payload
            if isinstance(item, Mapping)
            and str(item.get("key") or "").startswith("soccer_")
            and item.get("active", True)
        }
    )
    if not sport_keys:
        diagnostics.append("the_odds_api_no_soccer_sports")
        return []

    rows: list[OddsAPIMatch] = []
    requests_used = 1
    for sport_key in sport_keys:
        if requests_used >= max_requests:
            diagnostics.append(f"the_odds_api_budget_exhausted:{requests_used}")
            break
        rows.extend(fetch_the_odds_api_sport_matches(api_key, sport_key, diagnostics))
        requests_used += 1
    diagnostics.append(f"the_odds_api_requests_used:{requests_used}")
    if not rows:
        diagnostics.append("the_odds_api_empty_h2h_payload")
    return rows


def fetch_the_odds_api_sport_matches(
    api_key: str,
    sport_key: str,
    diagnostics: list[str],
) -> list[OddsAPIMatch]:
    """Fetch one The Odds API soccer sport key and parse complete H2H rows."""
    params = {
        "apiKey": api_key,
        "regions": "uk,eu,us",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "bookmakers": "pinnacle,bet365",
    }
    url = f"{DEFAULT_ODDS_API_BASE}/sports/{sport_key}/odds/?{urlencode(params)}"
    payload = fetch_the_odds_api_json(url, sport_key, diagnostics)
    if not isinstance(payload, list):
        return []
    return parse_the_odds_api_payload(payload)


def fetch_the_odds_api_json(
    url: str,
    label: str,
    diagnostics: list[str],
) -> object | None:
    """Fetch one The Odds API JSON payload with redacted diagnostics."""
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        diagnostics.append(f"the_odds_api_http_error:{label}:{exc.code}")
        return []
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        diagnostics.append(f"the_odds_api_fetch_failed:{label}:{type(exc).__name__}")
        return []


def parse_the_odds_api_payload(payload: Sequence[Mapping[str, Any]]) -> list[OddsAPIMatch]:
    """Parse The Odds API H2H response into no-vig 1X2 probability rows."""
    rows = []
    for item in payload:
        home = str(item.get("home_team") or "")
        away = str(item.get("away_team") or "")
        commence = parse_datetime(item.get("commence_time"))
        if not home or not away or commence is None:
            continue
        parsed = parse_bookmaker_probs(item, home, away)
        if parsed is None:
            continue
        bookmaker, probs = parsed
        rows.append(
            OddsAPIMatch(
                commence_time=commence,
                home_team=home,
                away_team=away,
                bookmaker=bookmaker,
                probs=probs,
            )
        )
    return rows


def parse_bookmaker_probs(
    item: Mapping[str, Any],
    home: str,
    away: str,
) -> tuple[str, dict[str, float]] | None:
    """Parse one preferred bookmaker's football H2H odds into fair probabilities."""
    bookmakers = item.get("bookmakers")
    if not isinstance(bookmakers, list):
        return None
    for preferred in ("pinnacle", "bet365"):
        for bookmaker in bookmakers:
            if str(bookmaker.get("key") or "").lower() != preferred:
                continue
            markets = bookmaker.get("markets")
            if not isinstance(markets, list):
                continue
            for market in markets:
                if str(market.get("key") or "") != "h2h":
                    continue
                odds = outcomes_to_1x2(market.get("outcomes"), home, away)
                if odds is not None:
                    return str(bookmaker.get("key") or preferred), no_vig_probabilities(odds)
    return None


def outcomes_to_1x2(outcomes: object, home: str, away: str) -> dict[str, float] | None:
    """Map The Odds API outcomes into H/D/A decimal odds."""
    if not isinstance(outcomes, list):
        return None
    odds: dict[str, float] = {}
    for outcome in outcomes:
        name = normalize_name(str(outcome.get("name") or ""))
        price = outcome.get("price")
        try:
            value = float(price)
        except (TypeError, ValueError):
            continue
        if name == normalize_name(home):
            odds["H"] = value
        elif name == normalize_name(away):
            odds["A"] = value
        elif name == "draw":
            odds["D"] = value
    return odds if all(selection in odds for selection in SELECTIONS) else None


def build_odds_api_lookup(matches: Sequence[OddsAPIMatch]) -> dict[str, OddsAPIMatch]:
    """Build a normalized lookup for fast The Odds API fixture matching."""
    lookup = {}
    for item in matches:
        key = match_key(item.home_team, item.away_team, item.commence_time.date())
        lookup[key] = item
    return lookup


def find_odds_api_match(
    fixture: Fixture,
    lookup: Mapping[str, OddsAPIMatch],
) -> OddsAPIMatch | None:
    """Return the best The Odds API row for one API-Football fixture."""
    exact = lookup.get(match_key(fixture.home_team, fixture.away_team, fixture.kickoff.date()))
    if exact is not None:
        return exact
    candidates = []
    for item in lookup.values():
        if abs((fixture.kickoff - item.commence_time).total_seconds()) / 3600 > 36:
            continue
        score = (
            name_score(fixture.home_team, item.home_team)
            + name_score(fixture.away_team, item.away_team)
        ) / 2.0
        candidates.append((score, item))
    if not candidates:
        return None
    candidates.sort(key=lambda row: row[0], reverse=True)
    return candidates[0][1] if candidates[0][0] >= DEFAULT_MATCH_THRESHOLD else None


def match_key(home: str, away: str, match_date: date) -> str:
    """Return a stable normalized match key."""
    return f"{match_date.isoformat()}|{normalize_name(home)}|{normalize_name(away)}"


def discover_polymarket_markets(
    target_date: date,
    diagnostics: list[str],
    fixtures: Sequence[Fixture] | None = None,
) -> list[PolymarketMatchMarket]:
    """Discover today's public Polymarket football markets and read CLOB quotes."""
    try:
        gamma = PolymarketGammaConnector.from_env()
        clob = PolymarketClobConnector.from_env()
    except Exception as exc:
        diagnostics.append(f"polymarket_connector_unavailable:{type(exc).__name__}")
        return []
    raw_items = collect_gamma_football_items(gamma, diagnostics)
    if fixtures:
        raw_items.extend(search_gamma_fixture_items(gamma, fixtures, diagnostics))
    markets = []
    seen: set[str] = set()
    skipped_missing_kickoff = 0
    skipped_other_date = 0
    skipped_not_tradable = 0
    for item in raw_items:
        kickoff = gamma_item_kickoff(item)
        if kickoff is None:
            skipped_missing_kickoff += 1
            continue
        if gamma_kickoff_local_date(kickoff) != target_date:
            skipped_other_date += 1
            continue
        if not gamma_item_can_have_order_book(item):
            skipped_not_tradable += 1
            continue
        market = gamma_item_to_match_market(item, clob, diagnostics)
        if market is None:
            continue
        if market.slug in seen:
            continue
        seen.add(market.slug)
        markets.append(market)
    if skipped_missing_kickoff:
        diagnostics.append(f"polymarket_skipped_missing_kickoff:{skipped_missing_kickoff}")
    if skipped_other_date:
        diagnostics.append(f"polymarket_skipped_other_date:{skipped_other_date}")
    if skipped_not_tradable:
        diagnostics.append(f"polymarket_skipped_not_tradable:{skipped_not_tradable}")
    return markets


def gamma_kickoff_local_date(
    kickoff: datetime,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> date:
    """Convert a Gamma kickoff timestamp into the report timezone's calendar date."""
    return kickoff.astimezone(ZoneInfo(timezone_name)).date()


def search_gamma_fixture_items(
    gamma: PolymarketGammaConnector,
    fixtures: Sequence[Fixture],
    diagnostics: list[str],
) -> list[Mapping[str, Any]]:
    """Search Gamma by same-day fixture team names when broad discovery misses match markets."""
    items: list[Mapping[str, Any]] = []
    seen_queries: set[str] = set()
    query_count = 0
    for fixture in fixtures:
        query = build_fixture_search_query(fixture.home_team, fixture.away_team)
        if not query or query in seen_queries:
            continue
        seen_queries.add(query)
        try:
            payload = gamma.search(query, limit_per_type=10)
        except Exception as exc:
            diagnostics.append(f"polymarket_gamma_fixture_search_failed:{query}:{type(exc).__name__}")
            continue
        query_count += 1
        items.extend(extract_gamma_items(payload))
    if query_count:
        diagnostics.append(f"polymarket_gamma_fixture_search_queries:{query_count}")
    return items


def build_fixture_search_query(home_team: str, away_team: str) -> str:
    """Build a deterministic Gamma search query from a fixture's team names."""
    home = normalize_name(home_team).strip()
    away = normalize_name(away_team).strip()
    return f"{home} {away}".strip()


def collect_gamma_football_items(
    gamma: PolymarketGammaConnector,
    diagnostics: list[str],
) -> list[Mapping[str, Any]]:
    """Collect candidate football market/event payloads from Gamma API."""
    items: list[Mapping[str, Any]] = []
    for term in SOCCER_SEARCH_TERMS:
        try:
            search_payload = gamma.search(term, limit_per_type=50)
            items.extend(extract_gamma_items(search_payload))
        except Exception as exc:
            diagnostics.append(f"polymarket_gamma_search_failed:{term}:{type(exc).__name__}")
    try:
        list_payload = gamma.list_markets(limit=500, active=True, closed=False)
        items.extend(extract_gamma_items(list_payload))
    except Exception as exc:
        diagnostics.append(f"polymarket_gamma_markets_failed:{type(exc).__name__}")
    return [item for item in items if looks_like_football_item(item)]


def extract_gamma_items(payload: object) -> list[Mapping[str, Any]]:
    """Extract event or market dicts from common Gamma response shapes."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    rows: list[Mapping[str, Any]] = []
    for key in ("markets", "events", "data", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, Mapping))
    if not rows and any(key in payload for key in ("question", "title", "slug")):
        rows.append(payload)
    return rows


def looks_like_football_item(item: Mapping[str, Any]) -> bool:
    """Return True for public Gamma rows that appear to be football/soccer markets."""
    text = " ".join(
        str(item.get(key) or "") for key in ("title", "question", "slug", "description")
    ).lower()
    if not text or any(term in text for term in NON_SOCCER_TERMS):
        return False
    if any(term in text for term in SOCCER_INCLUDE_TERMS):
        return True
    has_match_title = re.search(r"\b[a-z0-9 .'-]+\s+(?:vs\.?|v)\s+[a-z0-9 .'-]+\b", text)
    return bool(has_match_title and item_has_draw_outcome(item))


def item_has_draw_outcome(item: Mapping[str, Any]) -> bool:
    """Return True when a Gamma row exposes a draw outcome, implying a 1X2 market."""
    markets = item.get("markets") if isinstance(item.get("markets"), list) else [item]
    for market in markets:
        if not isinstance(market, Mapping):
            continue
        outcomes = market.get("outcomes")
        if not isinstance(outcomes, list):
            continue
        for outcome in outcomes:
            if isinstance(outcome, Mapping):
                name = str(outcome.get("name") or outcome.get("title") or "")
            else:
                name = str(outcome)
            if normalize_name(name) in {"draw", "tie"}:
                return True
    return False


def gamma_item_to_match_market(
    item: Mapping[str, Any],
    clob: PolymarketClobConnector,
    diagnostics: list[str],
) -> PolymarketMatchMarket | None:
    """Convert a Gamma event or market row into grouped 1X2 Polymarket quotes."""
    if not gamma_item_can_have_order_book(item):
        return None
    title = str(item.get("title") or item.get("question") or item.get("slug") or "")
    slug = str(item.get("slug") or stable_hash(title))
    kickoff = gamma_item_kickoff(item)
    home, away = infer_teams(title)
    markets = item.get("markets") if isinstance(item.get("markets"), list) else [item]
    selections: dict[str, PolymarketSelection] = {}
    for market in markets:
        if not isinstance(market, Mapping):
            continue
        if not gamma_row_can_have_order_book(market):
            continue
        market_slug = str(market.get("slug") or slug)
        question = str(market.get("question") or market.get("title") or title)
        outcome_names = parse_jsonish_list(market.get("outcomes"))
        token_ids = parse_jsonish_list(market.get("clobTokenIds") or market.get("tokenIds"))
        if not token_ids:
            token_ids = token_ids_from_tokens(market.get("tokens"))
        classified = classify_market_outcomes(question, outcome_names, token_ids, home, away)
        for selection, token_id in classified.items():
            if selection in selections:
                continue
            quote = read_clob_quote(clob, token_id, market_slug, question, selection, diagnostics)
            if quote is None:
                continue
            selections[selection] = quote
    if not home or not away or set(selections) != set(SELECTIONS):
        return None
    return PolymarketMatchMarket(title, slug, kickoff, home, away, selections)


def gamma_item_can_have_order_book(item: Mapping[str, Any]) -> bool:
    """Return whether a Gamma event or market can still expose a CLOB order book."""
    if not gamma_row_can_have_order_book(item):
        return False
    markets = item.get("markets") if isinstance(item.get("markets"), list) else [item]
    rows = [market for market in markets if isinstance(market, Mapping)]
    if not rows:
        return True
    return any(gamma_row_can_have_order_book(market) for market in rows)


def gamma_row_can_have_order_book(row: Mapping[str, Any]) -> bool:
    """Fail closed only on explicit Gamma flags that mean no live order book."""
    if gamma_bool(row.get("active")) is False:
        return False
    if gamma_bool(row.get("closed")) is True:
        return False
    if gamma_bool(row.get("archived")) is True:
        return False
    if gamma_bool(row.get("acceptingOrders")) is False:
        return False
    if gamma_bool(row.get("ready")) is False:
        return False
    return True


def gamma_bool(value: object) -> bool | None:
    """Parse Gamma booleans while treating missing fields as unknown."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def gamma_item_kickoff(item: Mapping[str, Any]) -> datetime | None:
    """Extract a Polymarket event kickoff timestamp when Gamma exposes one."""
    return parse_datetime(
        item.get("gameStartTime")
        or item.get("eventStartTime")
        or item.get("endDate")
        or item.get("startDate")
        or item.get("start_date")
    )


def classify_market_outcomes(
    question: str,
    outcomes: Sequence[str],
    token_ids: Sequence[str],
    home: str,
    away: str,
) -> dict[str, str]:
    """Classify Gamma outcomes or binary YES markets into H/D/A token IDs."""
    mapping: dict[str, str] = {}
    if len(outcomes) >= 3 and len(token_ids) >= 3:
        for outcome, token_id in zip(outcomes, token_ids, strict=False):
            selection = classify_selection_text(outcome, home, away)
            if selection:
                mapping[selection] = token_id
    elif len(token_ids) >= 1:
        selection = classify_selection_text(question, home, away)
        if selection:
            yes_token = yes_token_id(outcomes, token_ids)
            if yes_token:
                mapping[selection] = yes_token
    return mapping


def classify_selection_text(text: str, home: str, away: str) -> str | None:
    """Classify a text fragment as home, draw, or away selection."""
    cleaned = strip_event_prefix(text)
    normalized = normalize_name(cleaned)
    if "draw" in normalized or "tie" in normalized:
        return "D"

    explicit_team = explicit_binary_market_team(cleaned)
    if explicit_team:
        explicit = classify_team_name(explicit_team, home, away)
        if explicit:
            return explicit

    home_norm = normalize_name(home)
    away_norm = normalize_name(away)
    if home_norm and normalized == home_norm:
        return "H"
    if away_norm and normalized == away_norm:
        return "A"
    if normalized in {"home", "home win"}:
        return "H"
    if normalized in {"away", "away win"}:
        return "A"
    fallback = classify_team_name(cleaned, home, away)
    if fallback:
        return fallback
    return None


def read_clob_quote(
    clob: PolymarketClobConnector,
    token_id: str,
    market_slug: str,
    question: str,
    selection: str,
    diagnostics: list[str] | None = None,
) -> PolymarketSelection | None:
    """Read one token's CLOB bid/ask/mid quote from the public order book."""
    try:
        book = clob.get_order_book(token_id)
    except Exception as exc:
        append_clob_diagnostic(diagnostics, "clob_book_unavailable", market_slug, selection, exc)
        return None
    if not isinstance(book, Mapping):
        append_clob_diagnostic(diagnostics, "clob_book_invalid", market_slug, selection)
        return None
    bid = best_price(book.get("bids"), is_bid=True)
    ask = best_price(book.get("asks"), is_bid=False)
    if bid is None:
        append_clob_diagnostic(diagnostics, "clob_bid_missing", market_slug, selection)
        return None
    if ask is None:
        append_clob_diagnostic(diagnostics, "clob_ask_missing", market_slug, selection)
        return None
    mid = (bid + ask) / 2.0
    spread = max(ask - bid, 0.0)
    liquidity = parse_optional_float(book.get("liquidity"))
    return PolymarketSelection(selection, token_id, market_slug, question, bid, ask, mid, spread, liquidity)


def best_price(levels: object, is_bid: bool) -> float | None:
    """Return best bid or ask from CLOB book levels."""
    if not isinstance(levels, list) or not levels:
        return None
    prices = []
    for level in levels:
        if not isinstance(level, Mapping):
            continue
        price = parse_optional_float(level.get("price"))
        if price is not None:
            prices.append(price)
    if not prices:
        return None
    return max(prices) if is_bid else min(prices)


def infer_teams(text: str) -> tuple[str, str]:
    """Infer home and away names from common football market titles."""
    cleaned = strip_event_prefix(re.sub(r"\s+", " ", text).strip())
    patterns = (r"(.+?)\s+vs\.?\s+(.+)", r"(.+?)\s+v\s+(.+)")
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            left = strip_market_suffix(match.group(1))
            right = strip_market_suffix(match.group(2))
            return left, right
    return "", ""


def strip_market_suffix(value: str) -> str:
    """Remove common market wording from a team-name fragment."""
    value = re.sub(r"\([^)]*\)", "", value)
    value = re.sub(
        r"\bto\s+(?:advance|lift(?:\s+the)?\s+trophy)\b",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\bon\s+\d{4}-\d{2}-\d{2}\b", "", value, flags=re.IGNORECASE)
    value = re.sub(
        r"\bon\s+[a-z]{3,9}\s+\d{1,2}(?:,\s*\d{4})?\b",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", value)
    value = re.sub(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", "", value)
    value = re.sub(r"\b(match|winner|win|draw|tie|yes|no)\b", "", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip(" -?:")


def strip_event_prefix(value: str) -> str:
    """Remove competition labels before a match title without touching team names."""
    cleaned = re.sub(r"\s+", " ", value).strip()
    if ":" in cleaned:
        return cleaned.split(":", 1)[1].strip()
    return cleaned


def yes_token_id(outcomes: Sequence[str], token_ids: Sequence[str]) -> str:
    """Return the YES token for binary markets, falling back only for legacy payloads."""
    if not token_ids:
        return ""
    if not outcomes:
        return token_ids[0]
    for outcome, token_id in zip(outcomes, token_ids, strict=False):
        if normalize_name(outcome) == "yes":
            return token_id
    return ""


def explicit_binary_market_team(text: str) -> str:
    """Extract the team being backed in common binary football market questions."""
    cleaned = strip_event_prefix(text).strip(" ?")
    patterns = (
        r"^will\s+(.+?)\s+beat\s+.+$",
        r"^will\s+(.+?)\s+win(?:\s+on\s+\d{4}-\d{2}-\d{2})?.*$",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return strip_market_suffix(match.group(1))
    return ""


def classify_team_name(value: str, home: str, away: str) -> str | None:
    """Classify a team fragment against home/away names using fuzzy matching."""
    candidate = strip_market_suffix(value)
    if not candidate:
        return None
    scores = []
    if home:
        scores.append(("H", name_score(candidate, home)))
    if away:
        scores.append(("A", name_score(candidate, away)))
    if not scores:
        return None
    scores.sort(key=lambda item: item[1], reverse=True)
    best_selection, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else -1.0
    if best_score >= 0.86 and best_score - second_score >= 0.05:
        return best_selection
    return None


def append_clob_diagnostic(
    diagnostics: list[str] | None,
    reason: str,
    market_slug: str,
    selection: str,
    exc: Exception | None = None,
) -> None:
    """Append a structured CLOB quote diagnostic when the caller wants evidence."""
    if diagnostics is None:
        return
    suffix = ""
    if exc is not None:
        status = re.search(r"HTTP\s+(\d{3})", str(exc))
        status_suffix = f":http_{status.group(1)}" if status else ""
        suffix = f":{type(exc).__name__}{status_suffix}"
    diagnostics.append(f"{reason}:{market_slug}:{selection}{suffix}")


def parse_jsonish_list(value: object) -> list[str]:
    """Parse Gamma fields that may be JSON strings or lists."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        value = parsed
    if isinstance(value, list):
        rows = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, Mapping):
                label = item.get("name") or item.get("value") or item.get("title")
                if label is not None:
                    rows.append(str(label))
                continue
            rows.append(str(item))
        return rows
    return []


def token_ids_from_tokens(value: object) -> list[str]:
    """Extract CLOB token IDs from common Gamma token payload shapes."""
    if not isinstance(value, list):
        return []
    rows = []
    for item in value:
        if isinstance(item, Mapping):
            token = item.get("token_id") or item.get("tokenId") or item.get("id")
            if token:
                rows.append(str(token))
    return rows


def render_report(payload: ReportPayload) -> str:
    """Render the full Markdown report."""
    candidate_rows = "\n".join(result_row(item) for item in payload.candidates) or "无。"
    rejected_rows = "\n".join(result_row(item) for item in payload.rejected[:30]) or "无。"
    diagnostics = "\n".join(f"- {item}" for item in payload.diagnostics) or "- 无。"
    return "\n".join(
        [
            f"# Polymarket 足球日报 {payload.target_date}",
            "",
            "## 执行摘要",
            "",
            f"- data_gate: `{payload.data_gate}`",
            f"- send_gate: `{payload.send_gate}`",
            f"- candidates: `{len(payload.candidates)}`",
            f"- generated_at_utc: `{payload.generated_at}`",
            "",
            "## 下注建议",
            "",
            candidate_rows,
            "",
            "## 拒绝清单",
            "",
            rejected_rows,
            "",
            "## 数据覆盖",
            "",
            "\n".join(f"- {key}: `{value}`" for key, value in payload.coverage.items()),
            "",
            "## 方法与风控",
            "",
            "- 只读 Polymarket Gamma/CLOB，不执行真实下单。",
            "- 只分析未开赛、完整三路 1X2、盘口和外部赔率都可读的市场。",
            "- 候选必须通过 EV、价差、外部概率分歧和流动性门禁。",
            "- stake_units 是纸面研究单位，不是交易指令。",
            "",
            "## 数据缺口",
            "",
            diagnostics,
            "",
            "## 飞书回读",
            "",
            f"- status: `{payload.feishu.get('status', 'pending')}`",
            f"- file_token: `{payload.feishu.get('file_token', '')}`",
            f"- doc_url: `{payload.feishu.get('doc_url', '')}`",
            f"- message_id: `{payload.feishu.get('message_id', '')}`",
            f"- document_readback: `{payload.feishu.get('document_readback', '')}`",
            f"- message_readback: `{payload.feishu.get('message_readback', '')}`",
            "",
        ]
    )


def render_card(payload: ReportPayload) -> str:
    """Render a concise Markdown card for Feishu group messages."""
    top = payload.candidates[:3]
    if top:
        summary_parts = [
            f"候选: {item.match_name} {SELECTION_LABELS[item.selection]}, EV {item.ev:.3f}, "
            f"ask {item.polymarket_ask:.3f}, stake {item.stake_units:.2f}u"
            for item in top
        ]
    else:
        summary_parts = ["结论: 今日无通过门禁的下注候选。"]
    if payload.rejected:
        first_rejected = payload.rejected[0]
        summary_parts.append(
            f"首个拦截: {first_rejected.match_name} {SELECTION_LABELS.get(first_rejected.selection, first_rejected.selection)}"
            f" -> {','.join(first_rejected.reasons) or 'blocked'}"
        )
    summary_parts.append(
        f"覆盖: polymarket_markets={payload.coverage.get('polymarket_markets', 0)}, matched_rows={payload.coverage.get('matched_rows', 0)}"
    )
    doc_url = str(payload.feishu.get("doc_url", ""))
    if doc_url:
        summary_parts.append(f"文档: {doc_url}")
    return " | ".join(
        [
            f"Polymarket 足球日报 {payload.target_date}",
            f"data_gate={payload.data_gate}",
            f"candidates={len(payload.candidates)}",
            *summary_parts,
        ]
    ).strip()


def result_row(item: MatchResult) -> str:
    """Render one candidate or rejected row as Markdown."""
    reasons = ",".join(item.reasons) if item.reasons else "pass"
    return (
        f"- `{item.gate_status}` {item.match_name} {SELECTION_LABELS.get(item.selection, item.selection)} "
        f"| fair={item.fair_prob:.3f} api={item.api_prob:.3f} odds_api={item.odds_api_prob:.3f} "
        f"| ask={item.polymarket_ask:.3f} mid={item.polymarket_mid:.3f} "
        f"| EV={item.ev:.3f} spread={item.spread:.3f} max_price={item.max_price:.3f} "
        f"| confidence={item.confidence} stake={item.stake_units:.2f}u | reason={reasons}"
    )


def write_artifacts(payload: ReportPayload, output_dir: Path) -> dict[str, str]:
    """Write Markdown, card, and JSON artifacts for a report payload."""
    stem = f"polymarket_football_daily_{payload.target_date.replace('-', '')}"
    report_path = output_dir / f"{stem}.md"
    card_path = output_dir / f"{stem}_card.md"
    data_path = output_dir / f"{stem}_data.json"
    receipt_path = output_dir / f"{stem}_send_receipt.json"
    artifacts = {
        "report_md": str(report_path),
        "card_md": str(card_path),
        "data_json": str(data_path),
        "send_receipt_json": str(receipt_path),
    }
    payload = replace_payload(payload, artifacts=artifacts)
    report_path.write_text(render_report(payload), encoding="utf-8")
    card_path.write_text(render_card(payload), encoding="utf-8")
    write_json(data_path, payload_to_dict(payload))
    return artifacts


def send_to_feishu(
    payload: ReportPayload,
    report_path: Path,
    card_path: Path,
    chat_id: str | None,
) -> ReportPayload:
    """Create or overwrite a Feishu Markdown doc, send a card, and verify readback."""
    if payload.data_gate != "ok":
        return write_receipt(payload, "blocked:data_gate")
    active_chat_id = chat_id or os.environ.get(DEFAULT_CHAT_ENV)
    if not active_chat_id:
        return write_receipt(payload, "blocked:missing_chat_id")
    if not content_gate(report_path, card_path):
        return write_receipt(payload, "blocked:content_gate")
    state = load_lark_state()
    state_key = payload.target_date
    file_token = str(state.get(state_key, {}).get("file_token", ""))
    doc_result = (
        lark_markdown_overwrite(report_path, file_token)
        if file_token
        else lark_markdown_create(report_path)
    )
    file_token = extract_first(doc_result, ("file_token", "token", "obj_token"))
    doc_url = extract_first(doc_result, ("url", "doc_url", "preview_url", "web_url"))
    if not file_token and doc_url:
        file_token = extract_token_from_url(doc_url)
    if not file_token:
        return write_receipt(payload, "blocked:lark_doc_create_failed", {"doc_result": doc_result})
    state[state_key] = {"file_token": file_token, "doc_url": doc_url}
    save_lark_state(state)
    payload = replace_payload(payload, feishu={"file_token": file_token, "doc_url": doc_url})
    card_path.write_text(render_card(payload), encoding="utf-8")
    message_result = lark_send_message(active_chat_id, card_path.read_text(encoding="utf-8"), payload.target_date)
    message_id = extract_first(message_result, ("message_id", "id"))
    if not message_id:
        return write_receipt(payload, "blocked:lark_message_send_failed", {"message_result": message_result})
    doc_readback = lark_markdown_fetch(file_token)
    msg_readback = lark_message_mget(message_id)
    feishu = {
        "status": "sent",
        "file_token": file_token,
        "doc_url": doc_url,
        "message_id": message_id,
        "document_readback": "ok" if readback_contains(doc_readback, payload.target_date) else "failed",
        "message_readback": "ok" if readback_contains(msg_readback, message_id) else "failed",
        "doc_result": doc_result,
        "message_result": message_result,
    }
    send_gate = (
        "sent"
        if feishu["document_readback"] == "ok" and feishu["message_readback"] == "ok"
        else "blocked:readback_failed"
    )
    return write_receipt(replace_payload(payload, send_gate=send_gate, feishu=feishu), send_gate)


def content_gate(report_path: Path, card_path: Path) -> bool:
    """Validate required report/card sections before Feishu sending."""
    report = report_path.read_text(encoding="utf-8")
    card = card_path.read_text(encoding="utf-8")
    required = ("## 执行摘要", "## 下注建议", "## 数据覆盖", "## 方法与风控")
    return bool(card.strip()) and all(section in report for section in required)


def write_receipt(
    payload: ReportPayload,
    status: str,
    extra: Mapping[str, object] | None = None,
) -> ReportPayload:
    """Persist send receipt and update report JSON with final send status."""
    feishu = dict(payload.feishu)
    feishu.setdefault("status", status)
    if extra:
        feishu.update(extra)
    payload = replace_payload(payload, send_gate=status, feishu=feishu)
    artifacts = payload.artifacts
    if "send_receipt_json" in artifacts:
        write_json(Path(artifacts["send_receipt_json"]), {"send_gate": status, "feishu": feishu})
    if "data_json" in artifacts:
        write_json(Path(artifacts["data_json"]), payload_to_dict(payload))
    if "report_md" in artifacts:
        Path(artifacts["report_md"]).write_text(render_report(payload), encoding="utf-8")
    return payload


def lark_markdown_create(report_path: Path) -> dict[str, object]:
    """Create a Feishu Markdown file through lark-cli."""
    return run_lark_json(
        [
            lark_cli_executable(),
            "markdown",
            "+create",
            "--as",
            "bot",
            "--file",
            str(report_path),
            "--name",
            report_path.name,
        ]
    )


def lark_markdown_overwrite(report_path: Path, file_token: str) -> dict[str, object]:
    """Overwrite an existing Feishu Markdown file through lark-cli."""
    return run_lark_json(
        [
            lark_cli_executable(),
            "markdown",
            "+overwrite",
            "--as",
            "bot",
            "--file-token",
            file_token,
            "--file",
            str(report_path),
            "--name",
            report_path.name,
        ]
    )


def lark_markdown_fetch(file_token: str) -> dict[str, object]:
    """Fetch a Feishu Markdown document through lark-cli for readback verification."""
    return run_lark_json(
        [lark_cli_executable(), "markdown", "+fetch", "--as", "bot", "--file-token", file_token]
    )


def lark_send_message(chat_id: str, markdown: str, target_date: str) -> dict[str, object]:
    """Send a concise plain-text summary to a Feishu group through lark-cli."""
    return run_lark_json(
        [
            lark_cli_executable(),
            "im",
            "+messages-send",
            "--as",
            "bot",
            "--chat-id",
            chat_id,
            "--text",
            markdown,
            "--idempotency-key",
            f"pm-football-{target_date.replace('-', '')}",
        ]
    )


def lark_message_mget(message_id: str) -> dict[str, object]:
    """Fetch one Feishu message by ID through lark-cli."""
    return run_lark_json(
        [lark_cli_executable(), "im", "+messages-mget", "--as", "bot", "--message-ids", message_id]
    )


def lark_cli_executable() -> str:
    """Return the platform-appropriate lark-cli launcher."""
    return "lark-cli.cmd" if os.name == "nt" else "lark-cli"


def run_lark_json(args: Sequence[str]) -> dict[str, object]:
    """Run lark-cli and parse JSON output without logging secrets."""
    try:
        completed = subprocess.run(
            list(args),
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=90,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": type(exc).__name__}
    if completed.returncode != 0:
        return {"ok": False, "returncode": completed.returncode, "stderr": completed.stderr[-1000:]}
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"ok": True, "stdout": completed.stdout}
    return payload if isinstance(payload, dict) else {"data": payload}


def readback_contains(payload: Mapping[str, object], needle: str) -> bool:
    """Return True when a lark-cli readback payload contains an expected marker."""
    return needle in json.dumps(payload, ensure_ascii=False)


def extract_first(payload: Mapping[str, object], keys: Sequence[str]) -> str:
    """Recursively extract the first string value matching any key."""
    for key, value in walk_mapping(payload):
        if key in keys and isinstance(value, str) and value:
            return value
    return ""


def walk_mapping(payload: object) -> Iterable[tuple[str, object]]:
    """Yield nested mapping key-value pairs."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            yield str(key), value
            yield from walk_mapping(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from walk_mapping(item)


def extract_token_from_url(url: str) -> str:
    """Extract a Feishu token-like path segment from a document URL."""
    parts = [part for part in re.split(r"[/?#]", url) if part]
    return parts[-1] if parts else ""


def load_lark_state(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Load same-day Feishu Markdown file state for overwrite reuse."""
    state_path = path or REPO_ROOT / DEFAULT_OUTPUT_DIR / "polymarket_football_daily_lark_state.json"
    if not state_path.exists():
        return {}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_lark_state(state: Mapping[str, object], path: Path | None = None) -> None:
    """Save same-day Feishu Markdown file state."""
    state_path = path or REPO_ROOT / DEFAULT_OUTPUT_DIR / "polymarket_football_daily_lark_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(state_path, dict(state))


def replace_payload(
    payload: ReportPayload,
    artifacts: dict[str, str] | None = None,
    send_gate: str | None = None,
    feishu: dict[str, object] | None = None,
) -> ReportPayload:
    """Return a copy of a report payload with selected fields replaced."""
    return ReportPayload(
        target_date=payload.target_date,
        generated_at=payload.generated_at,
        data_gate=payload.data_gate,
        send_gate=send_gate if send_gate is not None else payload.send_gate,
        candidates=payload.candidates,
        rejected=payload.rejected,
        coverage=payload.coverage,
        diagnostics=payload.diagnostics,
        artifacts=artifacts if artifacts is not None else payload.artifacts,
        feishu=feishu if feishu is not None else payload.feishu,
    )


def payload_to_dict(payload: ReportPayload) -> dict[str, object]:
    """Convert nested dataclass payload to plain JSON-compatible objects."""
    return asdict(payload)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write UTF-8 JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_datetime(value: object) -> datetime | None:
    """Parse API timestamps into UTC-aware datetimes."""
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
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_optional_float(value: object) -> float | None:
    """Parse a finite float from a flexible API value."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def stable_hash(value: str) -> str:
    """Return a short stable ASCII hash for idempotent local identifiers."""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Polymarket football daily report.")
    parser.add_argument("--date", default="today")
    parser.add_argument("--chat-id", default=None)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--no-send", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-api-requests", type=int, default=250)
    parser.add_argument("--min-ev", type=float, default=DEFAULT_MIN_EV)
    parser.add_argument("--max-spread", type=float, default=DEFAULT_MAX_SPREAD)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    """CLI entry point for daily report generation and optional Feishu delivery."""
    os.chdir(REPO_ROOT)
    load_env_file(REPO_ROOT / ".env")
    args = _parse_args()
    target = parse_target_date(args.date)
    send = bool(args.send and not args.no_send)
    payload = run_daily_report(
        target_date=target,
        output_dir=Path(args.output_dir),
        chat_id=args.chat_id,
        send=send,
        dry_run=args.dry_run,
        max_api_requests=args.max_api_requests,
        min_ev=args.min_ev,
        max_spread=args.max_spread,
    )
    print(json.dumps(payload_to_dict(payload), ensure_ascii=False, indent=2))
    return 0 if not payload.send_gate.startswith("blocked") else 1


if __name__ == "__main__":
    raise SystemExit(main())
