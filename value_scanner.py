"""Scan today's football odds for model-based value candidates."""

import json
import sqlite3
from typing import Any, NamedTuple

import pandas as pd
import xgboost as xgb

from api_client import FootballAPIClient
from features import FeatureEngine


class ValueBet(NamedTuple):
    """A single value-bet candidate ready for reporting and persistence."""

    fixture_id: int
    match_name: str
    bet_type: str
    odds: float
    model_prob: float
    bookie_prob: float
    ev: float
    kelly: float
    home_rating: float
    away_rating: float


def remove_margin(odds: dict[str, float]) -> dict[str, float]:
    """Convert 1X2 decimal odds into margin-free probabilities."""
    implied = {
        "home": 1.0 / odds["home"],
        "draw": 1.0 / odds["draw"],
        "away": 1.0 / odds["away"],
    }
    total = sum(implied.values())
    return {side: prob / total for side, prob in implied.items()}


def calculate_kelly(
    p_win: float, decimal_odds: float, fraction: float = 0.25
) -> float:
    """Return fractional Kelly bankroll allocation for decimal odds."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p_win
    f_star = (b * p_win - q) / b
    if f_star <= 0:
        return 0.0
    return f_star * fraction


def _ensure_column(
    cursor: sqlite3.Cursor, table: str, column: str, definition: str
) -> None:
    columns = {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db(db_path: str = "bet_history.db") -> sqlite3.Connection:
    """Open bet history DB and migrate the CLV tracking schema in place."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS clv_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            fixture_id INTEGER,
            match_name TEXT,
            bet_type TEXT,
            model_prob REAL,
            bookie_odds REAL,
            ev REAL,
            kelly_pct REAL,
            home_caps REAL,
            away_caps REAL,
            closing_odds REAL
        )
        """
    )
    _ensure_column(cursor, "clv_tracking", "home_caps", "REAL")
    _ensure_column(cursor, "clv_tracking", "away_caps", "REAL")
    _ensure_column(cursor, "clv_tracking", "closing_odds", "REAL")
    conn.commit()
    return conn


def _rating_sum(lineup: dict[str, Any], engine: FeatureEngine) -> float:
    return sum(
        engine.get_avg_rating(player["player"]["id"])
        for player in lineup.get("startXI", [])
    )


def _lineup_ratings(
    client: FootballAPIClient,
    engine: FeatureEngine,
    fixture_id: int,
    home_id: int,
    away_id: int,
) -> tuple[float, float]:
    start_rating_h = 0.0
    start_rating_a = 0.0
    lineup_res = client.get("/fixtures/lineups", {"fixture": fixture_id})
    if not (lineup_res and lineup_res.get("response")):
        return start_rating_h, start_rating_a

    for team_lineup in lineup_res["response"]:
        team_id = team_lineup["team"]["id"]
        rating_sum = _rating_sum(team_lineup, engine)
        if team_id == home_id:
            start_rating_h = rating_sum
        elif team_id == away_id:
            start_rating_a = rating_sum
    return start_rating_h, start_rating_a


def _injury_mif(
    client: FootballAPIClient,
    engine: FeatureEngine,
    fixture_id: int,
    home_id: int,
    away_id: int,
) -> tuple[float, float]:
    mif_h = 0.0
    mif_a = 0.0
    injuries_res = client.get("/injuries", {"fixture": fixture_id})
    if not (injuries_res and injuries_res.get("response")):
        return mif_h, mif_a

    for injury in injuries_res["response"]:
        team_id = injury["team"]["id"]
        player_id = injury["player"]["id"]
        missed_rating = engine.get_avg_rating(player_id)
        if missed_rating <= 6.5:
            continue
        if team_id == home_id:
            mif_h += missed_rating
        elif team_id == away_id:
            mif_a += missed_rating
    return mif_h, mif_a


def _append_value_bets(
    value_bets: list[ValueBet],
    fixture_id: int,
    match_title: str,
    odds: dict[str, float],
    bookie_probs: dict[str, float],
    probs: tuple[float, float, float],
    ratings: tuple[float, float],
    threshold: float,
) -> None:
    specs = [
        ("Home Win", "home", probs[2]),
        ("Draw", "draw", probs[1]),
        ("Away Win", "away", probs[0]),
    ]
    for bet_type, side, model_prob in specs:
        ev = (model_prob * odds[side]) - 1.0
        if ev <= threshold:
            continue
        value_bets.append(
            ValueBet(
                fixture_id=fixture_id,
                match_name=match_title,
                bet_type=bet_type,
                odds=odds[side],
                model_prob=model_prob,
                bookie_prob=bookie_probs[side],
                ev=ev,
                kelly=calculate_kelly(model_prob, odds[side], fraction=0.25),
                home_rating=ratings[0],
                away_rating=ratings[1],
            )
        )


def scan_value_bets() -> None:
    """Load today's odds, score them with XGBoost, and persist candidates."""
    client = FootballAPIClient()
    conn = init_db()
    cursor = conn.cursor()

    print("Loading V6 AI Quant System (XGBoost + Feature Engine)...")
    engine = FeatureEngine()
    try:
        engine.build_dataset("raw_fixtures_v5.json")
    except Exception as exc:
        print(f"Warning: Could not warm up FeatureEngine: {exc}")

    model = xgb.XGBClassifier()
    try:
        model.load_model("xgboost_model.json")
    except Exception as exc:
        print(f"Error loading XGBoost model: {exc}")
        conn.close()
        return

    try:
        with open("today_odds.json", "r", encoding="utf-8") as file:
            today_odds = json.load(file)
    except FileNotFoundError:
        print("today_odds.json not found. Please run fetch_odds.py first.")
        conn.close()
        return

    print(f"\n--- Scanning {len(today_odds)} fixtures for Value Bets ---")
    value_bets: list[ValueBet] = []
    threshold = 0.05

    for item in today_odds:
        fixture_id = item["fixture_id"]
        odds = item["odds"]
        fixture_res = client.get("/fixtures", {"id": fixture_id})
        if not fixture_res or not fixture_res.get("response"):
            continue

        fixture = fixture_res["response"][0]
        home_id = fixture["teams"]["home"]["id"]
        away_id = fixture["teams"]["away"]["id"]
        home_name = fixture["teams"]["home"]["name"]
        away_name = fixture["teams"]["away"]["name"]

        start_rating_h, start_rating_a = _lineup_ratings(
            client, engine, fixture_id, home_id, away_id
        )
        mif_h, mif_a = _injury_mif(client, engine, fixture_id, home_id, away_id)

        features = pd.DataFrame(
            [
                {
                    "elo_diff": engine.elo_ratings[home_id]
                    - engine.elo_ratings[away_id],
                    "mom_diff": engine.get_team_momentum(home_id)
                    - engine.get_team_momentum(away_id),
                    "rating_diff": start_rating_h - start_rating_a,
                    "mif_home": mif_h,
                    "mif_away": mif_a,
                }
            ]
        )
        p_away, p_draw, p_home = (float(p) for p in model.predict_proba(features)[0])
        if p_home == 0 and p_draw == 0 and p_away == 0:
            continue

        _append_value_bets(
            value_bets=value_bets,
            fixture_id=fixture_id,
            match_title=f"{home_name} vs {away_name}",
            odds=odds,
            bookie_probs=remove_margin(odds),
            probs=(p_away, p_draw, p_home),
            ratings=(start_rating_h, start_rating_a),
            threshold=threshold,
        )

    value_bets.sort(key=lambda bet: bet.ev, reverse=True)
    print("\n--- TOP VALUE BETS & KELLY SIZING ---\n")
    if not value_bets:
        print("No positive Expected Value bets found today.")
    else:
        for bet in value_bets:
            print(
                f"Match: {bet.match_name} "
                f"[Avg Rating - H:{bet.home_rating:.1f} A:{bet.away_rating:.1f}]"
            )
            print(f"Bet On: {bet.bet_type} | Odds: {bet.odds}")
            print(
                f"Model Prob: {bet.model_prob * 100:.1f}% | "
                f"Bookie Prob: {bet.bookie_prob * 100:.1f}% | "
                f"EV: +{bet.ev * 100:.1f}%"
            )
            print(f"Suggested Stake (1/4 Kelly): {bet.kelly * 100:.2f}%")
            print("-" * 50)
            cursor.execute(
                """
                INSERT INTO clv_tracking (
                    fixture_id, match_name, bet_type, model_prob, bookie_odds,
                    ev, kelly_pct, home_caps, away_caps, closing_odds
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bet.fixture_id,
                    bet.match_name,
                    bet.bet_type,
                    bet.model_prob,
                    bet.odds,
                    bet.ev,
                    bet.kelly,
                    bet.home_rating,
                    bet.away_rating,
                    None,
                ),
            )
        conn.commit()
        print(f"[*] Logged {len(value_bets)} bets to bet_history.db.")

    conn.close()


if __name__ == "__main__":
    scan_value_bets()
