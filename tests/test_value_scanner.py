"""Regression tests for value scanner persistence helpers."""

import sqlite3

from value_scanner import init_db


def test_init_db_migrates_old_clv_schema(tmp_path) -> None:
    """Old CLV tables should gain nullable cap and closing-odds columns."""
    db_path = tmp_path / "old_bet_history.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE clv_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            fixture_id INTEGER,
            match_name TEXT,
            bet_type TEXT,
            model_prob REAL,
            bookie_odds REAL,
            ev REAL,
            kelly_pct REAL
        )
        """
    )
    conn.commit()
    conn.close()

    migrated = init_db(str(db_path))
    columns = {
        row[1] for row in migrated.execute("PRAGMA table_info(clv_tracking)")
    }
    migrated.close()

    assert {"home_caps", "away_caps", "closing_odds"} <= columns
