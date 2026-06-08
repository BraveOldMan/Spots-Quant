"""Tests for the Polymarket football daily report skill script."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".agents"
    / "skills"
    / "polymarket-football-daily-report"
    / "scripts"
    / "run_polymarket_football_daily_report.py"
)
SPEC = importlib.util.spec_from_file_location("pm_football_daily", SCRIPT_PATH)
assert SPEC is not None
pm = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["pm_football_daily"] = pm
SPEC.loader.exec_module(pm)

OPEN_SCRIPT_PATH = SCRIPT_PATH.with_name("run_polymarket_football_open_markets_report.py")
OPEN_SPEC = importlib.util.spec_from_file_location("pm_football_open", OPEN_SCRIPT_PATH)
assert OPEN_SPEC is not None
pm_open = importlib.util.module_from_spec(OPEN_SPEC)
assert OPEN_SPEC.loader is not None
sys.modules["pm_football_open"] = pm_open
OPEN_SPEC.loader.exec_module(pm_open)


def _fixture(
    home: str = "Arsenal",
    away: str = "Chelsea",
    kickoff: datetime | None = None,
) -> Any:
    return pm.Fixture(
        fixture_id=1,
        kickoff=kickoff or (datetime.now(timezone.utc) + timedelta(hours=6)),
        status_short="NS",
        home_team=home,
        away_team=away,
        api_probs={"H": 0.55, "D": 0.25, "A": 0.20},
    )


def _quote(selection: str, ask: float = 0.45, bid: float = 0.42) -> Any:
    return pm.PolymarketSelection(
        selection=selection,
        token_id=f"token-{selection}",
        market_slug="arsenal-chelsea",
        question=f"{selection} question",
        bid=bid,
        ask=ask,
        mid=(bid + ask) / 2,
        spread=ask - bid,
        liquidity=1000.0,
    )


def _market(kickoff: datetime | None = None) -> Any:
    return pm.PolymarketMatchMarket(
        title="Arsenal vs Chelsea",
        slug="arsenal-chelsea",
        kickoff=kickoff or (datetime.now(timezone.utc) + timedelta(hours=6)),
        home_team="Arsenal",
        away_team="Chelsea",
        selections={"H": _quote("H"), "D": _quote("D", 0.27, 0.25), "A": _quote("A", 0.30, 0.28)},
    )


class FakeClob:
    """Minimal CLOB stub returning deterministic order books by token id."""

    def __init__(
        self,
        books: dict[str, object] | None = None,
        exc: Exception | None = None,
    ) -> None:
        self.books = books or {}
        self.exc = exc

    def get_order_book(self, token_id: str) -> object:
        """Return a public order-book payload for the requested token."""
        if self.exc is not None:
            raise self.exc
        return self.books[token_id]


class ExplodingClob:
    """CLOB stub that fails if a closed Gamma market reaches quote reading."""

    def get_order_book(self, _token_id: str) -> object:
        """Raise to prove closed markets are filtered before CLOB access."""
        raise AssertionError("closed market should not query CLOB")


def _book(bid: str = "0.42", ask: str = "0.45") -> dict[str, object]:
    return {
        "bids": [{"price": bid, "size": "100"}],
        "asks": [{"price": ask, "size": "100"}],
        "liquidity": "1000",
    }


def test_no_vig_probabilities_normalize_three_way_odds() -> None:
    """Decimal odds should become normalized fair probabilities."""
    probs = pm.no_vig_probabilities({"H": 2.0, "D": 4.0, "A": 4.0})

    assert probs["H"] == pytest.approx(0.5)
    assert probs["D"] == pytest.approx(0.25)
    assert probs["A"] == pytest.approx(0.25)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_match_polymarket_to_fixture_accepts_exact_and_rejects_ambiguous() -> None:
    """Fixture matching should accept exact rows and fail closed on ties."""
    market = _market()
    fixture = _fixture()

    matched, score, reason = pm.match_polymarket_to_fixture(market, [fixture])

    assert matched == fixture
    assert score == pytest.approx(1.0)
    assert reason == ""

    matched, _, reason = pm.match_polymarket_to_fixture(market, [fixture, fixture])

    assert matched is None
    assert reason == "ambiguous_match"


def test_evaluate_selection_candidate_and_rejections() -> None:
    """EV and spread gates should separate candidates from observe-only rows."""
    quote = _quote("H", ask=0.45, bid=0.42)

    gate, reasons, ev, max_price, confidence, stake = pm.evaluate_selection(
        "H", 0.55, 0.55, 0.55, quote
    )

    assert gate == "candidate"
    assert reasons == []
    assert ev == pytest.approx(1.222222)
    assert max_price == pytest.approx(0.55 / 1.05)
    assert confidence == "high"
    assert stake == pytest.approx(0.75)

    wide = _quote("H", ask=0.55, bid=0.40)
    gate, reasons, *_ = pm.evaluate_selection("H", 0.56, 0.56, 0.56, wide)

    assert gate == "observe_only"
    assert "spread_too_wide" in reasons


def test_analyze_market_blocks_in_play_and_missing_odds_api() -> None:
    """In-play fixtures or missing The Odds API rows must fail closed."""
    fixture = _fixture(kickoff=datetime.now(timezone.utc) + timedelta(hours=6))
    in_play = pm.Fixture(
        fixture_id=2,
        kickoff=datetime.now(timezone.utc) - timedelta(minutes=5),
        status_short="1H",
        home_team="Arsenal",
        away_team="Chelsea",
        api_probs=fixture.api_probs,
    )

    rows = pm.analyze_market(_market(), [in_play], {}, min_ev=1.05, max_spread=0.08)

    assert rows[0].gate_status == "blocked"
    assert rows[0].reasons == ["fixture_in_play_or_closed"]

    rows = pm.analyze_market(_market(), [fixture], {}, min_ev=1.05, max_spread=0.08)

    assert rows[0].gate_status == "blocked"
    assert rows[0].reasons == ["odds_api_match_missing"]


def test_parse_the_odds_api_payload_and_lookup() -> None:
    """The Odds API payload should parse only complete football H2H rows."""
    payload = [
        {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "commence_time": "2026-06-07T15:00:00Z",
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Arsenal", "price": 2.0},
                                {"name": "Draw", "price": 4.0},
                                {"name": "Chelsea", "price": 4.0},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    rows = pm.parse_the_odds_api_payload(payload)
    lookup = pm.build_odds_api_lookup(rows)
    fixture = _fixture(kickoff=datetime(2026, 6, 7, 15, 0, tzinfo=timezone.utc))
    match = pm.find_odds_api_match(fixture, lookup)

    assert len(rows) == 1
    assert match is not None
    assert match.probs["H"] == pytest.approx(0.5)


def test_fetch_the_odds_api_records_empty_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Odds API empty H2H responses should leave explicit diagnostics."""

    class FakeResponse:
        def __init__(self, payload: object) -> None:
            self.payload = payload

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(self.payload).encode("utf-8")

    payloads: list[object] = [[{"key": "soccer_epl", "active": True}], []]
    monkeypatch.setenv("THE_ODDS_API_KEY", "test-key")

    def fake_urlopen(*_args: object, **_kwargs: object) -> FakeResponse:
        return FakeResponse(payloads.pop(0))

    monkeypatch.setattr(pm.urllib.request, "urlopen", fake_urlopen)

    diagnostics: list[str] = []
    rows = pm.fetch_the_odds_api_matches(diagnostics)

    assert rows == []
    assert "the_odds_api_empty_h2h_payload" in diagnostics


def test_polymarket_filter_excludes_american_football_draft() -> None:
    """Polymarket discovery should not treat NFL draft markets as soccer."""
    draft_item = {
        "title": "Who will be the first overall pick in the 2027 pro-football draft?",
        "slug": "2027-nfl-draft-first-overall-pick",
    }
    soccer_item = {
        "title": "Arsenal vs Chelsea",
        "markets": [
            {
                "outcomes": [
                    {"name": "Arsenal"},
                    {"name": "Draw"},
                    {"name": "Chelsea"},
                ]
            }
        ],
    }
    basketball_item = {
        "title": "Lakers vs Celtics",
        "markets": [{"outcomes": [{"name": "Lakers"}, {"name": "Celtics"}]}],
    }

    assert not pm.looks_like_football_item(draft_item)
    assert pm.looks_like_football_item(soccer_item)
    assert not pm.looks_like_football_item(basketball_item)


def test_gamma_item_kickoff_prefers_end_date_over_creation_start_date() -> None:
    """Gamma event rows should use match endDate when startDate is just creation time."""
    item = {
        "title": "Liechtenstein vs. Cyprus",
        "startDate": "2026-05-10T13:24:30.941959Z",
        "endDate": "2026-06-07T13:00:00Z",
    }

    kickoff = pm.gamma_item_kickoff(item)

    assert kickoff == datetime(2026, 6, 7, 13, 0, tzinfo=timezone.utc)


def test_infer_teams_strips_competition_prefix_and_market_suffix() -> None:
    """Polymarket titles should not leak competition labels into team names."""
    assert pm.infer_teams("Soccer: England vs. Senegal") == ("England", "Senegal")
    assert pm.infer_teams("UEFA Nations League: England vs Greece") == (
        "England",
        "Greece",
    )
    assert pm.infer_teams("FIFA Club World Cup: Benfica vs Chelsea (To Advance)") == (
        "Benfica",
        "Chelsea",
    )


def test_classify_binary_yes_markets_maps_questions_to_1x2() -> None:
    """Three Polymarket binary YES questions should aggregate into H/D/A."""
    assert pm.classify_market_outcomes(
        "Will England beat Senegal?",
        ["Yes", "No"],
        ["eng-yes", "eng-no"],
        "England",
        "Senegal",
    ) == {"H": "eng-yes"}
    assert pm.classify_market_outcomes(
        "Will Senegal beat England?",
        ["Yes", "No"],
        ["sen-yes", "sen-no"],
        "England",
        "Senegal",
    ) == {"A": "sen-yes"}
    assert pm.classify_market_outcomes(
        "Will England vs. Senegal end in a draw?",
        ["Yes", "No"],
        ["draw-yes", "draw-no"],
        "England",
        "Senegal",
    ) == {"D": "draw-yes"}


def test_binary_yes_market_uses_explicit_yes_token() -> None:
    """YES token selection must respect outcome order instead of taking index 0."""
    assert pm.classify_market_outcomes(
        "Will England beat Senegal?",
        ["No", "Yes"],
        ["no-token", "yes-token"],
        "England",
        "Senegal",
    ) == {"H": "yes-token"}


def test_gamma_item_to_match_market_aggregates_split_yes_no_markets() -> None:
    """Gamma split YES/NO child markets should become one complete 1X2 market."""
    item = {
        "title": "Soccer: England vs. Senegal",
        "slug": "soccer-england-vs-senegal",
        "endDate": "2026-06-07T19:00:00Z",
        "markets": [
            {
                "slug": "england-beat-senegal",
                "question": "Will England beat Senegal?",
                "outcomes": ["Yes", "No"],
                "clobTokenIds": ["eng-yes", "eng-no"],
            },
            {
                "slug": "senegal-beat-england",
                "question": "Will Senegal beat England?",
                "outcomes": ["Yes", "No"],
                "clobTokenIds": ["sen-yes", "sen-no"],
            },
            {
                "slug": "england-senegal-draw",
                "question": "Will England vs. Senegal end in a draw?",
                "outcomes": ["No", "Yes"],
                "clobTokenIds": ["draw-no", "draw-yes"],
            },
        ],
    }
    clob = FakeClob(
        {
            "eng-yes": _book("0.41", "0.44"),
            "sen-yes": _book("0.30", "0.33"),
            "draw-yes": _book("0.24", "0.27"),
        }
    )
    diagnostics: list[str] = []

    market = pm.gamma_item_to_match_market(item, clob, diagnostics)

    assert market is not None
    assert market.home_team == "England"
    assert market.away_team == "Senegal"
    assert set(market.selections) == {"H", "D", "A"}
    assert market.selections["H"].token_id == "eng-yes"
    assert market.selections["D"].token_id == "draw-yes"
    assert market.selections["A"].token_id == "sen-yes"
    assert diagnostics == []


def test_gamma_closed_markets_are_filtered_before_clob() -> None:
    """Closed Gamma rows should not trigger CLOB /book requests."""
    item = {
        "title": "Soccer: England vs. Senegal",
        "slug": "soccer-england-vs-senegal",
        "endDate": "2025-06-10T00:00:00Z",
        "active": True,
        "closed": True,
        "archived": False,
        "markets": [
            {
                "slug": "will-england-beat-senegal",
                "question": "Will England beat Senegal?",
                "outcomes": ["Yes", "No"],
                "clobTokenIds": ["eng-yes", "eng-no"],
                "active": True,
                "closed": True,
                "acceptingOrders": False,
                "ready": False,
            }
        ],
    }
    diagnostics: list[str] = []

    market = pm.gamma_item_to_match_market(item, ExplodingClob(), diagnostics)

    assert not pm.gamma_item_can_have_order_book(item)
    assert market is None
    assert diagnostics == []


def test_read_clob_quote_fail_closed_with_structured_diagnostics() -> None:
    """Unavailable or one-sided CLOB books should return None with reasons."""
    diagnostics: list[str] = []

    unavailable = pm.read_clob_quote(
        FakeClob(exc=RuntimeError("HTTP 404")),
        "token",
        "slug",
        "question",
        "H",
        diagnostics,
    )

    assert unavailable is None
    assert diagnostics == ["clob_book_unavailable:slug:H:RuntimeError:http_404"]

    diagnostics.clear()
    missing_bid = pm.read_clob_quote(
        FakeClob({"token": {"bids": [], "asks": [{"price": "0.55"}]}}),
        "token",
        "slug",
        "question",
        "H",
        diagnostics,
    )

    assert missing_bid is None
    assert diagnostics == ["clob_bid_missing:slug:H"]

    diagnostics.clear()
    missing_ask = pm.read_clob_quote(
        FakeClob({"token": {"bids": [{"price": "0.45"}], "asks": []}}),
        "token",
        "slug",
        "question",
        "H",
        diagnostics,
    )

    assert missing_ask is None
    assert diagnostics == ["clob_ask_missing:slug:H"]


def test_discover_polymarket_markets_falls_back_to_fixture_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-day fixture searches should rescue discovery when broad Gamma queries miss."""

    class FakeGamma:
        def search(self, query: str, limit_per_type: int = 10) -> Any:
            if query in {"soccer", "fifa", "uefa"}:
                return {"events": []}
            if query == "arsenal chelsea":
                return {
                    "events": [
                        {
                            "title": "Arsenal vs. Chelsea",
                            "slug": "arsenal-chelsea-2026-06-07",
                            "endDate": "2026-06-07T15:00:00Z",
                        }
                    ]
                }
            return {"events": []}

        def list_markets(self, **_kwargs: Any) -> Any:
            return []

    class FakeClob:
        pass

    target_date = date(2026, 6, 7)
    fixture = _fixture(kickoff=datetime(2026, 6, 7, 15, 0, tzinfo=timezone.utc))
    diagnostics: list[str] = []

    monkeypatch.setattr(pm.PolymarketGammaConnector, "from_env", lambda: FakeGamma())
    monkeypatch.setattr(pm.PolymarketClobConnector, "from_env", lambda: FakeClob())
    monkeypatch.setattr(
        pm,
        "gamma_item_to_match_market",
        lambda item, _clob, _diagnostics: pm.PolymarketMatchMarket(
            title=str(item["title"]),
            slug=str(item["slug"]),
            kickoff=pm.gamma_item_kickoff(item),
            home_team="Arsenal",
            away_team="Chelsea",
            selections={"H": _quote("H"), "D": _quote("D"), "A": _quote("A")},
        ),
    )

    markets = pm.discover_polymarket_markets(target_date, diagnostics, [fixture])

    assert [market.slug for market in markets] == ["arsenal-chelsea-2026-06-07"]
    assert "polymarket_gamma_fixture_search_queries:1" in diagnostics


def test_discover_polymarket_markets_uses_shanghai_date_for_gamma_kickoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UTC kickoff timestamps should be filtered by report timezone date, not raw UTC date."""

    class FakeGamma:
        def search(self, _query: str, limit_per_type: int = 10) -> Any:
            return {"events": []}

        def list_markets(self, **_kwargs: Any) -> Any:
            return [
                {
                    "title": "Soccer: Arsenal vs. Chelsea",
                    "slug": "arsenal-chelsea-2026-06-09",
                    "endDate": "2026-06-08T16:30:00Z",
                    "outcomes": ["Arsenal", "Draw", "Chelsea"],
                }
            ]

    class FakeClob:
        pass

    diagnostics: list[str] = []

    monkeypatch.setattr(pm.PolymarketGammaConnector, "from_env", lambda: FakeGamma())
    monkeypatch.setattr(pm.PolymarketClobConnector, "from_env", lambda: FakeClob())
    monkeypatch.setattr(
        pm,
        "gamma_item_to_match_market",
        lambda item, _clob, _diagnostics: pm.PolymarketMatchMarket(
            title=str(item["title"]),
            slug=str(item["slug"]),
            kickoff=pm.gamma_item_kickoff(item),
            home_team="Arsenal",
            away_team="Chelsea",
            selections={"H": _quote("H"), "D": _quote("D"), "A": _quote("A")},
        ),
    )

    markets = pm.discover_polymarket_markets(date(2026, 6, 9), diagnostics, [])

    assert [market.slug for market in markets] == ["arsenal-chelsea-2026-06-09"]
    assert "polymarket_skipped_other_date" not in diagnostics


def test_lark_send_command_uses_chat_id_without_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Feishu command construction should use chat ID and stable idempotency key."""
    captured: dict[str, Any] = {}

    def fake_run(
        args: list[str],
        cwd: Path,
        text: bool,
        capture_output: bool,
        check: bool,
        timeout: int,
    ) -> Any:
        captured["args"] = args

        class Result:
            returncode = 0
            stdout = '{"message_id":"om_test"}'
            stderr = ""

        return Result()

    monkeypatch.setattr(pm.subprocess, "run", fake_run)

    result = pm.lark_send_message("oc_test", "hello", "2026-06-07")

    assert result["message_id"] == "om_test"
    assert "oc_test" in captured["args"]
    assert "pm-football-20260607" in captured["args"]
    assert not any("API_KEY" in item for item in captured["args"])


def test_open_discovery_connector_failure_returns_empty_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Open-market discovery should fail closed without crashing on connector setup."""

    def fail_from_env() -> Any:
        raise RuntimeError("offline")

    monkeypatch.setattr(pm_open.BASE.PolymarketGammaConnector, "from_env", fail_from_env)
    diagnostics: list[str] = []

    snapshots, markets = pm_open.discover_open_match_markets(diagnostics)

    assert snapshots == []
    assert markets == []
    assert diagnostics == ["polymarket_connector_unavailable:RuntimeError"]


def test_open_write_artifacts_passes_content_gate(tmp_path: Path) -> None:
    """Open-market reports should satisfy their own Feishu content gate."""
    payload = pm_open.OpenMarketsPayload(
        report_key="all-open-2026-06-07",
        generated_at="2026-06-07T00:00:00+00:00",
        data_gate="ok",
        send_gate="pending",
        candidates=[],
        rejected=[],
        coverage={
            "raw_open_matches": 0,
            "polymarket_markets": 0,
            "market_dates": 0,
            "api_football_fixtures": 0,
            "odds_api_matches": 0,
            "matched_rows": 0,
            "candidates": 0,
        },
        diagnostics=[],
        artifacts={},
        feishu={},
    )

    artifacts = pm_open.write_artifacts(payload, tmp_path)

    assert pm_open.content_gate(
        Path(artifacts["report_md"]),
        Path(artifacts["card_md"]),
    )


def test_write_artifacts_contains_required_sections(tmp_path: Path) -> None:
    """Local dry-run artifacts should include required report sections."""
    payload = pm.ReportPayload(
        target_date=date(2026, 6, 7).isoformat(),
        generated_at="2026-06-07T00:00:00+00:00",
        data_gate="ok",
        send_gate="pending",
        candidates=[],
        rejected=[],
        coverage={"polymarket_markets": 0},
        diagnostics=[],
        artifacts={},
        feishu={},
    )

    artifacts = pm.write_artifacts(payload, tmp_path)
    payload = pm.replace_payload(payload, artifacts=artifacts)
    final_payload = pm.write_receipt(payload, "dry_run")

    report = Path(artifacts["report_md"]).read_text(encoding="utf-8")
    card = Path(artifacts["card_md"]).read_text(encoding="utf-8")
    receipt = Path(artifacts["send_receipt_json"])
    assert "## 执行摘要" in report
    assert "## 下注建议" in report
    assert "Polymarket 足球日报" in card
    assert final_payload.send_gate == "dry_run"
    assert receipt.exists()
