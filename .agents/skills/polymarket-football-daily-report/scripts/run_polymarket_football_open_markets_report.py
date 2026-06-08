"""Generate and optionally send a full report for all currently open Polymarket football 1X2 markets.

The script reuses the existing read-only Polymarket/API-Football/The Odds API
analysis primitives. It never places orders or instantiates live trading flows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api_football_today_daemon import APIFootballDaemonConfig, sync_api_football_today  # noqa: E402
from config import load_env_file  # noqa: E402

from importlib.util import module_from_spec, spec_from_file_location  # noqa: E402


BASE_SCRIPT_PATH = Path(__file__).with_name("run_polymarket_football_daily_report.py")
BASE_SPEC = spec_from_file_location("pm_football_daily_base", BASE_SCRIPT_PATH)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load base script: {BASE_SCRIPT_PATH}")
BASE = module_from_spec(BASE_SPEC)
sys.modules["pm_football_daily_base"] = BASE
BASE_SPEC.loader.exec_module(BASE)

DEFAULT_OUTPUT_DIR = "reports"
DEFAULT_MAX_API_REQUESTS = 500
DEFAULT_STATE_FILE = "polymarket_football_open_markets_lark_state.json"


@dataclass(frozen=True)
class OpenMarketsPayload:
    """Serializable payload for the all-open football markets report."""

    report_key: str
    generated_at: str
    data_gate: str
    send_gate: str
    candidates: list[BASE.MatchResult]
    rejected: list[BASE.MatchResult]
    coverage: dict[str, int]
    diagnostics: list[str]
    artifacts: dict[str, str]
    feishu: dict[str, object]


@dataclass(frozen=True)
class OpenMatchSnapshot:
    """One current Polymarket football match-like event before quote completeness checks."""

    title: str
    slug: str
    kickoff: datetime
    home_team: str
    away_team: str


def discover_open_match_markets(
    diagnostics: list[str],
) -> tuple[list[OpenMatchSnapshot], list[BASE.PolymarketMatchMarket]]:
    """Return open football match snapshots and the subset with complete 1X2 CLOB quotes."""
    try:
        gamma = BASE.PolymarketGammaConnector.from_env()
        clob = BASE.PolymarketClobConnector.from_env()
    except Exception as exc:  # pragma: no cover - environment failure
        diagnostics.append(f"polymarket_connector_unavailable:{type(exc).__name__}")
        return [], []

    raw_items: list[dict[str, Any]] = []
    for term in BASE.SOCCER_SEARCH_TERMS:
        try:
            payload = gamma.search(term, limit_per_type=100)
            raw_items.extend(BASE.extract_gamma_items(payload))
        except Exception as exc:
            diagnostics.append(f"polymarket_gamma_search_failed:{term}:{type(exc).__name__}")
    for fetcher_name in ("list_markets", "list_events"):
        fetcher = getattr(gamma, fetcher_name)
        try:
            payload = fetcher(limit=1000, active=True, closed=False)
            raw_items.extend(BASE.extract_gamma_items(payload))
        except Exception as exc:
            diagnostics.append(f"polymarket_gamma_{fetcher_name}_failed:{type(exc).__name__}")

    now_utc = datetime.now(timezone.utc)
    seen_slugs: set[str] = set()
    snapshots: list[OpenMatchSnapshot] = []
    markets: list[BASE.PolymarketMatchMarket] = []
    skipped_no_kickoff = 0
    skipped_past_kickoff = 0
    skipped_not_tradable = 0
    filtered_items = [item for item in raw_items if BASE.looks_like_football_item(item)]
    for item in filtered_items:
        title = str(item.get("title") or item.get("question") or item.get("slug") or "")
        slug = str(item.get("slug") or BASE.stable_hash(title))
        kickoff = BASE.gamma_item_kickoff(item)
        home_team, away_team = BASE.infer_teams(title)
        if slug in seen_slugs:
            continue
        if kickoff is None:
            skipped_no_kickoff += 1
            continue
        if kickoff <= now_utc:
            skipped_past_kickoff += 1
            continue
        if not BASE.gamma_item_can_have_order_book(item):
            skipped_not_tradable += 1
            continue
        seen_slugs.add(slug)
        if home_team and away_team:
            snapshots.append(OpenMatchSnapshot(title, slug, kickoff, home_team, away_team))
        market = BASE.gamma_item_to_match_market(item, clob, diagnostics)
        if market is not None:
            markets.append(market)
    if skipped_no_kickoff:
        diagnostics.append(f"polymarket_open_skipped_no_kickoff:{skipped_no_kickoff}")
    if skipped_past_kickoff:
        diagnostics.append(f"polymarket_open_skipped_past_kickoff:{skipped_past_kickoff}")
    if skipped_not_tradable:
        diagnostics.append(f"polymarket_open_skipped_not_tradable:{skipped_not_tradable}")
    snapshots.sort(key=lambda item: (item.kickoff, item.slug))
    markets.sort(key=lambda item: (item.kickoff or now_utc, item.slug))
    return snapshots, markets


def sync_fixture_dates(
    market_dates: set[date],
    max_api_requests: int,
    diagnostics: list[str],
) -> dict[date, list[BASE.Fixture]]:
    """Sync and load API-Football fixtures for each distinct market date."""
    fixture_map: dict[date, list[BASE.Fixture]] = {}
    for market_date in sorted(market_dates):
        try:
            sync_api_football_today(
                APIFootballDaemonConfig(max_api_requests_per_run=max_api_requests),
                target_date=market_date,
            )
        except Exception as exc:
            diagnostics.append(
                f"api_football_sync_failed:{market_date.isoformat()}:{type(exc).__name__}"
            )
        rows = BASE.read_api_football_rows(BASE.DEFAULT_API_DB, market_date)
        fixture_map[market_date] = rows
        if not rows:
            diagnostics.append(f"api_football_rows_missing:{market_date.isoformat()}")
    return fixture_map


def build_open_markets_payload(
    output_dir: Path,
    max_api_requests: int,
    min_ev: float,
    max_spread: float,
) -> OpenMarketsPayload:
    """Analyze all currently open football match markets and build one report payload."""
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics: list[str] = []
    snapshots, markets = discover_open_match_markets(diagnostics)
    odds_api_matches = BASE.fetch_the_odds_api_matches(
        diagnostics,
        max_requests=max_api_requests,
    )
    odds_lookup = BASE.build_odds_api_lookup(odds_api_matches)
    market_dates = {market.kickoff.date() for market in markets if market.kickoff is not None}
    fixtures_by_date = sync_fixture_dates(market_dates, max_api_requests, diagnostics)

    candidates: list[BASE.MatchResult] = []
    rejected: list[BASE.MatchResult] = []
    analyzed_slugs = {market.slug for market in markets}
    for snapshot in snapshots:
        if snapshot.slug in analyzed_slugs:
            continue
        pseudo_market = BASE.PolymarketMatchMarket(
            title=snapshot.title,
            slug=snapshot.slug,
            kickoff=snapshot.kickoff,
            home_team=snapshot.home_team,
            away_team=snapshot.away_team,
            selections={},
        )
        rejected.append(
            BASE.blocked_result(
                pseudo_market,
                "H",
                "incomplete_polymarket_quotes_or_market_structure",
            )
        )
    for market in markets:
        if market.kickoff is None:
            rejected.append(BASE.blocked_result(market, "H", "polymarket_kickoff_missing"))
            continue
        fixtures = fixtures_by_date.get(market.kickoff.date(), [])
        if not fixtures:
            rejected.append(
                BASE.blocked_result(
                    market,
                    "H",
                    f"api_football_fixture_date_missing:{market.kickoff.date().isoformat()}",
                )
            )
            continue
        rows = BASE.analyze_market(
            market,
            fixtures,
            odds_lookup,
            min_ev=min_ev,
            max_spread=max_spread,
        )
        for row in rows:
            if row.gate_status == "candidate":
                candidates.append(row)
            else:
                rejected.append(row)

    data_gate = "ok"
    if not snapshots:
        data_gate = "blocked:no_open_polymarket_football_markets"
    elif not odds_api_matches:
        data_gate = "blocked:no_odds_api_prices"
    elif not fixtures_by_date and markets:
        data_gate = "blocked:no_api_football_fixtures"

    today_label = datetime.now(ZoneInfo(BASE.DEFAULT_TIMEZONE)).date().isoformat()
    payload = OpenMarketsPayload(
        report_key=f"all-open-{today_label}",
        generated_at=datetime.now(timezone.utc).isoformat(),
        data_gate=data_gate,
        send_gate="pending",
        candidates=sorted(candidates, key=lambda item: item.ev, reverse=True),
        rejected=rejected,
        coverage={
            "raw_open_matches": len(snapshots),
            "polymarket_markets": len(markets),
            "market_dates": len(market_dates),
            "api_football_fixtures": sum(len(rows) for rows in fixtures_by_date.values()),
            "odds_api_matches": len(odds_api_matches),
            "matched_rows": len(candidates) + len(rejected),
            "candidates": len(candidates),
        },
        diagnostics=diagnostics,
        artifacts={},
        feishu={},
    )
    artifacts = write_artifacts(payload, output_dir)
    payload = replace_payload(payload, artifacts=artifacts)
    return write_receipt(payload, "skipped")


def render_report(payload: OpenMarketsPayload) -> str:
    """Render the all-open markets Markdown report."""
    candidate_rows = "\n".join(BASE.result_row(item) for item in payload.candidates[:50]) or "无。"
    rejected_rows = "\n".join(BASE.result_row(item) for item in payload.rejected[:80]) or "无。"
    diagnostics = "\n".join(f"- {item}" for item in payload.diagnostics) or "- 无。"
    date_counter = Counter()
    reason_counter = Counter()
    for row in payload.candidates + payload.rejected:
        if row.kickoff:
            date_counter[row.kickoff[:10]] += 1
        for reason in row.reasons:
            reason_counter[reason] += 1
    date_lines = "\n".join(f"- {day}: `{count}`" for day, count in sorted(date_counter.items())) or "- 无。"
    reason_lines = "\n".join(
        f"- {reason}: `{count}`" for reason, count in reason_counter.most_common(20)
    ) or "- 无。"
    return "\n".join(
        [
            f"# Polymarket 足球全量市场研究报告 {payload.report_key}",
            "",
            "## 执行摘要",
            "",
            f"- data_gate: `{payload.data_gate}`",
            f"- send_gate: `{payload.send_gate}`",
            f"- polymarket_markets: `{payload.coverage['polymarket_markets']}`",
            f"- candidates: `{len(payload.candidates)}`",
            f"- generated_at_utc: `{payload.generated_at}`",
            "",
            "## 候选清单",
            "",
            candidate_rows,
            "",
            "## 拒绝清单",
            "",
            rejected_rows,
            "",
            "## 开赛日覆盖",
            "",
            date_lines,
            "",
            "## 拦截原因统计",
            "",
            reason_lines,
            "",
            "## 数据覆盖",
            "",
            "\n".join(f"- {key}: `{value}`" for key, value in payload.coverage.items()),
            "",
            "## 方法与风控",
            "",
            "- 只读 Polymarket Gamma/CLOB、API-Football、The Odds API，不执行真实下单。",
            "- 仅分析当前仍可下注、未开赛、完整三路 1X2 且可读取 CLOB 的足球市场。",
            "- 候选门禁沿用现有主策略口径：EV、价差、外部概率分歧、纸面 stake。",
            "- 本报告是研究输出，不是成交回报，也不代表任何实盘执行。",
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


def render_card(payload: OpenMarketsPayload) -> str:
    """Render a one-line Feishu summary for the all-open markets report."""
    parts = [
        f"Polymarket 足球全量市场报告 {payload.report_key}",
        f"data_gate={payload.data_gate}",
        f"markets={payload.coverage.get('polymarket_markets', 0)}",
        f"candidates={len(payload.candidates)}",
    ]
    if payload.candidates:
        top = payload.candidates[0]
        parts.append(
            f"top={top.match_name} {BASE.SELECTION_LABELS.get(top.selection, top.selection)} EV={top.ev:.3f}"
        )
    elif payload.rejected:
        first = payload.rejected[0]
        parts.append(
            f"first_blocked={first.match_name} {','.join(first.reasons) or 'blocked'}"
        )
    doc_url = str(payload.feishu.get("doc_url", ""))
    if doc_url:
        parts.append(f"doc={doc_url}")
    return " | ".join(parts)


def replace_payload(
    payload: OpenMarketsPayload,
    artifacts: dict[str, str] | None = None,
    send_gate: str | None = None,
    feishu: dict[str, object] | None = None,
) -> OpenMarketsPayload:
    """Return a payload copy with selected fields replaced."""
    return OpenMarketsPayload(
        report_key=payload.report_key,
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


def payload_to_dict(payload: OpenMarketsPayload) -> dict[str, object]:
    """Convert payload to plain JSON-compatible objects."""
    return asdict(payload)


def write_artifacts(payload: OpenMarketsPayload, output_dir: Path) -> dict[str, str]:
    """Write report markdown, card, data JSON, and send receipt paths."""
    stem = f"polymarket_football_open_markets_{payload.report_key.replace('-', '')}"
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
    BASE.write_json(data_path, payload_to_dict(payload))
    return artifacts


def content_gate(report_path: Path, card_path: Path) -> bool:
    """Check the report contains required sections before sending."""
    report = report_path.read_text(encoding="utf-8")
    card = card_path.read_text(encoding="utf-8")
    required = ("## 执行摘要", "## 候选清单", "## 数据覆盖", "## 方法与风控")
    return bool(card.strip()) and all(section in report for section in required)


def load_state(path: Path) -> dict[str, dict[str, str]]:
    """Load persistent Feishu file state for overwrite reuse."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_state(path: Path, state: dict[str, object]) -> None:
    """Save persistent Feishu file state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    BASE.write_json(path, state)


def send_payload(payload: OpenMarketsPayload, output_dir: Path, chat_id: str | None) -> OpenMarketsPayload:
    """Create or overwrite the Feishu doc, send the group summary, and verify readback."""
    if payload.data_gate != "ok":
        return write_receipt(payload, "blocked:data_gate")
    report_path = Path(payload.artifacts["report_md"])
    card_path = Path(payload.artifacts["card_md"])
    if not content_gate(report_path, card_path):
        return write_receipt(payload, "blocked:content_gate")
    active_chat_id = chat_id or os.environ.get(BASE.DEFAULT_CHAT_ENV)
    if not active_chat_id:
        return write_receipt(payload, "blocked:missing_chat_id")

    state_path = output_dir / DEFAULT_STATE_FILE
    state = load_state(state_path)
    state_key = payload.report_key
    prior = state.get(state_key, {})
    file_token = str(prior.get("file_token", ""))
    doc_url = str(prior.get("doc_url", ""))
    doc_result = (
        BASE.lark_markdown_overwrite(report_path, file_token)
        if file_token
        else BASE.lark_markdown_create(report_path)
    )
    new_token = BASE.extract_first(doc_result, ("file_token", "token", "obj_token"))
    new_url = BASE.extract_first(doc_result, ("url", "doc_url", "preview_url", "web_url"))
    file_token = new_token or file_token
    doc_url = new_url or doc_url
    if not file_token:
        return write_receipt(payload, "blocked:lark_doc_create_failed", {"doc_result": doc_result})
    state[state_key] = {"file_token": file_token, "doc_url": doc_url}
    save_state(state_path, state)

    payload = replace_payload(payload, feishu={"file_token": file_token, "doc_url": doc_url})
    card_path.write_text(render_card(payload), encoding="utf-8")
    message_result = BASE.lark_send_message(
        active_chat_id,
        card_path.read_text(encoding="utf-8"),
        payload.report_key,
    )
    message_id = BASE.extract_first(message_result, ("message_id", "id"))
    if not message_id:
        return write_receipt(
            payload,
            "blocked:lark_message_send_failed",
            {"message_result": message_result, "doc_result": doc_result},
        )
    doc_readback = BASE.lark_markdown_fetch(file_token)
    msg_readback = BASE.lark_message_mget(message_id)
    feishu = {
        "status": "sent",
        "file_token": file_token,
        "doc_url": doc_url,
        "message_id": message_id,
        "document_readback": "ok" if BASE.readback_contains(doc_readback, payload.report_key) else "failed",
        "message_readback": "ok" if BASE.readback_contains(msg_readback, message_id) else "failed",
        "doc_result": doc_result,
        "message_result": message_result,
    }
    send_gate = (
        "sent"
        if feishu["document_readback"] == "ok" and feishu["message_readback"] == "ok"
        else "blocked:readback_failed"
    )
    return write_receipt(replace_payload(payload, send_gate=send_gate, feishu=feishu), send_gate)


def write_receipt(
    payload: OpenMarketsPayload,
    status: str,
    extra: dict[str, object] | None = None,
) -> OpenMarketsPayload:
    """Persist send receipt and refresh report/data artifacts with final status."""
    feishu = dict(payload.feishu)
    feishu.setdefault("status", status)
    if extra:
        feishu.update(extra)
    payload = replace_payload(payload, send_gate=status, feishu=feishu)
    artifacts = payload.artifacts
    if "send_receipt_json" in artifacts:
        BASE.write_json(Path(artifacts["send_receipt_json"]), {"send_gate": status, "feishu": feishu})
    if "data_json" in artifacts:
        BASE.write_json(Path(artifacts["data_json"]), payload_to_dict(payload))
    if "report_md" in artifacts:
        Path(artifacts["report_md"]).write_text(render_report(payload), encoding="utf-8")
    if "card_md" in artifacts:
        Path(artifacts["card_md"]).write_text(render_card(payload), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a report for all currently open Polymarket football 1X2 markets."
    )
    parser.add_argument("--chat-id", default=None)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-api-requests", type=int, default=DEFAULT_MAX_API_REQUESTS)
    parser.add_argument("--min-ev", type=float, default=BASE.DEFAULT_MIN_EV)
    parser.add_argument("--max-spread", type=float, default=BASE.DEFAULT_MAX_SPREAD)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    """CLI entry point for the all-open football markets report."""
    os.chdir(REPO_ROOT)
    load_env_file(REPO_ROOT / ".env")
    args = _parse_args()
    payload = build_open_markets_payload(
        output_dir=Path(args.output_dir),
        max_api_requests=args.max_api_requests,
        min_ev=args.min_ev,
        max_spread=args.max_spread,
    )
    if args.send and not args.dry_run:
        payload = send_payload(payload, Path(args.output_dir), args.chat_id)
    elif args.send and args.dry_run:
        payload = write_receipt(payload, "dry_run")
    print(json.dumps(payload_to_dict(payload), ensure_ascii=False, indent=2))
    return 0 if not payload.send_gate.startswith("blocked") else 1


if __name__ == "__main__":
    raise SystemExit(main())
