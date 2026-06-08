---
name: polymarket-football-daily-report
description: Create and send a read-only daily football betting research report by combining today's Polymarket football markets, API-Football fixtures and odds, The Odds API prices, CLOB order-book liquidity, EV gates, and Feishu document/message delivery. Use when asked to analyze today's Polymarket football matches, produce betting recommendations, or send the football betting report to a Feishu group.
---

# Polymarket Football Daily Report

## Workflow

Run the deterministic script from the repository root:

```powershell
python .agents\skills\polymarket-football-daily-report\scripts\run_polymarket_football_daily_report.py --date today --send
```

Use `--dry-run` before changing logic or when credentials are uncertain. Dry-run writes local
Markdown/JSON artifacts and prints the send commands without creating Feishu documents or messages.

## Inputs

- Polymarket: use the repository's read-only Gamma and CLOB connectors. Never instantiate a live trading connector or call `place_order`.
- API-Football: read today's rows from `spots_quant.db`; if missing or stale, run the existing `sync_api_football_today()` with the configured request budget.
- The Odds API: read `THE_ODDS_API_KEY` from `.env` or the process environment and fetch `h2h` football odds. Missing key or quota failure is a data gap, not a value to fill.
- Feishu: read `FEISHU_POLYMARKET_FOOTBALL_CHAT_ID` unless `--chat-id` is supplied. The default group is configured in `.env`, not in code.

## Gates

- Block sending when API-Football fixtures are missing, Polymarket markets cannot be matched, The Odds API is unavailable, or Feishu readback fails.
- Only analyze complete three-way 1X2 markets. Skip in-play events, ambiguous fixture matches, missing CLOB order books, incomplete bid/ask data, and odds with decimal values outside `(1, 100]`.
- A betting candidate requires `EV >= --min-ev`, spread `<= --max-spread`, readable liquidity, and no excessive disagreement between external fair probabilities.
- If no candidate passes but data coverage is valid, send a "no bet" report with explicit rejection reasons.

## Outputs

The script writes:

- `reports/polymarket_football_daily_YYYYMMDD.md`
- `reports/polymarket_football_daily_YYYYMMDD_card.md`
- `reports/polymarket_football_daily_YYYYMMDD_data.json`
- `reports/polymarket_football_daily_YYYYMMDD_send_receipt.json`

Feishu delivery uses `lark-cli markdown +create/+overwrite`, `lark-cli im +messages-send`,
then verifies with `markdown +fetch` and `im +messages-mget`. Do not treat a send receipt as
complete until both readbacks pass.

## Safety

Recommendations are research outputs only. Use paper units such as `0.25u`; do not place real orders,
do not submit signed transactions, and do not claim a simulated recommendation is an executed trade.

For exact report fields and acceptance criteria, read `references/report_contract.md` when modifying
the script or tests.
