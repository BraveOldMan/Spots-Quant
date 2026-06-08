# Polymarket Football Daily Report Contract

## Required Sections

- `执行摘要`: target date, data gate, candidate count, and send status.
- `下注建议`: one row per candidate with match, selection, fair probability, ask, EV, spread, max price, confidence, and paper stake units.
- `拒绝清单`: rejected matches and reasons.
- `数据覆盖`: counts for Polymarket markets, matched fixtures, API-Football odds, The Odds API odds, and CLOB quotes.
- `方法与风控`: no real trading, no in-play bets, no fabricated odds, and fail-closed rules.
- `飞书回读`: document token or URL, message ID, and readback status after sending.

## Candidate Gate

A row can be marked `candidate` only when all are true:

- matched fixture score is at least `0.86`;
- the fixture has not kicked off;
- Polymarket 1X2 quotes have complete bid, ask, mid, and spread;
- external fair probability exists from API-Football and The Odds API;
- fair-probability disagreement is not excessive;
- `fair_prob / polymarket_ask >= min_ev`;
- `spread <= max_spread`.

Otherwise mark the row as `observe_only` or `blocked` and preserve the reason.

## Feishu Gate

Sending is allowed only when:

- the local Markdown report and card are non-empty;
- required sections are present;
- data gate is `ok`;
- `FEISHU_POLYMARKET_FOOTBALL_CHAT_ID` or `--chat-id` is present;
- `lark-cli markdown +create/+overwrite` returns a file token or URL;
- `lark-cli im +messages-send` returns a message ID;
- `lark-cli markdown +fetch` and `lark-cli im +messages-mget` confirm the document and message.

The receipt must store local artifact paths, file token or URL, message ID, and readback results.
