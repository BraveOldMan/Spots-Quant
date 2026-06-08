---
description: "Reproducible, no-lookahead quantitative research system for football 1X2 markets and Polymarket workflows."
---

# Spots-Quant

Spots-Quant is a quantitative research system for football 1X2 markets and
Polymarket research workflows. The project is built around reproducible,
no-lookahead backtesting and fail-closed research gates, not around marketing a
high-return story.

This repository is useful for:

- walk-forward backtesting and robustness audits for football 1X2 markets;
- opening odds, closing odds, CLV, calibration, and out-of-sample research;
- quality isolation for API-Football, The Odds API, Kaggle odds-series, and
  Betfair data;
- pre-match Polymarket football research reports and read-only market coverage
  checks;
- dry-run-first validation of trading-system safety boundaries.

This repository is not intended for:

- direct live trading;
- copying unverified betting recommendations;
- using historical equity curves as proof of production readiness;
- forcing trade signals when opening odds, target markets, or result-matching
  evidence is missing.

## Current Status

The default strategy remains observe-only. The default model is not upgraded,
and model artifacts are not overwritten.

Latest local verification:

| Item | Current result |
| --- | --- |
| Main backtest entrypoint | `run_ultimate_backtest.py` |
| Default model | `WalkForwardXGBoostModel` |
| Test command | `python -m pytest -q` |
| Test result | `131 passed` |
| Static check | `ruff check .` passed |
| Total matches | `1140` |
| Trades | `475` |
| Total return | `-0.09275522180421103` |
| Max drawdown | `0.1507934537643345` |
| Final capital | `9072.44778195789` |
| Per-bet Sharpe | `-0.023493880225375154` |
| Brier | `0.21753360665677235` |
| CLV mean | `0.0` |
| Beat close | `0.0` |

Interpretation:

- These metrics are the current default baseline, not a return recommendation.
- `per_bet_sharpe` is negative, so the default strategy has not proven it can
  consistently beat the market.
- `CLV mean=0.0` and `beat_close=0.0` do not support upgrading the default
  strategy.
- A research candidate is only allowed to move into further observation after
  passing CLV, beat-close, ROI confidence-interval, and per-season robustness
  gates.

## Core Principles

### No Lookahead

Signals, factors, model updates, bet selection, and equity curves must follow
this order:

```text
predict -> execution/settlement -> update
```

A betting decision must not read the match result, closing-only information, or
post-match technical statistics.

### Honest Backtesting

Backtests must separate:

- model probability;
- market odds;
- real match result;
- opening-bet execution mode;
- closing-line value evaluation;
- transaction costs and execution pressure.

Do not replace real market data with model-generated odds or random settlement.

### Fail Closed

The system must block instead of continuing with empty tables, default values,
or fabricated data when any of the following happens:

- data source failure;
- missing critical fields;
- trading date or match date mismatch;
- empty backtest window;
- missing target-date Polymarket football 1X2 markets;
- missing odds or ambiguous matching;
- failed Feishu/Lark document or message readback.

### Dry Run First

`ALLOW_REAL_MONEY=0` is the default. Even when Polymarket credentials are
configured, dry-run orders, simulated orders, and constructed orders must not be
described as real fills.

## Repository Layout

```text
.
|-- run_ultimate_backtest.py        # Main default backtest entrypoint
|-- research_strategy_audit.py      # Research grid, segment attribution, candidate gates
|-- api_backtest_dataset.py         # API/Kaggle/Betfair standardization and quality isolation
|-- config.py                       # Typed settings and legacy constants
|-- quant_core.py                   # Odds, Kelly, CLV, calibration, Dixon-Coles utilities
|-- wf_models.py                    # Walk-forward models
|-- alpha_model.py                  # Alpha models and stacking research
|-- backtest_engine/                # Event-driven backtester, odds pipeline, risk, strategy brain
|-- polymarket_*                    # Polymarket Gamma/Data/CLOB/Relayer connectors
|-- api_football_today_daemon.py    # API-Football today-data daemon
|-- .agents/skills/                 # Codex automation skills, including Polymarket football reports
|-- tests/                          # pytest tests
|-- reports/                        # Local report outputs, not recommended for public commits
|-- data_seasons/                   # Local season data, not recommended for public commits
`-- data_standardized/              # Standardized backtest data, not recommended for public commits
```

For a public repository, prefer committing source code, tests, documentation,
configuration templates, and required scripts only. Do not commit `.env`,
databases, historical logs, model artifacts, large CSV/JSON files, `reports/`
outputs, or cache directories.

## Requirements

Recommended environment:

- Python 3.11;
- Windows PowerShell or any shell that can run Python;
- dependencies listed in `requirements.txt`;
- optional `ruff`, only if it is already installed in the environment.

Create an environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you need API-Football, The Odds API, Polymarket, or Feishu/Lark delivery,
copy the environment template:

```powershell
Copy-Item .env.example .env
```

Fill only the values you actually need. Never commit `.env`.

## Configuration

Core configuration lives in `config.py` and `.env.example`.

Common variables:

| Variable | Description |
| --- | --- |
| `API_FOOTBALL_KEY` | API-Football key, optional |
| `THE_ODDS_API_KEY` | The Odds API key, optional |
| `SPOTS_BACKTEST_CSV_PATHS` | Default backtest data path list |
| `SPOTS_ODDS_MODE` | `opening` or `closing` |
| `SPOTS_EV_THRESHOLD` | EV threshold |
| `SPOTS_INITIAL_CAPITAL` | Initial capital |
| `SPOTS_KELLY_MULT` | Kelly scaling factor |
| `SPOTS_MAX_BET_FRACTION` | Maximum bankroll fraction per bet |
| `SPOTS_MAX_MATCH_EXPOSURE` | Maximum exposure per match |
| `SPOTS_MAX_DRAWDOWN_LIMIT` | Drawdown stop limit |
| `ALLOW_REAL_MONEY` | Defaults to `0`; real-money paths are disabled |
| `POLYMARKET_*` | Polymarket read-only or dry-run connection settings |
| `FEISHU_POLYMARKET_FOOTBALL_CHAT_ID` | Feishu/Lark chat target for delivery automation |

Secrets must only be read from environment variables, `.env`, or existing safe
configuration files. Logs and reports must not expose full secrets.

## Quick Start

Run the default backtest:

```powershell
python run_ultimate_backtest.py
```

Print the default baseline metrics:

```powershell
@'
from run_ultimate_backtest import run_real_backtest

m = run_real_backtest(verbose=False)
for k in [
    "total_matches",
    "trades",
    "total_return",
    "max_drawdown",
    "final_capital",
    "per_bet_sharpe",
    "brier",
    "clv_mean",
    "beat_close",
]:
    print(f"{k}={m[k]!r}")
'@ | python -
```

Run the default backtest audit and write outputs to `reports/`:

```powershell
@'
from run_ultimate_backtest import run_backtest_audit

result = run_backtest_audit("reports")
print(result.metrics["trades"])
print(result.metrics["per_bet_sharpe"])
'@ | python -
```

Run the full strategy research audit:

```powershell
python research_strategy_audit.py
```

The full research audit can take longer. Run it when research logic, data
definitions, candidate gates, or report logic changes.

## Tests and Quality Checks

Unit tests:

```powershell
python -m pytest -q
```

Static check:

```powershell
ruff check .
```

AST syntax check:

```powershell
@'
import ast
from pathlib import Path

errors = []
checked = 0
for path in sorted(Path(".").rglob("*.py")):
    if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
        continue
    checked += 1
    try:
        ast.parse(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            errors.append((str(path), repr(exc)))
    except Exception as exc:
        errors.append((str(path), repr(exc)))

print(f"checked={checked}")
if errors:
    for path, err in errors:
        print(path, err)
    raise SystemExit(1)
print("syntax_ok")
'@ | python -
```

Secret long-value scan before public release:

```powershell
rg -n --hidden --glob '!.env' --glob '!reports/*.csv' --glob '!reports/*.md' '(API_FOOTBALL_KEY\s*=\s*[''"]?[A-Za-z0-9_\-]{20,}|THE_ODDS_API_KEY\s*=\s*[''"]?[A-Za-z0-9_\-]{20,}|api_key\s*=\s*[''"][A-Za-z0-9_\-]{20,})' .
```

## Data and Artifact Boundaries

The following should not be committed by default:

- `.env`;
- `*.db`;
- `*.csv`;
- `*.json`;
- `*.pth`;
- `logs/`;
- `reports/`;
- `runtime/`;
- `.tmp/`;
- `data_seasons/`;
- `data_standardized/`;
- `kaggle_dataset/`;
- `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`.

Reasons:

- datasets, databases, and model artifacts are often large;
- some files may contain runtime state, third-party API responses, message
  receipts, or private local paths;
- a public repository should keep source code, tests, and configuration
  templates easy to audit.

If data must be published, document the source, license, generation command, and
hash separately. Do not mix large data artifacts into source commits.

## Research Workflow

Recommended workflow:

1. Define the hypothesis: where the signal comes from, when it is available,
   and whether it can read future information.
2. Add the smallest reproducible test or audit command.
3. Run the default main backtest and confirm the baseline did not degrade.
4. Run per-season robustness and candidate gates.
5. Check CLV, beat close, ROI confidence intervals, and max drawdown.
6. Keep the result as `observe_only` when candidates are insufficient.

Candidate gates should include at least:

- `clv_mean > 0`;
- `beat_close > 0.50`;
- `roi_ci_low > 0`;
- no failed per-season robustness checks.

When there is no candidate, reports and documentation must clearly state that
the default strategy is not upgraded.

## Polymarket Football Daily Report

The repository includes a Codex skill:

```text
.agents/skills/polymarket-football-daily-report/
```

Main entrypoint:

```powershell
$targetDate = (Get-Date).ToString("yyyy-MM-dd")
python .agents\skills\polymarket-football-daily-report\scripts\run_polymarket_football_daily_report.py --date $targetDate --no-send --max-api-requests 250 --output-dir reports
```

A delivery run must satisfy all of the following:

- target-date API-Football fixtures are available;
- target-date Polymarket football 1X2 markets are available;
- external odds coverage is sufficient;
- matching is unambiguous;
- local report generation succeeds;
- Feishu/Lark document creation or overwrite succeeds;
- Feishu/Lark group message sending succeeds;
- document and message readbacks both pass.

If any gate fails, keep the local blocked-state report and diagnostics. Do not
send an empty report, stale report, or fabricated report.

## Live-Trading Safety

This repository defaults to research, backtesting, paper trading, and dry-run
paths only.

Strictly prohibited:

- executing real-money order commands;
- describing dry-run results as real fills;
- allowing random prices, empty tables, or default values into live decisions;
- continuing a live path without credentials, market tokens, or order books;
- uploading code, strategy files, or backtest logs to unknown external servers.

If real trading is added in the future, it must be manually confirmed by a human
and preceded by security review, permission isolation, and minimal-capital
validation.

## GitHub Release Guidance

Before publishing, prefer a clean copy instead of pushing local Git history that
contains historical databases or model artifacts.

Recommended public release set:

- Python source files;
- `backtest_engine/`;
- source and docs under `.agents/skills/`;
- `tests/`;
- `.env.example` and `.env.template`;
- `pyproject.toml`;
- `requirements.txt`;
- `README.md`;
- `AGENTS.md` or project collaboration notes;
- other text-only documentation.

Not recommended for public release:

- `.env`;
- `spots_quant.db`, `api_cache.db`, `bet_history.db`;
- `xgboost_model.json`, `xgboost_distilled_model.json`, `*.pth`;
- `reports/` runtime outputs;
- Feishu/Lark receipts or message state;
- large historical data, raw API snapshots, and caches.

## Contributing

Before committing, confirm:

- no plaintext secrets were added;
- `.env` or real credentials were not committed;
- large data artifacts were not committed;
- `python -m pytest -q` passes;
- `ruff check .` passes, or the reason it cannot run is documented;
- strategy-logic changes do not degrade the default main backtest baseline;
- research-logic changes keep report row counts, candidate counts, and gate
  conclusions consistent.

If a strategy-related change lowers `per_bet_sharpe` below the current baseline,
roll it back or keep it as `observe_only`. Do not upgrade the default strategy.

## Disclaimer

This project is for quantitative research, software engineering validation, and
education only. It is not investment advice, trading advice, or betting advice.
Sports events and prediction markets are high risk, and historical backtests do
not imply future performance. Any real-money action is the user's sole
responsibility and must comply with applicable laws, regulations, and platform
rules.

## License

This repository does not currently include a `LICENSE` file. Add a license file
before public release according to the intended licensing terms. Do not claim
MIT, Apache-2.0, or any other open-source license without the corresponding
license file.
