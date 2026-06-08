# Changelog & Updates

## [v13.0.0] - 2026-06-06 (Honest Quant Era)

### 去伪存真 (方法论修复)
- **废弃造假回测**: 旧版回测用"真实概率+噪声"人造对手盘再随机结算，必然盈利（"近千万倍/夏普21"系伪造）。重写为无前视真实回测：模型概率 / 市场赔率 / 真实赛果三者独立。
- **真实知识蒸馏**: `train_xgboost_distillation.py` 改以真实关盘隐含概率为目标，并修复 `train_test_split` 时序泄漏（改时序切分）。
- **修复主场优势逻辑反转** (`models.py`)、**Betfair 概率列错位** (`robust_extractor.py` 语义映射，重生成数据)、**比分判空** (`features.py`)。

### 度量正确性 (T0)
- 新增 `analyze_clv.py`：CLV（关盘价值）技艺诊断。
- 回测新增 Brier / LogLoss / 可靠性曲线（校准）。

### Alpha 与模型 (T1/T2)
- `alpha_model.py`：学真实赛果 + 休息天数/赛程密度/动量 + Dixon-Coles stacking。
- `wf_models.py`：Dixon-Coles walk-forward（主客分离 + 贝叶斯收缩 + proxy-xG + rho）。
- 多赛季回测中 CLV 由负转正（Dixon-Coles WF +1.76%），但尚未统计显著——**不可据此实盘加杠杆**。

### 回测严谨 (T3)
- 多赛季支持（`data_seasons/` 6 季约 2280 场）、交易成本、Sortino/Calmar/利润因子/ROI 自助置信区间。

### 工程质量 (T4)
- `quant_core.py` 公共数学库（DRY）、`config.py` 参数中心、API 缓存 TTL、依赖版本锁定。
- `tests/` 38 项 pytest 单测（核心数学覆盖率 94%）；修复 PoissonPricer 高 λ 未归一化、Sortino 退化、`.gitignore` 乱码等缺陷。

### 风控 / 实盘 (T5)
- 同场敞口上限（相关性防过度下注）、三路全市场决策、幂等去重、真实订单簿（best ask）接入。

## [v5.0.0] - 2026-06-06 (Pro API Era)

### Deep Data Integration & Proxy-xG
- **Massive Historical Fetch**: Upgraded `fetch_data.py` to pull matches dating back to 2018 across all major continental tournaments, fully leveraging unlimited API requests.
- **Proxy-xG (Expected Goals) Blending**: Modified `models.py` to parse match statistics. It now synthesizes a Proxy-xG metric based on Shots on Goal and Off Goal, and blends it 40/60 with actual goals. This drastically neutralizes luck variance (finishing/goalkeeping anomalies) when training the Dixon-Coles MLE model.

### Squad Degradation & Anti-Rotation Radar
- **Player Caps Graphing**: `value_scanner.py` now parses historical lineups to dynamically build a local SQLite-like dictionary of every player's international caps.
- **B-Team Penalty Engine**: During live scanning, the engine fetches the starting XI for upcoming matches. If the team's average player caps fall below the veteran threshold (e.g., < 3.0 caps), a 30% `B-Team Degradation` penalty is automatically applied to their Attack and Defense strengths, preventing the scanner from falling into bookmaker rotation traps.

## [v4.0.0] - 2026-06-06

### Mathematical Engine V4
- **Time-Decay Factor (Half-Life)**: Introduced a mathematical half-life of 365 days into both the ELO K-factor equations and the Dixon-Coles MLE log-likelihood minimization. The model now exponentially penalizes matches played far in the past, adapting extremely fast to current form and squad aging.

### Funding & Execution Architecture
- **Kelly Criterion Integration**: Re-engineered `value_scanner.py` to calculate fractional Kelly Criterion (Quarter-Kelly) position sizing. The scanner now protects the bankroll mathematically from ruin by suggesting exact capital allocation percentages for positive EV bets.
- **Closing Line Value (CLV) Database**: Created `bet_history.db` using `sqlite3`. The scanner now automatically logs every discovered positive EV bet alongside the model's probabilities, bookie odds, EV, and Kelly size. This allows for rigorous retroactive tracking of our CLV beating performance.

## [v3.0.0] - 2026-06-06

### Mathematical Core Overhaul
- **Dixon-Coles MLE Implementation**: Entirely refactored the Poisson expected goals logic in `models.py`. Introduced `scipy.optimize` L-BFGS-B to fit Attack Strength, Defense Vulnerability, Rho ($\rho$ draw inflation), and Gamma ($\gamma$ home advantage) via Maximum Likelihood Estimation over historical data.
- **Strength of Schedule**: The model now heavily punishes teams that score inflated goals against weak opponents (e.g., friendlies) and rewards goals scored against elite defenses.
- **Home Advantage Filtering**: Introduced a structural Home Advantage modifier in the ELO algorithm and MLE regression, aggressively filtering out the "Friendly Match Home Field" anomaly.

### Simulation Topology Upgrades
- **Group Stage Mechanics**: Rebuilt `simulator.py` to perfectly mirror World Cup topology:
  - 32-team snake/pot seeding (simulated randomly for Monte Carlo).
  - 8 Groups of 4 (Round-robin point calculation with tiebreakers).
  - Knockout Stage feeding from Group runner-ups and winners.

### Data Expansion Ready
- Added triggers for **Asian Cup (League 7)** and **CONCACAF Gold Cup (League 22)** to `fetch_data.py` to fix intercontinental data voids (Pending API quota reset).

## [v2.0.0] - 2026-06-06
- Added Value Bet Scanner (`value_scanner.py`) and live Odds Fetcher (`fetch_odds.py`) with Margin Removal algorithm.

## [v1.0.0] - 2026-06-06
- Initial commit. Base ELO ranking and Poisson model.
