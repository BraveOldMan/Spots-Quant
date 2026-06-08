# 🏆 Spots-Quant V12 终极操作说明书 (The Holy Grail)

## 1. 系统概述 (System Overview)

**Spots-Quant** 是一套专为 Web3 去中心化预测市场（以 Polymarket 为主）量身定制的**全栈自主、高频多智能体量化交易系统**。
经过 12 个大版本的进化，系统已经从最初的爬虫脚本，蜕变为一座包含“知识蒸馏模型”、“分数凯利风控”、“九大智能体协同网络”以及“L2 链上物理直连”的工业级数字堡垒。

**核心目标**：彻底摒弃人类的主观情绪赌博，依靠纯粹的数学概率、计算力与异步并发，在预测市场中进行无情的套利剥头皮。

---

## 2. 核心流派与数学原论 (Core Mathematics)

本系统采用双轨制利润引擎：**单向价值投资 (Value Arb)** 与 **双向统计套利 (Stat Arb)**。

### 2.1 价值投资：XGBoost 知识蒸馏与期望值套利
*   **知识蒸馏**：我们不使用模型来预测“胜负分类”，而是让 `XGBRegressor` 学习必发 (Betfair) 5.5 万场历史赛事的关盘赔率（这个地球上最准的赔率）。模型输出的是绝对客观的真实胜率（$P_{true}$）。
*   **期望值计算 (EV)**：当预测市场的市价 (Market Odds) 开出错误定价时，如果 $EV = P_{true} \times MarketOdds > 1.05$，系统便判定存在价值洼地。
*   **泊松降维**：通过 L-BFGS-B 非线性优化泊松分布，系统能把单一的胜负概率，降维投射到成百上千种复杂的亚盘 (Asian Handicap) 和大小球盘口中进行无死角覆盖。

### 2.2 统计套利：Z-Score 均值回归
*   系统使用 `statsmodels` 对历史赔率序列进行 **ADF 协整检验**。
*   当发现具备强关联的资产对（例如某场比赛的买卖盘口，或两支高度关联球队的夺冠赔率），系统会实时计算价差的 **Z-Score**。
*   如果散户情绪导致 $Z > 2.0$（超过两个标准差），系统会触发**配对交易**：做空被高估标的，做多被低估标的。等待情绪消散回归均值时，双边平仓锁润。

### 2.3 风控装甲：Fractional Kelly
放弃激进的全仓凯利公式。系统采用 **1/20 分数凯利** 配合 **波动率惩罚项**。在胜率不足 30% 的深水区，依靠盈亏比和极致的仓位切分，将系统最大历史回撤死死压制在 11% 以下。

---

## 3. 架构与特工矩阵 (Multi-Agent System)

在 V11/V12 架构中，线性代码被废弃。系统的中枢神经被替换为基于 `asyncio` 的 **分布式异步事件总线 (Event Bus)**。

### 🧬 九大智能体 (Agents)
1. **🕵️ 情报特工 (Sensor Agent)**：全天候扫描全球赛事网络，提取特征，在总线上广播 `[MATCH_FOUND]`。
2. **🧮 量化特工 (Alpha Agent)**：数学大脑。接收情报后，调用 XGBoost 进行毫秒级解算。一旦发现 EV 漏洞，提交 `[TRADE_PROPOSAL]`。
3. **⚖️ 风控法庭 (Critique Agent)**：冷酷无情的审查官。任何开火提案必须通过它的凯利公式风控与资金池回撤计算。
4. **🥷 执行特工 (Execution Agent)**：系统的物理双手。调用 Polymarket 官方 SDK 构建区块链订单，并隐蔽挂入暗池。
5. **📰 舆情特工 (LLM Agent)**：利用 GPT-4 并行阅读推特等非结构化文本，若发现“核心受伤”等突发利空，下发熔断指令。
6. **🐋 巨鲸特工 (Whale Tracker)**：监听 Polygon 链上大户地址，若发现巨鲸逆向砸盘，立刻封锁交易通道避免成为接盘侠。
7. **💱 对冲特工 (Arbitrage Agent)**：对比外部博彩公司（如 Pinnacle）。若发现无风险价差，直接截胡原提案，将其转变为双边锁仓对冲单。
8. **🌊 做市特工 (Market Maker)**：无需预测比赛，长期在边缘买卖盘口双边挂单，吃取散户点差与滑点。
9. **🧲 统计特工 (StatArb Agent)**：负责上述 2.2 章节的 Z-Score 计算，寻找均值回归标的。

---

## 4. 部署与操作手册 (Deployment & Operations)

### 4.1 环境准备
系统依赖 `py-clob-client` (Polymarket 官方 SDK), `xgboost`, `statsmodels`, `sqlite3` 等。请确保 `requirements.txt` 已满足。

### 4.2 数据库初始化
运行前请务必确认 `spots_quant.db` 存在。如果数据尚未装载，请运行：
```bash
python fetch_missing_history.py  # 补充历史基建
python fetch_daily_omni.py       # 获取当日全景比赛
```

### 4.3 实盘与沙盒切换 (关键安全点)
为了防止私钥被盗用或发生灾难性亏损，本系统引入了极度严格的物理隔离机制。
请在根目录下创建 `.env` 文件：

```env
# =============== 核心权限开关 ===============
# 0 = 沙盒模拟模式 (DRY RUN)，只打印日志，不发请求，无资金风险
# 1 = 真实主网实盘模式 (LIVE TRADING)，将动用真金白银！
ALLOW_REAL_MONEY=0

# =============== 钱包配置 ===============
# 您用于签名的 Polygon 私钥 (切勿外泄)
POLYGON_PRIVATE_KEY="0x_your_private_key"
# 您在 Polymarket 绑定的公钥 / 代理合约地址
FUNDER_ADDRESS="0x_your_public_address"

# =============== 网络配置 ===============
POLYMARKET_HOST="https://clob.polymarket.com"
POLYMARKET_CHAIN_ID=137
```

> **注意**：如果不创建 `.env` 文件或 `ALLOW_REAL_MONEY=0`，执行特工在最后一步发单时会被拦截，仅在控制台打印模拟成功日志。

### 4.4 启动联合沙盘推演
要启动九大特工的实况博弈推演，请执行：
```bash
python multi_agent_framework.py
```
您将会在终端看到不同 Emoji 标识的特工互相发送警报、提案拦截与资金签发的全过程。

### 4.5 运行终极蒙特卡洛回测
要检验底层 XGBoost 的基础能力以及风控模型的抗压上限，请运行：
```bash
python run_ultimate_backtest.py
```

---

## 5. 日常维护指引
1. **模型迭代**：随着时间推移，庄家开盘手法会发生变化。建议每隔 3 个月，重新抓取新的赔率数据灌入 `train_xgboost_distillation.py` 重新蒸馏生成新的 `v12_xgboost_model.pkl`。
2. **防诱多陷阱**：在 Polymarket 这种流动性容易干涸的市场，执行单笔大额订单极易产生巨大滑点。在实盘使用大资金时，必须在 `execution_agent` 中加入冰山算法 (Iceberg Orders)，将单笔大单切碎为 10~50 份连续抛出。
3. **安全红线**：严禁将带有私钥的 `.env` 上传至 GitHub！如果产生内存不足 (OOM)，请检查是否意外禁用了 SQLite 转而使用了 JSON 在内存中解压数据。

**System Status: All Green. Happy Hunting! 🏆**
