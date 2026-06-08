import time

import pandas as pd
import xgboost as xgb

from polymarket_connector import PolymarketConnector
from asian_handicap_engine import PoissonPricer
from run_ultimate_backtest import RiskManagementV9
from features import FeatureEngine
from api_client import FootballAPIClient
from logger import QuantLogger
from datetime import datetime
from config import (
    DISTILL_FEATURES,
    EV_THRESHOLD,
    INITIAL_CAPITAL,
    MAX_DRAWDOWN_LIMIT,
    MAX_MATCH_EXPOSURE,
)

# 1X2 结果索引与对应的 Polymarket 价格字段
OUTCOMES = [("home", "home_price"), ("draw", "draw_price"), ("away", "away_price")]


class CentralOrchestrator:
    def __init__(self, warmup_file: str = "raw_fixtures.json"):
        self.logger = QuantLogger()
        self.logger.info("=======================================")
        self.logger.info("   V10 中央调度总线 (OMS) 启动中...")
        self.logger.info("=======================================")

        # 1. 挂载分析大脑 (The Brain)
        # 优先用 3 路分类器 (xgboost_model.json) 以支持 主/平/客 全市场决策 (T5.2)；
        # 退化时用蒸馏回归器 (仅主胜)。
        self.logger.info("[1/5] 正在唤醒 XGBoost 分析网络...")
        self.clf = xgb.XGBClassifier()
        try:
            self.clf.load_model("xgboost_model.json")
        except Exception:
            self.logger.warning("未找到 3 路分类器 xgboost_model.json。")
            self.clf = None
        self.reg = xgb.XGBRegressor()
        try:
            self.reg.load_model("xgboost_distilled_model.json")
        except Exception:
            self.reg = None
        if self.clf is None and self.reg is None:
            self.logger.warning("无任何可用模型，大脑离线，将跳过所有下注。")

        # 2. 预热特征工厂 (真实 ELO / 动量)，替代旧版硬编码假特征
        self.logger.info("[2/5] 正在预热 FeatureEngine (真实 ELO/动量)...")
        self.feature_engine = FeatureEngine()
        try:
            self.feature_engine.build_dataset(warmup_file)
            self.logger.info(
                f"    特征库预热完成，已覆盖 {len(self.feature_engine.elo_ratings)} 支球队。"
            )
        except Exception as e:
            self.logger.warning(
                f"特征库预热失败 ({e})，无法计算真实胜率，将跳过所有下注。"
            )
            self.feature_engine = None

        # 3. 挂载泊松降维器
        self.logger.info("[3/5] 正在挂载 PoissonPricer 高维解算器...")
        self.pricer = PoissonPricer()

        # 4. 挂载执行通道 (The Hands)
        # PolymarketConnector 内部自行读取 ALLOW_REAL_MONEY / POLYGON_PRIVATE_KEY / FUNDER_ADDRESS，
        # 默认 DRY RUN 沙盒，缺私钥时自动降级，无需外部注入。
        self.logger.info("[4/5] 正在连接 Polymarket CLOB...")
        self.exchange = PolymarketConnector()

        # 5. 挂载风控与资产管理 (The Shield)
        self.logger.info("[5/5] 正在初始化 Fractional Kelly 资金管理中枢...")
        self.risk_mgr = RiskManagementV9(
            initial_capital=INITIAL_CAPITAL,
            max_drawdown_limit=MAX_DRAWDOWN_LIMIT,
            max_match_exposure=MAX_MATCH_EXPOSURE,
        )
        # T5.3 幂等去重: 记录已下注的 (match_id, outcome)，防止轮询重复下单
        self._placed_bets = set()

    def get_brain_prediction(self, home_id: int, away_id: int):
        """
        基于两队【真实】滚动特征推断 1X2 概率 (p_home, p_draw, p_away)。
        未知球队或大脑离线时返回 None（跳过该场）。不再返回硬编码常数。
        """
        if self.feature_engine is None or (self.clf is None and self.reg is None):
            return None

        eng = self.feature_engine
        if home_id not in eng.elo_ratings or away_id not in eng.elo_ratings:
            return None

        elo_diff = eng.elo_ratings[home_id] - eng.elo_ratings[away_id]
        mom_diff = eng.get_team_momentum(home_id) - eng.get_team_momentum(away_id)
        X = pd.DataFrame(
            [
                {
                    "elo_diff": elo_diff,
                    "mom_diff": mom_diff,
                    "rating_diff": 0.0,
                    "mif_home": 0.0,
                    "mif_away": 0.0,
                }
            ]
        )[DISTILL_FEATURES]

        if self.clf is not None:
            # 分类器 classes_ 升序 [0,1,2] = [客,平,主]
            proba = self.clf.predict_proba(X)[0]
            cls = list(self.clf.classes_)
            p = {c: float(proba[i]) for i, c in enumerate(cls)}
            return (p.get(2, 0.0), p.get(1, 0.0), p.get(0, 0.0))

        # 退化: 仅蒸馏回归器 -> 只给主胜，平/客置 None 由调用方处理
        p_home = min(max(float(self.reg.predict(X)[0]), 0.01), 0.99)
        return (p_home, None, None)

    def fetch_todays_fixtures(self):
        """
        从 API-Football 拉取当日真实赛程（取代旧版写死的 Arsenal/Real Madrid 假赛程）。
        返回 [{id, home_id, away_id, name}]，失败时返回空列表。
        """
        try:
            client = FootballAPIClient()
            today = datetime.now().strftime("%Y-%m-%d")
            res = client.get("/fixtures", {"date": today})
        except Exception as e:
            self.logger.warning(f"无法获取当日赛程 ({e})。")
            return []

        if not res or "response" not in res:
            return []

        fixtures = []
        for f in res["response"]:
            fixtures.append(
                {
                    "id": f["fixture"]["id"],
                    "home_id": f["teams"]["home"]["id"],
                    "away_id": f["teams"]["away"]["id"],
                    "name": f"{f['teams']['home']['name']} vs {f['teams']['away']['name']}",
                }
            )
        return fixtures

    def live_scan_loop(self):
        """全天候实盘高频扫描主循环。"""
        self.logger.info("🟢 系统已进入实盘高频扫描监听模式 (Polling)...")

        todays_fixtures = self.fetch_todays_fixtures()
        if not todays_fixtures:
            self.logger.warning("当日无可用赛程（或 API 不可用），系统进入挂机状态。")
            return

        self.logger.info(f"📡 当日锁定 {len(todays_fixtures)} 场赛事，开始逐场扫描。")

        for fixture in todays_fixtures:
            match_name = fixture["name"]

            if self.risk_mgr.trading_frozen:
                self.logger.alert("资金池回撤已触发熔断，雷达系统强行休眠！")
                return

            # [A] 真实特征 -> 1X2 概率
            preds = self.get_brain_prediction(fixture["home_id"], fixture["away_id"])
            if preds is None:
                self.logger.info(f"--- 跳过 {match_name}：球队不在特征库或大脑离线。")
                continue

            # [B] 市场实时价格 (主/平/客)
            market_odds = self.exchange.fetch_market_odds(match_name)
            self.logger.info(f"\n--- 锁定: {match_name} ---")

            # [C] 三路全市场扫描 (T5.2): 对 主/平/客 分别评估 EV
            for idx, (outcome, price_key) in enumerate(OUTCOMES):
                p_model = preds[idx]
                if p_model is None:  # 退化的回归器只给主胜
                    continue
                if self.risk_mgr.trading_frozen:
                    self.logger.alert("资金池回撤已触发熔断，雷达系统强行休眠！")
                    return

                # T5.3 幂等去重: 同场同结果不重复下注
                bet_key = (fixture["id"], outcome)
                if bet_key in self._placed_bets:
                    continue

                pm_price = market_odds.get(price_key, 0.0)
                if pm_price <= 0:
                    continue
                implied_odds = 1.0 / pm_price  # 1 share 赢 $1, 成本 pm_price
                ev = p_model * implied_odds
                if ev <= EV_THRESHOLD:
                    continue

                # T5.1 仓位含同场敞口上限 (match_id 维度)
                bet_amount = self.risk_mgr.calculate_bet_size(
                    p_model, implied_odds, match_id=fixture["id"]
                )
                if bet_amount <= 0:
                    self.logger.info(
                        f"  [{outcome}] EV={ev:.3f} 命中，但同场敞口已满，跳过。"
                    )
                    continue
                shares_to_buy = bet_amount / pm_price

                self.logger.info(
                    f"🔥 [{outcome}] EV={ev:.3f} | P_model={p_model * 100:.1f}% | 价={pm_price:.3f}"
                )
                self.logger.alert(
                    f"执行买入! {match_name} [{outcome}] {shares_to_buy:.2f} 份。"
                )
                success = self.exchange.place_order(
                    market_token=f"0x{fixture['id']}_{outcome}",
                    side="BUY",
                    size=shares_to_buy,
                    price=pm_price,
                )
                if success:
                    self.risk_mgr.register_exposure(fixture["id"], bet_amount)
                    self._placed_bets.add(bet_key)
                    self.risk_mgr.trades += 1
                    self.logger.info(
                        f"✅ 已提交。累计出单 {self.risk_mgr.trades} 笔。资金池 ${self.risk_mgr.current_capital:,.2f}"
                    )

            time.sleep(1)  # 防限频

        self.logger.info("\n今日赛程扫尾完毕。系统进入挂机状态...")


if __name__ == "__main__":
    oms = CentralOrchestrator()
    oms.live_scan_loop()
