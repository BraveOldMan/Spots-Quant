import json
import time
import pandas as pd
import xgboost as xgb

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import FeatureEngine
from backtest_engine.odds_pipeline import OddsDataPipeline
from backtest_engine.strategy_brain import StrategyBrain
from backtest_engine.risk_management import RiskManagement
from backtest_engine.event_driven_backtester import EventDrivenBacktester

def main():
    print("==================================================")
    print("   V6 AI 量化引擎 联合 盘口回测沙盒 (全量跑批)")
    print("   执行模式: 严谨务实派 (仅采用真实可查的历史盘口)")
    print("==================================================")
    
    # 1. 预热特征库
    print("[1] 正在加载并预热 V6 基础特征库...")
    engine = FeatureEngine()
    try:
        engine.build_dataset("raw_fixtures_v5.json")
    except Exception as e:
        print(f"Error warming up FeatureEngine: {e}")
        return
        
    # 2. 载入预测大脑
    print("[2] 正在唤醒 XGBoost Meta-Learner...")
    model = xgb.XGBClassifier()
    try:
        model.load_model("xgboost_model.json")
    except Exception as e:
        print(f"Error loading XGBoost model: {e}")
        return

    # 3. 初始化回测组件
    pipeline = OddsDataPipeline()
    strategy = StrategyBrain(ev_threshold=0.05, momentum_threshold=-0.02)
    risk_mgr = RiskManagement(initial_capital=10000.0, max_drawdown_limit=0.30)
    
    # 4. 加载原始历史比赛库
    try:
        with open("raw_fixtures_v5.json", "r", encoding="utf-8") as f:
            fixtures = json.load(f)
    except Exception as e:
        print("未找到比赛数据库:", e)
        return
        
    # 按时间戳排序
    fixtures.sort(key=lambda x: x["fixture"]["timestamp"])
    
    # 为了防止 API 限流被封锁，我们挑选最近的 50 场比赛进行真实 API 查询探测
    recent_fixtures = fixtures[-50:]
    
    processed_count = 0
    skipped_count = 0
    
    print(f"\n[3] 抽取最近 {len(recent_fixtures)} 场赛事开启探针扫库...")
    
    # 用来聚合全部交易记录
    master_trade_log = []
    
    for match in recent_fixtures:
        # 如果触发了回撤硬约束拔线，直接中断全量回测
        if risk_mgr.trading_frozen:
            print("\n🚨 [止损警报] 触发资金最大回撤保护！沙盒强制停止运行！")
            break
            
        fixture_id = match["fixture"]["id"]
        home_name = match["teams"]["home"]["name"]
        away_name = match["teams"]["away"]["name"]
        
        # 获取基础比分与判定
        goals_h = match["goals"].get("home")
        goals_a = match["goals"].get("away")
        if goals_h is None or goals_a is None:
            continue
        home_win = int(goals_h) > int(goals_a)
        
        # ----- 特征工程推断真实胜率 P_model -----
        home_id = match["teams"]["home"]["id"]
        away_id = match["teams"]["away"]["id"]
        
        elo_h = engine.elo_ratings.get(home_id, 1500)
        elo_a = engine.elo_ratings.get(away_id, 1500)
        elo_diff = elo_h - elo_a
        mom_diff = engine.get_team_momentum(home_id) - engine.get_team_momentum(away_id)
        
        rating_diff = 0.0 # 简化：由于没有获取当时首发球员的列表，忽略 rating_diff
        mif_home = 0.0
        mif_away = 0.0
        
        X_test = pd.DataFrame([{
            'elo_diff': elo_diff, 'mom_diff': mom_diff, 'rating_diff': rating_diff,
            'mif_home': mif_home, 'mif_away': mif_away
        }])
        
        probs = model.predict_proba(X_test)[0]
        p_model = float(probs[2]) # home win probability
        
        # ----- 拉取历史盘口并清洗 -----
        api_res = pipeline.fetch_odds_history(fixture_id)
        raw_df = pipeline.parse_api_response_to_dataframe(api_res)
        panel_data = pipeline.resample_time_series(raw_df, freq='1h')
        
        if panel_data.empty:
            # 记录历史数据缺失
            skipped_count += 1
            time.sleep(0.3) # API 限速
            continue
            
        print(f"\n[{fixture_id}] {home_name} vs {away_name} -> API 数据命中！启动逐笔撮合...")
        processed_count += 1
        
        # 实例化单场比赛沙盒
        sandbox = EventDrivenBacktester(panel_data, strategy, risk_mgr)
        sandbox.run_backtest(p_model, home_win)
        
        if sandbox.trade_log:
            master_trade_log.extend(sandbox.trade_log)
            
        time.sleep(0.3)
        
    print("\n==================================================")
    print("              量 化 跑 批 财 报")
    print("==================================================")
    print(f"探针扫描总场次: {len(recent_fixtures)}")
    print(f"历史赔率缺失场次 (安全跳过): {skipped_count}")
    print(f"成功进入沙盒撮合场次: {processed_count}")
    print(f"最终资金池: ${risk_mgr.current_capital:.2f} (起步 ${risk_mgr.initial_capital:.2f})")
    
    total_return = (risk_mgr.current_capital - risk_mgr.initial_capital) / risk_mgr.initial_capital
    print(f"累计绝对收益率 (Total Return): {total_return*100:.2f}%")
    print("最大回撤断电控制: 已开启 (30%)")
    print("==================================================")
    
if __name__ == "__main__":
    main()
