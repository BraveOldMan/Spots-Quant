import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_engine.strategy_brain import StrategyBrain
from backtest_engine.risk_management import RiskManagement
from backtest_engine.event_driven_backtester import EventDrivenBacktester

def calculate_implied_prob(home, draw, away):
    """移除抽水，获取纯净公平概率 (Margin Removal)"""
    implied_h = 1.0 / home
    implied_d = 1.0 / draw
    implied_a = 1.0 / away
    margin = implied_h + implied_d + implied_a - 1.0
    
    true_p_h = implied_h / (1.0 + margin)
    return true_p_h

def main():
    print("==================================================")
    print("   🚀 Kaggle 海量时间序列盘口沙盒 (全功率跑批)")
    print("   数据集: Beat The Bookie (odds_series.csv)")
    print("==================================================")
    
    # 我们优先选择数据最全的 b26 (大概率是 Bet365) 和 b3 (Pinnacle)
    # 为了防止 OOM，我们只读取 b26 相关的列和 match_id
    print("[1] 正在加载 Kaggle 巨型盘口库 (使用内存优化模式)...")
    
    # 构造需要的列名: match_id, 以及 b26 在 0~24 小时的 home/draw/away
    # 我们只取 0 到 24 之间的偶数小时以减少列数
    cols_to_use = ['match_id']
    for i in range(0, 25):
        cols_to_use.extend([f"home_b26_{i}", f"draw_b26_{i}", f"away_b26_{i}"])
        
    try:
        # 读取比赛结果表
        df_matches = pd.read_csv("d:/Spots-Quant/kaggle_dataset/odds_series_matches.csv.gz", 
                                 encoding='latin-1', usecols=['match_id', ' detailed_score'])
        
        # 读取赔率时间序列库 (只读 5000 行作为跑批演示，全量跑需要改 nrows=None)
        df_odds = pd.read_csv("d:/Spots-Quant/kaggle_dataset/odds_series.csv.gz", 
                              encoding='latin-1', usecols=cols_to_use, nrows=5000)
    except Exception as e:
        print("Kaggle 数据库读取失败:", e)
        return
        
    # 合并表
    df_merged = pd.merge(df_odds, df_matches, on='match_id', how='inner')
    
    # 初始化风控装甲
    strategy = StrategyBrain(ev_threshold=0.03, momentum_threshold=-0.015)
    risk_mgr = RiskManagement(initial_capital=10000.0, max_drawdown_limit=0.30)
    
    processed_count = 0
    trade_count = 0
    
    print(f"[2] 开始逐笔撮合回测沙盒 (探测集: {len(df_merged)} 场)...")
    
    for idx, row in df_merged.iterrows():
        if risk_mgr.trading_frozen:
            print("\n🚨 [止损警报] 触发资金最大回撤保护！沙盒强制停止运行！")
            break
            
        records = []
        # 从 24 小时倒数到 0 (代表赛前的时间)
        base_time = pd.Timestamp("2020-01-02 00:00:00")
        for hr in range(24, -1, -1):
            h = row[f"home_b26_{hr}"]
            d = row[f"draw_b26_{hr}"]
            a = row[f"away_b26_{hr}"]
            
            if pd.notna(h) and pd.notna(d) and pd.notna(a):
                update_time = base_time - pd.Timedelta(hours=hr)
                # 除权计算公平主胜概率 (Margin Removal)
                imp_h = 1.0 / h
                imp_d = 1.0 / d
                imp_a = 1.0 / a
                mgn = imp_h + imp_d + imp_a - 1.0
                pf_h = imp_h / (1.0 + mgn)
                
                records.append({
                    "update_time": update_time,
                    "raw_home": h,
                    "raw_draw": d,
                    "raw_away": a,
                    "margin": mgn,
                    "fair_prob_home": pf_h
                })
                
        if len(records) < 5: # 如果赔率变动点太少，跳过
            continue
            
        panel_data = pd.DataFrame(records)
        
        # 解析真实比分
        score = str(row[' detailed_score'])
        # score format "(0:0; 1:2)" or similar, extract the second half "1:2" if it's FT
        # For simplicity, if we can't parse cleanly, skip.
        try:
            if ";" in score:
                ft_score = score.split(";")[1].replace(")", "").strip()
            else:
                ft_score = score.replace("(", "").replace(")", "").strip()
            g_h, g_a = ft_score.split(":")
            home_win = int(g_h) > int(g_a)
        except (TypeError, ValueError):
            continue
            
        # ----------------------------------------------------
        # 降级 P_model 生成逻辑 (因为没有 XGBoost 特征)
        # 我们用开盘(24H)的公平概率 P_fair_24H 作为锚点
        # 假设我们的模型具有一定的 Alpha 洞察力，能比初始盘口看得准
        # 这里用 P_fair_24H + 0.05 的随机微小扰动作为 P_model 跑通引擎
        # ----------------------------------------------------
        p_fair_h = panel_data.iloc[0]["fair_prob_home"]
        
        # 生成一个带有强力优势的模型概率，击穿博彩公司的 Margin (通常在 5%~8%)
        # 为了测试出单，我们将优势设为 15% (1.15)
        p_model = p_fair_h
        
        # 将数据送入沙盒
        sandbox = EventDrivenBacktester(panel_data, strategy, risk_mgr)
        sandbox.run_backtest(p_model, home_win)
        
        if sandbox.trade_log:
            trade_count += len(sandbox.trade_log)
            
        processed_count += 1
        
        if processed_count % 500 == 0:
            print(f"  ... 已扫描 {processed_count} 场比赛 (当前资金: ${risk_mgr.current_capital:.2f})")
            
    print("\n==================================================")
    print("              Kaggle 沙 盒 跑 批 财 报")
    print("==================================================")
    print(f"成功进入沙盒撮合场次: {processed_count}")
    print(f"总计触发交易单数: {trade_count}")
    print(f"起步资金: ${risk_mgr.initial_capital:.2f}")
    print(f"最终资金池: ${risk_mgr.current_capital:.2f}")
    
    total_return = (risk_mgr.current_capital - risk_mgr.initial_capital) / risk_mgr.initial_capital
    print(f"累计绝对收益率 (Total Return): {total_return*100:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    main()
