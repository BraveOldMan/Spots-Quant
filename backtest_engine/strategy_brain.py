import pandas as pd

class StrategyBrain:
    """
    模块 2：双因子信号引擎 (Strategy Brain)
    核心发单引擎，基于 "绝对价值做底 + 微观变盘动量共振确认" (A+B 双因子模型) 生成交易信号。
    """
    def __init__(self, ev_threshold: float = 0.05, momentum_threshold: float = -0.02):
        self.ev_threshold = ev_threshold
        self.momentum_threshold = momentum_threshold

    def calculate_alpha1_value(self, p_model: float, current_odds: float, p_fair: float) -> dict:
        """
        Alpha 1: 价值剥离因子
        计算期望收益 (EV)。
        """
        ev = (p_model * current_odds) - 1.0
        
        # 偏离度检查：如果博彩公司公允概率与模型概率差距超过 30%，可能是信息不对称
        extreme_deviation = False
        if abs(p_model - p_fair) > 0.30:
            extreme_deviation = True
            
        is_triggered = (ev > self.ev_threshold) and not extreme_deviation
        
        return {
            "ev": ev,
            "extreme_deviation": extreme_deviation,
            "alpha1_triggered": is_triggered
        }

    def calculate_alpha2_momentum(self, odds_series: pd.Series) -> dict:
        """
        Alpha 2: 盘口动量因子
        提取开赛前 24H 至 1H 的赔率一阶导数（变化斜率）与累计变盘幅度。
        odds_series: 赔率的时间序列，索引为距离开赛的小时数（负数），如 -24H 到 -1H
        """
        if len(odds_series) < 2:
            return {"slope": 0.0, "total_drop": 0.0, "alpha2_triggered": False, "trap_detected": False}
            
        # 简单一阶导数：末位值减去首位值 (也可以用线性回归斜率)
        start_odds = odds_series.iloc[0]
        end_odds = odds_series.iloc[-1]
        
        total_drop = end_odds - start_odds
        # 如果赔率阶梯式下降，代表资金涌入，total_drop 为负数
        
        is_triggered = total_drop < self.momentum_threshold
        
        # 量价背离识别（诱多检测）：赔率不降反升（遭遇抛售）
        trap_detected = total_drop > abs(self.momentum_threshold)
        
        return {
            "slope": total_drop,
            "total_drop": total_drop,
            "alpha2_triggered": is_triggered,
            "trap_detected": trap_detected
        }

    def generate_signal(self, selection: str, p_model: float, odds_series: pd.Series, p_fair: float) -> dict:
        """
        共振决策树 (发单)
        评估一个特定的选项（例如主胜 "Home"），生成交易信号。
        """
        if odds_series.empty:
            return {"signal": "PASS", "reason": "No odds data"}
            
        current_odds = odds_series.iloc[-1]
        
        alpha1 = self.calculate_alpha1_value(p_model, current_odds, p_fair)
        alpha2 = self.calculate_alpha2_momentum(odds_series)
        
        # 1. 分歧过滤 (防诱多) 熔断机制
        if alpha1["alpha1_triggered"] and alpha2["trap_detected"]:
            return {
                "signal": "ABORT_TRAP",
                "reason": f"[阻断] 价值因子EV={alpha1['ev']:.3f}极高, 但动量因子斜率={alpha2['slope']:.3f}显示赛前遭遇抛售攀升，判定为主力资金/博彩公司诱导，触发硬性熔断，放弃交易。",
                "ev": alpha1["ev"],
                "momentum": alpha2["slope"]
            }
            
        # 2. 强多头共振
        if alpha1["alpha1_triggered"] and alpha2["alpha2_triggered"]:
            return {
                "signal": "LONG",
                "reason": f"[共振] Alpha 1 (EV={alpha1['ev']:.3f}) 与 Alpha 2 (资金同向涌入, Drop={alpha2['total_drop']:.3f}) 强烈共振，允许开仓。",
                "ev": alpha1["ev"],
                "momentum": alpha2["slope"]
            }
            
        # 3. 不满足条件
        return {
            "signal": "PASS",
            "reason": f"未满足双因子共振。Alpha 1: {alpha1['alpha1_triggered']}, Alpha 2: {alpha2['alpha2_triggered']}",
            "ev": alpha1["ev"],
            "momentum": alpha2["slope"]
        }

if __name__ == "__main__":
    print("[Module 2] StrategyBrain 基础类测试初始化...")
    brain = StrategyBrain(ev_threshold=0.05, momentum_threshold=-0.1)
    
    # 模拟数据
    # 假设预测胜率 60% (P_model = 0.6)
    # 赔率序列 (开赛前24h到1h)，赔率从 2.10 降到了 1.90 (资金涌入)
    sim_odds_series_good = pd.Series([2.10, 2.05, 1.98, 1.90], index=[-24, -12, -6, -1])
    # 真实市场公允概率 (通过赔率推导)
    p_fair = 0.52 
    
    print("\n测试用例 1：强多头共振 (赔率下降，资金涌入)")
    res1 = brain.generate_signal("Home", 0.60, sim_odds_series_good, p_fair)
    print(res1["signal"], "->", res1["reason"])
    
    print("\n测试用例 2：诱多陷阱过滤 (价值高，但赔率攀升)")
    # 赔率序列从 2.10 升到了 2.30 (资金抛售)，但模型依然觉得价值很高
    sim_odds_series_trap = pd.Series([2.10, 2.15, 2.25, 2.30], index=[-24, -12, -6, -1])
    res2 = brain.generate_signal("Home", 0.60, sim_odds_series_trap, p_fair)
    print(res2["signal"], "->", res2["reason"])
