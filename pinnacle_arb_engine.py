"""
Pinnacle Arbitrage Engine (平博套利数学核心)
处理跨平台的 Value Betting (价值下注) 与 Surebet (无风险双边套利)
"""

class PinnacleArbitrageEngine:
    def __init__(self, ev_threshold: float = 1.05):
        self.ev_threshold = ev_threshold

    def calculate_true_probabilities(self, pinny_h: float, pinny_d: float, pinny_a: float):
        """
        [除权算法]
        去除 Pinnacle 盘口的抽水 (Margin)，还原最锋利的物理胜率。
        返回 (P_true_h, P_true_d, P_true_a)
        """
        imp_h = 1.0 / pinny_h
        imp_d = 1.0 / pinny_d
        imp_a = 1.0 / pinny_a
        
        margin = imp_h + imp_d + imp_a
        if margin <= 0:
            return 0, 0, 0
            
        return imp_h / margin, imp_d / margin, imp_a / margin

    def check_value_bet(self, pinny_odds: tuple, poly_odds: dict):
        """
        Mode 1: Value Betting (价值下注 / 截胡单边)
        利用 Pinnacle 作为真理，审判 Polymarket 的赔率。
        pinny_odds: (H_odds, D_odds, A_odds)
        poly_odds: {"home": 2.50, "draw": 3.10, "away": 2.80}
        """
        ph, pd, pa = self.calculate_true_probabilities(*pinny_odds)
        
        opportunities = []
        
        if "home" in poly_odds:
            ev_h = ph * poly_odds["home"]
            if ev_h > self.ev_threshold:
                opportunities.append({"side": "home", "ev": ev_h, "poly_odds": poly_odds["home"], "true_prob": ph})
                
        if "draw" in poly_odds:
            ev_d = pd * poly_odds["draw"]
            if ev_d > self.ev_threshold:
                opportunities.append({"side": "draw", "ev": ev_d, "poly_odds": poly_odds["draw"], "true_prob": pd})
                
        if "away" in poly_odds:
            ev_a = pa * poly_odds["away"]
            if ev_a > self.ev_threshold:
                opportunities.append({"side": "away", "ev": ev_a, "poly_odds": poly_odds["away"], "true_prob": pa})
                
        return opportunities

    def check_surebet(self, poly_odds: float, pinny_opposite_odds: float, total_capital: float = 1000.0):
        """
        Mode 2: Surebet (跨平台无风险锁仓对冲)
        例如：Polymarket 卖主胜 (2.10)，Pinnacle 卖平局或客胜双选 (2.05)
        返回是否触发套利，及两边的配资情况。
        """
        arb_margin = (1.0 / poly_odds) + (1.0 / pinny_opposite_odds)
        
        if arb_margin < 1.0:
            # 存在无风险套利空间
            poly_stake = total_capital * (1.0 / poly_odds) / arb_margin
            pinny_stake = total_capital * (1.0 / pinny_opposite_odds) / arb_margin
            
            guaranteed_return = total_capital / arb_margin
            profit = guaranteed_return - total_capital
            roi = profit / total_capital
            
            return {
                "is_surebet": True,
                "arb_margin": arb_margin,
                "profit": profit,
                "roi_percent": roi * 100,
                "poly_stake": poly_stake,
                "pinny_stake": pinny_stake
            }
            
        return {"is_surebet": False}

if __name__ == "__main__":
    engine = PinnacleArbitrageEngine()
    
    # 模拟 Mode 1
    pinny = (1.95, 3.50, 4.00)
    poly = {"home": 2.20, "away": 3.80}
    vals = engine.check_value_bet(pinny, poly)
    print("Mode 1 (Value Bets):", vals)
    
    # 模拟 Mode 2
    sb = engine.check_surebet(poly_odds=2.10, pinny_opposite_odds=2.05)
    print("Mode 2 (Surebet):", sb)
