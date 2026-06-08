import math
from scipy.optimize import minimize


class PoissonPricer:
    """
    泊松分布定价引擎 (Poisson Pricer)
    将 1X2 独赢盘概率 (P_home, P_draw, P_away) 转化为理论的进球期望 (Expected Goals: lambda_h, lambda_a)，
    从而推导出所有亚洲让球盘 (Asian Handicap) 和大小球 (Over/Under) 的公平赔率。
    """

    def __init__(self, max_goals=10):
        self.max_goals = max_goals

    def poisson_prob(self, lam, k):
        return (lam**k * math.exp(-lam)) / math.factorial(k)

    def calculate_1x2_from_lambdas(self, lam_h, lam_a):
        """已知期望进球数，计算胜平负概率"""
        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0

        for h in range(self.max_goals):
            for a in range(self.max_goals):
                prob = self.poisson_prob(lam_h, h) * self.poisson_prob(lam_a, a)
                if h > a:
                    p_home += prob
                elif h == a:
                    p_draw += prob
                else:
                    p_away += prob

        # 归一化：max_goals 截断会丢失高比分尾部概率 (高 λ 时显著)，
        # 不归一化会使三路概率和 < 1，导致 EV 系统性偏低。
        total = p_home + p_draw + p_away
        if total > 0:
            p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total

        return p_home, p_draw, p_away

    def infer_lambdas(self, p_home_target, p_draw_target, p_away_target):
        """
        逆向推导: 寻找最符合 1X2 概率的 lambda_h 和 lambda_a
        使用 L-BFGS-B 最小化平方误差
        """

        def loss_function(params):
            lam_h, lam_a = params
            ph, pd, pa = self.calculate_1x2_from_lambdas(lam_h, lam_a)
            # 损失为目标概率与计算概率的均方误差
            return (
                (ph - p_home_target) ** 2
                + (pd - p_draw_target) ** 2
                + (pa - p_away_target) ** 2
            )

        # 初始猜测：根据经验，平均每场进球约 2.6 个
        initial_guess = [1.5, 1.1]
        bounds = ((0.1, 5.0), (0.1, 5.0))

        result = minimize(
            loss_function, initial_guess, bounds=bounds, method="L-BFGS-B"
        )
        return result.x[0], result.x[1]

    def price_asian_handicap(self, lam_h, lam_a, handicap):
        """
        计算亚洲让球盘 (Asian Handicap) 概率
        handicap: 对主队而言的让球数。例如 -0.5, -0.25, +1.0
        返回: p_home_win_bet, p_away_win_bet (包含退本金情况的处理)
        """
        p_home_win = 0.0
        p_away_win = 0.0
        p_push = 0.0  # 退水
        p_half_win = 0.0
        p_half_lose = 0.0

        for h in range(self.max_goals):
            for a in range(self.max_goals):
                prob = self.poisson_prob(lam_h, h) * self.poisson_prob(lam_a, a)
                net_score = h - a + handicap

                if net_score > 0:
                    if net_score == 0.25:
                        p_half_win += prob
                        p_push += prob * 0.5
                    else:
                        p_home_win += prob
                elif net_score < 0:
                    if net_score == -0.25:
                        p_half_lose += prob
                        p_push += prob * 0.5
                    else:
                        p_away_win += prob
                else:  # net_score == 0
                    p_push += prob

        # 标准化盘口胜率（忽略退本金的部分，计算投资回报率有效胜率）
        # 实际全赢的概率: p_home_win + 0.5 * p_half_win
        effective_p_home = p_home_win + 0.5 * p_half_win
        effective_p_away = p_away_win + 0.5 * p_half_lose

        return effective_p_home, effective_p_away

    def price_over_under(self, lam_h, lam_a, total_line=2.5):
        """
        计算大小球盘口 (Over/Under)
        """
        p_over = 0.0
        p_under = 0.0
        p_push = 0.0

        for h in range(self.max_goals):
            for a in range(self.max_goals):
                prob = self.poisson_prob(lam_h, h) * self.poisson_prob(lam_a, a)
                total_goals = h + a

                if total_goals > total_line:
                    p_over += prob
                elif total_goals < total_line:
                    p_under += prob
                else:
                    p_push += prob

        return p_over, p_under


if __name__ == "__main__":
    print("==================================================")
    print("   [Phase 3] 亚盘与大小球高维解析引擎 (Asian Handicap)")
    print("==================================================")

    pricer = PoissonPricer()

    # 假设我们的 XGBoost 算出了主场胜率 60%，平局 22%，客胜 18%
    p_h, p_d, p_a = 0.60, 0.22, 0.18
    print(f"XGBoost 预测胜率 -> 主:{p_h * 100}%, 平:{p_d * 100}%, 客:{p_a * 100}%")

    # 逆向推导期望进球
    lam_h, lam_a = pricer.infer_lambdas(p_h, p_d, p_a)
    print(f"\n推导出的泊松期望进球 -> 主队: {lam_h:.3f} 球, 客队: {lam_a:.3f} 球")

    print("\n--- 亚盘降维打击 (Asian Handicap) ---")
    handicaps = [0.0, -0.25, -0.5, -0.75, -1.0, -1.5]
    for hc in handicaps:
        p_h_ah, p_a_ah = pricer.price_asian_handicap(lam_h, lam_a, hc)
        fair_odds_h = 1.0 / p_h_ah if p_h_ah > 0 else 0
        fair_odds_a = 1.0 / p_a_ah if p_a_ah > 0 else 0
        print(
            f"让球盘 [{hc:>5}]: 主队公平水位 {fair_odds_h:.2f} | 客队公平水位 {fair_odds_a:.2f}"
        )

    print("\n--- 大小球降维打击 (Over/Under) ---")
    lines = [1.5, 2.5, 3.5]
    for line in lines:
        p_over, p_under = pricer.price_over_under(lam_h, lam_a, line)
        fair_odds_over = 1.0 / p_over if p_over > 0 else 0
        fair_odds_under = 1.0 / p_under if p_under > 0 else 0
        print(
            f"大小球 [{line:>3}]: 大球公平水位 {fair_odds_over:.2f} | 小球公平水位 {fair_odds_under:.2f}"
        )
    print("==================================================")
