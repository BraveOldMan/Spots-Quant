"""
wf_models — 进阶 walk-forward 概率模型 (路线图 T2)

DixonColesWFModel 相对朴素泊松 (run_ultimate_backtest.WalkForwardPoissonModel) 的升级：
  T2.1 Dixon-Coles 低分修正 (rho)：修正独立泊松对平局/低比分的系统性低估。
  T2.2 主客分离强度 + 贝叶斯收缩：分别估计主/客场进攻防守，并向联赛均值收缩，稳住早赛季小样本。
  T2.3 proxy-xG 融合目标：用 0.6*真实进球 + 0.4*proxy-xG 估强度，抑制"运气球"方差。

接口与 run_ultimate_backtest 的 model_factory 兼容: predict(home,away,...) / update(home,away,hg,ag,row=...)。
仅用"该场之前"的数据，无前视。
"""

from collections import defaultdict

import quant_core as qc

GOALS_BLEND = 0.6  # 真实进球权重
PXG_BLEND = 0.4  # proxy-xG 权重
SHRINK_K = 5.0  # 贝叶斯收缩伪计数 (越大越向联赛均值收缩)
DC_RHO = -0.05  # Dixon-Coles 低分修正系数


class DixonColesWFModel:
    def __init__(
        self, min_history: int = 2, rho: float = DC_RHO, shrink_k: float = SHRINK_K
    ):
        self.min_history = min_history
        self.rho = rho
        self.shrink_k = shrink_k
        # 主场视角累计 (blended goals)
        self.h_gf = defaultdict(float)
        self.h_ga = defaultdict(float)
        self.h_played = defaultdict(int)
        # 客场视角累计
        self.a_gf = defaultdict(float)
        self.a_ga = defaultdict(float)
        self.a_played = defaultdict(int)
        # 联赛基线
        self.tot_home = 0.0
        self.tot_away = 0.0
        self.n = 0

    def _blend(self, goals, shots, shots_on):
        if shots is None or shots_on is None:
            return float(goals)
        return GOALS_BLEND * goals + PXG_BLEND * qc.proxy_xg(
            float(shots), float(shots_on)
        )

    def _shrunk_rate(self, total, played, league_rate):
        """向联赛均值收缩的每场速率。"""
        return (total + self.shrink_k * league_rate) / (played + self.shrink_k)

    def predict(self, home, away, date=None, row=None, **kwargs):
        if self.n < 20:
            return None
        if (
            self.h_played[home] < self.min_history
            or self.a_played[away] < self.min_history
        ):
            return None

        league_home = self.tot_home / self.n  # 主队平均进球
        league_away = self.tot_away / self.n  # 客队平均进球
        if league_home <= 0 or league_away <= 0:
            return None

        # 主客分离 + 收缩
        home_scoring = self._shrunk_rate(
            self.h_gf[home], self.h_played[home], league_home
        )
        home_conceding = self._shrunk_rate(
            self.h_ga[home], self.h_played[home], league_away
        )
        away_scoring = self._shrunk_rate(
            self.a_gf[away], self.a_played[away], league_away
        )
        away_conceding = self._shrunk_rate(
            self.a_ga[away], self.a_played[away], league_home
        )

        home_attack = home_scoring / league_home
        home_defense = home_conceding / league_away
        away_attack = away_scoring / league_away
        away_defense = away_conceding / league_home

        lam_h = league_home * home_attack * away_defense
        lam_a = league_away * away_attack * home_defense
        lam_h = min(max(lam_h, 0.05), 8.0)
        lam_a = min(max(lam_a, 0.05), 8.0)

        return qc.dixon_coles_1x2(lam_h, lam_a, rho=self.rho)

    def update(self, home, away, hg, ag, date=None, row=None, **kwargs):
        hs = hst = as_ = ast = None
        if row is not None:
            hs, hst = row.get("HS"), row.get("HST")
            as_, ast = row.get("AS"), row.get("AST")
        h_blend = self._blend(hg, hs, hst)
        a_blend = self._blend(ag, as_, ast)

        self.h_gf[home] += h_blend
        self.h_ga[home] += a_blend
        self.h_played[home] += 1
        self.a_gf[away] += a_blend
        self.a_ga[away] += h_blend
        self.a_played[away] += 1
        self.tot_home += h_blend
        self.tot_away += a_blend
        self.n += 1


if __name__ == "__main__":
    from run_ultimate_backtest import run_real_backtest, WalkForwardPoissonModel

    for name, factory in [
        ("朴素泊松 (基线)", WalkForwardPoissonModel),
        ("Dixon-Coles WF (T2)", DixonColesWFModel),
    ]:
        print(f"\n########## {name} ##########")
        run_real_backtest(model_factory=factory, odds_mode="opening", ev_threshold=1.05)
