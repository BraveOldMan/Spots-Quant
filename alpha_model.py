"""
alpha_model — 基于 Alpha 特征的无前视结果预测模型 (路线图 T1)

与"蒸馏模型(模仿市场)"不同，本模型学习【真实赛果】，并融入【市场反应慢/易低估】的特征，
目标是产生真正的边际 (Alpha) 而非复述市场价格。

T1.1 学真实赛果 + 复用已有强度/动量特征。
T1.2 新增市场低估类特征: 休息天数差、赛程密度(近 14 天场次)、proxy-xG 动量差。
T1.3 stacking: 把 Dixon-Coles/泊松的 1X2 概率作为 XGBoost 的输入特征，融合基本面与机器学习。

接口与 run_ultimate_backtest 的 model_factory 兼容: predict(home,away,date,row) / update(...)。
模型采用 walk-forward 周期性重训 (refit_every)，严格只用过去样本，无前视。
"""

from collections import defaultdict, deque

import numpy as np
import xgboost as xgb

import quant_core as qc
from asian_handicap_engine import PoissonPricer

FEATURE_ORDER = [
    "elo_diff",
    "mom_diff",
    "rest_diff",
    "cong_diff",
    "dc_p_home",
    "dc_p_draw",
]
# XGB 多分类: 0=客胜, 1=平, 2=主胜 (与 train_xgboost.py 一致)
_RESULT_TO_CLASS = {"A": 0, "D": 1, "H": 2}


class _PoissonStrength:
    """内嵌的无前视泊松强度估计，用于产出 stacking 用的 Dixon-Coles 概率。"""

    def __init__(self):
        self.pricer = PoissonPricer(max_goals=10)
        self.gf = defaultdict(float)
        self.ga = defaultdict(float)
        self.played = defaultdict(int)
        self.thg = self.tag = 0.0
        self.n = 0

    def probs(self, home, away):
        if self.n < 20 or self.played[home] < 3 or self.played[away] < 3:
            return (0.45, 0.27, 0.28)  # 缺数据时给联赛先验
        lh = self.thg / self.n
        la = self.tag / self.n
        ov = (self.thg + self.tag) / (2 * self.n)
        if ov <= 0:
            return (0.45, 0.27, 0.28)
        atk_h = (self.gf[home] / self.played[home]) / ov
        def_h = (self.ga[home] / self.played[home]) / ov
        atk_a = (self.gf[away] / self.played[away]) / ov
        def_a = (self.ga[away] / self.played[away]) / ov
        lam_h = float(np.clip(atk_h * def_a * lh, 0.05, 8.0))
        lam_a = float(np.clip(atk_a * def_h * la, 0.05, 8.0))
        return self.pricer.calculate_1x2_from_lambdas(lam_h, lam_a)

    def update(self, home, away, hg, ag):
        self.gf[home] += hg
        self.ga[home] += ag
        self.played[home] += 1
        self.gf[away] += ag
        self.ga[away] += hg
        self.played[away] += 1
        self.thg += hg
        self.tag += ag
        self.n += 1


class AlphaXGBModel:
    def __init__(
        self,
        refit_every: int = 15,
        min_train: int = 80,
        congestion_window_days: int = 14,
    ):
        self.refit_every = refit_every
        self.min_train = min_train
        self.cong_window = congestion_window_days

        self.elo = defaultdict(lambda: 1500.0)
        self.pxg = defaultdict(lambda: deque(maxlen=5))
        self.last_dates = defaultdict(list)  # team -> [match dates]
        self.dc = _PoissonStrength()

        self._X = []  # 训练特征 (历史)
        self._y = []  # 训练标签 (历史真实赛果类)
        self.model = None
        self._since_fit = 0

    # ---- 特征 (全部赛前可得) ----
    def _momentum(self, team):
        h = self.pxg[team]
        return sum(h) / len(h) if h else 1.0

    def _rest_days(self, team, date):
        ds = self.last_dates[team]
        if not ds:
            return 7.0  # 默认一周
        return max((date - ds[-1]).days, 0)

    def _congestion(self, team, date):
        ds = self.last_dates[team]
        return sum(1 for d in ds if 0 <= (date - d).days <= self.cong_window)

    def _make_features(self, home, away, date):
        dc_h, dc_d, _ = self.dc.probs(home, away)
        return {
            "elo_diff": self.elo[home] - self.elo[away],
            "mom_diff": self._momentum(home) - self._momentum(away),
            "rest_diff": self._rest_days(home, date) - self._rest_days(away, date),
            "cong_diff": self._congestion(home, date) - self._congestion(away, date),
            "dc_p_home": dc_h,
            "dc_p_draw": dc_d,
        }

    def predict(self, home, away, date=None, row=None, **kwargs):
        if self.model is None:
            return None
        feats = self._make_features(home, away, date)
        x = np.array([[feats[k] for k in FEATURE_ORDER]], dtype=float)
        proba = self.model.predict_proba(x)[0]
        # classes_ 升序 [0,1,2] = [客,平,主] -> 返回 (主,平,客)
        cls = list(self.model.classes_)
        p = {c: proba[i] for i, c in enumerate(cls)}
        return (float(p.get(2, 0.0)), float(p.get(1, 0.0)), float(p.get(0, 0.0)))

    def update(self, home, away, hg, ag, date=None, row=None, **kwargs):
        # 1) 先用"赛前特征 + 真实结果"沉淀训练样本 (无前视: 特征基于更新前状态)
        feats = self._make_features(home, away, date)
        ftr = "H" if hg > ag else ("D" if hg == ag else "A")
        self._X.append([feats[k] for k in FEATURE_ORDER])
        self._y.append(_RESULT_TO_CLASS[ftr])
        self._since_fit += 1

        # 2) 周期性重训 (只用已积累的过去样本)
        if len(self._y) >= self.min_train and self._since_fit >= self.refit_every:
            self._fit()
            self._since_fit = 0

        # 3) 更新滚动状态供后续比赛使用
        self.dc.update(home, away, hg, ag)
        exp_h = qc.elo_expected(self.elo[home], self.elo[away])
        s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        self.elo[home] = qc.elo_update(self.elo[home], s_h, exp_h, 40.0)
        self.elo[away] = qc.elo_update(self.elo[away], 1 - s_h, 1 - exp_h, 40.0)
        if row is not None:
            try:
                self.pxg[home].append(qc.proxy_xg(float(row["HS"]), float(row["HST"])))
                self.pxg[away].append(qc.proxy_xg(float(row["AS"]), float(row["AST"])))
            except (KeyError, TypeError, ValueError):
                pass
        if date is not None:
            self.last_dates[home].append(date)
            self.last_dates[away].append(date)

    def _fit(self):
        X = np.asarray(self._X, dtype=float)
        y = np.asarray(self._y, dtype=int)
        if len(np.unique(y)) < 2:
            return
        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            learning_rate=0.05,
            max_depth=3,
            n_estimators=120,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        self.model.fit(X, y)


if __name__ == "__main__":
    from run_ultimate_backtest import run_real_backtest

    print("########## 基线: 朴素泊松 ##########")
    run_real_backtest(odds_mode="opening", ev_threshold=1.05)
    print("\n########## T1: Alpha XGBoost (学赛果 + 低估特征 + DC stacking) ##########")
    run_real_backtest(
        model_factory=AlphaXGBModel, odds_mode="opening", ev_threshold=1.05
    )
