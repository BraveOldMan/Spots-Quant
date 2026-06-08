"""
quant_core — 量化公共工具模块 (T4.2 DRY 去重的落点)

把先前散落在 models.py / features.py / run_ultimate_backtest.py /
train_xgboost_distillation.py 各写一遍的核心数学统一到这里：
  - 赔率除权 (margin removal) 与隐含概率
  - 凯利仓位与 EV
  - ELO 期望/更新
  - proxy-xG
  - 校准与 CLV 指标

所有函数无副作用、可独立单测（见 tests/）。
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# ----------------------------------------------------------------------
# 赔率 / 概率
# ----------------------------------------------------------------------


def implied_prob(decimal_odds: float) -> float:
    """单一 decimal 赔率 -> 原始隐含概率 (未除权)。"""
    if decimal_odds is None or decimal_odds <= 0:
        return 0.0
    return 1.0 / decimal_odds


def remove_margin(odds: Sequence[float]) -> list[float]:
    """
    多路 decimal 赔率 -> 除权后的公平概率 (归一化, 和为 1)。
    适用于 1X2 (3 路) 或任意路数。
    """
    imps = [implied_prob(o) for o in odds]
    total = sum(imps)
    if total <= 0:
        return [0.0 for _ in odds]
    return [i / total for i in imps]


def margin(odds: Sequence[float]) -> float:
    """博彩公司抽水率 = sum(1/odds) - 1。"""
    return sum(implied_prob(o) for o in odds) - 1.0


# ----------------------------------------------------------------------
# EV / 凯利
# ----------------------------------------------------------------------


def expected_value(prob: float, decimal_odds: float) -> float:
    """期望价值乘数 EV = p * odds (>1 即正期望)。"""
    return prob * decimal_odds


def kelly_fraction(prob: float, decimal_odds: float, fraction: float = 1.0) -> float:
    """
    凯利下注比例 (返回应投入本金的比例, 已乘以 fraction)。
    prob: 真实胜率; decimal_odds: 赔率; fraction: 分数凯利系数。
    f* = (b*p - q) / b, b = odds - 1。f*<=0 返回 0。
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - prob
    f_star = (b * prob - q) / b
    if f_star <= 0:
        return 0.0
    return f_star * fraction


# ----------------------------------------------------------------------
# ELO
# ----------------------------------------------------------------------


def elo_expected(rating_a: float, rating_b: float, home_adv: float = 0.0) -> float:
    """A 对 B 的期望胜率 (含可选主场 ELO 加成)。"""
    return 1.0 / (1.0 + 10 ** ((rating_b - (rating_a + home_adv)) / 400.0))


def elo_update(rating: float, score: float, expected: float, k: float) -> float:
    """单方 ELO 更新: rating + k*(score - expected)。"""
    return rating + k * (score - expected)


# ----------------------------------------------------------------------
# proxy-xG
# ----------------------------------------------------------------------


def proxy_xg(
    shots_total: float, shots_on_target: float, w_on: float = 0.25, w_off: float = 0.05
) -> float:
    """proxy 期望进球 = 射正*w_on + 射偏*w_off。"""
    sog = max(shots_on_target, 0.0)
    soff = max(shots_total - shots_on_target, 0.0)
    return sog * w_on + soff * w_off


# ----------------------------------------------------------------------
# 泊松 / Dixon-Coles 1X2
# ----------------------------------------------------------------------


def dixon_coles_1x2(
    lam_h: float, lam_a: float, rho: float = 0.0, max_goals: int = 10
) -> tuple[float, float, float]:
    """
    由主/客期望进球 (lam_h, lam_a) 计算 1X2 概率，含 Dixon-Coles 低分修正 (rho)。
    rho<0 提升 0-0/1-1 等低比分平局概率，修正独立泊松对平局的低估。
    返回 (p_home, p_draw, p_away)。
    """
    lam_h = max(lam_h, 1e-6)
    lam_a = max(lam_a, 1e-6)
    # 泊松 pmf 向量
    ph = np.array(
        [math.exp(-lam_h) * lam_h**k / math.factorial(k) for k in range(max_goals)]
    )
    pa = np.array(
        [math.exp(-lam_a) * lam_a**k / math.factorial(k) for k in range(max_goals)]
    )
    mat = np.outer(ph, pa)
    # DC tau 低分修正
    if rho != 0.0:
        mat[0, 0] *= 1 - lam_h * lam_a * rho
        mat[1, 0] *= 1 + lam_a * rho
        mat[0, 1] *= 1 + lam_h * rho
        mat[1, 1] *= 1 - rho
        mat = np.clip(mat, 0.0, None)
    total = mat.sum()
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    mat /= total
    p_home = float(np.tril(mat, -1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.triu(mat, 1).sum())
    return (p_home, p_draw, p_away)


# ----------------------------------------------------------------------
# 校准 (Calibration)
# ----------------------------------------------------------------------


def brier_score(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """二分类 Brier 分数 (越小越好)。outcomes ∈ {0,1}。"""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(p) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def log_loss(
    probs: Sequence[float], outcomes: Sequence[int], eps: float = 1e-12
) -> float:
    """二分类对数损失 (越小越好)。"""
    p = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
    y = np.asarray(outcomes, dtype=float)
    if len(p) == 0:
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_curve(
    probs: Sequence[float], outcomes: Sequence[int], n_bins: int = 10
):
    """
    可靠性曲线: 返回 [(bin_mid, mean_pred, empirical_freq, count), ...]。
    用于检查"模型说 X% 时, 实际是否约 X% 发生"。
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    out = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        out.append(((lo + hi) / 2, float(p[mask].mean()), float(y[mask].mean()), cnt))
    return out


# ----------------------------------------------------------------------
# CLV (Closing Line Value)
# ----------------------------------------------------------------------


def clv_pct(odds_taken: float, odds_close: float) -> float:
    """
    单注 CLV%: 你拿到的赔率相对关盘赔率的优势。
    CLV% = odds_taken / odds_close - 1。正值表示你拿到了比关盘更好的价格。
    """
    if odds_close is None or odds_close <= 0:
        return 0.0
    return odds_taken / odds_close - 1.0


def clv_prob_edge(
    odds_taken_set: Sequence[float], odds_close_set: Sequence[float], sel_idx: int
) -> float:
    """
    基于除权概率的 CLV: 关盘公平概率 - 入场公平概率 (针对所投选项 sel_idx)。
    正值表示市场在你下注后向你的方向移动 (你领先于市场)。
    """
    p_entry = remove_margin(odds_taken_set)[sel_idx]
    p_close = remove_margin(odds_close_set)[sel_idx]
    return p_close - p_entry
