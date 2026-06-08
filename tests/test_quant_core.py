"""quant_core 单元测试 (T4.1)。"""


import pytest

import quant_core as qc


# ---- 赔率 / 概率 ----


def test_implied_prob_basic():
    assert qc.implied_prob(2.0) == pytest.approx(0.5)
    assert qc.implied_prob(0) == 0.0
    assert qc.implied_prob(-1) == 0.0


def test_remove_margin_sums_to_one():
    probs = qc.remove_margin([2.0, 4.0, 4.0])
    assert sum(probs) == pytest.approx(1.0)
    # 赔率越低概率越高
    assert probs[0] > probs[1]


def test_margin_positive_for_bookmaker():
    # 含抽水的盘口 margin > 0
    assert qc.margin([1.9, 3.5, 4.0]) > 0
    # 公平盘 (1/p 和为 1) margin ≈ 0
    assert qc.margin([2.0, 2.0]) == pytest.approx(0.0)


# ---- EV / 凯利 ----


def test_expected_value():
    assert qc.expected_value(0.6, 2.0) == pytest.approx(1.2)


def test_kelly_zero_when_no_edge():
    # p*odds <= 1 时不下注
    assert qc.kelly_fraction(0.5, 2.0) == 0.0
    assert qc.kelly_fraction(0.4, 2.0) == 0.0


def test_kelly_positive_with_edge():
    f = qc.kelly_fraction(0.6, 2.0)  # b=1,p=0.6,q=0.4 -> f*=0.2
    assert f == pytest.approx(0.2, abs=1e-9)


def test_kelly_fraction_scaling():
    full = qc.kelly_fraction(0.6, 2.0, fraction=1.0)
    half = qc.kelly_fraction(0.6, 2.0, fraction=0.5)
    assert half == pytest.approx(full * 0.5)


# ---- ELO ----


def test_elo_expected_symmetry():
    assert qc.elo_expected(1500, 1500) == pytest.approx(0.5)
    assert qc.elo_expected(1900, 1500) > 0.5
    # 主场加成提升期望
    assert qc.elo_expected(1500, 1500, home_adv=100) > 0.5


def test_elo_update_direction():
    # 赢了且期望低于 1 -> 评分上升
    new = qc.elo_update(1500, score=1.0, expected=0.5, k=20)
    assert new > 1500
    new2 = qc.elo_update(1500, score=0.0, expected=0.5, k=20)
    assert new2 < 1500


# ---- proxy-xG ----


def test_proxy_xg():
    # 10 射 4 射正 -> 4*0.25 + 6*0.05 = 1.3
    assert qc.proxy_xg(10, 4) == pytest.approx(1.3)
    assert qc.proxy_xg(0, 0) == 0.0


# ---- Dixon-Coles 1X2 ----


def test_dc_1x2_normalized():
    p = qc.dixon_coles_1x2(1.5, 1.1, rho=-0.05)
    assert sum(p) == pytest.approx(1.0, abs=1e-6)
    assert all(0 <= x <= 1 for x in p)


def test_dc_home_favorite():
    # 主队期望进球远高 -> 主胜概率最高
    p_home, p_draw, p_away = qc.dixon_coles_1x2(2.5, 0.6)
    assert p_home > p_away
    assert p_home > p_draw


def test_dc_rho_increases_draw():
    # rho<0 应提升平局概率 (相对 rho=0)
    _, draw0, _ = qc.dixon_coles_1x2(1.2, 1.2, rho=0.0)
    _, draw_neg, _ = qc.dixon_coles_1x2(1.2, 1.2, rho=-0.1)
    assert draw_neg > draw0


# ---- 校准 ----


def test_brier_perfect():
    assert qc.brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == pytest.approx(0.0)


def test_brier_worst():
    assert qc.brier_score([0.0, 1.0], [1, 0]) == pytest.approx(1.0)


def test_log_loss_better_when_confident_correct():
    good = qc.log_loss([0.9, 0.1], [1, 0])
    bad = qc.log_loss([0.6, 0.4], [1, 0])
    assert good < bad


def test_reliability_curve_bins():
    probs = [0.05, 0.15, 0.95, 0.85]
    outcomes = [0, 0, 1, 1]
    curve = qc.reliability_curve(probs, outcomes, n_bins=10)
    assert len(curve) > 0
    for _mid, mean_pred, emp, cnt in curve:
        assert 0 <= mean_pred <= 1
        assert 0 <= emp <= 1
        assert cnt >= 1


# ---- CLV ----


def test_clv_pct_positive_when_beat_close():
    # 拿到 2.2, 关盘 2.0 -> 正 CLV
    assert qc.clv_pct(2.2, 2.0) == pytest.approx(0.1)
    # 拿到比关盘差的价格 -> 负 CLV
    assert qc.clv_pct(1.9, 2.0) < 0
    assert qc.clv_pct(2.0, 0) == 0.0


def test_clv_prob_edge():
    # 关盘把所投选项概率抬高 -> 正 edge
    taken = [2.5, 3.4, 3.0]
    close = [2.1, 3.4, 3.6]  # 主队关盘赔率下降(概率升)
    edge = qc.clv_prob_edge(taken, close, sel_idx=0)
    assert edge > 0
