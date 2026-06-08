"""PoissonPricer / Betfair 映射 / 回测结算 单元测试 (T4.1)。"""

import pytest

from asian_handicap_engine import PoissonPricer
from robust_extractor import map_odds_to_hda
from run_ultimate_backtest import RiskManagementV9, WalkForwardPoissonModel


# ---- PoissonPricer ----


def test_1x2_normalized():
    pricer = PoissonPricer()
    p = pricer.calculate_1x2_from_lambdas(1.5, 1.2)
    assert sum(p) == pytest.approx(1.0, abs=1e-3)


def test_over_under_normalized():
    pricer = PoissonPricer()
    over, under = pricer.price_over_under(1.5, 1.2, 2.5)
    # 大 + 小 应接近 1 (无 push, 因为 2.5 非整数)
    assert over + under == pytest.approx(1.0, abs=1e-3)


def test_infer_lambdas_roundtrip():
    pricer = PoissonPricer()
    lam_h, lam_a = pricer.infer_lambdas(0.5, 0.27, 0.23)
    ph, pd, pa = pricer.calculate_1x2_from_lambdas(lam_h, lam_a)
    assert ph == pytest.approx(0.5, abs=0.03)
    assert pa == pytest.approx(0.23, abs=0.03)


def test_higher_home_lambda_raises_home_prob():
    pricer = PoissonPricer()
    p_low = pricer.calculate_1x2_from_lambdas(1.0, 1.0)[0]
    p_high = pricer.calculate_1x2_from_lambdas(2.5, 1.0)[0]
    assert p_high > p_low


def test_1x2_normalized_high_lambda():
    # 高 λ 截断回归测试: 仍需归一化到 1
    pricer = PoissonPricer()
    p = pricer.calculate_1x2_from_lambdas(6.0, 1.0)
    assert sum(p) == pytest.approx(1.0, abs=1e-6)


def test_asian_handicap_minus_half():
    pricer = PoissonPricer()
    # 让球 -0.5 主队: 强主队有效胜率应较高
    eff_home, eff_away = pricer.price_asian_handicap(2.2, 0.8, -0.5)
    assert 0 < eff_home < 1
    assert eff_home > eff_away


def test_asian_handicap_symmetry_level():
    pricer = PoissonPricer()
    # 平手盘 (0): 实力相当, 主客有效胜率接近
    eff_home, eff_away = pricer.price_asian_handicap(1.3, 1.3, 0.0)
    assert eff_home == pytest.approx(eff_away, abs=0.05)


def test_over_under_higher_line_lowers_over():
    pricer = PoissonPricer()
    over_25, _ = pricer.price_over_under(1.5, 1.5, 2.5)
    over_35, _ = pricer.price_over_under(1.5, 1.5, 3.5)
    assert over_25 > over_35


def test_infer_lambdas_within_bounds():
    pricer = PoissonPricer()
    lam_h, lam_a = pricer.infer_lambdas(0.45, 0.27, 0.28)
    assert 0.1 <= lam_h <= 5.0
    assert 0.1 <= lam_a <= 5.0


# ---- Betfair 语义映射 (核心数据修复回归测试) ----


def test_mapper_assigns_by_name_not_id_order():
    # selectionId 数值顺序与语义相反, 必须按名称映射
    closing = {58805: 27.0, 47999: 1.03, 48224: 8.5}
    names = {58805: "The Draw", 47999: "Tottenham", 48224: "Man City"}
    h, d, a = map_odds_to_hda(closing, names, "Tottenham v Man City")
    # Tottenham@1.03 是主队大热 -> home_prob 最大
    assert h > a > d


def test_mapper_reversed_event_name():
    closing = {58805: 27.0, 47999: 1.03, 48224: 8.5}
    names = {58805: "The Draw", 47999: "Tottenham", 48224: "Man City"}
    h, d, a = map_odds_to_hda(closing, names, "Man City v Tottenham")
    # 现在 Tottenham 是客队 -> away_prob 最大
    assert a > h


def test_mapper_returns_none_without_draw():
    closing = {1: 2.0, 2: 2.0, 3: 2.0}
    names = {1: "Team A", 2: "Team B", 3: "Team C"}
    assert map_odds_to_hda(closing, names, "Team A v Team B") is None


def test_mapper_probs_sum_to_one():
    closing = {58805: 3.4, 47999: 2.1, 48224: 3.6}
    names = {58805: "The Draw", 47999: "Arsenal", 48224: "Chelsea"}
    h, d, a = map_odds_to_hda(closing, names, "Arsenal v Chelsea")
    assert h + d + a == pytest.approx(1.0, abs=1e-9)


# ---- 风控 / 凯利 ----


def test_kelly_caps_at_max_fraction():
    rm = RiskManagementV9(initial_capital=10000.0, max_fraction=0.03)
    # 极强优势也不应超过 3% 上限
    stake = rm.calculate_bet_size(0.99, 5.0)
    assert stake <= 10000.0 * 0.03 + 1e-9


def test_drawdown_freeze_triggers():
    rm = RiskManagementV9(initial_capital=1000.0, max_drawdown_limit=0.15)
    rm.update_capital(-200)  # -20% > 15%
    assert rm.trading_frozen is True


def test_no_bet_when_no_edge():
    rm = RiskManagementV9()
    # p*b<=1 时凯利为 0
    assert rm.calculate_bet_size(0.4, 2.0) == 0.0


def test_match_exposure_cap_binds():
    # 同场敞口上限应限制后续同场下注 (T5.1)
    rm = RiskManagementV9(initial_capital=10000.0, max_match_exposure=0.05)
    # 先登记已用满 480
    rm.register_exposure(1, 480.0)
    # 再下一注: 即使凯利想下更多, 也只剩 20 的额度
    stake = rm.calculate_bet_size(0.99, 5.0, match_id=1)
    assert stake <= 20.0 + 1e-9
    # 不同比赛不受影响
    other = rm.calculate_bet_size(0.99, 5.0, match_id=2)
    assert other > 20.0


def test_match_exposure_no_cap_when_unset():
    rm = RiskManagementV9(initial_capital=10000.0)  # max_match_exposure=None
    rm.register_exposure(1, 480.0)
    stake_capped = rm.calculate_bet_size(0.99, 5.0, match_id=1)
    stake_plain = rm.calculate_bet_size(0.99, 5.0, match_id=None)
    # 未设同场上限时, 登记的敞口不应削减下注额
    assert stake_capped == pytest.approx(stake_plain)
    assert stake_capped > 20.0


# ---- 回测无前视: 模型 predict 仅依赖 update 之前的状态 ----


def test_walkforward_no_lookahead():
    m = WalkForwardPoissonModel(min_history=1)
    # 冷启动: 历史不足应返回 None
    assert m.predict("A", "B") is None
    # 喂足历史
    for _ in range(25):
        m.update("A", "B", 2, 0)
        m.update("C", "D", 1, 1)
    preds = m.predict("A", "B")
    assert preds is not None
    assert sum(preds) == pytest.approx(1.0, abs=1e-3)
