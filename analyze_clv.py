"""
analyze_clv — CLV (Closing Line Value) 分析 (路线图 T0.1)

CLV 是检验下注【技艺】的领先指标：方差远小于 P&L。
若长期 CLV 显著为正，说明你的模型能在市场修正前识别价值——这是真正 Alpha 的标志；
反之，再漂亮的短期 P&L 也多半是运气。

两种数据来源：
  1. 回测 CLV (推荐, 数据完整)：在【开盘价】下注、与【关盘价】比较。
     直接复用 run_ultimate_backtest 的 opening 模式逐注记录。
  2. 实盘 CLV：分析 bet_history.db 的 clv_tracking 表 (需含关盘赔率列, 见下方说明)。
"""

import sqlite3

import numpy as np

from run_ultimate_backtest import run_real_backtest


def _report(clv_pcts, label):
    arr = np.asarray(clv_pcts, dtype=float)
    print(f"\n===== CLV 分析: {label} =====")
    if len(arr) == 0:
        print("无可分析的下注记录。")
        return
    print(f"样本注数        : {len(arr)}")
    print(f"平均 CLV        : {arr.mean() * 100:+.2f}%")
    print(f"CLV 中位数      : {np.median(arr) * 100:+.2f}%")
    print(f"击败关盘比例    : {(arr > 0).mean() * 100:.1f}%")
    print(f"CLV 标准差      : {arr.std() * 100:.2f}%")
    # 简易显著性: 平均 CLV 的 t 统计 (>2 约等于 95% 显著为正)
    if arr.std() > 0:
        t = arr.mean() / (arr.std() / np.sqrt(len(arr)))
        verdict = (
            "显著为正 ✅ (疑似真实 Alpha)"
            if t > 2
            else ("显著为负 ❌" if t < -2 else "不显著 ⚪ (尚无法证明技艺)")
        )
        print(f"t 统计量        : {t:.2f} -> {verdict}")


def analyze_backtest_clv(csv_paths=("EPL_2324.csv",), ev_threshold: float = 1.02):
    """在开盘价下注、与关盘比较，度量回测 CLV。"""
    res = run_real_backtest(
        csv_paths=csv_paths,
        ev_threshold=ev_threshold,
        odds_mode="opening",
        verbose=False,
    )
    clvs = [b["clv_pct"] for b in res["bet_records"]]
    _report(clvs, f"回测开盘→关盘 (EV>{ev_threshold})")
    # 额外: 按是否盈利分组看 CLV 与结果的一致性
    if res["bet_records"]:
        won = [b["clv_pct"] for b in res["bet_records"] if b["won"]]
        lost = [b["clv_pct"] for b in res["bet_records"] if not b["won"]]
        if won:
            print(f"  命中注平均 CLV : {np.mean(won) * 100:+.2f}%")
        if lost:
            print(f"  未中注平均 CLV : {np.mean(lost) * 100:+.2f}%")
    return clvs


def analyze_live_clv(db_path: str = "bet_history.db"):
    """
    分析实盘 bet_history.db 的 CLV。
    注意: 当前 clv_tracking 表只存了下注时赔率 (bookie_odds)，未存关盘赔率。
    若要启用实盘 CLV，需在 value_scanner 落库时补一列 closing_odds，
    或赛后回填关盘赔率。此函数对缺列情况给出明确提示。
    """
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(clv_tracking)")
        cols = {r[1] for r in cur.fetchall()}
    except Exception as e:
        print(f"无法读取 {db_path}: {e}")
        return

    if "closing_odds" not in cols:
        print("\n===== 实盘 CLV =====")
        print("⚠️ bet_history.db 的 clv_tracking 表尚未存储关盘赔率 (closing_odds)，")
        print(
            "   无法计算实盘 CLV。建议: 在 value_scanner 落库时新增 closing_odds 列，"
        )
        print("   或编写赛后回填脚本。当前请使用回测 CLV (analyze_backtest_clv)。")
        conn.close()
        return

    rows = cur.execute(
        "SELECT bookie_odds, closing_odds FROM clv_tracking "
        "WHERE bookie_odds > 0 AND closing_odds > 0"
    ).fetchall()
    conn.close()
    clvs = [bo / co - 1.0 for bo, co in rows]
    _report(clvs, "实盘 bet_history.db")


if __name__ == "__main__":
    print("=" * 50)
    print("   CLV 技艺诊断 (Closing Line Value)")
    print("=" * 50)
    analyze_backtest_clv()
    analyze_live_clv()
