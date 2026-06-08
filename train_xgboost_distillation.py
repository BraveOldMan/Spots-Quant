"""
XGBoost 知识蒸馏引擎 (V12 — 真实三向蒸馏版)

【原理】
1. 数据源: 前3个赛季 (19/20, 20/21, 21/22) 的真实博彩公司关盘赔率 (B365CH/CD/CA)。
2. 特征 X : walk-forward ELO 差 + proxy-xG 动量差。
3. 目标 y : 由真实关盘赔率除权得到的【主/平/客】三个方向的市场关盘概率。
4. 产出: 训练 3 个独立的 XGBoost Regressor，输出 xgb_h_distilled.json, xgb_d_distilled.json, xgb_a_distilled.json。
"""

from collections import defaultdict, deque

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

FEATURES = ["elo_diff", "mom_diff", "rating_diff", "mif_home", "mif_away"]

ELO_START = 1500.0
ELO_K = 40.0


def _market_probs(oh, od, oa):
    """关盘 1X2 赔率除权 -> (主, 平, 客) 真实概率。"""
    ih, idr, ia = 1.0 / oh, 1.0 / od, 1.0 / oa
    total = ih + idr + ia
    if total <= 0:
        return None, None, None
    return ih / total, idr / total, ia / total


def _proxy_xg(shots_total, shots_on_target):
    sog = max(shots_on_target, 0)
    soff = max(shots_total - shots_on_target, 0)
    return sog * 0.25 + soff * 0.05


def build_distillation_dataset(csv_paths) -> pd.DataFrame:
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df = df.assign(_date=pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"))
        dfs.append(df)
        
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("_date").reset_index(drop=True)

    elo = defaultdict(lambda: ELO_START)
    pxg_hist = defaultdict(lambda: deque(maxlen=5))

    def momentum(team):
        h = pxg_hist[team]
        return sum(h) / len(h) if h else 1.0

    rows = []
    for _, r in df.iterrows():
        home, away = r["HomeTeam"], r["AwayTeam"]

        # --- 赛前特征（无前视）---
        elo_diff = elo[home] - elo[away]
        mom_diff = momentum(home) - momentum(away)

        # --- 目标：真实关盘 1X2 概率 ---
        try:
            ph, pd_prob, pa = _market_probs(
                float(r["B365CH"]), float(r["B365CD"]), float(r["B365CA"])
            )
        except (TypeError, ValueError, KeyError):
            ph, pd_prob, pa = None, None, None

        if ph is not None and not pd.isna(elo_diff):
            rows.append(
                {
                    "elo_diff": elo_diff,
                    "mom_diff": mom_diff,
                    "rating_diff": 0.0,
                    "mif_home": 0.0,
                    "mif_away": 0.0,
                    "market_h": ph,
                    "market_d": pd_prob,
                    "market_a": pa,
                }
            )

        # --- 赛后滚动更新（结算后才更新，保证无前视）---
        hg, ag = r.get("FTHG"), r.get("FTAG")
        if pd.notna(hg) and pd.notna(ag):
            exp_h = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
            s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            elo[home] += ELO_K * (s_h - exp_h)
            elo[away] += ELO_K * ((1.0 - s_h) - (1.0 - exp_h))

        if pd.notna(r.get("HST")) and pd.notna(r.get("HS")):
            pxg_hist[home].append(_proxy_xg(float(r["HS"]), float(r["HST"])))
        if pd.notna(r.get("AST")) and pd.notna(r.get("AS")):
            pxg_hist[away].append(_proxy_xg(float(r["AS"]), float(r["AST"])))

    return pd.DataFrame(rows)


def train_single_model(X_train, y_train, X_test, y_test, model_name):
    print(f"\n[启动 XGBRegressor 蒸馏: {model_name}]...")
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        learning_rate=0.03,
        max_depth=4,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    corr = np.corrcoef(preds, y_test)[0, 1]

    print(f"✅ {model_name} 真实蒸馏完成！")
    print(f"MSE: {mse:.5f} | R^2: {r2:.4f} | Corr: {corr:.4f}")

    model.save_model(f"{model_name}.json")
    print(f"🚀 已保存至 {model_name}.json")


def fuzzy_match_and_train():
    print("==================================================")
    print("   [V12] XGBoost 三向知识蒸馏引擎 (主/平/客 独立分解)")
    print("==================================================")

    # 1. 严格划定训练集边界 (19/20, 20/21, 21/22)
    train_files = [
        "data_seasons/E0_1920.csv",
        "data_seasons/E0_2021.csv",
        "data_seasons/E0_2122.csv"
    ]
    
    print("[1] 构建蒸馏数据集 (walk-forward 特征 + 真实关盘概率标签)...")
    data = build_distillation_dataset(train_files)
    if len(data) < 30:
        print(f"有效样本过少 ({len(data)})，无法训练。")
        return
    print(f"    有效样本: {len(data)} 场。")

    X = data[FEATURES]
    
    # 划分验证集 (80/20 时序切分)
    split = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    
    # 2. 分别训练并保存三个维度的独立模型
    targets = {
        "xgb_h_distilled": "market_h",
        "xgb_d_distilled": "market_d",
        "xgb_a_distilled": "market_a"
    }
    
    for model_name, target_col in targets.items():
        y = data[target_col]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        train_single_model(X_train, y_train, X_test, y_test, model_name)


if __name__ == "__main__":
    fuzzy_match_and_train()
