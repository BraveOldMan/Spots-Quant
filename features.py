import json
import pandas as pd
from collections import defaultdict, deque


class FeatureEngine:
    def __init__(self):
        self.player_ratings = defaultdict(lambda: deque(maxlen=5))  # 近5场评分滚动均值
        self.team_pxg = defaultdict(
            lambda: deque(maxlen=5)
        )  # 球队近5场 proxy-xg 滚动均值
        self.elo_ratings = defaultdict(lambda: 1500.0)  # 滚动 ELO

    def get_avg_rating(self, player_id: int) -> float:
        history = self.player_ratings[player_id]
        if not history:
            return 6.0  # 默认及格分
        return sum(history) / len(history)

    def get_team_momentum(self, team_id: int) -> float:
        history = self.team_pxg[team_id]
        if not history:
            return 1.0  # 默认 xG
        return sum(history) / len(history)

    def extract_pxg(self, stats_array: list, team_id: int) -> float:
        sog, soff = 0.0, 0.0
        if not stats_array:
            return 0.0
        for ts in stats_array:
            if ts["team"]["id"] == team_id:
                for s in ts["statistics"]:
                    if s["type"] == "Shots on Goal":
                        sog = float(s["value"] or 0)
                    elif s["type"] == "Shots off Goal":
                        soff = float(s["value"] or 0)
        return sog * 0.25 + soff * 0.05

    def build_dataset(self, fixtures_file: str) -> pd.DataFrame:
        with open(fixtures_file, "r", encoding="utf-8") as f:
            fixtures = json.load(f)

        fixtures.sort(key=lambda x: x["fixture"]["timestamp"])

        dataset = []

        for match in fixtures:
            status = match["fixture"]["status"]["short"]
            if status not in ["FT", "AET", "PEN"]:
                continue

            home_id = match["teams"]["home"]["id"]
            away_id = match["teams"]["away"]["id"]

            # --- 1. 提取比赛前 (Pre-match) 的无未来函数特征 ---

            # ELO 特征
            elo_h = self.elo_ratings[home_id]
            elo_a = self.elo_ratings[away_id]
            elo_diff = elo_h - elo_a

            # Momentum 特征
            mom_h = self.get_team_momentum(home_id)
            mom_a = self.get_team_momentum(away_id)
            mom_diff = mom_h - mom_a

            # 首发阵容综合战力评估 (Micro Squad Rating)
            start_rating_h = 0.0
            start_rating_a = 0.0

            lineups = match.get("lineups", [])
            for lu in lineups:
                tid = lu["team"]["id"]
                rating_sum = 0.0
                for p in lu.get("startXI", []):
                    pid = p["player"]["id"]
                    rating_sum += self.get_avg_rating(pid)

                if tid == home_id:
                    start_rating_h = rating_sum
                elif tid == away_id:
                    start_rating_a = rating_sum

            rating_diff = start_rating_h - start_rating_a

            # 伤停折损因子 (MIF)
            injuries = match.get("injuries", [])
            mif_h = 0.0
            mif_a = 0.0
            for inj in injuries:
                tid = inj["team"]["id"]
                pid = inj["player"]["id"]
                missed_rating = self.get_avg_rating(pid)
                # 只有当缺阵的是高分主力时，MIF才显著
                if missed_rating > 6.5:
                    if tid == home_id:
                        mif_h += missed_rating
                    elif tid == away_id:
                        mif_a += missed_rating

            # 构造样本
            home_goals = match["goals"]["home"]
            away_goals = match["goals"]["away"]
            # 防御：完赛状态下仍可能缺失比分，None 比较会抛 TypeError（对齐 models.py 的判空）
            if home_goals is None or away_goals is None:
                continue
            if home_goals > away_goals:
                target = 1
            elif home_goals == away_goals:
                target = 0
            else:
                target = -1

            dataset.append(
                {
                    "fixture_id": match["fixture"]["id"],
                    "elo_diff": elo_diff,
                    "mom_diff": mom_diff,
                    "rating_diff": rating_diff,
                    "mif_home": mif_h,
                    "mif_away": mif_a,
                    "target": target,
                }
            )

            # --- 2. 比赛结束后，更新状态供下一场使用 (滚动更新) ---

            # 更新 ELO
            expected_h = 1 / (1 + 10 ** (-elo_diff / 400.0))
            expected_a = 1 - expected_h
            s_h = 1.0 if target == 1 else (0.5 if target == 0 else 0.0)
            s_a = 1.0 - s_h
            self.elo_ratings[home_id] = elo_h + 40.0 * (s_h - expected_h)
            self.elo_ratings[away_id] = elo_a + 40.0 * (s_a - expected_a)

            # 更新 Proxy-xG 动量
            stats = match.get("statistics", [])
            pxg_h = self.extract_pxg(stats, home_id)
            pxg_a = self.extract_pxg(stats, away_id)
            self.team_pxg[home_id].append(pxg_h)
            self.team_pxg[away_id].append(pxg_a)

            # 更新球员历史评分
            players_data = match.get("players", [])
            for team_data in players_data:
                for p in team_data.get("players", []):
                    pid = p["player"]["id"]
                    try:
                        r_str = p["statistics"][0]["games"]["rating"]
                        if r_str:
                            self.player_ratings[pid].append(float(r_str))
                    except (KeyError, IndexError, TypeError, ValueError):
                        pass

        df = pd.DataFrame(dataset)
        # 清洗掉初期的样本（此时 ELO 和动量还没收敛）
        # 假设前 20% 是 Burn-in 期
        burn_in = int(len(df) * 0.2)
        df_clean = df.iloc[burn_in:].copy()

        return df_clean


if __name__ == "__main__":
    engine = FeatureEngine()
    try:
        df = engine.build_dataset("raw_fixtures_v5.json")
        print("Feature Engineering Complete.")
        print(df.head())
        df.to_csv("xgboost_features.csv", index=False)
        print("Saved to xgboost_features.csv")
    except Exception as e:
        print(f"Waiting for dataset... ({e})")
