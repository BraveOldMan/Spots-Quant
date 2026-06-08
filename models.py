import json
import math
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict
from typing import Tuple

# 真正在中立场进行、应中和主场优势的赛事联赛 ID。
# league 10 = International Friendlies（见 get_k_factor 注释），多在中立或无真实主场氛围。
# 联赛/杯赛/世预赛均为主客场制，必须保留主场优势。
NEUTRAL_LEAGUE_IDS = {10}


class FootballModel:
    def __init__(self):
        self.elo_ratings = defaultdict(lambda: 1500.0)
        self.team_stats = defaultdict(
            lambda: {"matches": 0, "goals_for": 0, "goals_against": 0}
        )
        self.team_names = {}

        self.total_matches = 0
        self.total_goals = 0

        # MLE Parameters
        self.params = {}  # will hold att, def, rho, gamma
        self.team_ids = []
        self.id_to_idx = {}

        self.home_adv_elo = 80.0
        self.half_life_days = 365.0

    def expected_score(
        self,
        rating_a: float,
        rating_b: float,
        is_home: bool = False,
        is_neutral: bool = False,
    ) -> float:
        adv = self.home_adv_elo if (is_home and not is_neutral) else 0.0
        return 1 / (1 + 10 ** ((rating_b - (rating_a + adv)) / 400.0))

    def get_k_factor(self, league_id: int) -> float:
        if league_id == 1:
            return 60.0
        if league_id in [4, 7, 9, 22]:
            return 50.0
        if league_id == 6:
            return 40.0
        if league_id == 10:
            return 10.0
        return 20.0

    def _get_team_stat(self, stats_array: list, team_id: int, stat_type: str) -> float:
        if not stats_array:
            return 0.0
        for team_stat in stats_array:
            if team_stat["team"]["id"] == team_id:
                for s in team_stat["statistics"]:
                    if s["type"] == stat_type:
                        val = s["value"]
                        if val is None:
                            return 0.0
                        if isinstance(val, str) and "%" in val:
                            return float(val.replace("%", ""))
                        return float(val)
        return 0.0

    def _calc_proxy_xg(self, stats_array: list, team_id: int) -> float:
        sog = self._get_team_stat(stats_array, team_id, "Shots on Goal")
        soff = self._get_team_stat(stats_array, team_id, "Shots off Goal")
        # 代理模型：1个射正折算0.25球，1个射偏折算0.05球
        return sog * 0.25 + soff * 0.05

    def process_match(self, match: dict, matches_list: list, max_timestamp: int):
        status = match["fixture"]["status"]["short"]
        if status not in ["FT", "AET", "PEN"]:
            return

        league_id = match["league"]["id"]
        home_id = match["teams"]["home"]["id"]
        away_id = match["teams"]["away"]["id"]
        self.team_names[home_id] = match["teams"]["home"]["name"]
        self.team_names[away_id] = match["teams"]["away"]["name"]

        home_actual_goals = match["goals"]["home"]
        away_actual_goals = match["goals"]["away"]

        if home_actual_goals is None or away_actual_goals is None:
            return

        is_neutral = league_id in NEUTRAL_LEAGUE_IDS

        # --- V5: Proxy-xG Blending ---
        stats_array = match.get("statistics", [])
        if stats_array:
            home_pxg = self._calc_proxy_xg(stats_array, home_id)
            away_pxg = self._calc_proxy_xg(stats_array, away_id)

            # 融合真实进球与期望进球 (60% 真实进球 + 40% 期望进球)
            home_blended = home_actual_goals * 0.6 + home_pxg * 0.4
            away_blended = away_actual_goals * 0.6 + away_pxg * 0.4
        else:
            home_blended = float(home_actual_goals)
            away_blended = float(away_actual_goals)

        # --- V4: Time-Decay Weighting ---
        match_ts = match["fixture"]["timestamp"]
        delta_days = max(0, max_timestamp - match_ts) / 86400.0
        decay_lambda = math.log(2) / self.half_life_days
        weight = math.exp(-decay_lambda * delta_days)

        # --- ELO Update (使用真实胜负关系) ---
        if home_actual_goals > away_actual_goals:
            s_home, s_away = 1.0, 0.0
        elif home_actual_goals < away_actual_goals:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5
            if status == "PEN":
                pen_home = match["score"]["penalty"]["home"]
                pen_away = match["score"]["penalty"]["away"]
                if pen_home is not None and pen_away is not None:
                    if pen_home > pen_away:
                        s_home, s_away = 0.75, 0.25
                    elif pen_home < pen_away:
                        s_home, s_away = 0.25, 0.75

        r_home = self.elo_ratings[home_id]
        r_away = self.elo_ratings[away_id]

        e_home = self.expected_score(
            r_home, r_away, is_home=True, is_neutral=is_neutral
        )
        e_away = self.expected_score(
            r_away, r_home, is_home=False, is_neutral=is_neutral
        )

        k = self.get_k_factor(league_id) * weight
        self.elo_ratings[home_id] = r_home + k * (s_home - e_home)
        self.elo_ratings[away_id] = r_away + k * (s_away - e_away)

        # --- Stats Update (使用 blended goals 进行进攻能力评估) ---
        self.team_stats[home_id]["matches"] += 1 * weight
        self.team_stats[home_id]["goals_for"] += home_blended * weight
        self.team_stats[home_id]["goals_against"] += away_blended * weight

        self.team_stats[away_id]["matches"] += 1 * weight
        self.team_stats[away_id]["goals_for"] += away_blended * weight
        self.team_stats[away_id]["goals_against"] += home_blended * weight

        self.total_matches += 1
        self.total_goals += home_blended + away_blended

        # MLE 接收 blended goals
        matches_list.append(
            (home_id, away_id, home_blended, away_blended, is_neutral, weight)
        )

    def _dixon_coles_negll(self, params, match_data, num_teams):
        att = params[:num_teams]
        def_ = params[num_teams : 2 * num_teams]
        rho = params[-2]
        gamma = params[-1]

        att_mean = np.mean(att)
        penalty = 1000 * (att_mean - 1.0) ** 2

        h_idx = match_data[:, 0].astype(int)
        a_idx = match_data[:, 1].astype(int)
        h_goals = match_data[:, 2]  # 此时是 blended goals，是浮点数
        a_goals = match_data[:, 3]
        is_neutral = match_data[:, 4]
        weights = match_data[:, 5]

        home_adv = np.where(is_neutral == 1, 1.0, gamma)
        lam = att[h_idx] * def_[a_idx] * home_adv
        mu = att[a_idx] * def_[h_idx]

        lam = np.clip(lam, 1e-5, 20)
        mu = np.clip(mu, 1e-5, 20)

        # 泊松分布对数似然 (支持浮点数的 Gamma 函数等价部分，忽略常数项)
        ll = np.sum(weights * (h_goals * np.log(lam) - lam + a_goals * np.log(mu) - mu))

        # 浮点数进球不再严格适用 Rho adjustment (只有0和1才需要调整)
        # 为简化，V5 中我们仍然对极低比分的比赛（blended < 1.5）进行平局因子修正
        tau = np.ones_like(lam)

        m00 = (h_goals < 0.5) & (a_goals < 0.5)
        m10 = (h_goals >= 0.5) & (h_goals < 1.5) & (a_goals < 0.5)
        m01 = (h_goals < 0.5) & (a_goals >= 0.5) & (a_goals < 1.5)
        m11 = (h_goals >= 0.5) & (h_goals < 1.5) & (a_goals >= 0.5) & (a_goals < 1.5)

        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m10] = 1 + lam[m10] * rho
        tau[m01] = 1 + mu[m01] * rho
        tau[m11] = 1 - rho

        tau = np.clip(tau, 1e-5, None)
        ll += np.sum(weights * np.log(tau))

        return -ll + penalty

    def fit(self, fixtures_file: str):
        with open(fixtures_file, "r", encoding="utf-8") as f:
            fixtures = json.load(f)

        if not fixtures:
            return

        fixtures.sort(key=lambda x: x["fixture"]["timestamp"])
        max_timestamp = max(match["fixture"]["timestamp"] for match in fixtures)

        matches_list = []
        for match in fixtures:
            self.process_match(match, matches_list, max_timestamp)

        print(
            f"ELO updated on {self.total_matches} matches (with V4 Time-Decay & V5 Proxy-xG)."
        )

        valid_teams = [k for k, v in self.team_stats.items() if v["matches"] >= 1.5]
        self.team_ids = valid_teams
        self.id_to_idx = {tid: i for i, tid in enumerate(valid_teams)}

        num_teams = len(valid_teams)
        if num_teams == 0:
            return

        filtered_matches = []
        for h, a, hg, ag, n, w in matches_list:
            if h in self.id_to_idx and a in self.id_to_idx:
                filtered_matches.append(
                    [self.id_to_idx[h], self.id_to_idx[a], hg, ag, 1 if n else 0, w]
                )

        match_data = np.array(filtered_matches)

        print(
            f"Starting V5 Proxy-xG MLE fitting for {num_teams} teams over {len(match_data)} matches..."
        )

        init_params = np.ones(2 * num_teams + 2)
        init_params[-2] = 0.0
        init_params[-1] = 1.2

        bounds = [(0.1, 5.0)] * (2 * num_teams) + [(-0.2, 0.2), (1.0, 2.0)]

        res = minimize(
            self._dixon_coles_negll,
            init_params,
            args=(match_data, num_teams),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200},
        )

        if res.success:
            print("MLE optimization converged successfully.")
        else:
            print("MLE optimization reached limits, using best found parameters.")

        best_params = res.x
        self.params = {
            "att": {self.team_ids[i]: best_params[i] for i in range(num_teams)},
            "def": {
                self.team_ids[i]: best_params[num_teams + i] for i in range(num_teams)
            },
            "rho": best_params[-2],
            "gamma": best_params[-1],
        }

    def predict_poisson(
        self,
        team_a_id: int,
        team_b_id: int,
        is_neutral: bool = True,
        penalty_a: float = 1.0,
        penalty_b: float = 1.0,
    ) -> Tuple[float, float, float]:
        if (
            not self.params
            or team_a_id not in self.params["att"]
            or team_b_id not in self.params["att"]
        ):
            return (0.368, 0.264, 0.368)

        att_a = self.params["att"][team_a_id] * penalty_a
        def_a = self.params["def"][team_a_id] / penalty_a
        att_b = self.params["att"][team_b_id] * penalty_b
        def_b = self.params["def"][team_b_id] / penalty_b
        rho = self.params["rho"]
        gamma = self.params["gamma"]

        home_adv = 1.0 if is_neutral else gamma
        lam = att_a * def_b * home_adv
        mu = att_b * def_a

        max_goals = 10
        prob_matrix = np.zeros((max_goals, max_goals))

        for i in range(max_goals):
            for j in range(max_goals):
                p_base = (math.exp(-lam) * (lam**i) / math.factorial(i)) * (
                    math.exp(-mu) * (mu**j) / math.factorial(j)
                )

                tau = 1.0
                if i == 0 and j == 0:
                    tau = 1 - lam * mu * rho
                elif i == 1 and j == 0:
                    tau = 1 + lam * rho
                elif i == 0 and j == 1:
                    tau = 1 + mu * rho
                elif i == 1 and j == 1:
                    tau = 1 - rho

                prob_matrix[i, j] = max(0.0, p_base * tau)

        prob_matrix /= np.sum(prob_matrix)

        prob_a_win = np.sum(np.tril(prob_matrix, -1))
        prob_draw = np.trace(prob_matrix)
        prob_b_win = np.sum(np.triu(prob_matrix, 1))

        return (float(prob_a_win), float(prob_draw), float(prob_b_win))

    def get_top_teams(self, n: int = 20, min_matches: float = 2.0):
        valid_teams = {
            k: v
            for k, v in self.elo_ratings.items()
            if self.team_stats[k]["matches"] >= min_matches
        }
        sorted_teams = sorted(valid_teams.items(), key=lambda x: x[1], reverse=True)
        results = []
        for team_id, elo in sorted_teams[:n]:
            results.append((self.team_names[team_id], round(elo, 1)))
        return results


if __name__ == "__main__":
    model = FootballModel()
    # 我们先容错测试旧数据，旧数据没有 statistics 会回退到纯真实进球
    model.fit("raw_fixtures.json")
    print("\n--- Top 15 Teams by ELO (Time-Decayed & Proxy-xG) ---")
    top_teams = model.get_top_teams(15, min_matches=2.0)
    for i, (name, elo) in enumerate(top_teams, 1):
        print(f"{i:2d}. {name:20s} | ELO: {elo}")
