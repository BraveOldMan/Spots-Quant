import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_client import FootballAPIClient

class OddsDataPipeline:
    """
    模块 1: 数据基建与特征工程 (Data Pipeline)
    负责处理 API-Football 赔率数据，剥离抽水并重采样为连续时间序列矩阵。
    """
    def __init__(self, client: FootballAPIClient = None):
        self.client = client if client else FootballAPIClient()

    def fetch_odds_history(self, fixture_id: int) -> dict:
        """
        拉取特定赛事的历史盘口赔率（这里以演示和预留接口为主，实际视 API 权限调用 /odds 或相关历史接口）
        """
        # 注意: 如果需要深度历史轨迹，需调用 /odds 或 /odds/history 接口
        # 这里模拟返回请求
        return self.client.get("/odds", {"fixture": fixture_id})

    @staticmethod
    def calculate_margin_1x2(odds_home: float, odds_draw: float, odds_away: float) -> float:
        """
        计算 1X2 独赢盘的博彩公司抽水率 (Margin)
        Margin = (1 / Odds_home + 1 / Odds_draw + 1 / Odds_away) - 1
        """
        try:
            return (1.0 / odds_home + 1.0 / odds_draw + 1.0 / odds_away) - 1.0
        except ZeroDivisionError:
            return 0.0

    @staticmethod
    def calculate_fair_prob(odds_raw: float, margin: float) -> float:
        """
        剥离抽水，计算真实市场隐含公允概率 (P_fair)
        P_fair = (1 / Odds_raw) / (1 + Margin)
        """
        try:
            return (1.0 / odds_raw) / (1.0 + margin)
        except ZeroDivisionError:
            return 0.0

    def parse_api_response_to_dataframe(self, api_response: dict) -> pd.DataFrame:
        """
        将 API-Football 返回的嵌套 JSON 转化为扁平化的离散更新点 DataFrame。
        处理 1X2 (Match Winner) 盘口。
        """
        records = []
        
        if not api_response or "response" not in api_response:
            return pd.DataFrame()
            
        for match_odds in api_response["response"]:
            update_time = match_odds.get("update")
            if not update_time:
                continue
                
            for bookie in match_odds.get("bookmakers", []):
                bookie_name = bookie.get("name")
                for bet in bookie.get("bets", []):
                    # 提取独赢盘 (1x2)
                    if bet.get("name") == "Match Winner":
                        vals = {v["value"]: float(v["odd"]) for v in bet.get("values", [])}
                        if "Home" in vals and "Draw" in vals and "Away" in vals:
                            home_odd = vals["Home"]
                            draw_odd = vals["Draw"]
                            away_odd = vals["Away"]
                            
                            margin = self.calculate_margin_1x2(home_odd, draw_odd, away_odd)
                            p_fair_h = self.calculate_fair_prob(home_odd, margin)
                            p_fair_d = self.calculate_fair_prob(draw_odd, margin)
                            p_fair_a = self.calculate_fair_prob(away_odd, margin)
                            
                            records.append({
                                "update_time": pd.to_datetime(update_time),
                                "bookmaker": bookie_name,
                                "raw_home": home_odd,
                                "raw_draw": draw_odd,
                                "raw_away": away_odd,
                                "margin": margin,
                                "fair_prob_home": p_fair_h,
                                "fair_prob_draw": p_fair_d,
                                "fair_prob_away": p_fair_a
                            })
                            
        df = pd.DataFrame(records)
        return df

    def resample_time_series(self, df: pd.DataFrame, freq: str = '1h') -> pd.DataFrame:
        """
        时间序列对齐：按照 update_time 将离散的赔率更新点重采样为固定时间窗口（1H/4H）的连续时间序列特征矩阵。
        """
        if df.empty:
            return df
            
        # 按照时间和博彩公司排序
        df = df.sort_values("update_time")
        
        # 将 update_time 设为索引
        df.set_index("update_time", inplace=True)
        
        # 针对每个 bookmaker 进行重采样。前向填充(ffill)处理缺失数据，保证连续性
        resampled_dfs = []
        for bookie, group in df.groupby("bookmaker"):
            # 删除重复索引，保留最后一个
            group = group[~group.index.duplicated(keep='last')]
            
            # 使用指定频率重采样，取区间内最后一个变动值，若无变动则前向填充
            resampled = group.resample(freq).last().ffill()
            resampled["bookmaker"] = bookie
            resampled_dfs.append(resampled)
            
        if not resampled_dfs:
            return pd.DataFrame()
            
        final_df = pd.concat(resampled_dfs).reset_index()
        return final_df

if __name__ == "__main__":
    print("[Module 1] OddsDataPipeline 基础类测试初始化...")
    pipeline = OddsDataPipeline()
    # 模拟输入测试数据以验证逻辑
    margin = pipeline.calculate_margin_1x2(1.5, 4.0, 6.0)
    print(f"Test Margin: {margin:.4f}")
    print(f"Test Fair Prob Home: {pipeline.calculate_fair_prob(1.5, margin):.4f}")
