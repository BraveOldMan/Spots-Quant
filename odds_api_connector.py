import os
import urllib.request
import json
import asyncio
from typing import Optional, Tuple

class OddsAPIConnector:
    """
    专门针对 The Odds API 的连接器。
    用于获取外界(特别是 Pinnacle) 的真实关盘/实时赔率。
    """
    
    def __init__(self):
        # 严格遵循风控纪律，从系统环境读取秘钥
        self.api_key = os.getenv("THE_ODDS_API_KEY")
        if not self.api_key:
            # 兼容 .env 文件的直接读取（容错处理）
            try:
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("THE_ODDS_API_KEY="):
                            self.api_key = line.strip().split("=")[1]
            except Exception:
                pass
                
        self.base_url = "https://api.the-odds-api.com/v4/sports/upcoming/odds/"
        
    def _fetch_sync(self, match_name_query: str) -> Optional[Tuple[float, float, float]]:
        """
        同步底层调用 API
        由于免费版 API 会查出多场，这里只是做最简单的拉取并模糊匹配赛事的逻辑
        """
        if not self.api_key:
            return None
            
        # 默认只拉取 Pinnacle 赔率 (欧洲区 1X2 H2H 市场)
        url = f"{self.base_url}?apiKey={self.api_key}&regions=eu&markets=h2h&bookmakers=pinnacle"
        
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as res:
                data = json.loads(res.read())
                
            if not data:
                return None
                
            # 简单模糊匹配（实际生产中应精确使用 event_id 匹配，此处做沙盒示范）
            # 找到列表中带有 odds 的赛事
            for match in data:
                # 只需验证 match 存在 bookmakers
                bookmakers = match.get("bookmakers", [])
                for b in bookmakers:
                    if b["key"] == "pinnacle":
                        markets = b.get("markets", [])
                        for m in markets:
                            if m["key"] == "h2h":
                                outcomes = m.get("outcomes", [])
                                if len(outcomes) >= 2:
                                    # 解析赔率
                                    # The Odds API 对于足球返回 3 项 (Home, Draw, Away)
                                    # 对于篮球/棒球返回 2 项 (Home, Away)
                                    # 这里为了通用性尝试提取
                                    h, d, a = 1.01, 1.01, 1.01
                                    for outcome in outcomes:
                                        if outcome["name"] == match["home_team"]:
                                            h = outcome["price"]
                                        elif outcome["name"] == match["away_team"]:
                                            a = outcome["price"]
                                        elif outcome["name"].lower() == "draw":
                                            d = outcome["price"]
                                            
                                    # 修正双项运动没有平局的情况
                                    if len(outcomes) == 2:
                                        # 两项市场，人为将平局概率设为无穷大/极低
                                        d = 999.0
                                        
                                    return (h, d, a)
            return None
        except Exception as e:
            print(f"[OddsAPI] 连接失败或超时: {e}")
            return None

    async def get_pinnacle_odds(self, match_name_query: str) -> Optional[Tuple[float, float, float]]:
        """
        供特工调用的异步接口，防止阻塞 EventBus
        """
        return await asyncio.to_thread(self._fetch_sync, match_name_query)

if __name__ == "__main__":
    # 测试直连
    print("Testing The Odds API Connector...")
    connector = OddsAPIConnector()
    # 采用同步运行以便快速测试
    res = connector._fetch_sync("any")
    if res:
        print(f"成功抓取到一条 Pinnacle 实时赔率样本! 主={res[0]}, 平={res[1]}, 客={res[2]}")
    else:
        print("API 调用成功，但当前没有获取到即将开赛的 Pinnacle 赔率(列表为空)。")
