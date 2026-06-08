import json
from api_client import FootballAPIClient

def fetch_massive_fixtures():
    client = FootballAPIClient()
    
    # V5 Pro Era: 海量扩容
    # 包含了自 2018 以来的所有核心国际赛事
    leagues_to_fetch = [
        {"id": 1, "season": 2018, "name": "World Cup 2018"},
        {"id": 1, "season": 2022, "name": "World Cup 2022"},
        {"id": 1, "season": 2026, "name": "World Cup 2026"},
        {"id": 4, "season": 2020, "name": "Euro Championship 2020"},
        {"id": 4, "season": 2024, "name": "Euro Championship 2024"},
        {"id": 9, "season": 2019, "name": "Copa America 2019"},
        {"id": 9, "season": 2021, "name": "Copa America 2021"},
        {"id": 9, "season": 2024, "name": "Copa America 2024"},
        {"id": 6, "season": 2019, "name": "Africa Cup 2019"},
        {"id": 6, "season": 2021, "name": "Africa Cup 2021"},
        {"id": 6, "season": 2023, "name": "Africa Cup 2023"},
        {"id": 7, "season": 2019, "name": "Asian Cup 2019"},
        {"id": 7, "season": 2023, "name": "Asian Cup 2023"},
        {"id": 22, "season": 2019, "name": "Gold Cup 2019"},
        {"id": 22, "season": 2021, "name": "Gold Cup 2021"},
        {"id": 22, "season": 2023, "name": "Gold Cup 2023"}
    ]
    
    all_fixtures = []
    
    # 1. 抓取所有基础赛程
    print(">>> Phase 1: Fetching all fixture metadata...")
    for lq in leagues_to_fetch:
        print(f"Fetching data for {lq['name']} ...")
        res = client.get("/fixtures", {"league": lq["id"], "season": lq["season"]})
        if res and "response" in res:
            fixtures = res["response"]
            print(f" -> Found {len(fixtures)} matches.")
            all_fixtures.extend(fixtures)
            
    # 过滤掉未完成的比赛
    valid_fixtures = [m for m in all_fixtures if m["fixture"]["status"]["short"] in ["FT", "AET", "PEN"]]
    print(f"\nTotal completed matches found: {len(valid_fixtures)}")
    
    # 2. 深度充实数据 (Statistics & Lineups)
    print("\n>>> Phase 2: Enriching with Proxy-xG Statistics & Lineup Data...")
    enriched_fixtures = []
    
    total = len(valid_fixtures)
    for i, match in enumerate(valid_fixtures, 1):
        fid = match["fixture"]["id"]
        if i % 100 == 0:
            print(f"Processing Match {i}/{total} ...")
            
        # 抓取技术统计
        stat_res = client.get("/fixtures/statistics", {"fixture": fid})
        if stat_res and "response" in stat_res:
            match["statistics"] = stat_res["response"]
        else:
            match["statistics"] = []
            
        # 抓取首发阵容
        lineup_res = client.get("/fixtures/lineups", {"fixture": fid})
        if lineup_res and "response" in lineup_res:
            match["lineups"] = lineup_res["response"]
        else:
            match["lineups"] = []
            
        # V6 新增：抓取球员技术评分
        players_res = client.get("/fixtures/players", {"fixture": fid})
        if players_res and "response" in players_res:
            match["players"] = players_res["response"]
        else:
            match["players"] = []
            
        # V6 新增：抓取伤停情报
        injuries_res = client.get("/injuries", {"fixture": fid})
        if injuries_res and "response" in injuries_res:
            match["injuries"] = injuries_res["response"]
        else:
            match["injuries"] = []
            
        enriched_fixtures.append(match)
            
    # 3. 保存最终尊享版数据集
    print("\n>>> Phase 3: Saving to V5 massive dataset...")
    with open("raw_fixtures_v5.json", "w", encoding="utf-8") as f:
        json.dump(enriched_fixtures, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ SUCCESS: {len(enriched_fixtures)} matches fully enriched and saved to raw_fixtures_v5.json")

if __name__ == "__main__":
    fetch_massive_fixtures()
