import json
from api_client import FootballAPIClient

def fetch_2015_premier_league():
    client = FootballAPIClient()
    
    print("==================================================")
    print("   [Phase 1 Hotfix] 补录 2015 赛季历史基座数据")
    print("==================================================")
    
    # 获取英超 (League 39) 2015 赛季的数据
    leagues_to_fetch = [
        {"id": 39, "season": 2015, "name": "Premier League 2015"}
    ]
    
    all_fixtures = []
    
    # 1. 抓取基础赛程
    print(">>> 正在抓取赛程元数据...")
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
    
    # 2. 深度充实数据 (Statistics & Lineups) - 为了省额度和速度，我们暂时只抓取基础特征所需的 Lineups 和 Players
    # 其实 ID Mapping 只需要比赛 ID 和队伍名称，但为了后续能喂给 XGBoost，我们要尽量补全。
    print("\n>>> 正在抓取阵容和球员数据 (消耗大量 API)...")
    enriched_fixtures = []
    
    total = len(valid_fixtures)
    for i, match in enumerate(valid_fixtures, 1):
        fid = match["fixture"]["id"]
        if i % 50 == 0:
            print(f"Processing Match {i}/{total} ...")
            
        # 抓取首发阵容
        lineup_res = client.get("/fixtures/lineups", {"fixture": fid})
        match["lineups"] = lineup_res.get("response", []) if lineup_res else []
            
        # 抓取球员技术评分
        players_res = client.get("/fixtures/players", {"fixture": fid})
        match["players"] = players_res.get("response", []) if players_res else []
            
        # 伤停情报 (早期赛季可能没有)
        injuries_res = client.get("/injuries", {"fixture": fid})
        match["injuries"] = injuries_res.get("response", []) if injuries_res else []
            
        enriched_fixtures.append(match)
            
    # 3. 追加合并到 raw_fixtures_v5.json
    print("\n>>> 正在将 2015 赛季数据并入主库...")
    try:
        with open("raw_fixtures_v5.json", "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []
        
    existing_data.extend(enriched_fixtures)
    
    with open("raw_fixtures_v5.json", "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ SUCCESS: 成功补录 {len(enriched_fixtures)} 场 2015 赛季比赛！")
    print("现在可以重新运行 align_datasets.py 进行 ID 匹配了。")

if __name__ == "__main__":
    fetch_2015_premier_league()
