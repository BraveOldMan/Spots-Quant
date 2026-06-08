import json
from datetime import datetime
from api_client import FootballAPIClient

def fetch_today_odds():
    client = FootballAPIClient()
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Fetching odds for date: {today_str} ...")
    # API 免费版限制: 必须提供有效的、接近当前的日期
    res = client.get("/odds", {"date": today_str})
    
    if not res or "response" not in res:
        print("Failed to fetch odds or no response.")
        return

    odds_data = res["response"]
    print(f"Total fixtures with odds found: {len(odds_data)}")
    
    parsed_odds = []
    
    for item in odds_data:
        fixture_id = item["fixture"]["id"]
        bookmakers = item["bookmakers"]
        
        # 优先选择 Bet365 (id: 8) 或其他主流博彩公司，取第一个可用的
        selected_bookmaker = None
        for b in bookmakers:
            if b["id"] == 8: # Bet365
                selected_bookmaker = b
                break
        if not selected_bookmaker and bookmakers:
            selected_bookmaker = bookmakers[0]
            
        if not selected_bookmaker:
            continue
        
        # 提取 "Match Winner" 赔率 (Home, Draw, Away)
        bets = selected_bookmaker["bets"]
        match_winner_bet = next((b for b in bets if b["name"] == "Match Winner"), None)
        
        if not match_winner_bet:
            continue
        
        values = {v["value"]: float(v["odd"]) for v in match_winner_bet["values"]}
        
        if "Home" in values and "Away" in values and "Draw" in values:
            parsed_odds.append({
                "fixture_id": fixture_id,
                "bookmaker": selected_bookmaker["name"],
                "odds": {
                    "home": values["Home"],
                    "draw": values["Draw"],
                    "away": values["Away"]
                }
            })
            
    with open("today_odds.json", "w", encoding="utf-8") as f:
        json.dump(parsed_odds, f, indent=2)
        
    print(f"Successfully saved {len(parsed_odds)} valid match winner odds to today_odds.json.")

if __name__ == "__main__":
    fetch_today_odds()
