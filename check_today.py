from datetime import datetime
from api_client import FootballAPIClient

def check_todays_matches():
    client = FootballAPIClient()
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"Checking matches for today ({today_str})...")
    
    # We query major leagues or just all fixtures for today
    res = client.get("/fixtures", {"date": today_str})
    
    if not res or "response" not in res:
        print("Failed to fetch data or no response.")
        return
        
    fixtures = res["response"]
    print(f"Total matches scheduled globally for today: {len(fixtures)}")
    
    # Filter for major leagues or notable matches
    # Leagues: 39 (Premier League), 140 (La Liga), 135 (Serie A), 78 (Bundesliga), 61 (Ligue 1), 1 (World Cup), 4 (Euro)
    major_league_ids = [39, 140, 135, 78, 61, 1, 4, 9, 6, 7, 22]
    
    major_matches = [f for f in fixtures if f["league"]["id"] in major_league_ids]
    print(f"Major tournament matches today: {len(major_matches)}\n")
    
    for f in major_matches:
        league_name = f["league"]["name"]
        home = f["teams"]["home"]["name"]
        away = f["teams"]["away"]["name"]
        status = f["fixture"]["status"]["short"]
        time_str = f["fixture"]["date"]
        
        print(f"[{league_name}] {time_str} | {home} vs {away} | Status: {status}")
        
if __name__ == "__main__":
    check_todays_matches()
