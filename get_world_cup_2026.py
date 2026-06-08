import os
import json
import urllib.request
from typing import Optional, Dict, Any

def load_env(file_path: str = ".env") -> None:
    if not os.path.exists(file_path):
        return
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip()

def get_fixtures(league_id: int, season: int) -> Optional[Dict[str, Any]]:
    load_env()
    api_key = os.environ.get('API_FOOTBALL_KEY')
    
    url = f"https://v3.football.api-sports.io/fixtures?league={league_id}&season={season}"
    
    headers = {'x-apisports-key': api_key}
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    data = get_fixtures(1, 2026)
    if data and "response" in data:
        fixtures = data["response"]
        print(f"找到 {len(fixtures)} 场 2026 世界杯比赛数据。")
        for f in fixtures[:3]:
            fixture = f.get("fixture", {})
            teams = f.get("teams", {})
            goals = f.get("goals", {})
            
            date = fixture.get("date")
            home = teams.get("home", {}).get("name")
            away = teams.get("away", {}).get("name")
            status = fixture.get("status", {}).get("long")
            
            print(f"[{date}] {home} vs {away} - 状态: {status}")
