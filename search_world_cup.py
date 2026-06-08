import os
import json
import urllib.request
from urllib.parse import quote
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

def search_league(keyword: str) -> Optional[Dict[str, Any]]:
    load_env()
    api_key = os.environ.get('API_FOOTBALL_KEY')
    
    # URL 编码搜索词
    query = quote(keyword)
    url = f"https://v3.football.api-sports.io/leagues?search={query}"
    
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
    data = search_league("World Cup")
    if data and "response" in data:
        # 只打印前几个匹配的联赛
        for item in data["response"][:5]:
            league = item.get("league", {})
            country = item.get("country", {})
            seasons = item.get("seasons", [])
            
            # 获取最近的几个赛季
            recent_seasons = [str(s.get("year")) for s in seasons[-3:]]
            
            print(f"ID: {league.get('id')} | Name: {league.get('name')} | Type: {league.get('type')}")
            print(f"Country: {country.get('name')}")
            print(f"Recent Seasons: {', '.join(recent_seasons)}")
            print("-" * 40)
