"""Search API-Football qualifier leagues using an environment API key."""

import json
import os
import urllib.parse
import urllib.request
from typing import Any

from api_client import load_env


def fetch_qualifier_leagues(keyword: str) -> list[dict[str, Any]]:
    """Return API-Football league search results for a qualifier keyword."""
    load_env()
    api_key = os.environ.get("API_FOOTBALL_KEY")
    if not api_key:
        raise RuntimeError(
            "API_FOOTBALL_KEY is missing. Set it in .env or the environment."
        )

    query = urllib.parse.quote(keyword)
    url = f"https://v3.football.api-sports.io/leagues?search={query}"
    request = urllib.request.Request(url, headers={"x-apisports-key": api_key})

    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    return payload.get("response", [])


def main() -> None:
    """Print matching World Cup qualification leagues."""
    leagues = fetch_qualifier_leagues("World Cup - Qualification")
    for league_info in leagues:
        league = league_info["league"]
        country = league_info["country"]
        print(f"{league['id']}: {league['name']} - {country['name']}")


if __name__ == "__main__":
    main()
