from api_client import FootballAPIClient

client = FootballAPIClient()
res = client.get('/odds', {'date': '2024-06-05'})
if res and "response" in res:
    print(f"Found odds for {len(res['response'])} fixtures.")
else:
    print("No response or error:", res)
