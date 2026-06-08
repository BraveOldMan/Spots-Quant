import json
import sqlite3
from api_client import FootballAPIClient

class MissingHistoryFetcher:
    def __init__(self, db_path="spots_quant.db"):
        self.client = FootballAPIClient(db_path="api_cache.db")
        self.db_path = db_path
        
    def fetch_leagues(self):
        print("==================================================")
        print("   [V10] 历史断层数据全量修补程序启动")
        print("   (目标: 五大联赛 2023-2025 & 世预赛/友谊赛 2024-2026)")
        print("==================================================")
        
        targets = [
            # 1. 欧洲五大联赛 (2023 - 2025)
            {"id": 39, "season": 2023, "name": "Premier League"},
            {"id": 39, "season": 2024, "name": "Premier League"},
            {"id": 39, "season": 2025, "name": "Premier League"},
            {"id": 140, "season": 2023, "name": "La Liga"},
            {"id": 140, "season": 2024, "name": "La Liga"},
            {"id": 140, "season": 2025, "name": "La Liga"},
            {"id": 135, "season": 2023, "name": "Serie A"},
            {"id": 135, "season": 2024, "name": "Serie A"},
            {"id": 135, "season": 2025, "name": "Serie A"},
            {"id": 78, "season": 2023, "name": "Bundesliga"},
            {"id": 78, "season": 2024, "name": "Bundesliga"},
            {"id": 78, "season": 2025, "name": "Bundesliga"},
            {"id": 61, "season": 2023, "name": "Ligue 1"},
            {"id": 61, "season": 2024, "name": "Ligue 1"},
            {"id": 61, "season": 2025, "name": "Ligue 1"},
            
            # 2. 国际友谊赛与世预赛 (2024 - 2026) -> 用于本届世界杯的战前热身
            {"id": 10, "season": 2024, "name": "Friendlies"},
            {"id": 10, "season": 2025, "name": "Friendlies"},
            {"id": 10, "season": 2026, "name": "Friendlies"},
            # 假设各大洲世预赛ID (南美34, 欧洲32, 亚洲30, 非洲29, 北美33)
            {"id": 34, "season": 2024, "name": "WC Qualifiers CONMEBOL"},
            {"id": 34, "season": 2025, "name": "WC Qualifiers CONMEBOL"},
            {"id": 32, "season": 2024, "name": "WC Qualifiers Europe"},
            {"id": 32, "season": 2025, "name": "WC Qualifiers Europe"},
            {"id": 30, "season": 2024, "name": "WC Qualifiers Asia"},
            {"id": 30, "season": 2025, "name": "WC Qualifiers Asia"}
        ]
        
        all_fixtures = []
        for target in targets:
            print(f"📡 拉取赛程列表: {target['name']} - {target['season']}")
            res = self.client.get("/fixtures", {"league": target["id"], "season": target["season"]})
            if res and "response" in res:
                matches = res["response"]
                # 过滤出已完赛的
                finished = [m for m in matches if m["fixture"]["status"]["short"] in ["FT", "AET", "PEN"]]
                all_fixtures.extend(finished)
                print(f"   -> 找到 {len(finished)} 场完赛记录。")
                
        print(f"\n✅ 列表搜集完毕。总计待剥离赛事: {len(all_fixtures)} 场。")
        self._process_and_store(all_fixtures)
        
    def _is_fixture_in_db(self, fixture_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM raw_fixtures WHERE fixture_id = ?", (fixture_id,))
            return cursor.fetchone() is not None

    def _process_and_store(self, fixtures_list):
        records_to_insert = []
        total = len(fixtures_list)
        
        print("\n🚀 启动深度数据剥离与 SQLite 并行入库...")
        
        inserted_count = 0
        skipped_count = 0
        
        for i, match in enumerate(fixtures_list, 1):
            fid = match["fixture"]["id"]
            if i % 100 == 0:
                print(f" -> 进度: {i}/{total} (已入库 {inserted_count}, 缓存跳过 {skipped_count}) ...")
                
            if self._is_fixture_in_db(fid):
                skipped_count += 1
                continue
                
            # 抓取技术统计 (Statistics)
            stat_res = self.client.get("/fixtures/statistics", {"fixture": fid})
            match["statistics"] = stat_res["response"] if stat_res and "response" in stat_res else []
            
            # 抓取首发阵容 (Lineups)
            lineup_res = self.client.get("/fixtures/lineups", {"fixture": fid})
            match["lineups"] = lineup_res["response"] if lineup_res and "response" in lineup_res else []
            
            f_data = match.get("fixture", {})
            t_data = match.get("teams", {})
            g_data = match.get("goals", {})
            
            ts = f_data.get("timestamp")
            status = f_data.get("status", {}).get("short")
            h_id = t_data.get("home", {}).get("id")
            a_id = t_data.get("away", {}).get("id")
            h_name = t_data.get("home", {}).get("name")
            a_name = t_data.get("away", {}).get("name")
            h_g = g_data.get("home")
            a_g = g_data.get("away")
            
            raw_json_str = json.dumps(match)
            records_to_insert.append((fid, ts, status, h_id, a_id, h_name, a_name, h_g, a_g, raw_json_str))
            inserted_count += 1
            
            # 每积累 100 条写入一次数据库，防止断电丢失
            if len(records_to_insert) >= 100:
                self._write_to_db(records_to_insert)
                records_to_insert.clear()
                
        # 写入剩余的记录
        if records_to_insert:
            self._write_to_db(records_to_insert)
            
        print("\n==================================================")
        print("🏆 历史断层修补计划完美竣工！")
        print(f"本次新入库: {inserted_count} 场，跳过已有记录: {skipped_count} 场。")
        print("您的 spots_quant.db 已拥有全球顶级联赛和世界杯赛事的绝对记忆。")
        print("==================================================")

    def _write_to_db(self, records):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT OR REPLACE INTO raw_fixtures 
                (fixture_id, timestamp, status, home_team_id, away_team_id, home_team_name, away_team_name, home_goals, away_goals, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)
            conn.commit()

if __name__ == "__main__":
    fetcher = MissingHistoryFetcher()
    fetcher.fetch_leagues()
