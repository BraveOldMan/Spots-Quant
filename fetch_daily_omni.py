import json
import sqlite3
from datetime import datetime, timedelta
from api_client import FootballAPIClient

class OmniDataPipeline:
    def __init__(self, db_path="spots_quant.db"):
        self.client = FootballAPIClient(db_path="api_cache.db")
        self.db_path = db_path
        
    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS raw_fixtures (
                    fixture_id INTEGER PRIMARY KEY,
                    timestamp INTEGER,
                    status TEXT,
                    home_team_id INTEGER,
                    away_team_id INTEGER,
                    home_team_name TEXT,
                    away_team_name TEXT,
                    home_goals INTEGER,
                    away_goals INTEGER,
                    raw_json TEXT
                )
            ''')
            conn.commit()

    def fetch_date(self, target_date_str: str):
        """
        拉取指定日期的所有全球比赛
        """
        print("==================================================")
        print(f"🌍 Omni-Pipeline: 正在全栈扫描全球赛程 ({target_date_str})")
        print("==================================================")
        
        # 1. 获取当天的全量对阵名单
        res = self.client.get("/fixtures", {"date": target_date_str})
        if not res or "response" not in res:
            print("❌ 无法获取赛事列表。请检查 API_KEY 或网络状态。")
            return
            
        all_fixtures = res["response"]
        print(f"📡 雷达锁定: 当天全球共有 {len(all_fixtures)} 场比赛。")
        
        # 挑选出已经完赛的，我们需要深度抓取它们的技术统计用于更新数据库
        finished_fixtures = [f for f in all_fixtures if f["fixture"]["status"]["short"] in ["FT", "AET", "PEN"]]
        
        print(f"✅ 已完赛: {len(finished_fixtures)} 场 (准备深度入库)")
        
        # 2. 深度抓取完赛数据并直接入库
        self._process_and_store(finished_fixtures)
        
    def _process_and_store(self, fixtures_list):
        if not fixtures_list:
            return
        
        self.init_db()
        records_to_insert = []
        total = len(fixtures_list)
        
        print("\n🚀 启动深度数据剥离与 SQLite 入库流水线...")
        
        # 为了演示和不耗尽额度，如果太多，我们可以选择性限制或者提醒 (目前先全量尝试，因为 client 有限速)
        # 注意: api_client_v3 中的 get 会有 0.25 秒延迟
        
        for i, match in enumerate(fixtures_list, 1):
            fid = match["fixture"]["id"]
            if i % 50 == 0:
                print(f" -> 进度: {i}/{total} ...")
                
            # 我们仅对数据库里没有的赛事进行深度查库
            if self._is_fixture_in_db(fid):
                continue
                
            # 抓取技术统计 (Statistics)
            stat_res = self.client.get("/fixtures/statistics", {"fixture": fid})
            match["statistics"] = stat_res["response"] if stat_res and "response" in stat_res else []
            
            # 抓取首发阵容 (Lineups)
            lineup_res = self.client.get("/fixtures/lineups", {"fixture": fid})
            match["lineups"] = lineup_res["response"] if lineup_res and "response" in lineup_res else []
            
            # 提取基础字段
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
            
        if records_to_insert:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.executemany('''
                    INSERT OR REPLACE INTO raw_fixtures 
                    (fixture_id, timestamp, status, home_team_id, away_team_id, home_team_name, away_team_name, home_goals, away_goals, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', records_to_insert)
                conn.commit()
            print(f"💾 写入完成！成功将 {len(records_to_insert)} 场全新赛果沉淀入 spots_quant.db 数据库！")
        else:
            print("ℹ️ 数据库已是最新，无新增赛事入库。")

    def _is_fixture_in_db(self, fixture_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM raw_fixtures WHERE fixture_id = ?", (fixture_id,))
            return cursor.fetchone() is not None

if __name__ == "__main__":
    # 默认抓取昨天的数据，因为今天的数据可能还没打完
    # 这样每天跑一次，就能确保前一天全球比赛全部入库
    pipeline = OmniDataPipeline()
    
    # 获取昨天的日期
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"由于需要抓取已完赛的技术统计，系统默认拉取 {yesterday} 的全球赛果。")
    
    pipeline.fetch_date(yesterday)
