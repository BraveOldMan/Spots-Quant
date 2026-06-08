import sqlite3
import pandas as pd
import json
import time
import os

def migrate_database():
    db_path = "spots_quant.db"
    print("==================================================")
    print("   [V10] 数据基建升维：JSON/CSV -> SQLite")
    print("   (彻底消除大文件 OOM 隐患，支持毫秒级 SQL 穿透)")
    print("==================================================")
    
    # 建立数据库连接
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    start_time = time.time()
    
    # ==========================================
    # 1. 迁移必发 P_model 矩阵 (CSV -> SQLite)
    # ==========================================
    csv_file = "betfair_closing_odds_full.csv"
    if os.path.exists(csv_file):
        print(f"\n[1] 正在迁移 {csv_file} (列式并行装载)...")
        # Pandas 直接 to_sql 是最快的方式
        df_csv = pd.read_csv(csv_file)
        
        # 将原始数据表写入 SQLite
        df_csv.to_sql('betfair_odds', conn, if_exists='replace', index=False)
        
        # 建立索引加速查询
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_match_name ON betfair_odds(match_name)")
        print(f"✅ 必发矩阵迁移完成！共计 {len(df_csv):,} 行。")
    else:
        print(f"⚠️ 未找到 {csv_file}")
        
    # ==========================================
    # 2. 迁移 Kaggle / API-Football 庞大 JSON (JSON -> SQLite)
    # ==========================================
    json_file = "raw_fixtures_v5.json"
    if os.path.exists(json_file):
        print(f"\n[2] 正在解析巨型对象 {json_file} (结构化扁平装载)...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 建表
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
        
        # 清空旧数据
        cursor.execute("DELETE FROM raw_fixtures")
        
        # 批量插入准备
        records = []
        for match in data:
            f_data = match.get("fixture", {})
            t_data = match.get("teams", {})
            g_data = match.get("goals", {})
            
            fid = f_data.get("id")
            ts = f_data.get("timestamp")
            status = f_data.get("status", {}).get("short")
            h_id = t_data.get("home", {}).get("id")
            a_id = t_data.get("away", {}).get("id")
            h_name = t_data.get("home", {}).get("name")
            a_name = t_data.get("away", {}).get("name")
            h_g = g_data.get("home")
            a_g = g_data.get("away")
            
            # 把深层嵌套的结构作为文本存入，方便后续用 JSON 函数解析
            raw_json_str = json.dumps(match)
            
            records.append((fid, ts, status, h_id, a_id, h_name, a_name, h_g, a_g, raw_json_str))
            
        cursor.executemany('''
            INSERT INTO raw_fixtures (fixture_id, timestamp, status, home_team_id, away_team_id, home_team_name, away_team_name, home_goals, away_goals, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', records)
        
        # 建立索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON raw_fixtures(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_teams ON raw_fixtures(home_team_id, away_team_id)")
        
        conn.commit()
        print(f"✅ 基础库迁移完成！共计 {len(records):,} 场历史赛事。")
    else:
        print(f"⚠️ 未找到 {json_file}")
        
    conn.close()
    
    elapsed = time.time() - start_time
    print("\n==================================================")
    print(f"🏆 V10 数据底座搭建完毕！(耗时 {elapsed:.2f} 秒)")
    print("   现在的系统拥有了毫秒级的 SQL 联表穿透能力！")
    print("==================================================")

if __name__ == "__main__":
    migrate_database()
