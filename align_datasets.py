import json
import time
import pandas as pd
from difflib import SequenceMatcher
from datetime import datetime

def similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def main():
    print("==================================================")
    print("   🧩 Betfair 关盘赔率 ↔ API-Football 赛果 模糊缝合")
    print("==================================================")

    # 1. 载入 API-Football 数据作为基准库
    print("[1] 加载 API-Football raw_fixtures_v5.json ...")
    try:
        with open("d:/Spots-Quant/raw_fixtures_v5.json", "r", encoding="utf-8") as f:
            fixtures = json.load(f)
    except Exception as e:
        print(f"读取 raw_fixtures_v5.json 失败: {e}")
        return

    # 构建快速查询内存池 (缓存 timestamp, home, away, hg, ag)
    api_db = []
    for match in fixtures:
        status = match["fixture"]["status"]["short"]
        if status not in ["FT", "AET", "PEN"]:
            continue
            
        home_goals = match["goals"]["home"]
        away_goals = match["goals"]["away"]
        if home_goals is None or away_goals is None:
            continue
            
        api_db.append({
            "ts": match["fixture"]["timestamp"],
            "home": match["teams"]["home"]["name"],
            "away": match["teams"]["away"]["name"],
            "hg": int(home_goals),
            "ag": int(away_goals)
        })
        
    print(f"    --> 已缓存 {len(api_db)} 场完赛比分数据。")

    # 按时间戳排序，支持更快的二分查找或直接遍历
    api_db.sort(key=lambda x: x["ts"])

    # 2. 读取 Betfair 巨型 CSV
    print("[2] 载入 betfair_closing_odds_full.csv ...")
    df_bf = pd.read_csv("d:/Spots-Quant/betfair_closing_odds_full.csv")
    print(f"    --> 共发现 {len(df_bf)} 笔关盘赔率事件。")

    matched_rows = []
    skipped_count = 0
    total_count = len(df_bf)

    print("[3] 启动时间窗 + Fuzzy Match 双核过滤 (请耐心等待) ...")
    start_time = time.time()
    
    # 转换为 dict records 加速遍历
    bf_records = df_bf.to_dict('records')
    
    for idx, row in enumerate(bf_records):
        if idx % 5000 == 0 and idx > 0:
            print(f"  ... 处理进度: {idx}/{total_count} (匹配成功率: {len(matched_rows)/idx*100:.1f}%)")

        try:
            # 解析时间 (2015-05-02T17:00:00.000Z)
            bf_time_str = str(row["market_time"])
            bf_dt = datetime.strptime(bf_time_str, "%Y-%m-%dT%H:%M:%S.000Z")
            bf_ts = int(bf_dt.timestamp())
        except ValueError:
            skipped_count += 1
            continue
            
        match_name = str(row["match_name"])
        if " v " not in match_name:
            skipped_count += 1
            continue
            
        bf_home, bf_away = match_name.split(" v ", 1)
        
        # O(N) 遍历筛选时间窗 (±24小时 = 86400秒)
        candidates = [m for m in api_db if abs(m["ts"] - bf_ts) <= 86400]
        
        if not candidates:
            skipped_count += 1
            continue
            
        best_match = None
        best_score = 0.0
        
        for cand in candidates:
            score_h = similar(bf_home, cand["home"])
            score_a = similar(bf_away, cand["away"])
            avg_score = (score_h + score_a) / 2.0
            
            if avg_score > best_score:
                best_score = avg_score
                best_match = cand
                
        # 相似度阈值判定
        if best_score > 0.65:
            # 反算十进制赔率 (去除 margin 影响，模拟 B365)
            # 因为数据是 prob，十进制赔率 = 1 / prob
            ph, pd_prob, pa = row["home_prob"], row["draw_prob"], row["away_prob"]
            if pd.isna(ph) or ph <= 0:
                skipped_count += 1
                continue
                
            odds_h = round(1.0 / ph, 2)
            odds_d = round(1.0 / pd_prob, 2)
            odds_a = round(1.0 / pa, 2)
            
            matched_rows.append({
                "Date": bf_dt.strftime("%d/%m/%Y"), # 适配回测格式
                "HomeTeam": best_match["home"],     # 统一使用 API 队名
                "AwayTeam": best_match["away"],
                "FTHG": best_match["hg"],
                "FTAG": best_match["ag"],
                "B365CH": odds_h,
                "B365CD": odds_d,
                "B365CA": odds_a
            })
        else:
            skipped_count += 1

    print("\n==================================================")
    print("               数据缝合 手术财报")
    print("==================================================")
    print(f"Betfair 原始数据量: {total_count}")
    print(f"成功匹配比分数量: {len(matched_rows)}")
    print(f"因时间窗或队名差异丢弃: {skipped_count}")
    print(f"整体缝合成功率: {(len(matched_rows)/total_count)*100:.2f}%")
    print(f"运算耗时: {time.time() - start_time:.2f} 秒")
    print("==================================================")

    if matched_rows:
        df_out = pd.DataFrame(matched_rows)
        # 按时间排序
        df_out["_dt"] = pd.to_datetime(df_out["Date"], format="%d/%m/%Y", errors="coerce")
        df_out = df_out.sort_values("_dt").drop(columns=["_dt"])
        
        out_path = "d:/Spots-Quant/betfair_matched_with_results.csv"
        df_out.to_csv(out_path, index=False)
        print(f"🚀 已保存纯净大算力数据集至: {out_path}")
    else:
        print("😭 匹配失败，未生成数据集。")

if __name__ == "__main__":
    main()
