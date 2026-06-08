import tarfile
import bz2
import json
import time
import os
import csv
from difflib import SequenceMatcher


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def map_odds_to_hda(closing_odds: dict, runner_names: dict, event_name: str):
    """
    将 Betfair runner 赔率按【语义】映射为 (home_prob, draw_prob, away_prob)。

    关键修复：旧逻辑用 sorted(selectionId) 决定列顺序，与主/平/客语义无关，
    导致 "The Draw" 常被错写进 away_prob。此函数改用 runner 名称 + eventName 做语义对齐：
      - 平局：runner 名称含 "draw"；
      - 主/客：用 eventName "Home v Away" 拆出的队名与 runner 名称做相似度匹配。

    返回 (home_prob, draw_prob, away_prob) 或在无法可靠识别时返回 None。
    """
    if len(closing_odds) < 3:
        return None

    implied_sum = sum(1.0 / p for p in closing_odds.values() if p and p > 0)
    if implied_sum <= 0:
        return None

    # eventName 形如 "Tottenham v Man City"
    parts = event_name.split(" v ")
    if len(parts) != 2:
        return None
    home_name, away_name = parts[0].strip(), parts[1].strip()

    draw_rid = None
    contender_rids = []
    for rid in closing_odds:
        name = str(runner_names.get(rid, "")).lower()
        if "draw" in name:
            draw_rid = rid
        else:
            contender_rids.append(rid)

    # 必须恰好识别出 1 个平局 + 2 个非平局选项
    if draw_rid is None or len(contender_rids) != 2:
        return None

    r1, r2 = contender_rids
    n1 = runner_names.get(r1, "")
    n2 = runner_names.get(r2, "")

    # 判断 r1/r2 哪个是主队：比较两种指派下的总相似度
    score_r1_home = _similar(n1, home_name) + _similar(n2, away_name)
    score_r2_home = _similar(n2, home_name) + _similar(n1, away_name)
    if score_r1_home >= score_r2_home:
        home_rid, away_rid = r1, r2
    else:
        home_rid, away_rid = r2, r1

    def prob(rid):
        return (1.0 / closing_odds[rid]) / implied_sum

    return prob(home_rid), prob(draw_rid), prob(away_rid)


def extract_all_betfair_odds(
    tar_path,
    output_csv="betfair_closing_odds_full.csv",
    progress_file="betfair_progress.txt",
):
    print("==================================================")
    print("   [企业级提纯器] Betfair 全量清洗线启动")
    print("==================================================")

    if not os.path.exists(tar_path):
        print(f"❌ 找不到文件 {tar_path}")
        return

    # 获取上一次的断点
    processed_count = 0
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            try:
                processed_count = int(f.read().strip())
                print(f"🔄 检测到断点，将跳过前 {processed_count} 个已处理的数据块。")
            except ValueError:
                pass

    start_time = time.time()

    # 准备 CSV 头
    file_exists = os.path.exists(output_csv)

    try:
        tar = tarfile.open(tar_path, "r:")
    except Exception as e:
        print(f"Tar 穿透失败: {e}")
        return

    extracted_count = 0
    scanned_count = 0

    with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "match_name",
            "market_time",
            "home_prob",
            "draw_prob",
            "away_prob",
            "raw_odds",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists or processed_count == 0:
            writer.writeheader()

        print("✅ 开始深度流水线作业 (每 1000 场自动 Checkpoint 落盘)...")

        for member in tar:
            if not (member.isfile() and member.name.endswith(".bz2")):
                continue

            scanned_count += 1
            if scanned_count <= processed_count:
                continue  # 跳过断点之前的文件

            f = tar.extractfile(member)
            if f is None:
                continue

            try:
                bz2_data = f.read()
                json_str_data = bz2.decompress(bz2_data).decode("utf-8")
                lines = [line for line in json_str_data.strip().split("\n") if line]

                if len(lines) == 0:
                    continue

                first_tick = json.loads(lines[0])
                if "mc" not in first_tick or len(first_tick["mc"]) == 0:
                    continue

                market_def = first_tick["mc"][0].get("marketDefinition", {})
                market_type = market_def.get("marketType", "")

                if market_type == "MATCH_ODDS":
                    event_name = market_def.get("eventName", "Unknown")
                    market_time = market_def.get("marketTime", "Unknown")

                    runners = market_def.get("runners", [])
                    runner_names = {r["id"]: r["name"] for r in runners}

                    closing_odds = {}

                    for line in reversed(lines):
                        tick = json.loads(line)
                        if "mc" in tick:
                            for mc in tick["mc"]:
                                if "rc" in mc:
                                    for rc in mc["rc"]:
                                        rid = rc["id"]
                                        if "ltp" in rc and rid not in closing_odds:
                                            closing_odds[rid] = rc["ltp"]
                        if len(closing_odds) >= 3:
                            break

                    if len(closing_odds) >= 3:
                        # 语义映射：按 runner 名称 + eventName 还原 主/平/客，杜绝列错位
                        hda = map_odds_to_hda(closing_odds, runner_names, event_name)
                        if hda is not None:
                            home_prob, draw_prob, away_prob = hda
                            writer.writerow(
                                {
                                    "match_name": event_name,
                                    "market_time": market_time,
                                    "home_prob": home_prob,
                                    "draw_prob": draw_prob,
                                    "away_prob": away_prob,
                                    # 同时保存 id->name 映射，便于事后审计与复现
                                    "raw_odds": json.dumps(
                                        {
                                            str(rid): {
                                                "price": closing_odds[rid],
                                                "name": runner_names.get(rid, str(rid)),
                                            }
                                            for rid in closing_odds
                                        }
                                    ),
                                }
                            )
                            extracted_count += 1

            except Exception:
                pass  # 静默忽略单点损坏，保证流水线不中断

            # 每扫描 1000 个流，保存一次断点
            if scanned_count % 1000 == 0:
                with open(progress_file, "w") as pf:
                    pf.write(str(scanned_count))
                print(
                    f"[{time.strftime('%H:%M:%S')}] 进度播报: 已扫描 {scanned_count} 个流，成功提纯 {extracted_count} 场比赛。 (Checkpoint 已更新)"
                )

    tar.close()

    # 彻底完成（断点写入失败不应让整个提纯任务以非零码退出，CSV 此时已落盘）
    try:
        with open(progress_file, "w") as pf:
            pf.write(str(scanned_count))
    except OSError as e:
        print(f"⚠️ 断点文件写入失败（可忽略，CSV 已完整落盘）: {e}")

    print("\n==================================================")
    print("🏆 全量提纯彻底完成！")
    print(f"总计扫描数据流: {scanned_count}")
    print(f"成功萃取 Match Odds 场次: {extracted_count}")
    print(f"总耗时: {(time.time() - start_time) / 60:.1f} 分钟。")
    print("==================================================")


if __name__ == "__main__":
    TAR_PATH = r"C:\Users\MrLee\Downloads\betfair_data.tar"
    extract_all_betfair_odds(TAR_PATH)
