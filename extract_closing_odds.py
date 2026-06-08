import tarfile
import bz2
import json
import time
import os
import pandas as pd

def extract_betfair_closing_odds(tar_path, max_matches=5000):
    print("==================================================")
    print("   [P_model 提纯器] Betfair 关盘赔率提取引擎")
    print("==================================================")
    
    if not os.path.exists(tar_path):
        print(f"❌ 找不到文件 {tar_path}")
        return
        
    start_time = time.time()
    extracted_data = []
    
    try:
        tar = tarfile.open(tar_path, 'r:')
    except Exception as e:
        print(f"Tar 穿透失败: {e}")
        return

    print("✅ 开始深度遍历提取 MATCH_ODDS (胜平负) 关盘水位...")
    
    match_count = 0
    
    for member in tar:
        if member.isfile() and member.name.endswith('.bz2'):
            f = tar.extractfile(member)
            if f is None:
                continue
            
            try:
                bz2_data = f.read()
                json_str_data = bz2.decompress(bz2_data).decode('utf-8')
                lines = [line for line in json_str_data.strip().split('\n') if line]
                
                if len(lines) == 0:
                    continue
                
                # 检查是否为胜平负市场
                first_tick = json.loads(lines[0])
                if 'mc' not in first_tick or len(first_tick['mc']) == 0:
                    continue
                
                market_def = first_tick['mc'][0].get('marketDefinition', {})
                market_type = market_def.get('marketType', '')
                
                # 只提纯 MATCH_ODDS (足球胜平负)
                if market_type == 'MATCH_ODDS':
                    event_name = market_def.get('eventName', 'Unknown')
                    market_time = market_def.get('marketTime', 'Unknown')
                    
                    # 获取该市场的选项 (Home, Draw, Away)
                    runners = market_def.get('runners', [])
                    runner_names = {r['id']: r['name'] for r in runners}
                    
                    # 寻找关盘赔率 (即最后一次有交易的赔率)
                    # 倒序遍历 Tick，寻找最后的最佳可买入赔率 (LTP/BATP)
                    closing_odds = {}
                    
                    for line in reversed(lines):
                        tick = json.loads(line)
                        if 'mc' in tick:
                            for mc in tick['mc']:
                                if 'rc' in mc:  # Runner Changes
                                    for rc in mc['rc']:
                                        rid = rc['id']
                                        # LTP = Last Traded Price (最后成交价)
                                        if 'ltp' in rc and rid not in closing_odds:
                                            closing_odds[rid] = rc['ltp']
                        # 如果找齐了全部三个选项的赔率，就停止
                        if len(closing_odds) >= 3:
                            break
                            
                    if len(closing_odds) >= 2:
                        # 计算除权真实概率 (True Probability)
                        # 假设赔率分别为 O1, O2, O3
                        # Implied Prob = 1/O1 + 1/O2 + 1/O3
                        # True Prob = (1/O_i) / Implied Prob
                        implied_sum = sum(1.0/price for price in closing_odds.values() if price > 0)
                        
                        true_probs = {}
                        if implied_sum > 0:
                            for rid, price in closing_odds.items():
                                name = runner_names.get(rid, str(rid))
                                true_prob = (1.0 / price) / implied_sum
                                true_probs[name] = true_prob
                                
                        extracted_data.append({
                            'match_name': event_name,
                            'market_time': market_time,
                            'closing_odds': closing_odds,
                            'true_probabilities': true_probs
                        })
                        match_count += 1
                        
                        if match_count % 100 == 0:
                            print(f" -> 已提纯 {match_count} 场比赛的关盘真理概率...")
                            
                    if match_count >= max_matches:
                        break
                        
            except Exception:
                pass # 忽略损坏的流
                
    tar.close()
    
    # 保存结果
    df = pd.DataFrame(extracted_data)
    df.to_csv("betfair_closing_odds.csv", index=False)
    
    print("\n==================================================")
    print(f"✅ 提纯任务完成！共提取 {len(extracted_data)} 场比赛。")
    print("纯净 P_model 数据已保存至: betfair_closing_odds.csv")
    print(f"耗时: {time.time() - start_time:.2f} 秒。")
    print("==================================================")

if __name__ == "__main__":
    TAR_PATH = r"C:\Users\MrLee\Downloads\betfair_data.tar"
    # 为了演示速度，这里先提取 200 场比赛作为 P_model 雏形
    extract_betfair_closing_odds(TAR_PATH, max_matches=200)
