import tarfile
import bz2
import json
import time
import os

def inspect_betfair_tar(tar_path):
    print("==================================================")
    print("   [Betfair 探针] 内存流式穿透引擎启动")
    print(f"   目标黑洞: {tar_path}")
    print("==================================================")
    
    if not os.path.exists(tar_path):
        print(f"❌ 找不到文件 {tar_path}")
        return
        
    start_time = time.time()
    
    # 使用流式读取 ('r|') 避免将完整的 index 加载到内存
    # 但如果是标准 tar 可以用 'r:' 
    try:
        tar = tarfile.open(tar_path, 'r:')
    except Exception as e:
        print(f"Tar 穿透失败: {e}")
        return

    print("✅ 成功连接压缩流，正在下潜深度提取第一个有效行情碎片...")
    
    count = 0
    max_inspect = 5  # 只查看前 5 个文件
    
    for member in tar:
        # 只处理 bz2 格式的独立市场数据文件
        if member.isfile() and member.name.endswith('.bz2'):
            print(f"\n>> 发现市场流: {member.name} (大小: {member.size} 字节)")
            
            # 在内存中直接读取 bz2 数据流
            f = tar.extractfile(member)
            if f is not None:
                bz2_data = f.read()
                try:
                    # 解压 bz2
                    json_str_data = bz2.decompress(bz2_data).decode('utf-8')
                    
                    # Betfair 数据通常是由换行符分隔的多行 JSON 
                    lines = json_str_data.strip().split('\n')
                    print(f"   该市场流包含了 {len(lines)} 次高频 Tick 跳动。")
                    
                    if len(lines) > 0:
                        # 解析第一个和最后一个 Tick 以判断这是什么比赛的什么市场
                        first_tick = json.loads(lines[0])
                        print("   [Tick 1 (初始状态)]:")
                        # 打印部分关键信息，避免刷屏
                        if 'mc' in first_tick and len(first_tick['mc']) > 0:
                            market_def = first_tick['mc'][0].get('marketDefinition', {})
                            print(f"     赛事名称: {market_def.get('eventName', 'Unknown')}")
                            print(f"     市场类型: {market_def.get('marketType', 'Unknown')}")
                            print(f"     开赛时间: {market_def.get('marketTime', 'Unknown')}")
                        
                        last_tick = json.loads(lines[-1])
                        print(f"   [Tick N (最终状态)]: Timestamp={last_tick.get('pt')}")
                except Exception as e:
                    print(f"   解析失败: {e}")
                    
            count += 1
            if count >= max_inspect:
                break
                
    tar.close()
    print("\n==================================================")
    print(f"探针任务完成，耗时 {time.time() - start_time:.2f} 秒。")
    print("==================================================")

if __name__ == "__main__":
    TAR_PATH = r"C:\Users\MrLee\Downloads\betfair_data.tar"
    inspect_betfair_tar(TAR_PATH)
