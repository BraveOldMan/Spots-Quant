import os
import json
import urllib.request
from urllib.error import URLError, HTTPError
from typing import Optional, Dict, Any

def load_env(file_path: str = ".env") -> None:
    """简易的 .env 文件加载器，避免引入外部依赖"""
    if not os.path.exists(file_path):
        return
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

def get_api_football_status() -> Optional[Dict[str, Any]]:
    """测试连接 API Football 并获取状态"""
    # 加载环境变量
    load_env()
    
    # 获取 API Key，避免硬编码
    api_key = os.environ.get('API_FOOTBALL_KEY')
    if not api_key:
        print("错误: 未找到 API_FOOTBALL_KEY 环境变量。")
        print("请将 .env.example 复制为 .env，并填入您的 API Key。")
        return None
    
    # API-Football 接口要求
    url = "https://v3.football.api-sports.io/status"
    headers = {
        'x-apisports-key': api_key
    }
    
    req = urllib.request.Request(url, headers=headers)
    
    try:
        print(f"正在连接 API Football: {url} ...")
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            print("连接成功！")
            return result
            
    except HTTPError as e:
        print(f"HTTP 错误: {e.code} - {e.reason}")
        if e.code == 401:
            print("提示: 401 未授权，请检查您的 API Key 是否正确。")
        elif e.code == 403:
            print("提示: 403 被拒绝，请检查您的订阅状态。")
    except URLError as e:
        print(f"网络连接错误: {e.reason}")
    except Exception as e:
        print(f"发生未知错误: {e}")
        
    return None

if __name__ == "__main__":
    status_data = get_api_football_status()
    if status_data:
        print("\nAPI 状态返回如下：")
        print(json.dumps(status_data, indent=4, ensure_ascii=False))
