import asyncio
import random
import time
import torch
import torch.nn as nn
from datetime import datetime

# 引入我们在 Phase 2 训练的 LSTM 模型架构
class OddsLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super(OddsLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out)

class LiveTradingEngine:
    def __init__(self):
        print(">>> 初始化 Live Trading Engine (高频猎手)...")
        self.device = torch.device('cpu')
        self.model = OddsLSTM().to(self.device)
        
        try:
            # 尝试加载 Phase 2 训练出的权重
            self.model.load_state_dict(torch.load("lstm_momentum.pth", map_location=self.device))
            self.model.eval()
            print("✅ LSTM 盘口形态记忆模型加载成功！")
        except Exception:
            print("⚠️ 警告: 未找到 lstm_momentum.pth，将使用未预训练的初始化权重进行演示。")
            self.model.eval()
            
        self.active_matches = {}
        # V9 架构新增：模拟一个 P_model 真理预测器
        self.mock_p_model = {}
        # ⚠️ 绝对红线：风控禁止实盘
        self.IS_LIVE_TRADING = False
        self.current_capital = 10000.0
        
    def process_tick(self, msg_data):
        match_id = msg_data['match_id']
        tick_time = msg_data['timestamp']
        odds = msg_data['odds']
        
        # 维护每场比赛的时间序列 (模拟凑够 13 个截面)
        if match_id not in self.active_matches:
            self.active_matches[match_id] = {'history': [], 'name': msg_data['match_name']}
            
        # [1/home, 1/draw, 1/away] 隐含概率
        implied_probs = [1.0/odds['home'], 1.0/odds['draw'], 1.0/odds['away']]
        self.active_matches[match_id]['history'].append(implied_probs)
        
        # 只保留最近 13 个截面 (模拟 24H -> 0H)
        if len(self.active_matches[match_id]['history']) > 13:
            self.active_matches[match_id]['history'].pop(0)
            
        # 满足序列长度后，送入 LSTM 嗅探
        if len(self.active_matches[match_id]['history']) == 13:
            seq_tensor = torch.tensor([self.active_matches[match_id]['history']], dtype=torch.float32)
            
            with torch.no_grad():
                momentum_score = self.model(seq_tensor).item()
                
            # 发送买入信号的双重联防条件：
            # 1. 动量异动 (LSTM Momentum Score > 0.40)
            # 2. 价值套利 (EV > 1.05)
            true_p_h = self.mock_p_model.get(match_id, 0.35) # 模拟 P_model 给出的真实胜率
            ev = true_p_h * odds['home']
            
            if ev > 1.05 and momentum_score > 0.40:
                print("\n[🚨 高频猎手双重预警] 发现猎物！")
                print(f"时间: {datetime.fromtimestamp(tick_time).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"赛事: {self.active_matches[match_id]['name']}")
                print(f"当前盘口: 主胜 @{odds['home']:.2f} | 机构 P_model: {true_p_h*100:.1f}%")
                print(f"LSTM 异动评分: {momentum_score*100:.1f}% | 理论 EV: {ev:.2f}")
                
                self.execute_trade(match_id, odds['home'], true_p_h)
                
                # 冷却：清空序列防止重复报警
                self.active_matches[match_id]['history'].clear()

    def execute_trade(self, match_id, price, true_p):
        print(">>> 正在启动动态仓位分配引擎...")
        
        # V9 Fractional Kelly + 波动率惩罚
        b = price
        p = true_p
        if p * b <= 1:
            return
            
        kelly_fraction = (p * b - 1) / (b - 1)
        base_fraction = kelly_fraction * 0.05  # 1/20 Kelly
        penalty = 1.0 / (b ** 0.5) if b > 2.0 else 1.0
        final_fraction = min(max(base_fraction * penalty, 0.001), 0.03) # 封顶 3%
        bet_size = self.current_capital * final_fraction
        
        if self.IS_LIVE_TRADING:
            print("❌ 错误：实盘交易未被人类授权！自动拦截！")
        else:
            print(f"✅ DRY RUN (沙盒模式): 动态计算下注总资金的 {final_fraction*100:.2f}% (${bet_size:.2f}) 买入主队 @{price:.2f}。")

async def mock_websocket_stream(engine):
    """
    模拟一个高频跳动的交易所 WebSocket (例如 Pinnacle 或 Betfair API)
    """
    print("==================================================")
    print("   [Phase 4] 开启 WebSocket 异步行情流监听器")
    print("   (模拟 Betfair / Pinnacle 毫秒级 Tick 流)")
    print("==================================================")
    
    matches = [
        {"id": 101, "name": "Arsenal vs Chelsea", "base": [2.10, 3.40, 3.50]},
        {"id": 102, "name": "Real Madrid vs Barcelona", "base": [1.95, 3.60, 3.80]},
        {"id": 103, "name": "Bayern Munich vs Dortmund", "base": [1.40, 4.50, 6.00]}
    ]
    
    # 模拟推送 100 个 Tick
    for i in range(1, 101):
        # 随机挑选一场比赛更新赔率
        m = random.choice(matches)
        
        # 模拟盘口跳动 (微调赔率)
        h_jump = random.uniform(-0.05, 0.05)
        d_jump = random.uniform(-0.02, 0.02)
        a_jump = random.uniform(-0.05, 0.05)
        
        m['base'][0] = max(1.01, m['base'][0] + h_jump)
        m['base'][1] = max(1.01, m['base'][1] + d_jump)
        m['base'][2] = max(1.01, m['base'][2] + a_jump)
        
        msg = {
            "type": "odds_update",
            "match_id": m["id"],
            "match_name": m["name"],
            "timestamp": time.time(),
            "odds": {
                "home": m['base'][0],
                "draw": m['base'][1],
                "away": m['base'][2]
            }
        }
        
        # 抛给引擎处理
        engine.process_tick(msg)
        
        # 模拟高频延迟
        await asyncio.sleep(0.02)
        
    print("\n==================================================")
    print("   WebSocket 模拟数据流推送结束。")
    print("==================================================")

if __name__ == "__main__":
    engine = LiveTradingEngine()
    # 启动异步事件循环
    asyncio.run(mock_websocket_stream(engine))
