import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import pandas as pd


class OddsLSTM(nn.Module):
    """
    LSTM 时序网络：自动学习赛前 24H -> 0H 的盘口跳动规律，取代人工设定的斜率动量规则。
    """
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super(OddsLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # 全连接输出层
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # out shape: (batch, seq_len, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的隐状态
        out = out[:, -1, :]
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out)


def sequential_train_val_split(
    dataset: TensorDataset, train_fraction: float = 0.8
) -> tuple[Subset, Subset]:
    """Split time-ordered samples into train then validation subsets."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    train_size = int(len(dataset) * train_fraction)
    if train_size <= 0 or train_size >= len(dataset):
        raise ValueError("dataset is too small for a non-empty split.")
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def train_lstm_engine():
    print("==================================================")
    print("   [Phase 2] 点火：LSTM 盘口形态深度学习炼丹炉")
    print("==================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"检测到运算硬件: {device}")
    
    print("1. 加载 Kaggle 盘口时间轴数据...")
    # 抽取 24H 到 0H 之间的偶数小时数据，以节省内存
    cols_to_use = ['match_id']
    for i in range(24, -1, -2):
        cols_to_use.extend([f"home_b26_{i}", f"draw_b26_{i}", f"away_b26_{i}"])
        
    try:
        df_matches = pd.read_csv("kaggle_dataset/odds_series_matches.csv.gz", 
                                 encoding='latin-1', usecols=['match_id', ' detailed_score'])
        
        # 取 20000 行用于快速炼丹验证
        df_odds = pd.read_csv("kaggle_dataset/odds_series.csv.gz", 
                              encoding='latin-1', usecols=cols_to_use, nrows=20000)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
        
    df_merged = pd.merge(df_odds, df_matches, on='match_id', how='inner')
    
    print("2. 组装 3D 时序张量 (Tensor Data Preparation)...")
    X_list = []
    y_list = []
    
    for _, row in df_merged.iterrows():
        seq = []
        valid_seq = True
        
        # 按照时间从远到近 (24H -> 0H) 组装
        for hr in range(24, -1, -2):
            h = row[f"home_b26_{hr}"]
            d = row[f"draw_b26_{hr}"]
            a = row[f"away_b26_{hr}"]
            if pd.isna(h) or pd.isna(d) or pd.isna(a):
                valid_seq = False
                break
            # 可以计算 margin 或直接输入原始赔率。为了方便网络学习，取倒数 (隐含概率)
            seq.append([1.0/h, 1.0/d, 1.0/a])
            
        if not valid_seq:
            continue
            
        # 解析真实比分作为 Label
        score_str = str(row[' detailed_score'])
        try:
            if ";" in score_str:
                ft_score = score_str.split(";")[1].replace(")", "").strip()
            else:
                ft_score = score_str.replace("(", "").replace(")", "").strip()
            g_h, g_a = ft_score.split(":")
            home_win = 1.0 if int(g_h) > int(g_a) else 0.0
        except (TypeError, ValueError):
            continue
            
        X_list.append(seq)
        y_list.append([home_win])
        
    if not X_list:
        print("有效时序样本为 0，程序终止。")
        return
        
    X_tensor = torch.tensor(X_list, dtype=torch.float32)
    y_tensor = torch.tensor(y_list, dtype=torch.float32)
    
    print(f"样本组装完毕! Shape: {X_tensor.shape} -> 对应 (样本数, 序列长度(13个时间截面), 特征数(主/平/客3路))")
    
    dataset = TensorDataset(X_tensor, y_tensor)
    # 留 20% 作为验证集
    train_dataset, val_dataset = sequential_train_val_split(dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    print("3. 初始化 LSTM 架构与 Adam 优化器...")
    model = OddsLSTM().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15
    print(f"4. 启动深度炼丹 (Epochs: {epochs})...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
                preds = (outputs >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            
    print("5. 保存盘口形态记忆模型...")
    torch.save(model.state_dict(), "lstm_momentum.pth")
    print("✅ SUCCESS: 模型权重已冻结并保存至 lstm_momentum.pth")
    print("==================================================")

if __name__ == "__main__":
    train_lstm_engine()
