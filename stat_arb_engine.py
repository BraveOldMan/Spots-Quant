import numpy as np
import statsmodels.tsa.stattools as ts

class StatArbMathEngine:
    """
    纯数学引擎：用于计算配对交易的协整性与实时 Z-Score 偏离度。
    """
    def __init__(self):
        self.lookback_period = 60 # 用于计算均值的历史滑动窗口 (如 60 分钟)
        
    def check_cointegration(self, series_a, series_b):
        """
        使用 Engle-Granger (ADF test) 检验两个时间序列是否协整。
        """
        if len(series_a) < 30 or len(series_b) < 30:
            return False, 1.0
            
        score, pvalue, _ = ts.coint(series_a, series_b)
        is_cointegrated = pvalue < 0.05
        return is_cointegrated, pvalue
        
    def calculate_z_score(self, current_spread, historical_spreads):
        """
        计算当前价差的 Z-Score (标准化偏离度)
        """
        if len(historical_spreads) < 2:
            return 0.0
            
        mean_spread = np.mean(historical_spreads)
        std_spread = np.std(historical_spreads)
        
        if std_spread == 0:
            return 0.0
            
        z_score = (current_spread - mean_spread) / std_spread
        return z_score

    def analyze_pair_state(self, price_a, price_b, historical_a, historical_b):
        """
        全流程分析：计算协整性，提取当前 Z-Score，并给出多空信号。
        """
        hist_spreads = np.array(historical_a) - np.array(historical_b)
        current_spread = price_a - price_b
        
        # 实时更新后的历史序列
        hist_spreads_updated = np.append(hist_spreads[-self.lookback_period+1:], current_spread)
        
        z_score = self.calculate_z_score(current_spread, hist_spreads_updated)
        
        signal = "HOLD"
        # 设定阈值：Z-Score > 2.0 代表 A 涨幅超标，做空 A 做多 B
        if z_score > 2.0:
            signal = "SHORT_A_LONG_B"
        elif z_score < -2.0:
            signal = "LONG_A_SHORT_B"
        elif abs(z_score) < 0.2: # 回归到均值附近，触发平仓
            signal = "CLOSE_POSITIONS"
            
        return z_score, signal
