class RiskManagement:
    """
    模块 3：风控与动态仓位管理 (Risk Management)
    负责将生成的信号转化为具体的持仓指令，首要任务是控制极限情况下的资金回撤。
    """
    def __init__(self, initial_capital: float = 10000.0, max_drawdown_limit: float = 0.30):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        self.max_drawdown_limit = max_drawdown_limit
        self.trading_frozen = False
        
        self.max_position_per_match = 0.05 # 单场最高 5%

    def calculate_kelly_position(self, p_model: float, decimal_odds: float) -> float:
        """
        采用凯利公式进行资金分配，返回建议投入的资金绝对值
        """
        b = decimal_odds - 1.0
        if b <= 0:
            return 0.0
        
        q = 1.0 - p_model
        f_star = (p_model * b - q) / b
        
        if f_star <= 0:
            return 0.0
        
        # 乘以保守系数 0.25 (四分之一凯利)
        fraction = f_star * 0.25
        
        # 单场最高限额黑天鹅过滤
        fraction = min(fraction, self.max_position_per_match)
        
        return self.current_capital * fraction

    def check_drawdown_constraint(self) -> bool:
        """
        最大回撤硬约束
        """
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
            
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        if drawdown >= self.max_drawdown_limit:
            if not self.trading_frozen:
                print(f"!!! [重大警报] 触发最大回撤硬约束 (DD: {drawdown*100:.2f}%)，自动冻结发单引擎 !!!")
                self.trading_frozen = True
            return False
            
        return True

    def process_trade_result(self, pnl: float):
        """
        更新资金曲线
        """
        self.current_capital += pnl
        self.check_drawdown_constraint()
