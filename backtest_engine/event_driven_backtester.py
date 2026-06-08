import pandas as pd
from backtest_engine.strategy_brain import StrategyBrain
from backtest_engine.risk_management import RiskManagement

class EventDrivenBacktester:
    """
    模块 4：事件驱动回测沙盒 (Event-driven Backtester)
    逐行解析时间戳的 Tick 事件驱动器，模拟真实的撮合与滑点场景。
    """
    def __init__(self, data: pd.DataFrame, strategy: StrategyBrain, risk_mgr: RiskManagement, slippage_tolerance: float = 0.05):
        self.data = data
        self.strategy = strategy
        self.risk_mgr = risk_mgr
        self.slippage_tolerance = slippage_tolerance
        
        self.equity_curve = []
        self.trade_log = []
        
    def run_backtest(self, p_model: float, match_result_home_win: bool):
        """
        开始回测主循环。为简单起见，此函数针对单场比赛的整个时间序列进行模拟。
        p_model: 模型给出的胜率预测
        match_result_home_win: 比赛最终结果（True表示主胜）
        """
        if self.data.empty:
            print("No data for backtest.")
            return

        print("--- Starting Event-Driven Backtest Simulation ---")
        
        position_open = False
        entry_price = 0.0
        staked_amount = 0.0
        
        # 将数据转为字典列表，实现 Tick 级遍历
        ticks = self.data.to_dict('records')
        
        for i, tick in enumerate(ticks):
            if self.risk_mgr.trading_frozen:
                break
                
            current_time = tick['update_time']
            current_odds = tick['raw_home']
            p_fair = tick['fair_prob_home']
            
            # 记录当前净值
            self.equity_curve.append({
                "time": current_time,
                "capital": self.risk_mgr.current_capital
            })
            
            if position_open:
                # 真实业务中可能会有止损/对冲逻辑，目前简化为持有到期
                continue

            # 构建截止到当前 Tick 的历史序列，供动量因子提取
            # (模拟实时截面，只能获取到目前为止的数据)
            history_series = pd.Series([t['raw_home'] for t in ticks[:i+1]])
            
            # 生成信号
            signal_obj = self.strategy.generate_signal("Home", p_model, history_series, p_fair)
            
            if signal_obj["signal"] == "LONG":
                # 检查下一跳滑点撮合
                if i + 1 < len(ticks):
                    next_tick = ticks[i+1]
                    next_odds = next_tick['raw_home']
                    
                    # 滑点判定
                    if abs(next_odds - current_odds) > self.slippage_tolerance:
                        print(f"[{current_time}] 订单流产：赔率发生剧烈滑点 {current_odds} -> {next_odds}")
                        continue
                        
                    # 成交！
                    execution_price = next_odds
                else:
                    # 最后一个 tick，假设能以当前价格成交
                    execution_price = current_odds
                    
                # 计算仓位
                staked_amount = self.risk_mgr.calculate_kelly_position(p_model, execution_price)
                if staked_amount > 0:
                    position_open = True
                    entry_price = execution_price
                    print(f"[{current_time}] 执行建仓: LONG 赔率 {entry_price:.2f}, 资金: ${staked_amount:.2f}")
                    self.trade_log.append({
                        "time": current_time,
                        "odds": entry_price,
                        "stake": staked_amount
                    })
                    # 扣除本金
                    self.risk_mgr.current_capital -= staked_amount
                    
        # 比赛结束，结算 PnL
        if position_open:
            if match_result_home_win:
                payout = staked_amount * entry_price
                self.risk_mgr.process_trade_result(payout)
                print(f"[比赛结算] 主队获胜！赢得红利: +${payout - staked_amount:.2f}")
            else:
                print(f"[比赛结算] 预测失败。损失: -${staked_amount:.2f}")

        # 结算后记录终态
        self.equity_curve.append({
            "time": "End_of_Match",
            "capital": self.risk_mgr.current_capital
        })

    def generate_report(self):
        """
        性能基准输出
        """
        initial = self.risk_mgr.initial_capital
        final = self.risk_mgr.current_capital
        total_return = (final - initial) / initial
        
        print("\n======== BACKTEST REPORT ========")
        print(f"Total Return: {total_return*100:.2f}%")
        print(f"Max Drawdown Limit Configured: {self.risk_mgr.max_drawdown_limit*100:.2f}%")
        print(f"Final Capital: ${final:.2f} (from ${initial:.2f})")
        print("=================================")

if __name__ == "__main__":
    from odds_pipeline import OddsDataPipeline
    
    # 构建 Mock 数据测试沙盒
    mock_api_res = {
        "response": [
            {
                "update": "2026-06-05T10:00:00+00:00",
                "bookmakers": [{"name": "Bet365", "bets": [{"name": "Match Winner", "values": [{"value": "Home", "odd": "2.10"}, {"value": "Draw", "odd": "3.4"}, {"value": "Away", "odd": "3.5"}]}]}]
            },
            {
                "update": "2026-06-05T12:00:00+00:00",
                "bookmakers": [{"name": "Bet365", "bets": [{"name": "Match Winner", "values": [{"value": "Home", "odd": "2.05"}, {"value": "Draw", "odd": "3.4"}, {"value": "Away", "odd": "3.6"}]}]}]
            },
            {
                "update": "2026-06-05T16:00:00+00:00",
                "bookmakers": [{"name": "Bet365", "bets": [{"name": "Match Winner", "values": [{"value": "Home", "odd": "1.95"}, {"value": "Draw", "odd": "3.5"}, {"value": "Away", "odd": "3.8"}]}]}]
            }
        ]
    }
    
    pipeline = OddsDataPipeline()
    raw_df = pipeline.parse_api_response_to_dataframe(mock_api_res)
    panel_data = pipeline.resample_time_series(raw_df, freq='1h')
    
    strategy = StrategyBrain(ev_threshold=0.05, momentum_threshold=-0.1)
    risk = RiskManagement(initial_capital=10000.0)
    
    backtester = EventDrivenBacktester(panel_data, strategy, risk)
    
    # 模拟一场模型胜率 0.60，真实比赛打出主胜的赛事回测
    backtester.run_backtest(p_model=0.60, match_result_home_win=True)
    backtester.generate_report()
