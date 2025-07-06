import backtrader as bt
import pandas as pd
import numpy as np

class BaseStrategy(bt.Strategy):
    """Base strategy class with common functionality"""
    
    params = (
        ('stop_loss', 0.05),
        ('take_profit', 0.10),
        ('position_size', 0.95),
    )
    
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.trades = []
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.trades.append({
                    'date': self.datas[0].datetime.date(0),
                    'type': 'SELL',
                    'price': order.executed.price,
                    'size': order.executed.size,
                    'comm': order.executed.comm
                })
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            pass
        
        self.order = None
    
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        
        self.trades.append({
            'date': self.datas[0].datetime.date(0),
            'type': 'TRADE_CLOSED',
            'pnl': trade.pnl,
            'pnlcomm': trade.pnlcomm
        })

class SMAStrategy(BaseStrategy):
    """Simple Moving Average Crossover Strategy"""
    
    params = (
        ('short_window', 10),
        ('long_window', 50),
        ('stop_loss', 0.05),
        ('take_profit', 0.10),
        ('position_size', 0.95),
    )
    
    def __init__(self):
        super().__init__()
        self.sma_short = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.short_window
        )
        self.sma_long = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.long_window
        )
        self.crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)
    
    def next(self):
        if self.order:
            return
        
        if not self.position:
            if self.crossover > 0:
                size = int(self.broker.getcash() * self.params.position_size / self.dataclose[0])
                self.order = self.buy(size=size)
                
                # Set stop loss and take profit
                if self.params.stop_loss > 0:
                    self.sell(size=size, exectype=bt.Order.Stop, 
                             price=self.dataclose[0] * (1 - self.params.stop_loss))
                if self.params.take_profit > 0:
                    self.sell(size=size, exectype=bt.Order.Limit,
                             price=self.dataclose[0] * (1 + self.params.take_profit))
        
        else:
            if self.crossover < 0:
                self.order = self.sell(size=self.position.size)

class EMAStrategy(BaseStrategy):
    """Exponential Moving Average Crossover Strategy"""
    
    params = (
        ('short_window', 12),
        ('long_window', 26),
        ('stop_loss', 0.05),
        ('take_profit', 0.10),
        ('position_size', 0.95),
    )
    
    def __init__(self):
        super().__init__()
        self.ema_short = bt.indicators.ExponentialMovingAverage(
            self.datas[0], period=self.params.short_window
        )
        self.ema_long = bt.indicators.ExponentialMovingAverage(
            self.datas[0], period=self.params.long_window
        )
        self.crossover = bt.indicators.CrossOver(self.ema_short, self.ema_long)
    
    def next(self):
        if self.order:
            return
        
        if not self.position:
            if self.crossover > 0:
                size = int(self.broker.getcash() * self.params.position_size / self.dataclose[0])
                self.order = self.buy(size=size)
                
                # Set stop loss and take profit
                if self.params.stop_loss > 0:
                    self.sell(size=size, exectype=bt.Order.Stop,
                             price=self.dataclose[0] * (1 - self.params.stop_loss))
                if self.params.take_profit > 0:
                    self.sell(size=size, exectype=bt.Order.Limit,
                             price=self.dataclose[0] * (1 + self.params.take_profit))
        
        else:
            if self.crossover < 0:
                self.order = self.sell(size=self.position.size)

class RSIStrategy(BaseStrategy):
    """RSI Strategy"""
    
    params = (
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('stop_loss', 0.05),
        ('take_profit', 0.10),
        ('position_size', 0.95),
    )
    
    def __init__(self):
        super().__init__()
        self.rsi = bt.indicators.RSI(
            self.datas[0], period=self.params.rsi_period
        )
    
    def next(self):
        if self.order:
            return
        
        if not self.position:
            if self.rsi < self.params.rsi_oversold:
                size = int(self.broker.getcash() * self.params.position_size / self.dataclose[0])
                self.order = self.buy(size=size)
                
                # Set stop loss and take profit
                if self.params.stop_loss > 0:
                    self.sell(size=size, exectype=bt.Order.Stop,
                             price=self.dataclose[0] * (1 - self.params.stop_loss))
                if self.params.take_profit > 0:
                    self.sell(size=size, exectype=bt.Order.Limit,
                             price=self.dataclose[0] * (1 + self.params.take_profit))
        
        else:
            if self.rsi > self.params.rsi_overbought:
                self.order = self.sell(size=self.position.size)

class MACDStrategy(BaseStrategy):
    """MACD Strategy"""
    
    params = (
        ('fast_period', 12),
        ('slow_period', 26),
        ('signal_period', 9),
        ('stop_loss', 0.05),
        ('take_profit', 0.10),
        ('position_size', 0.95),
    )
    
    def __init__(self):
        super().__init__()
        self.macd = bt.indicators.MACD(
            self.datas[0],
            period_me1=self.params.fast_period,
            period_me2=self.params.slow_period,
            period_signal=self.params.signal_period
        )
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
    
    def next(self):
        if self.order:
            return
        
        if not self.position:
            if self.crossover > 0:
                size = int(self.broker.getcash() * self.params.position_size / self.dataclose[0])
                self.order = self.buy(size=size)
                
                # Set stop loss and take profit
                if self.params.stop_loss > 0:
                    self.sell(size=size, exectype=bt.Order.Stop,
                             price=self.dataclose[0] * (1 - self.params.stop_loss))
                if self.params.take_profit > 0:
                    self.sell(size=size, exectype=bt.Order.Limit,
                             price=self.dataclose[0] * (1 + self.params.take_profit))
        
        else:
            if self.crossover < 0:
                self.order = self.sell(size=self.position.size)

class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands Strategy"""
    
    params = (
        ('bb_period', 20),
        ('bb_std', 2.0),
        ('stop_loss', 0.05),
        ('take_profit', 0.10),
        ('position_size', 0.95),
    )
    
    def __init__(self):
        super().__init__()
        self.bb = bt.indicators.BollingerBands(
            self.datas[0],
            period=self.params.bb_period,
            devfactor=self.params.bb_std
        )
    
    def next(self):
        if self.order:
            return
        
        if not self.position:
            if self.dataclose[0] < self.bb.lines.bot[0]:
                size = int(self.broker.getcash() * self.params.position_size / self.dataclose[0])
                self.order = self.buy(size=size)
                
                # Set stop loss and take profit
                if self.params.stop_loss > 0:
                    self.sell(size=size, exectype=bt.Order.Stop,
                             price=self.dataclose[0] * (1 - self.params.stop_loss))
                if self.params.take_profit > 0:
                    self.sell(size=size, exectype=bt.Order.Limit,
                             price=self.dataclose[0] * (1 + self.params.take_profit))
        
        else:
            if self.dataclose[0] > self.bb.lines.top[0]:
                self.order = self.sell(size=self.position.size)

class StochasticStrategy(BaseStrategy):
    """Stochastic Oscillator Strategy"""
    
    params = (
        ('k_period', 14),
        ('d_period', 3),
        ('overbought', 80),
        ('oversold', 20),
        ('stop_loss', 0.05),
        ('take_profit', 0.10),
        ('position_size', 0.95),
    )
    
    def __init__(self):
        super().__init__()
        self.stoch = bt.indicators.Stochastic(
            self.datas[0],
            period=self.params.k_period,
            period_dfast=self.params.d_period
        )
    
    def next(self):
        if self.order:
            return
        
        if not self.position:
            if self.stoch.percK[0] < self.params.oversold:
                size = int(self.broker.getcash() * self.params.position_size / self.dataclose[0])
                self.order = self.buy(size=size)
                
                # Set stop loss and take profit
                if self.params.stop_loss > 0:
                    self.sell(size=size, exectype=bt.Order.Stop,
                             price=self.dataclose[0] * (1 - self.params.stop_loss))
                if self.params.take_profit > 0:
                    self.sell(size=size, exectype=bt.Order.Limit,
                             price=self.dataclose[0] * (1 + self.params.take_profit))
        
        else:
            if self.stoch.percK[0] > self.params.overbought:
                self.order = self.sell(size=self.position.size)

class WilliamsRStrategy(BaseStrategy):
    """Williams %R Strategy"""
    
    params = (
        ('wr_period', 14),
        ('wr_overbought', -20),
        ('wr_oversold', -80),
        ('stop_loss', 0.05),
        ('take_profit', 0.10),
        ('position_size', 0.95),
    )
    
    def __init__(self):
        super().__init__()
        self.williams_r = bt.indicators.WilliamsR(
            self.datas[0],
            period=self.params.wr_period
        )
    
    def next(self):
        if self.order:
            return
        
        if not self.position:
            if self.williams_r[0] < self.params.wr_oversold:
                size = int(self.broker.getcash() * self.params.position_size / self.dataclose[0])
                self.order = self.buy(size=size)
                
                # Set stop loss and take profit
                if self.params.stop_loss > 0:
                    self.sell(size=size, exectype=bt.Order.Stop,
                             price=self.dataclose[0] * (1 - self.params.stop_loss))
                if self.params.take_profit > 0:
                    self.sell(size=size, exectype=bt.Order.Limit,
                             price=self.dataclose[0] * (1 + self.params.take_profit))
        
        else:
            if self.williams_r[0] > self.params.wr_overbought:
                self.order = self.sell(size=self.position.size)
