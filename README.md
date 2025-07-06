# 🚀 Trading Strategy Backtester

A comprehensive web-based trading strategy backtesting application built with Streamlit. Test and analyze various trading strategies on financial data with advanced portfolio optimization and risk management features.

## 📋 Features

### Trading Strategies
- **Simple Moving Average (SMA)** - Classic crossover strategy
- **Exponential Moving Average (EMA)** - Weight recent prices more heavily
- **Relative Strength Index (RSI)** - Momentum oscillator strategy
- **MACD** - Moving Average Convergence Divergence
- **Bollinger Bands** - Volatility-based trading signals
- **Stochastic Oscillator** - Momentum indicator for overbought/oversold conditions
- **Williams %R** - Momentum indicator similar to stochastic

### Data Sources
- **Yahoo Finance Integration** - Real-time and historical market data
- **CSV Upload Support** - Use your own custom datasets
- **Multiple Timeframes** - Daily, weekly, monthly data analysis

### Advanced Analytics
- **Portfolio Optimization** - Modern Portfolio Theory implementation
- **Risk Management** - VaR, Expected Shortfall, maximum drawdown analysis
- **Performance Metrics** - Sharpe ratio, Sortino ratio, and more
- **Interactive Visualizations** - Dynamic charts and plots

### Risk Management
- **Stop Loss & Take Profit** - Configurable risk controls
- **Position Sizing** - Adjustable position sizes
- **Drawdown Analysis** - Track and analyze portfolio drawdowns
- **Monte Carlo Simulations** - Risk scenario modeling

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- UV package manager (or pip)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd trading-backtester
```

2. Install dependencies:
```bash
uv add yfinance backtrader matplotlib plotly scipy streamlit
```

3. Run the application:
```bash
streamlit run app.py --server.port 5000
```

4. Open your browser and navigate to `http://localhost:5000`

## 📊 Usage Guide

### Getting Started
1. **Select Data Source**: Choose between Yahoo Finance or upload a CSV file
2. **Configure Parameters**: Set your ticker symbol, date range, and initial capital
3. **Choose Strategy**: Select from available trading strategies
4. **Customize Settings**: Adjust strategy parameters and risk management rules
5. **Run Backtest**: Execute the backtest and analyze results

### Data Requirements
For CSV uploads, ensure your file contains these columns:
- `Date` - Trading date (YYYY-MM-DD format)
- `Open` - Opening price
- `High` - Highest price
- `Low` - Lowest price
- `Close` - Closing price
- `Volume` - Trading volume

### Strategy Parameters

#### SMA Strategy
- **Short Window**: Fast moving average period (default: 10)
- **Long Window**: Slow moving average period (default: 50)

#### RSI Strategy
- **RSI Period**: Calculation period (default: 14)
- **Overbought Level**: Upper threshold (default: 70)
- **Oversold Level**: Lower threshold (default: 30)

#### MACD Strategy
- **Fast Period**: Fast EMA period (default: 12)
- **Slow Period**: Slow EMA period (default: 26)
- **Signal Period**: Signal line EMA period (default: 9)

## 📈 Performance Metrics

The application calculates comprehensive performance metrics including:

### Return Metrics
- **Total Return**: Overall portfolio performance
- **Annualized Return**: Yearly performance average
- **CAGR**: Compound Annual Growth Rate

### Risk Metrics
- **Volatility**: Standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline

### Trade Analytics
- **Total Trades**: Number of completed trades
- **Win Rate**: Percentage of profitable trades
- **Average Win/Loss**: Average profit and loss per trade
- **Profit Factor**: Ratio of gross profit to gross loss

## 🏗️ Architecture

### Core Components

- **`app.py`** - Main Streamlit application and user interface
- **`strategies.py`** - Trading strategy implementations
- **`portfolio_optimizer.py`** - Portfolio optimization algorithms
- **`risk_manager.py`** - Risk analysis and management tools
- **`utils.py`** - Utility functions and performance calculations
- **`backtrader_data_feed.py`** - Custom data feed for Backtrader

### Technology Stack
- **Frontend**: Streamlit web framework
- **Backtesting Engine**: Backtrader library
- **Data Processing**: Pandas and NumPy
- **Visualization**: Plotly and Matplotlib
- **Data Source**: Yahoo Finance API (yfinance)
- **Optimization**: SciPy optimization algorithms

## 🔧 Configuration

### Streamlit Configuration
The application uses a custom Streamlit configuration in `.streamlit/config.toml`:

```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
runOnSave = false
enableStaticServing = true

[browser]
gatherUsageStats = false
serverAddress = "0.0.0.0"
serverPort = 5000

[theme]
base = "dark"
```

### Environment Variables
No environment variables are required for basic functionality. All configuration is done through the web interface.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## 🚀 Working Application

https://trading-strategy-tester.streamlit.app

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not intended as financial advice. Trading and investing involve substantial risk of loss and are not suitable for all investors. Past performance does not guarantee future results.
