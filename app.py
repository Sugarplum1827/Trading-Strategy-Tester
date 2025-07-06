import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import backtrader as bt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
import base64
from strategies import (SMAStrategy, EMAStrategy, RSIStrategy, MACDStrategy,
                        BollingerBandsStrategy, StochasticStrategy,
                        WilliamsRStrategy)
from utils import calculate_performance_metrics, create_equity_curve_plot, create_trade_summary
from portfolio_optimizer import PortfolioOptimizer
from risk_manager import RiskManager
from backtrader_data_feed import PandasDataFeed

# Page configuration
st.set_page_config(page_title="Trading Strategy Backtester",
                   page_icon="📈",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Main title
st.title("🚀 Comprehensive Trading Strategy Backtester")
st.markdown("---")

# Sidebar for user inputs
st.sidebar.header("🔧 Configuration")

# Apply dark theme only
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #ffffff;
}
.stSidebar {
    background-color: #262730;
}
.stSidebar .stSelectbox > div > div,
.stSidebar .stTextInput > div > div,
.stSidebar .stNumberInput > div > div,
.stSidebar .stSlider > div > div {
    background-color: #3a3d4a;
    color: #ffffff;
}
.stDataFrame {
    background-color: #262730;
}
.stMetric {
    background-color: #262730;
    border: 1px solid #3a3d4a;
    border-radius: 8px;
    padding: 10px;
}
.stTabs [data-baseweb="tab-list"] {
    background-color: #262730;
}
.stTabs [data-baseweb="tab"] {
    background-color: #3a3d4a;
    color: #ffffff;
}
.stTabs [aria-selected="true"] {
    background-color: #ff6b6b;
}
.stButton > button {
    background-color: #ff6b6b;
    color: #ffffff;
    border: none;
    border-radius: 8px;
}
.stButton > button:hover {
    background-color: #ff5252;
}
h1, h2, h3, h4, h5, h6 {
    color: #ffffff;
}
</style>
""",
            unsafe_allow_html=True)

# Data source selection
data_source = st.sidebar.radio(
    "Data Source", ["Yahoo Finance", "Upload CSV"],
    help=
    "Choose between fetching data from Yahoo Finance or uploading your own CSV file"
)

# Initialize session state
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = {}

# Initialize ticker variable
ticker = "NONE"

if data_source == "Yahoo Finance":
    # Ticker input
    ticker = st.sidebar.text_input(
        "Ticker Symbol",
        value="AAPL",
        help="Enter stock ticker symbol (e.g., AAPL, TSLA, GOOGL)").upper()

    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365 * 2),
            help="Select the start date for backtesting")
    with col2:
        end_date = st.date_input("End Date",
                                 value=datetime.now(),
                                 help="Select the end date for backtesting")

    # Fetch data
    @st.cache_data
    def fetch_data(ticker, start, end):
        try:
            data = yf.download(ticker, start=start, end=end)
            if data is None or data.empty:
                st.error(f"No data found for ticker {ticker}")
                return None

            # Fix MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                st.error(
                    f"Missing required columns. Available: {list(data.columns)}"
                )
                return None

            return data
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None

    data = fetch_data(ticker, start_date, end_date)

else:  # Upload CSV
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help=
        "Upload a CSV file with columns: Date, Open, High, Low, Close, Volume")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            # Ensure proper column names
            expected_columns = [
                'Date', 'Open', 'High', 'Low', 'Close', 'Volume'
            ]
            if not all(col in data.columns for col in expected_columns):
                st.error(f"CSV must contain columns: {expected_columns}")
                data = None
            else:
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
                data = data.sort_index()
                ticker = "CUSTOM"
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            data = None
    else:
        data = None

# Strategy selection
st.sidebar.subheader("📊 Strategy Selection")
strategy_type = st.sidebar.selectbox(
    "Choose Strategy", [
        "SMA Crossover", "EMA Crossover", "RSI", "MACD", "Bollinger Bands",
        "Stochastic", "Williams %R"
    ],
    help="Select the trading strategy to backtest")

# Strategy parameters
st.sidebar.subheader("⚙️ Strategy Parameters")

# Common parameters
initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=1000,
    max_value=10000000,
    value=100000,
    step=1000,
    help="Starting capital for the backtest")

commission = st.sidebar.number_input("Commission (%)",
                                     min_value=0.0,
                                     max_value=5.0,
                                     value=0.1,
                                     step=0.01,
                                     help="Commission per trade as percentage")

# Strategy-specific parameters
strategy_params = {}

if strategy_type == "SMA Crossover":
    strategy_params['short_window'] = st.sidebar.slider(
        "Short SMA Period", 5, 50, 10)
    strategy_params['long_window'] = st.sidebar.slider("Long SMA Period", 20,
                                                       200, 50)

elif strategy_type == "EMA Crossover":
    strategy_params['short_window'] = st.sidebar.slider(
        "Short EMA Period", 5, 50, 12)
    strategy_params['long_window'] = st.sidebar.slider("Long EMA Period", 20,
                                                       200, 26)

elif strategy_type == "RSI":
    strategy_params['rsi_period'] = st.sidebar.slider("RSI Period", 5, 50, 14)
    strategy_params['rsi_overbought'] = st.sidebar.slider(
        "RSI Overbought", 60, 90, 70)
    strategy_params['rsi_oversold'] = st.sidebar.slider(
        "RSI Oversold", 10, 40, 30)

elif strategy_type == "MACD":
    strategy_params['fast_period'] = st.sidebar.slider("Fast EMA", 5, 20, 12)
    strategy_params['slow_period'] = st.sidebar.slider("Slow EMA", 20, 50, 26)
    strategy_params['signal_period'] = st.sidebar.slider(
        "Signal Period", 5, 20, 9)

elif strategy_type == "Bollinger Bands":
    strategy_params['bb_period'] = st.sidebar.slider("BB Period", 10, 50, 20)
    strategy_params['bb_std'] = st.sidebar.slider("BB Standard Deviation", 1.0,
                                                  3.0, 2.0, 0.1)

elif strategy_type == "Stochastic":
    strategy_params['k_period'] = st.sidebar.slider("K Period", 5, 30, 14)
    strategy_params['d_period'] = st.sidebar.slider("D Period", 3, 10, 3)
    strategy_params['overbought'] = st.sidebar.slider("Overbought Level", 70,
                                                      90, 80)
    strategy_params['oversold'] = st.sidebar.slider("Oversold Level", 10, 30,
                                                    20)

elif strategy_type == "Williams %R":
    strategy_params['wr_period'] = st.sidebar.slider("Williams %R Period", 5,
                                                     30, 14)
    strategy_params['wr_overbought'] = st.sidebar.slider(
        "Overbought Level", -30, -10, -20)
    strategy_params['wr_oversold'] = st.sidebar.slider("Oversold Level", -90,
                                                       -70, -80)

# Risk management parameters
st.sidebar.subheader("🛡️ Risk Management")
stop_loss = st.sidebar.number_input(
    "Stop Loss (%)",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=0.5,
    help="Stop loss as percentage of entry price")

take_profit = st.sidebar.number_input(
    "Take Profit (%)",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=0.5,
    help="Take profit as percentage of entry price")

position_size = st.sidebar.slider(
    "Position Size (%)",
    min_value=10,
    max_value=100,
    value=95,
    step=5,
    help="Percentage of available capital to use per trade")

# Run backtest button
run_backtest = st.sidebar.button("🚀 Run Backtest", type="primary")

# Main content area
if data is not None and not data.empty:
    # Display basic data info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", len(data))
    with col2:
        try:
            start_date_str = pd.to_datetime(data.index[0]).strftime('%Y-%m-%d')
        except Exception as e:
            st.warning(f"Failed to parse start date: {e}")
            start_date_str = str(data.index[0])[:10]
    with col3:
        try:
            end_date_value = pd.to_datetime(data.index[-1])
            end_date_str = end_date_value.strftime('%Y-%m-%d')
        except Exception as e:
            st.warning(f"Could not format end date: {e}")
            end_date_str = str(data.index[-1])[:10]
        st.write("**End Date:**", end_date_str)
    with col4:
        if data_source == "Yahoo Finance":
            st.metric("Ticker", ticker)
        else:
            st.metric("Source", "Custom CSV")

    # Show raw data
    with st.expander("📋 Raw Data Preview"):
        st.dataframe(data.head(10))

    # Price chart
    st.subheader("📈 Price Chart")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=data.index,
                   y=data['Close'],
                   mode='lines',
                   name='Close Price',
                   line=dict(color='#00d4aa')))

    # Apply theme-specific layout
    chart_bg = '#0e1117'
    text_color = '#ffffff'
    grid_color = '#3a3d4a'

    fig.update_layout(title=f"{ticker} Price Chart",
                      xaxis_title="Date",
                      yaxis_title="Price ($)",
                      height=400,
                      plot_bgcolor=chart_bg,
                      paper_bgcolor=chart_bg,
                      font_color=text_color,
                      xaxis=dict(gridcolor=grid_color),
                      yaxis=dict(gridcolor=grid_color))
    st.plotly_chart(fig, use_container_width=True)

    # Run backtest
    if run_backtest:
        with st.spinner("Running backtest..."):
            try:
                # Initialize Cerebro
                cerebro = bt.Cerebro()

                # Add strategy
                strategy_map = {
                    "SMA Crossover": SMAStrategy,
                    "EMA Crossover": EMAStrategy,
                    "RSI": RSIStrategy,
                    "MACD": MACDStrategy,
                    "Bollinger Bands": BollingerBandsStrategy,
                    "Stochastic": StochasticStrategy,
                    "Williams %R": WilliamsRStrategy
                }

                strategy_class = strategy_map[strategy_type]

                # Add risk management parameters
                strategy_params.update({
                    'stop_loss': stop_loss / 100,
                    'take_profit': take_profit / 100,
                    'position_size': position_size / 100
                })

                cerebro.addstrategy(strategy_class, **strategy_params)

                # Prepare data for backtrader
                bt_data = PandasDataFeed.create_from_dataframe(data)
                cerebro.adddata(bt_data)

                # Set initial capital
                cerebro.broker.setcash(initial_capital)

                # Set commission
                cerebro.broker.setcommission(commission=commission / 100)

                # Add analyzers
                cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
                cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
                cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
                cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

                # Run backtest
                results = cerebro.run()

                # Store results in session state
                st.session_state.backtest_results = {
                    'cerebro': cerebro,
                    'results': results,
                    'strategy_type': strategy_type,
                    'strategy_params': strategy_params,
                    'initial_capital': initial_capital,
                    'final_value': cerebro.broker.getvalue(),
                    'data': data,
                    'ticker': ticker
                }

                st.success("Backtest completed successfully!")

            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")
                st.exception(e)

# Display results if available
if st.session_state.backtest_results:
    results_data = st.session_state.backtest_results

    st.markdown("---")
    st.header("📊 Backtest Results")

    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)

    initial_value = results_data['initial_capital']
    final_value = results_data['final_value']
    total_return = (final_value - initial_value) / initial_value * 100

    with col1:
        st.metric("Initial Capital", f"${initial_value:,.2f}")
    with col2:
        st.metric("Final Value", f"${final_value:,.2f}")
    with col3:
        st.metric("Total Return", f"{total_return:.2f}%")
    with col4:
        st.metric("Profit/Loss", f"${final_value - initial_value:,.2f}")

    # Detailed analytics
    result = results_data['results'][0]

    # Extract analyzer results
    sharpe_ratio = result.analyzers.sharpe.get_analysis().get('sharperatio', 0)
    max_drawdown = result.analyzers.drawdown.get_analysis().get('max', {}).get(
        'drawdown', 0)
    trade_analysis = result.analyzers.trades.get_analysis()

    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sharpe Ratio",
                  f"{sharpe_ratio:.3f}" if sharpe_ratio else "N/A")
    with col2:
        st.metric("Max Drawdown",
                  f"{max_drawdown:.2f}%" if max_drawdown else "N/A")
    with col3:
        total_trades = trade_analysis.get('total', {}).get('total', 0)
        st.metric("Total Trades", total_trades)
    with col4:
        won_trades = trade_analysis.get('won', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")

    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Equity Curve", "📋 Trade Summary", "🔍 Risk Analysis",
        "💼 Portfolio Metrics"
    ])

    with tab1:
        st.subheader("Equity Curve")

        # Create equity curve
        equity_curve = create_equity_curve_plot(results_data['cerebro'])
        if equity_curve is not None:
            st.plotly_chart(equity_curve, use_container_width=True)
        else:
            st.warning("Could not generate equity curve plot")

    with tab2:
        st.subheader("Trade Summary")

        # Create trade summary
        trade_summary = create_trade_summary(result)
        if not trade_summary.empty:
            st.dataframe(trade_summary)

            # Download CSV
            csv = trade_summary.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="trade_summary.csv">Download Trade Summary CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("No trades executed during the backtest period")

    with tab3:
        st.subheader("Risk Analysis")

        # Risk metrics
        risk_manager = RiskManager(results_data['data'])
        risk_metrics = risk_manager.calculate_risk_metrics(
            results_data['data']['Close'])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Annual Volatility",
                      f"{risk_metrics.get('volatility', 0):.2f}%")
            st.metric("VaR (95%)", f"{risk_metrics.get('var_95', 0):.2f}%")
            st.metric("CVaR (95%)", f"{risk_metrics.get('cvar_95', 0):.2f}%")

        with col2:
            st.metric("Sortino Ratio",
                      f"{risk_metrics.get('sortino_ratio', 0):.3f}")
            st.metric("Calmar Ratio",
                      f"{risk_metrics.get('calmar_ratio', 0):.3f}")
            st.metric("Maximum Drawdown Duration",
                      f"{risk_metrics.get('max_dd_duration', 0)} days")

        # Risk-return scatter plot
        risk_return_fig = risk_manager.create_risk_return_plot(
            results_data['data']['Close'])
        if risk_return_fig:
            st.plotly_chart(risk_return_fig, use_container_width=True)

    with tab4:
        st.subheader("Portfolio Metrics")

        # Portfolio optimization
        portfolio_optimizer = PortfolioOptimizer([results_data['data']])
        portfolio_metrics = portfolio_optimizer.calculate_portfolio_metrics()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Portfolio Beta",
                      f"{portfolio_metrics.get('beta', 0):.3f}")
            st.metric("Information Ratio",
                      f"{portfolio_metrics.get('information_ratio', 0):.3f}")
            st.metric("Treynor Ratio",
                      f"{portfolio_metrics.get('treynor_ratio', 0):.3f}")

        with col2:
            st.metric("Jensen's Alpha",
                      f"{portfolio_metrics.get('alpha', 0):.3f}")
            st.metric("Tracking Error",
                      f"{portfolio_metrics.get('tracking_error', 0):.2f}%")
            st.metric("R-Squared",
                      f"{portfolio_metrics.get('r_squared', 0):.3f}")

else:
    st.info(
        "Please configure your data source and parameters in the sidebar, then run the backtest."
    )

    # Show example data format for CSV upload
    if data_source == "Upload CSV":
        st.subheader("📋 CSV Format Example")
        example_data = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Open': [100.0, 101.0, 102.0],
            'High': [101.5, 102.5, 103.5],
            'Low': [99.5, 100.5, 101.5],
            'Close': [101.0, 102.0, 103.0],
            'Volume': [1000000, 1100000, 1200000]
        })
        st.dataframe(example_data)
        st.info(
            "Your CSV file should contain these exact column names: Date, Open, High, Low, Close, Volume"
        )

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Backtrader, and Python")
