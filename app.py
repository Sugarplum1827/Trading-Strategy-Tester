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

# Enhanced header with better styling
st.markdown("""
<div style="text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); border-radius: 10px; margin-bottom: 2rem;">
    <h1 style="color: white; font-size: 3rem; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
        🚀 Trading Strategy Backtester
    </h1>
    <p style="color: #e0e0e0; font-size: 1.2rem; margin: 0.5rem 0 0 0;">
        Professional-grade backtesting with advanced analytics
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.header("🔧 Configuration")

# Enhanced styling with modern design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #1a1f2e 50%, #2a2d3a 100%);
    color: #ffffff;
    font-family: 'Inter', sans-serif;
}

.stSidebar {
    background: linear-gradient(180deg, #1a1f2e 0%, #262730 100%);
    border-right: 2px solid #3a3d4a;
}

.stSidebar .stSelectbox > div > div,
.stSidebar .stTextInput > div > div,
.stSidebar .stNumberInput > div > div,
.stSidebar .stSlider > div > div {
    background-color: #3a3d4a;
    color: #ffffff;
    border: 1px solid #4a4d5a;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.stSidebar .stSelectbox > div > div:hover,
.stSidebar .stTextInput > div > div:hover,
.stSidebar .stNumberInput > div > div:hover {
    border-color: #6366f1;
    box-shadow: 0 4px 8px rgba(99, 102, 241, 0.2);
}

.stDataFrame {
    background-color: #1a1f2e;
    border-radius: 12px;
    border: 1px solid #3a3d4a;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.stMetric {
    background: linear-gradient(135deg, #2a2d3a 0%, #3a3d4a 100%);
    border: 1px solid #4a4d5a;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease;
}

.stMetric:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
}

.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(90deg, #1a1f2e 0%, #262730 100%);
    border-radius: 12px;
    padding: 8px;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #a0a3b5;
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: #ffffff;
    box-shadow: 0 4px 8px rgba(99, 102, 241, 0.3);
}

.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: #ffffff;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    padding: 12px 24px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(99, 102, 241, 0.2);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #5856eb 0%, #7c3aed 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(99, 102, 241, 0.3);
}

.ticker-card {
    background: linear-gradient(135deg, #2a2d3a 0%, #3a3d4a 100%);
    border: 1px solid #4a4d5a;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    cursor: pointer;
    transition: all 0.3s ease;
}

.ticker-card:hover {
    background: linear-gradient(135deg, #3a3d4a 0%, #4a4d5a 100%);
    border-color: #6366f1;
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(99, 102, 241, 0.2);
}

.ticker-selected {
    border-color: #6366f1 !important;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    box-shadow: 0 6px 12px rgba(99, 102, 241, 0.3) !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #ffffff;
    font-family: 'Inter', sans-serif;
}

.sidebar-section {
    background: linear-gradient(135deg, #2a2d3a 0%, #3a3d4a 100%);
    border-radius: 12px;
    padding: 16px;
    margin: 16px 0;
    border: 1px solid #4a4d5a;
}

.performance-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #2a2d3a 100%);
    border: 1px solid #3a3d4a;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)

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
    # Top 10 popular tickers
    top_tickers = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corp.", 
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "NVDA": "NVIDIA Corp.",
        "META": "Meta Platforms Inc.",
        "NFLX": "Netflix Inc.",
        "ORCL": "Oracle Corp.",
        "AMD": "Advanced Micro Devices"
    }
    
    # Enhanced ticker selection with popular choices
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown("### 📊 Stock Selection")
    
    # Radio button for selection method
    selection_method = st.sidebar.radio(
        "Choose selection method:",
        ["Popular Stocks", "Custom Ticker"],
        help="Select from top 10 popular stocks or enter a custom ticker"
    )
    
    if selection_method == "Popular Stocks":
        # Create a nice grid of popular tickers
        ticker = st.sidebar.selectbox(
            "Select Popular Stock:",
            options=list(top_tickers.keys()),
            format_func=lambda x: f"{x} - {top_tickers[x]}",
            help="Choose from the most popular stocks"
        )
        
        # Show additional info about selected ticker
        if ticker in top_tickers:
            st.sidebar.info(f"Selected: {ticker} - {top_tickers[ticker]}")
    else:
        # Custom ticker input
        ticker = st.sidebar.text_input(
            "Custom Ticker Symbol:",
            value="AAPL",
            help="Enter any stock ticker symbol (e.g., SPY, QQQ, SHOP)"
        ).upper()
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Date range section with enhanced styling
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown("### 📅 Date Range")
    
    # Quick date range presets
    preset_range = st.sidebar.selectbox(
        "Quick Date Presets:",
        ["Custom", "Last 1 Year", "Last 2 Years", "Last 3 Years", "Last 5 Years", "Last 6 Months"],
        help="Choose a preset date range or select custom"
    )
    
    if preset_range != "Custom":
        days_map = {
            "Last 6 Months": 180,
            "Last 1 Year": 365,
            "Last 2 Years": 365 * 2,
            "Last 3 Years": 365 * 3,
            "Last 5 Years": 365 * 5
        }
        start_date = datetime.now() - timedelta(days=days_map[preset_range])
        end_date = datetime.now()
        
        # Show the preset dates (read-only display)
        st.sidebar.success(f"**Start:** {start_date.strftime('%Y-%m-%d')}")
        st.sidebar.success(f"**End:** {end_date.strftime('%Y-%m-%d')}")
    else:
        # Custom date selection
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
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

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

# Strategy selection with enhanced design
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.markdown("### 📊 Strategy Selection")

# Strategy descriptions for better UX
strategy_descriptions = {
    "SMA Crossover": "🔄 Simple Moving Average crossover signals",
    "EMA Crossover": "⚡ Exponential Moving Average with recent price emphasis", 
    "RSI": "📈 Relative Strength Index momentum strategy",
    "MACD": "📊 Moving Average Convergence Divergence",
    "Bollinger Bands": "🎯 Volatility-based mean reversion",
    "Stochastic": "🔄 Overbought/oversold momentum indicator",
    "Williams %R": "📉 Williams Percent Range momentum"
}

strategy_type = st.sidebar.selectbox(
    "Choose Strategy:",
    options=list(strategy_descriptions.keys()),
    format_func=lambda x: strategy_descriptions[x],
    help="Select the trading strategy to backtest"
)

# Show strategy description
if strategy_type:
    st.sidebar.info(f"**Selected:** {strategy_descriptions[strategy_type]}")

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Trading parameters section
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.markdown("### 💰 Trading Parameters")

# Enhanced capital and commission inputs
col1, col2 = st.sidebar.columns(2)
with col1:
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=1000,
        help="Starting capital for the backtest")

with col2:
    commission = st.sidebar.number_input(
        "Commission (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.1,
        step=0.01,
        help="Commission per trade as percentage")

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Strategy-specific parameters section
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.markdown("### ⚙️ Strategy Parameters")

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

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Risk management parameters with enhanced styling
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.markdown("### 🛡️ Risk Management")

# Risk management in columns for better layout
col1, col2 = st.sidebar.columns(2)
with col1:
    stop_loss = st.sidebar.number_input(
        "Stop Loss (%)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5,
        help="Stop loss as percentage of entry price")

with col2:
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

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Enhanced run button with better styling
st.sidebar.markdown('<div style="padding: 20px 0;">', unsafe_allow_html=True)
run_backtest = st.sidebar.button(
    "🚀 Start Backtest Analysis", 
    type="primary",
    use_container_width=True,
    help="Execute the backtesting strategy with current parameters"
)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Main content area with enhanced design
if data is not None and not data.empty:
    # Enhanced data overview section
    st.markdown('<div class="performance-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Data Overview")
    
    # Enhanced metrics display
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📈 Ticker", ticker)
    with col2:
        st.metric("📊 Data Points", f"{len(data):,}")
    with col3:
        st.metric("📅 Start Date", data.index[0].strftime('%Y-%m-%d'))
    with col4:
        st.metric("📅 End Date", data.index[-1].strftime('%Y-%m-%d'))
    with col5:
        latest_price = data['Close'].iloc[-1]
        st.metric("💰 Latest Price", f"${latest_price:.2f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced data preview section
    st.markdown('<div class="performance-card">', unsafe_allow_html=True)
    with st.expander("📋 Raw Data Preview", expanded=False):
        # Show basic statistics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📊 Price Statistics**")
            price_stats = data['Close'].describe()
            st.write(f"**Min:** ${price_stats['min']:.2f}")
            st.write(f"**Max:** ${price_stats['max']:.2f}")
            st.write(f"**Mean:** ${price_stats['mean']:.2f}")
            st.write(f"**Std:** ${price_stats['std']:.2f}")
        
        with col2:
            st.markdown("**📈 Returns Analysis**")
            returns = data['Close'].pct_change().dropna()
            st.write(f"**Daily Return Mean:** {returns.mean():.4f}")
            st.write(f"**Daily Return Std:** {returns.std():.4f}")
            st.write(f"**Best Day:** {returns.max():.4f}")
            st.write(f"**Worst Day:** {returns.min():.4f}")
        
        # Data table
        st.markdown("**🗂️ Sample Data**")
        st.dataframe(data.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced price chart
    st.markdown('<div class="performance-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Interactive Price Chart")
    
    # Create enhanced candlestick chart
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="OHLC",
        increasing=dict(line=dict(color='#00d4aa'), fillcolor='#00d4aa'),
        decreasing=dict(line=dict(color='#ff6b6b'), fillcolor='#ff6b6b')
    ))

    # Apply enhanced theme
    chart_bg = '#1a1f2e'
    text_color = '#ffffff'
    grid_color = '#3a3d4a'

    fig.update_layout(
        title=f"{ticker} - OHLC Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        plot_bgcolor=chart_bg,
        paper_bgcolor=chart_bg,
        font_color=text_color,
        xaxis=dict(gridcolor=grid_color, showgrid=True),
        yaxis=dict(gridcolor=grid_color, showgrid=True),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=text_color))
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

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
