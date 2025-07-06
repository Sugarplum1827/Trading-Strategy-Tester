# Trading Strategy Backtester

## Overview

This is a comprehensive trading strategy backtesting application built with Streamlit that allows users to test various trading strategies on financial data. The system provides a complete framework for strategy development, backtesting, portfolio optimization, and risk management analysis.

## System Architecture

### Frontend Architecture
- **Streamlit Web Interface**: Single-page application with sidebar configuration
- **Interactive Visualization**: Uses Plotly for dynamic charts and graphs
- **Real-time Data Integration**: Connects to Yahoo Finance API for live market data
- **File Upload Support**: Allows users to upload custom CSV datasets

### Backend Architecture
- **Modular Strategy Framework**: Object-oriented design with base strategy class and specific implementations
- **Backtrader Integration**: Professional backtesting engine for strategy execution
- **Data Processing Pipeline**: Pandas-based data manipulation and analysis
- **Performance Analytics**: Comprehensive metrics calculation and reporting

## Key Components

### 1. Main Application (app.py)
- **Purpose**: Central application controller and user interface
- **Key Features**:
  - Configuration management through Streamlit sidebar
  - Data source selection (Yahoo Finance or CSV upload)
  - Strategy selection and parameter configuration
  - Results visualization and reporting

### 2. Trading Strategies (strategies.py)
- **Base Strategy Class**: Common functionality for all trading strategies
- **Implemented Strategies**:
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
  - Relative Strength Index (RSI)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Stochastic Oscillator
  - Williams %R
- **Risk Management**: Built-in stop-loss and take-profit mechanisms

### 3. Portfolio Optimizer (portfolio_optimizer.py)
- **Purpose**: Portfolio construction and optimization analysis
- **Key Features**:
  - Multi-asset return calculations
  - Portfolio metrics computation
  - Performance analysis across multiple assets
  - Risk-return optimization capabilities

### 4. Risk Manager (risk_manager.py)
- **Purpose**: Comprehensive risk analysis and management
- **Key Features**:
  - Value at Risk (VaR) calculations
  - Conditional VaR (Expected Shortfall)
  - Maximum drawdown analysis
  - Volatility measurements
  - Risk metric visualization

### 5. Utilities (utils.py)
- **Purpose**: Common utility functions and performance calculations
- **Key Features**:
  - Performance metrics calculation
  - Equity curve generation
  - Trade summary creation
  - Statistical analysis functions

## Data Flow

1. **Data Acquisition**: Users select data source (Yahoo Finance API or CSV upload)
2. **Data Processing**: Raw price data is cleaned and formatted using Pandas
3. **Strategy Selection**: Users choose trading strategy and configure parameters
4. **Backtesting Execution**: Backtrader engine executes strategy on historical data
5. **Performance Analysis**: Results are processed through performance metrics calculations
6. **Risk Assessment**: Risk manager analyzes portfolio risk characteristics
7. **Visualization**: Results are displayed through interactive Plotly charts
8. **Reporting**: Comprehensive performance and risk reports are generated

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Backtrader**: Trading strategy backtesting engine

### Data Sources
- **Yahoo Finance (yfinance)**: Real-time and historical market data
- **CSV Upload**: Custom dataset support

### Visualization
- **Plotly**: Interactive charting and visualization
- **Matplotlib**: Static plotting capabilities

### Scientific Computing
- **SciPy**: Statistical analysis and optimization
- **NumPy**: Mathematical operations

## Deployment Strategy

### Current Architecture
- **Single-file Deployment**: Streamlit application runs as a single process
- **Local Execution**: Designed for local development and testing
- **Memory-based Storage**: Uses Streamlit session state for temporary data

### Scalability Considerations
- **Stateless Design**: Each session maintains independent state
- **Modular Components**: Easy to extend with additional strategies
- **API-ready Structure**: Backend components can be easily converted to REST APIs

## Changelog

- July 06, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.