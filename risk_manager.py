import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RiskManager:
    """Risk management and analysis"""
    
    def __init__(self, data):
        """
        Initialize with price data
        data: pandas DataFrame with price data
        """
        self.data = data
        self.returns = self._calculate_returns()
        
    def _calculate_returns(self):
        """Calculate returns from price data"""
        if 'Close' in self.data.columns:
            return self.data['Close'].pct_change().dropna()
        return pd.Series()
    
    def calculate_risk_metrics(self, price_series):
        """Calculate comprehensive risk metrics"""
        
        if len(price_series) == 0:
            return {}
        
        returns = price_series.pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        # Basic risk metrics
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        cvar_99 = returns[returns <= np.percentile(returns, 1)].mean() * 100
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Maximum Drawdown Duration
        drawdown_duration = self._calculate_drawdown_duration(drawdown)
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) * 100
        
        # Sortino ratio
        annual_return = returns.mean() * 252 * 100
        risk_free_rate = 2.0  # 2% risk-free rate
        sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Beta (simplified - using market proxy)
        beta = 1.0  # Simplified
        
        return {
            'volatility': volatility,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_drawdown': max_drawdown,
            'max_dd_duration': drawdown_duration,
            'downside_deviation': downside_deviation,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'beta': beta
        }
    
    def _calculate_drawdown_duration(self, drawdown):
        """Calculate maximum drawdown duration in days"""
        
        is_in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for in_dd in is_in_drawdown:
            if in_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                    current_period = 0
        
        # Don't forget the last period if it ends in drawdown
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def calculate_var_es(self, confidence_level=0.05):
        """Calculate Value at Risk and Expected Shortfall"""
        
        if len(self.returns) == 0:
            return None, None
        
        # Historical simulation
        var_historical = np.percentile(self.returns, confidence_level * 100)
        es_historical = self.returns[self.returns <= var_historical].mean()
        
        # Parametric VaR (assuming normal distribution)
        mean_return = self.returns.mean()
        std_return = self.returns.std()
        var_parametric = stats.norm.ppf(confidence_level, mean_return, std_return)
        
        # Expected Shortfall for parametric
        es_parametric = mean_return - std_return * stats.norm.pdf(stats.norm.ppf(confidence_level)) / confidence_level
        
        return {
            'var_historical': var_historical * 100,
            'es_historical': es_historical * 100,
            'var_parametric': var_parametric * 100,
            'es_parametric': es_parametric * 100
        }
    
    def calculate_rolling_risk_metrics(self, window=252):
        """Calculate rolling risk metrics"""
        
        if len(self.returns) < window:
            return {}
        
        rolling_metrics = {}
        
        # Rolling volatility
        rolling_metrics['volatility'] = self.returns.rolling(window=window).std() * np.sqrt(252) * 100
        
        # Rolling VaR
        rolling_metrics['var_95'] = self.returns.rolling(window=window).apply(
            lambda x: np.percentile(x, 5) * 100
        )
        
        # Rolling maximum drawdown
        rolling_metrics['max_drawdown'] = self.returns.rolling(window=window).apply(
            lambda x: self._calculate_max_drawdown(x)
        )
        
        # Rolling Sortino ratio
        rolling_annual_return = self.returns.rolling(window=window).mean() * 252 * 100
        rolling_downside_dev = self.returns.rolling(window=window).apply(
            lambda x: x[x < 0].std() * np.sqrt(252) * 100
        )
        rolling_metrics['sortino_ratio'] = (rolling_annual_return - 2) / rolling_downside_dev
        
        return rolling_metrics
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown for a series"""
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.min() * 100
    
    def create_risk_return_plot(self, price_series):
        """Create risk-return scatter plot"""
        
        if len(price_series) == 0:
            return None
        
        returns = price_series.pct_change().dropna()
        
        if len(returns) == 0:
            return None
        
        # Calculate metrics
        annual_return = returns.mean() * 252 * 100
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Create plot
        fig = go.Figure()
        
        # Add asset point
        fig.add_trace(go.Scatter(
            x=[volatility],
            y=[annual_return],
            mode='markers',
            name='Asset',
            marker=dict(size=15, color='blue')
        ))
        
        # Add risk-free rate line
        risk_free_rate = 2.0
        max_vol = volatility * 1.5
        fig.add_trace(go.Scatter(
            x=[0, max_vol],
            y=[risk_free_rate, risk_free_rate],
            mode='lines',
            name='Risk-Free Rate',
            line=dict(color='red', dash='dash')
        ))
        
        # Add Sharpe ratio line
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        fig.add_trace(go.Scatter(
            x=[0, volatility],
            y=[risk_free_rate, annual_return],
            mode='lines',
            name=f'Sharpe Ratio: {sharpe_ratio:.3f}',
            line=dict(color='green', dash='dot')
        ))
        
        fig.update_layout(
            title='Risk-Return Analysis',
            xaxis_title='Volatility (% per year)',
            yaxis_title='Expected Return (% per year)',
            height=500
        )
        
        return fig
    
    def calculate_stress_test_scenarios(self, scenarios=None):
        """Calculate stress test scenarios"""
        
        if scenarios is None:
            scenarios = {
                'market_crash': -0.20,  # 20% market crash
                'moderate_decline': -0.10,  # 10% decline
                'volatility_spike': 2.0,  # 2x volatility
                'black_swan': -0.30  # 30% crash (black swan)
            }
        
        if len(self.returns) == 0:
            return {}
        
        current_price = self.data['Close'].iloc[-1] if 'Close' in self.data.columns else 100
        stress_results = {}
        
        for scenario_name, shock in scenarios.items():
            if scenario_name == 'volatility_spike':
                # Simulate high volatility period
                stressed_returns = self.returns * shock
                stressed_price = current_price * (1 + stressed_returns.iloc[-1])
            else:
                # Price shock scenario
                stressed_price = current_price * (1 + shock)
            
            stress_results[scenario_name] = {
                'price_change': shock * 100,
                'stressed_price': stressed_price,
                'portfolio_impact': shock * 100  # Simplified
            }
        
        return stress_results
    
    def calculate_correlation_risk(self, other_assets=None):
        """Calculate correlation risk metrics"""
        
        if other_assets is None or len(other_assets) == 0:
            return {}
        
        correlations = {}
        
        for asset_name, asset_data in other_assets.items():
            if 'Close' in asset_data.columns:
                other_returns = asset_data['Close'].pct_change().dropna()
                
                # Align returns
                aligned_returns = pd.concat([self.returns, other_returns], axis=1).dropna()
                
                if len(aligned_returns) > 0:
                    correlation = aligned_returns.corr().iloc[0, 1]
                    correlations[asset_name] = correlation
        
        return correlations
    
    def generate_monte_carlo_simulation(self, num_simulations=1000, time_horizon=252):
        """Generate Monte Carlo simulation for risk analysis"""
        
        if len(self.returns) == 0:
            return None
        
        # Calculate parameters from historical data
        mean_return = self.returns.mean()
        std_return = self.returns.std()
        
        # Generate random returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(mean_return, std_return, 
                                           (num_simulations, time_horizon))
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + simulated_returns, axis=1)
        
        # Calculate final values
        final_values = cumulative_returns[:, -1]
        
        # Calculate risk metrics
        var_95 = np.percentile(final_values, 5)
        var_99 = np.percentile(final_values, 1)
        expected_value = np.mean(final_values)
        
        return {
            'simulated_paths': cumulative_returns,
            'final_values': final_values,
            'var_95': (var_95 - 1) * 100,
            'var_99': (var_99 - 1) * 100,
            'expected_return': (expected_value - 1) * 100,
            'probability_of_loss': np.mean(final_values < 1) * 100
        }
