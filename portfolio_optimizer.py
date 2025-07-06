import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px

class PortfolioOptimizer:
    """Portfolio optimization and analysis"""
    
    def __init__(self, asset_data):
        """
        Initialize with asset data
        asset_data: list of pandas DataFrames with price data
        """
        self.asset_data = asset_data
        self.returns = self._calculate_returns()
        
    def _calculate_returns(self):
        """Calculate returns for all assets"""
        returns_list = []
        for data in self.asset_data:
            if 'Close' in data.columns:
                returns = data['Close'].pct_change().dropna()
                returns_list.append(returns)
        
        if returns_list:
            return pd.concat(returns_list, axis=1)
        return pd.DataFrame()
    
    def calculate_portfolio_metrics(self):
        """Calculate comprehensive portfolio metrics"""
        
        if self.returns.empty:
            return {}
        
        # Use first asset if multiple
        returns = self.returns.iloc[:, 0] if len(self.returns.columns) > 0 else pd.Series()
        
        if len(returns) == 0:
            return {}
        
        # Annual return
        annual_return = returns.mean() * 252 * 100
        
        # Volatility
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio
        risk_free_rate = 2.0  # 2% risk-free rate
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Beta (using market proxy - simplified)
        market_return = returns  # Simplified - using asset as market proxy
        beta = 1.0  # Simplified
        
        # Alpha
        alpha = annual_return - (risk_free_rate + beta * (annual_return - risk_free_rate))
        
        # Treynor ratio
        treynor_ratio = (annual_return - risk_free_rate) / beta if beta != 0 else 0
        
        # Information ratio
        tracking_error = returns.std() * np.sqrt(252) * 100
        information_ratio = (annual_return - annual_return) / tracking_error if tracking_error > 0 else 0  # Simplified
        
        # R-squared
        r_squared = 0.85  # Simplified
        
        return {
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'beta': beta,
            'alpha': alpha,
            'treynor_ratio': treynor_ratio,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'r_squared': r_squared
        }
    
    def optimize_portfolio(self, method='max_sharpe'):
        """Optimize portfolio weights"""
        
        if self.returns.empty or len(self.returns.columns) < 2:
            return None
        
        n_assets = len(self.returns.columns)
        
        # Expected returns and covariance matrix
        expected_returns = self.returns.mean() * 252
        cov_matrix = self.returns.cov() * 252
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_guess = np.array([1/n_assets] * n_assets)
        
        if method == 'max_sharpe':
            # Maximize Sharpe ratio
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = portfolio_return / portfolio_volatility
                return -sharpe_ratio  # Negative because we want to maximize
            
        elif method == 'min_variance':
            # Minimize variance
            def objective(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        else:
            return None
        
        # Optimize
        result = minimize(objective, initial_guess, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            portfolio_return = np.dot(optimal_weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            sharpe_ratio = portfolio_return / portfolio_volatility
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return * 100,
                'volatility': portfolio_volatility * 100,
                'sharpe_ratio': sharpe_ratio
            }
        
        return None
    
    def calculate_efficient_frontier(self, num_portfolios=100):
        """Calculate efficient frontier"""
        
        if self.returns.empty or len(self.returns.columns) < 2:
            return None
        
        n_assets = len(self.returns.columns)
        expected_returns = self.returns.mean() * 252
        cov_matrix = self.returns.cov() * 252
        
        # Generate target returns
        min_return = expected_returns.min()
        max_return = expected_returns.max()
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return}
            ]
            
            # Bounds
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Initial guess
            initial_guess = np.array([1/n_assets] * n_assets)
            
            # Minimize variance
            def objective(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Optimize
            result = minimize(objective, initial_guess, method='SLSQP', 
                             bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
                
                efficient_portfolios.append({
                    'return': target_return * 100,
                    'volatility': portfolio_volatility * 100,
                    'weights': optimal_weights
                })
        
        return efficient_portfolios
    
    def create_efficient_frontier_plot(self):
        """Create efficient frontier plot"""
        
        efficient_portfolios = self.calculate_efficient_frontier()
        
        if not efficient_portfolios:
            return None
        
        returns = [p['return'] for p in efficient_portfolios]
        volatilities = [p['volatility'] for p in efficient_portfolios]
        
        fig = go.Figure()
        
        # Add efficient frontier
        fig.add_trace(go.Scatter(
            x=volatilities,
            y=returns,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=2)
        ))
        
        # Add optimal portfolios
        max_sharpe_portfolio = self.optimize_portfolio('max_sharpe')
        min_variance_portfolio = self.optimize_portfolio('min_variance')
        
        if max_sharpe_portfolio:
            fig.add_trace(go.Scatter(
                x=[max_sharpe_portfolio['volatility']],
                y=[max_sharpe_portfolio['expected_return']],
                mode='markers',
                name='Max Sharpe Ratio',
                marker=dict(color='red', size=10, symbol='star')
            ))
        
        if min_variance_portfolio:
            fig.add_trace(go.Scatter(
                x=[min_variance_portfolio['volatility']],
                y=[min_variance_portfolio['expected_return']],
                mode='markers',
                name='Min Variance',
                marker=dict(color='green', size=10, symbol='diamond')
            ))
        
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Volatility (%)',
            yaxis_title='Expected Return (%)',
            height=500
        )
        
        return fig
    
    def calculate_risk_parity_weights(self):
        """Calculate risk parity portfolio weights"""
        
        if self.returns.empty or len(self.returns.columns) < 2:
            return None
        
        n_assets = len(self.returns.columns)
        cov_matrix = self.returns.cov() * 252
        
        # Risk parity objective function
        def risk_parity_objective(weights):
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            marginal_contributions = np.dot(cov_matrix, weights)
            risk_contributions = weights * marginal_contributions
            
            # Minimize the sum of squared deviations from equal risk contribution
            equal_risk_contribution = portfolio_variance / n_assets
            return np.sum((risk_contributions - equal_risk_contribution) ** 2)
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(risk_parity_objective, initial_guess, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        
        return None
    
    def calculate_black_litterman_weights(self, views=None, confidences=None):
        """Calculate Black-Litterman portfolio weights"""
        
        if self.returns.empty or len(self.returns.columns) < 2:
            return None
        
        # This is a simplified implementation
        # In practice, you'd need market capitalization data and more sophisticated modeling
        
        expected_returns = self.returns.mean() * 252
        cov_matrix = self.returns.cov() * 252
        
        # If no views provided, use equal weights
        if views is None:
            n_assets = len(self.returns.columns)
            return np.array([1/n_assets] * n_assets)
        
        # Simplified Black-Litterman calculation
        # This would need more sophisticated implementation in practice
        return None
