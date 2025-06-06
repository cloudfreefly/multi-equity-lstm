# 投资组合优化模块
from AlgorithmImports import *
import numpy as np
import pandas as pd
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # 创建一个简单的minimize函数替代
    def minimize(objective, x0, method=None, bounds=None, constraints=None, options=None):
        # 简单的随机搜索优化
        best_x = x0
        best_val = objective(x0)
        
        for _ in range(100):  # 简单的随机搜索
            x = np.random.uniform(0, 1, len(x0))
            x = x / np.sum(x)  # 归一化到和为1
            
            # 应用边界约束
            if bounds:
                for i, (low, high) in enumerate(bounds):
                    x[i] = max(low, min(high, x[i]))
                x = x / np.sum(x)  # 重新归一化
            
            val = objective(x)
            if val < best_val:
                best_x = x
                best_val = val
        
        # 创建模拟的优化结果对象
        class Result:
            def __init__(self, x, success=True):
                self.x = x
                self.success = success
        
        return Result(best_x)

from config import AlgorithmConfig
from risk_management import ConcentrationLimiter

class PortfolioOptimizer:
    """投资组合优化器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        self.concentration_limiter = ConcentrationLimiter(algorithm_instance)
        
    def optimize_portfolio(self, expected_returns, 
                         covariance_matrix, 
                         symbols):
        """优化投资组合权重"""
        try:
            n = len(symbols)
            self.algorithm.Debug(f"Starting portfolio optimization for {n} symbols")
            
            if n == 0:
                return np.array([]), []
            
            if n == 1:
                return np.array([1.0]), symbols
            
            # 验证输入数据
            if not self._validate_optimization_inputs(expected_returns, covariance_matrix, n):
                self.algorithm.Debug("Invalid optimization inputs, using equal weights")
                return np.ones(n) / n, symbols
            
            # 尝试多种优化方法
            optimization_methods = [
                self._mean_variance_optimization,
                self._risk_parity_optimization,
                self._maximum_diversification_optimization
            ]
            
            best_weights = None
            best_score = -np.inf
            
            for method in optimization_methods:
                try:
                    weights = method(expected_returns, covariance_matrix)
                    if weights is not None:
                        score = self._evaluate_portfolio_quality(weights, expected_returns, covariance_matrix)
                        
                        if score > best_score:
                            best_weights = weights
                            best_score = score
                            
                except Exception as e:
                    self.algorithm.Debug(f"Optimization method failed: {e}")
                    continue
            
            if best_weights is None:
                self.algorithm.Debug("All optimization methods failed, using equal weights")
                best_weights = np.ones(n) / n
            
            # 应用约束和限制
            constrained_weights = self._apply_constraints(best_weights, symbols)
            
            # 最终筛选
            final_weights, final_symbols = self._final_screening(constrained_weights, symbols)
            
            self.algorithm.Debug(f"Portfolio optimization completed:")
            self.algorithm.Debug(f"  Method score: {best_score:.4f}")
            self.algorithm.Debug(f"  Final symbols: {len(final_symbols)}")
            self.algorithm.Debug(f"  Scipy available: {SCIPY_AVAILABLE}")
            
            return final_weights, final_symbols
            
        except Exception as e:
            self.algorithm.Debug(f"Critical error in portfolio optimization: {e}")
            n = len(symbols)
            return np.ones(n) / n if n > 0 else np.array([]), symbols
    
    def _validate_optimization_inputs(self, expected_returns, 
                                    covariance_matrix, n):
        """验证优化输入数据"""
        if len(expected_returns) != n or covariance_matrix.shape != (n, n):
            return False
        
        if np.any(np.isnan(expected_returns)) or np.any(np.isinf(expected_returns)):
            return False
        
        if np.any(np.isnan(covariance_matrix)) or np.any(np.isinf(covariance_matrix)):
            return False
        
        try:
            eigenvals = np.linalg.eigvals(covariance_matrix)
            if np.any(eigenvals <= 0):
                return False
        except:
            return False
        
        return True
    
    def _mean_variance_optimization(self, expected_returns, 
                                  covariance_matrix):
        """均值-方差优化"""
        try:
            n = len(expected_returns)
            
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
                portfolio_std = np.sqrt(portfolio_variance)
                
                if portfolio_std < 1e-8:
                    return -np.inf
                
                sharpe_ratio = portfolio_return / portfolio_std
                return -sharpe_ratio
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            min_weight = self.config.PORTFOLIO_CONFIG['min_weight']
            max_weight = self.config.PORTFOLIO_CONFIG['max_weight']
            bounds = [(min_weight, max_weight) for _ in range(n)]
            
            x0 = np.ones(n) / n
            
            result = minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                return result.x
            else:
                return None
                
        except Exception as e:
            self.algorithm.Debug(f"Error in mean-variance optimization: {e}")
            return None
    
    def _risk_parity_optimization(self, expected_returns, 
                                covariance_matrix):
        """风险平价优化"""
        try:
            n = len(expected_returns)
            
            def objective(weights):
                portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
                marginal_contrib = np.dot(covariance_matrix, weights)
                risk_contrib = weights * marginal_contrib / portfolio_variance
                target_contrib = np.ones(n) / n
                return np.sum((risk_contrib - target_contrib) ** 2)
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            min_weight = self.config.PORTFOLIO_CONFIG['min_weight']
            max_weight = self.config.PORTFOLIO_CONFIG['max_weight']
            bounds = [(min_weight, max_weight) for _ in range(n)]
            
            x0 = np.ones(n) / n
            
            result = minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x
            else:
                return None
                
        except Exception as e:
            self.algorithm.Debug(f"Error in risk parity optimization: {e}")
            return None
    
    def _maximum_diversification_optimization(self, expected_returns, 
                                            covariance_matrix):
        """最大分散化优化"""
        try:
            n = len(expected_returns)
            asset_volatilities = np.sqrt(np.diag(covariance_matrix))
            
            def objective(weights):
                weighted_avg_vol = np.dot(weights, asset_volatilities)
                portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
                portfolio_vol = np.sqrt(portfolio_variance)
                
                if portfolio_vol < 1e-8:
                    return -np.inf
                
                diversification_ratio = weighted_avg_vol / portfolio_vol
                return -diversification_ratio
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            min_weight = self.config.PORTFOLIO_CONFIG['min_weight']
            max_weight = self.config.PORTFOLIO_CONFIG['max_weight']
            bounds = [(min_weight, max_weight) for _ in range(n)]
            
            x0 = np.ones(n) / n
            
            result = minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x
            else:
                return None
                
        except Exception as e:
            self.algorithm.Debug(f"Error in maximum diversification optimization: {e}")
            return None
    
    def _evaluate_portfolio_quality(self, weights, 
                                  expected_returns, 
                                  covariance_matrix):
        """评估投资组合质量"""
        try:
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_vol = np.sqrt(portfolio_variance)
            
            sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            concentration_ratio = np.sum(weights ** 2)
            diversification_score = 1 - concentration_ratio
            weight_reasonableness = 1 - np.std(weights) / np.mean(weights) if np.mean(weights) > 0 else 0
            
            quality_score = (
                0.5 * sharpe_ratio +
                0.3 * diversification_score +
                0.2 * weight_reasonableness
            )
            
            return quality_score
            
        except Exception as e:
            self.algorithm.Debug(f"Error evaluating portfolio quality: {e}")
            return 0.0
    
    def _apply_constraints(self, weights, symbols):
        """应用各种约束条件"""
        constrained_weights = self.concentration_limiter.apply_concentration_limits(weights, symbols)
        
        min_weight = self.config.PORTFOLIO_CONFIG['min_weight']
        constrained_weights = np.maximum(constrained_weights, min_weight)
        
        total_weight = np.sum(constrained_weights)
        if total_weight > 0:
            cash_buffer = self.config.PORTFOLIO_CONFIG['cash_buffer']
            target_ratio = 1.0 - cash_buffer
            constrained_weights = constrained_weights * (target_ratio / total_weight)
        
        return constrained_weights
    
    def _final_screening(self, weights, symbols):
        """最终筛选和权重调整"""
        threshold = self.config.PORTFOLIO_CONFIG['weight_threshold']
        valid_mask = weights >= threshold
        
        if np.sum(valid_mask) == 0:
            top_n = min(5, len(weights))
            top_indices = np.argsort(weights)[-top_n:]
            valid_mask = np.zeros(len(weights), dtype=bool)
            valid_mask[top_indices] = True
        
        final_symbols = [symbols[i] for i in range(len(symbols)) if valid_mask[i]]
        final_weights = weights[valid_mask]
        
        if np.sum(final_weights) > 0:
            final_weights = final_weights / np.sum(final_weights)
            cash_buffer = self.config.PORTFOLIO_CONFIG['cash_buffer']
            final_weights = final_weights * (1.0 - cash_buffer)
        
        return final_weights, final_symbols

class CovarianceCalculator:
    """协方差矩阵计算器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        
    def calculate_covariance_matrix(self, symbols):
        """计算协方差矩阵"""
        try:
            historical_returns = self._get_historical_returns(symbols)
            
            if historical_returns is None or len(historical_returns.columns) == 0:
                self.algorithm.Debug("Failed to get historical returns for covariance calculation")
                return None
            
            covariance_matrix = historical_returns.cov() * 252
            covariance_matrix = self._clean_covariance_matrix(covariance_matrix)
            
            self.algorithm.Debug(f"Calculated covariance matrix for {len(covariance_matrix)} symbols")
            
            return covariance_matrix
            
        except Exception as e:
            self.algorithm.Debug(f"Error calculating covariance matrix: {e}")
            return None
    
    def _get_historical_returns(self, symbols):
        """获取历史收益率数据"""
        returns_data = {}
        
        for symbol in symbols:
            try:
                history = self.algorithm.History(symbol, self.config.LOOKBACK_DAYS, Resolution.Daily)
                history_list = list(history)
                if len(history_list) < self.config.LOOKBACK_DAYS // 2:
                    continue
                
                prices = np.array([x.Close for x in history_list])
                returns = np.diff(prices) / prices[:-1]
                
                if self._validate_returns(returns, symbol):
                    returns_data[symbol] = returns
                    
            except Exception as e:
                self.algorithm.Debug(f"Error getting returns for {symbol}: {e}")
                continue
        
        if not returns_data:
            return None
        
        min_length = min(len(returns) for returns in returns_data.values())
        aligned_returns = {}
        
        for symbol, returns in returns_data.items():
            aligned_returns[symbol] = returns[-min_length:]
        
        returns_df = pd.DataFrame(aligned_returns)
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 50:
            self.algorithm.Debug("Insufficient aligned return observations")
            return None
        
        return returns_df
    
    def _validate_returns(self, returns, symbol):
        """验证收益率数据质量"""
        if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
            return False
        
        if np.any(np.abs(returns) > 0.5):
            return False
        
        if np.var(returns) == 0:
            return False
        
        return True
    
    def _clean_covariance_matrix(self, cov_matrix):
        """清理协方差矩阵"""
        cov_matrix = cov_matrix.fillna(0)
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        
        try:
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix.values)
            min_eigenval = 1e-8
            eigenvals = np.maximum(eigenvals, min_eigenval)
            cleaned_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            cov_matrix.iloc[:, :] = cleaned_matrix
        except Exception as e:
            self.algorithm.Debug(f"Error cleaning covariance matrix: {e}")
        
        return cov_matrix

class SmartRebalancer:
    """智能调仓器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        
    def execute_smart_rebalance(self, target_weights, 
                              symbols):
        """执行智能调仓"""
        if not hasattr(self.algorithm, 'current_slice') or self.algorithm.current_slice is None:
            self.algorithm.Debug("No current data slice available for rebalancing")
            return {}
        
        try:
            current_holdings = self._get_current_holdings()
            target_holdings = self._calculate_target_holdings(target_weights, symbols)
            trades = self._generate_trade_instructions(current_holdings, target_holdings)
            executed_trades = self._execute_trades(trades)
            self._log_rebalance_summary(executed_trades)
            
            return executed_trades
            
        except Exception as e:
            self.algorithm.Debug(f"Error in smart rebalancing: {e}")
            return {}
    
    def _get_current_holdings(self):
        """获取当前持仓状态"""
        current_holdings = {}
        total_value = self.algorithm.Portfolio.TotalPortfolioValue
        
        for holding in self.algorithm.Portfolio.Values:
            if holding.Invested:
                symbol_str = str(holding.Symbol)
                current_holdings[symbol_str] = {
                    'quantity': holding.Quantity,
                    'value': holding.HoldingsValue,
                    'weight': holding.HoldingsValue / total_value,
                    'price': holding.Price
                }
        
        return current_holdings
    
    def _calculate_target_holdings(self, target_weights, 
                                 symbols):
        """计算目标持仓"""
        target_holdings = {}
        total_value = self.algorithm.Portfolio.TotalPortfolioValue
        
        for i, symbol in enumerate(symbols):
            if target_weights[i] <= 0.001:
                continue
            
            symbol_str = str(symbol)
            
            if not self.algorithm.current_slice.ContainsKey(symbol_str):
                continue
            
            current_price = self.algorithm.current_slice[symbol_str].Price
            target_value = target_weights[i] * total_value
            target_quantity = int(target_value / current_price)
            
            if target_quantity > 0:
                target_holdings[symbol_str] = {
                    'quantity': target_quantity,
                    'value': target_value,
                    'weight': target_weights[i],
                    'price': current_price
                }
        
        return target_holdings
    
    def _generate_trade_instructions(self, current_holdings, 
                                   target_holdings):
        """生成交易指令"""
        trades = {}
        tolerance = self.config.PORTFOLIO_CONFIG['rebalance_tolerance']
        
        # 处理需要卖出的持仓
        for symbol, current in current_holdings.items():
            target = target_holdings.get(symbol, {'quantity': 0, 'weight': 0})
            weight_diff = current['weight'] - target['weight']
            
            if weight_diff > tolerance:
                quantity_diff = current['quantity'] - target['quantity']
                
                trades[symbol] = {
                    'action': 'sell' if target['quantity'] == 0 else 'reduce',
                    'quantity': quantity_diff,
                    'current_quantity': current['quantity'],
                    'target_quantity': target['quantity'],
                    'weight_diff': weight_diff
                }
        
        # 处理需要买入的持仓
        for symbol, target in target_holdings.items():
            current = current_holdings.get(symbol, {'quantity': 0, 'weight': 0})
            weight_diff = target['weight'] - current['weight']
            
            if weight_diff > tolerance:
                quantity_diff = target['quantity'] - current['quantity']
                
                trades[symbol] = {
                    'action': 'buy' if current['quantity'] == 0 else 'increase',
                    'quantity': quantity_diff,
                    'current_quantity': current['quantity'],
                    'target_quantity': target['quantity'],
                    'weight_diff': weight_diff
                }
        
        return trades
    
    def _execute_trades(self, trades):
        """执行交易指令"""
        executed_trades = {}
        
        # 先执行卖出指令
        sell_trades = {k: v for k, v in trades.items() if v['action'] in ['sell', 'reduce']}
        for symbol, trade in sell_trades.items():
            try:
                if trade['action'] == 'sell':
                    order_ticket = self.algorithm.Liquidate(symbol)
                    executed_trades[symbol] = -trade['current_quantity']
                    self.algorithm.Debug(f"Liquidated {symbol}: {trade['current_quantity']} shares")
                else:
                    order_ticket = self.algorithm.MarketOrder(symbol, -trade['quantity'])
                    executed_trades[symbol] = -trade['quantity']
                    self.algorithm.Debug(f"Reduced {symbol}: -{trade['quantity']} shares")
                    
            except Exception as e:
                self.algorithm.Debug(f"Error executing sell order for {symbol}: {e}")
        
        # 再执行买入指令
        buy_trades = {k: v for k, v in trades.items() if v['action'] in ['buy', 'increase']}
        for symbol, trade in buy_trades.items():
            try:
                order_ticket = self.algorithm.MarketOrder(symbol, trade['quantity'])
                executed_trades[symbol] = trade['quantity']
                
                action_text = "Bought" if trade['action'] == 'buy' else "Increased"
                self.algorithm.Debug(f"{action_text} {symbol}: {trade['quantity']} shares")
                
            except Exception as e:
                self.algorithm.Debug(f"Error executing buy order for {symbol}: {e}")
        
        return executed_trades
    
    def _log_rebalance_summary(self, executed_trades):
        """记录调仓总结"""
        total_trades = len(executed_trades)
        buy_trades = sum(1 for qty in executed_trades.values() if qty > 0)
        sell_trades = sum(1 for qty in executed_trades.values() if qty < 0)
        
        self.algorithm.Debug(f"Rebalance summary:")
        self.algorithm.Debug(f"  Total trades: {total_trades}")
        self.algorithm.Debug(f"  Buy orders: {buy_trades}")
        self.algorithm.Debug(f"  Sell orders: {sell_trades}")
        
        total_value = self.algorithm.Portfolio.TotalPortfolioValue
        cash_ratio = self.algorithm.Portfolio.Cash / total_value
        
        self.algorithm.Debug(f"  Cash ratio: {cash_ratio:.4f}")
        self.algorithm.Debug(f"  Portfolio value: ${total_value:,.2f}") 