# 风险管理模块
from AlgorithmImports import *
import numpy as np
import pandas as pd
# 简化类型注解以兼容QuantConnect云端
from config import AlgorithmConfig
from collections import defaultdict

class RiskManager:
    """风险管理器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        self.risk_metrics_history = []
        
    def apply_risk_controls(self, expected_returns, symbols):
        """应用风险控制"""
        self.algorithm.Debug("Applying risk management controls...")
        
        # 1. 流动性筛选
        liquid_symbols = self._filter_by_liquidity(symbols)
        
        # 2. 波动率筛选
        volatility_filtered = self._filter_by_volatility(liquid_symbols, expected_returns)
        
        # 3. 相关性检查
        correlation_filtered = self._filter_by_correlation(volatility_filtered)
        
        # 4. 调整预期收益（风险调整）
        risk_adjusted_returns = self._adjust_returns_for_risk(expected_returns, correlation_filtered)
        
        # 5. 最终有效性检查
        final_symbols = self._final_validity_check(correlation_filtered)
        
        self.algorithm.Debug(f"Risk filtering results:")
        self.algorithm.Debug(f"  Original: {len(symbols)} -> Liquidity: {len(liquid_symbols)}")
        self.algorithm.Debug(f"  Volatility: {len(volatility_filtered)} -> Correlation: {len(correlation_filtered)}")
        self.algorithm.Debug(f"  Final: {len(final_symbols)}")
        
        return risk_adjusted_returns, final_symbols
    
    def _filter_by_liquidity(self, symbols):
        """基于流动性过滤股票"""
        liquid_symbols = []
        
        for symbol in symbols:
            try:
                # 获取最近的成交量数据
                history = self.algorithm.History(symbol, 20, Resolution.Daily)
                history_list = list(history)
                if len(history_list) < 10:
                    continue
                
                volumes = [x.Volume for x in history_list]
                avg_volume = np.mean(volumes)
                
                # 检查平均成交量是否满足要求
                if avg_volume >= self.config.RISK_CONFIG['liquidity_min_volume']:
                    liquid_symbols.append(symbol)
                else:
                    self.algorithm.Debug(f"Filtered out {symbol} due to low liquidity: {avg_volume:,.0f}")
                    
            except Exception as e:
                self.algorithm.Debug(f"Error checking liquidity for {symbol}: {e}")
                continue
        
        return liquid_symbols
    
    def _filter_by_volatility(self, symbols, expected_returns):
        """基于波动率过滤股票"""
        volatility_filtered = []
        
        for symbol in symbols:
            try:
                # 计算历史波动率
                history = self.algorithm.History(symbol, 60, Resolution.Daily)
                history_list = list(history)
                if len(history_list) < 30:
                    continue
                
                prices = np.array([x.Close for x in history_list])
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns) * np.sqrt(252)  # 年化波动率
                
                # 检查波动率是否在合理范围内
                if volatility <= self.config.RISK_CONFIG['volatility_threshold']:
                    volatility_filtered.append(symbol)
                else:
                    self.algorithm.Debug(f"Filtered out {symbol} due to high volatility: {volatility:.3f}")
                    
            except Exception as e:
                self.algorithm.Debug(f"Error calculating volatility for {symbol}: {e}")
                continue
        
        return volatility_filtered
    
    def _filter_by_correlation(self, symbols):
        """基于相关性过滤股票"""
        if len(symbols) <= 1:
            return symbols
        
        try:
            # 计算相关性矩阵
            correlation_matrix = self._calculate_correlation_matrix(symbols)
            
            if correlation_matrix is None:
                return symbols
            
            # 识别高相关性的股票对
            high_corr_pairs = self._find_high_correlation_pairs(
                correlation_matrix, 
                self.config.RISK_CONFIG['correlation_threshold']
            )
            
            # 从高相关性对中选择保留的股票
            filtered_symbols = self._select_from_correlated_pairs(symbols, high_corr_pairs)
            
            return filtered_symbols
            
        except Exception as e:
            self.algorithm.Debug(f"Error in correlation filtering: {e}")
            return symbols
    
    def _calculate_correlation_matrix(self, symbols):
        """计算相关性矩阵"""
        try:
            returns_data = {}
            
            for symbol in symbols:
                history = self.algorithm.History(symbol, 60, Resolution.Daily)
                history_list = list(history)
                if len(history_list) < 30:
                    continue
                
                prices = np.array([x.Close for x in history_list])
                returns = np.diff(prices) / prices[:-1]
                returns_data[symbol] = returns
            
            if len(returns_data) < 2:
                return None
            
            # 构建收益率DataFrame
            min_length = min(len(returns) for returns in returns_data.values())
            aligned_returns = {}
            
            for symbol, returns in returns_data.items():
                aligned_returns[symbol] = returns[-min_length:]
            
            returns_df = pd.DataFrame(aligned_returns)
            correlation_matrix = returns_df.corr().values
            
            return correlation_matrix
            
        except Exception as e:
            self.algorithm.Debug(f"Error calculating correlation matrix: {e}")
            return None
    
    def _find_high_correlation_pairs(self, correlation_matrix, 
                                   threshold):
        """找到高相关性的股票对"""
        high_corr_pairs = []
        n = correlation_matrix.shape[0]
        
        for i in range(n):
            for j in range(i + 1, n):
                if abs(correlation_matrix[i, j]) > threshold:
                    high_corr_pairs.append((i, j))
        
        return high_corr_pairs
    
    def _select_from_correlated_pairs(self, symbols, 
                                    high_corr_pairs):
        """从高相关性对中选择保留的股票"""
        if not high_corr_pairs:
            return symbols
        
        # 简单策略：从每个高相关性对中随机选择一个
        to_remove = set()
        
        for i, j in high_corr_pairs:
            if i not in to_remove and j not in to_remove:
                # 随机选择移除其中一个
                to_remove.add(j)  # 保留索引较小的
        
        filtered_symbols = [symbols[i] for i in range(len(symbols)) if i not in to_remove]
        
        if len(to_remove) > 0:
            removed_symbols = [symbols[i] for i in to_remove]
            self.algorithm.Debug(f"Removed due to high correlation: {removed_symbols}")
        
        return filtered_symbols
    
    def _adjust_returns_for_risk(self, expected_returns, symbols):
        """基于风险调整预期收益"""
        risk_adjusted_returns = {}
        
        for symbol in symbols:
            if symbol not in expected_returns:
                continue
            
            base_return = expected_returns[symbol]
            
            try:
                # 计算风险调整因子
                risk_factor = self._calculate_risk_factor(symbol)
                
                # 应用风险调整
                adjusted_return = base_return * risk_factor
                risk_adjusted_returns[symbol] = adjusted_return
                
                self.algorithm.Debug(f"{symbol} risk adjustment: {base_return:.6f} -> {adjusted_return:.6f} (factor: {risk_factor:.3f})")
                
            except Exception as e:
                self.algorithm.Debug(f"Error adjusting return for {symbol}: {e}")
                risk_adjusted_returns[symbol] = base_return
        
        return risk_adjusted_returns
    
    def _calculate_risk_factor(self, symbol):
        """计算个股风险调整因子"""
        try:
            # 获取历史数据
            history = self.algorithm.History(symbol, 60, Resolution.Daily)
            history_list = list(history)
            if len(history_list) < 30:
                return 1.0
            
            prices = np.array([x.Close for x in history_list])
            returns = np.diff(prices) / prices[:-1]
            
            # 计算多个风险指标
            volatility = np.std(returns) * np.sqrt(252)
            skewness = self._calculate_skewness(returns)
            max_drawdown = self._calculate_max_drawdown(prices)
            
            # 综合风险评分 (0-1, 1表示低风险)
            volatility_score = max(0, 1 - volatility / 0.5)  # 50%波动率为基准
            skewness_score = max(0, 1 - abs(skewness) / 2)   # 偏度绝对值2为基准
            drawdown_score = max(0, 1 - max_drawdown / 0.3)  # 30%回撤为基准
            
            # 加权平均
            risk_factor = (0.5 * volatility_score + 0.3 * drawdown_score + 0.2 * skewness_score)
            
            # 确保在合理范围内
            risk_factor = max(0.5, min(1.2, risk_factor))
            
            return risk_factor
            
        except Exception as e:
            self.algorithm.Debug(f"Error calculating risk factor for {symbol}: {e}")
            return 1.0
    
    def _calculate_skewness(self, returns):
        """计算偏度"""
        if len(returns) < 3:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        skewness = np.mean(((returns - mean_return) / std_return) ** 3)
        return skewness
    
    def _calculate_max_drawdown(self, prices):
        """计算最大回撤"""
        if len(prices) < 2:
            return 0.0
        
        peak = prices[0]
        max_drawdown = 0.0
        
        for price in prices:
            if price > peak:
                peak = price
            
            drawdown = (peak - price) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def _final_validity_check(self, symbols):
        """最终有效性检查"""
        valid_symbols = []
        
        for symbol in symbols:
            try:
                # 检查当前是否有价格数据
                if hasattr(self.algorithm, 'current_slice') and self.algorithm.current_slice:
                    if self.algorithm.current_slice.ContainsKey(symbol):
                        current_price = self.algorithm.current_slice[symbol].Price
                        if current_price > 0:
                            valid_symbols.append(symbol)
                        else:
                            self.algorithm.Debug(f"Invalid current price for {symbol}: {current_price}")
                    else:
                        self.algorithm.Debug(f"No current price data for {symbol}")
                else:
                    # 如果没有current_slice，尝试其他方式验证
                    valid_symbols.append(symbol)
                    
            except Exception as e:
                self.algorithm.Debug(f"Error in final validity check for {symbol}: {e}")
                continue
        
        return valid_symbols

class ConcentrationLimiter:
    """集中度限制器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        
    def apply_concentration_limits(self, weights, symbols):
        """应用集中度限制"""
        if len(weights) == 0:
            return weights
        
        # 1. 单个持仓权重限制
        weights = self._limit_individual_weights(weights)
        
        # 2. 行业集中度限制（简化版）
        weights = self._limit_sector_concentration(weights, symbols)
        
        # 3. 重新归一化
        weights = self._renormalize_weights(weights)
        
        return weights
    
    def _limit_individual_weights(self, weights):
        """限制单个持仓权重"""
        max_weight = self.config.PORTFOLIO_CONFIG['max_weight']
        
        # 将超过最大权重的部分按比例分配给其他股票
        excess_weights = np.maximum(0, weights - max_weight)
        total_excess = np.sum(excess_weights)
        
        if total_excess > 0:
            # 限制最大权重
            constrained_weights = np.minimum(weights, max_weight)
            
            # 找到未达到最大权重限制的股票
            available_capacity = max_weight - constrained_weights
            total_capacity = np.sum(available_capacity)
            
            if total_capacity > 0:
                # 按剩余容量比例分配超额权重
                redistribution = excess_weights * (available_capacity / total_capacity)
                constrained_weights += redistribution * (total_excess / np.sum(redistribution))
            
            return constrained_weights
        
        return weights
    
    def _limit_sector_concentration(self, weights, symbols):
        """限制行业集中度（简化版本）"""
        # 简化的行业分类
        sector_mapping = self._get_simple_sector_mapping()
        
        # 计算每个行业的权重
        sector_weights = defaultdict(float)
        symbol_sectors = {}
        
        for i, symbol in enumerate(symbols):
            sector = sector_mapping.get(symbol, 'Other')
            sector_weights[sector] += weights[i]
            symbol_sectors[symbol] = sector
        
        # 检查是否有行业超过限制
        max_sector_weight = self.config.PORTFOLIO_CONFIG['sector_max_weight']
        
        for sector, total_weight in sector_weights.items():
            if total_weight > max_sector_weight:
                # 计算需要削减的权重
                excess = total_weight - max_sector_weight
                sector_symbols = [s for s in symbols if symbol_sectors[s] == sector]
                
                # 按当前权重比例削减
                for i, symbol in enumerate(symbols):
                    if symbol in sector_symbols:
                        reduction_ratio = excess / total_weight
                        weights[i] *= (1 - reduction_ratio)
        
        return weights
    
    def _get_simple_sector_mapping(self):
        """获取简化的行业分类"""
        # 这是一个简化的映射，实际应用中应该使用更准确的行业分类数据
        sector_mapping = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOG': 'Technology',
            'META': 'Technology',
            'TSLA': 'Technology',
            'NVDA': 'Technology',
            'AVGO': 'Technology',
            'INTC': 'Technology',
            'NFLX': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'SPY': 'ETF',
            'GLD': 'Commodities',
            'LLY': 'Healthcare'
        }
        return sector_mapping
    
    def _renormalize_weights(self, weights):
        """重新归一化权重"""
        total_weight = np.sum(weights)
        
        if total_weight > 0:
            # 保留现金缓冲
            cash_buffer = self.config.PORTFOLIO_CONFIG['cash_buffer']
            target_investment_ratio = 1.0 - cash_buffer
            
            normalized_weights = weights * (target_investment_ratio / total_weight)
            return normalized_weights
        
        return weights

class DrawdownMonitor:
    """回撤监控器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        self.portfolio_values = []
        self.max_portfolio_value = 0
        
    def update_portfolio_value(self, current_value):
        """更新组合价值并计算回撤指标"""
        self.portfolio_values.append(current_value)
        
        if current_value > self.max_portfolio_value:
            self.max_portfolio_value = current_value
        
        # 计算当前回撤
        current_drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        
        # 计算历史最大回撤
        max_drawdown = self._calculate_historical_max_drawdown()
        
        # 检查回撤警告
        drawdown_alert = self._check_drawdown_alert(current_drawdown)
        
        metrics = {
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'portfolio_value': current_value,
            'peak_value': self.max_portfolio_value,
            'drawdown_alert': drawdown_alert
        }
        
        return metrics
    
    def _calculate_historical_max_drawdown(self):
        """计算历史最大回撤"""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        values = np.array(self.portfolio_values)
        peak = values[0]
        max_drawdown = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def _check_drawdown_alert(self, current_drawdown):
        """检查是否需要回撤警告"""
        max_allowed_drawdown = self.config.RISK_CONFIG['max_drawdown']
        return current_drawdown > max_allowed_drawdown

class VolatilityMonitor:
    """波动率监控器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.returns_history = []
        
    def update_return(self, daily_return):
        """更新日收益率并计算波动率指标"""
        self.returns_history.append(daily_return)
        
        # 保持最近252个交易日的数据
        if len(self.returns_history) > 252:
            self.returns_history = self.returns_history[-252:]
        
        if len(self.returns_history) < 20:
            return {'volatility': 0, 'volatility_alert': False}
        
        # 计算年化波动率
        returns_array = np.array(self.returns_history)
        volatility = np.std(returns_array) * np.sqrt(252)
        
        # 计算滚动波动率（最近20天）
        recent_volatility = np.std(returns_array[-20:]) * np.sqrt(252)
        
        # 波动率警告
        volatility_threshold = AlgorithmConfig.RISK_CONFIG['volatility_threshold']
        volatility_alert = recent_volatility > volatility_threshold
        
        return {
            'annual_volatility': volatility,
            'recent_volatility': recent_volatility,
            'volatility_alert': volatility_alert
        } 