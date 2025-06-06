# 主算法文件 - 多时间跨度CNN+LSTM+Attention策略
from AlgorithmImports import *
import numpy as np
import pandas as pd
import time

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# 导入自定义模块
from config import AlgorithmConfig
from data_processing import DataProcessor
from model_training import ModelTrainer
from prediction import (PredictionEngine, ExpectedReturnCalculator, 
                       PredictionValidator)
from risk_management import (RiskManager, DrawdownMonitor, VolatilityMonitor, 
                           ConcentrationLimiter)
from portfolio_optimization import (PortfolioOptimizer, CovarianceCalculator, 
                                  SmartRebalancer)


class MultiHorizonTradingAlgorithm(QCAlgorithm):

    def Initialize(self):
        """
        初始化多时间跨度预测策略：
        1. 设置基础参数和配置
        2. 初始化各功能模块
        3. 设置调度和预热期
        """
        # 检查TensorFlow可用性
        if not TF_AVAILABLE:
            self.Debug("WARNING: TensorFlow not available, switching to simplified mode")
        
        # 加载配置
        self.config = AlgorithmConfig()
        
        # 设置基础参数
        self.SetStartDate(*self.config.START_DATE)
        self.SetEndDate(*self.config.END_DATE)
        self.SetCash(self.config.INITIAL_CASH)
        
        # 添加股票到算法环境
        for symbol in self.config.SYMBOLS:
            self.AddEquity(symbol, Resolution.Daily)
        
        # 初始化功能模块
        self._initialize_modules()
        
        # 设置预热期
        self.SetWarmUp(TimeSpan.FromDays(self.config.WARMUP_DAYS))
        self.Debug(f"Setting warmup period to {self.config.WARMUP_DAYS} days")
        
        # 存储当前数据slice
        self.current_slice = None
        
        # 性能监控
        self.performance_metrics = {
            'rebalance_count': 0,
            'total_training_time': 0,
            'prediction_success_rate': 0,
            'daily_returns': []
        }
        
        # 调度设置：每月月初执行调仓
        self.Schedule.On(
            self.DateRules.MonthStart("SPY"), 
            self.TimeRules.AfterMarketOpen("SPY"), 
            self.Rebalance
        )
        
        self.Debug("Multi-horizon trading algorithm initialized successfully")
        self._log_library_availability()
    
    def _initialize_modules(self):
        """初始化各功能模块"""
        # 数据处理模块
        self.data_processor = DataProcessor(self)
        
        # 模型训练模块
        self.model_trainer = ModelTrainer(self)
        
        # 预测模块
        self.prediction_engine = PredictionEngine(self)
        self.expected_return_calculator = ExpectedReturnCalculator(self)
        self.prediction_validator = PredictionValidator
        
        # 风险管理模块
        self.risk_manager = RiskManager(self)
        self.drawdown_monitor = DrawdownMonitor(self)
        self.volatility_monitor = VolatilityMonitor(self)
        self.concentration_limiter = ConcentrationLimiter(self)
        
        # 投资组合优化模块
        self.portfolio_optimizer = PortfolioOptimizer(self)
        self.covariance_calculator = CovarianceCalculator(self)
        self.smart_rebalancer = SmartRebalancer(self)
        
        self.Debug("All modules initialized")
    
    def _log_library_availability(self):
        """记录库的可用性状态"""
        self.Debug(f"TensorFlow available: {TF_AVAILABLE}")
        if TF_AVAILABLE:
            try:
                import tensorflow as tf
                self.Debug(f"TensorFlow version: {tf.__version__}")
            except:
                self.Debug("TensorFlow version check failed")
        
        try:
            import numpy as np
            self.Debug(f"NumPy version: {np.__version__}")
        except:
            self.Debug("NumPy not available")
        
        try:
            import pandas as pd
            self.Debug(f"Pandas version: {pd.__version__}")
        except:
            self.Debug("Pandas not available")
    
    def _should_retrain(self):
        """判断是否应该重新训练模型"""
        # 检查全局开关
        if not self.config.TRAINING_CONFIG['enable_retraining']:
            return False
        
        # 检查是否有现有模型
        if not hasattr(self.model_trainer, 'models') or not self.model_trainer.models:
            self.Debug("No existing models found, training required")
            return True
        
        # 检查训练频率
        frequency = self.config.TRAINING_CONFIG['retraining_frequency']
        
        if frequency == 'always':
            return True
        elif frequency == 'monthly':
            return True
        elif frequency == 'weekly':
            current_day = self.Time.weekday()
            return current_day == 0  # 周一
        elif frequency == 'quarterly':
            current_month = self.Time.month
            return current_month in [1, 4, 7, 10] and self.Time.day <= 7
        elif frequency == 'never':
            return False
        
        return True
    
    def _perform_training(self):
        """执行模型训练"""
        training_start_time = time.time()
        
        try:
            # 使用ModelTrainer进行训练
            models, successful_symbols = self.model_trainer.train_all_models()
            
            if successful_symbols:
                # 获取训练结果
                training_stats = self.model_trainer.get_training_statistics()
                
                # 记录训练统计
                total_time = time.time() - training_start_time
                self.performance_metrics['total_training_time'] += total_time
                
                self.Debug(f"Multi-horizon training completed successfully")
                self.Debug(f"Tradable symbols: {len(successful_symbols)}")
                self.Debug(f"Training time: {total_time:.1f}s")
                self.Debug(f"Training statistics: {training_stats}")
                
                return True
            else:
                self.Debug("Training failed, will use fallback strategy")
                return False
                
        except Exception as e:
            self.Debug(f"Error in training: {e}")
            return False
    
    def _perform_rebalancing(self):
        """执行投资组合调仓"""
        # 检查是否有可用的训练模型
        tradable_symbols = self.model_trainer.get_tradable_symbols()
        if tradable_symbols:
            # 使用训练好的模型进行预测和调仓
            self._rebalance_with_models()
        else:
            # 使用备用策略
            self._rebalance_with_fallback_strategy()
    
    def _rebalance_with_models(self):
        """使用训练好的模型进行调仓"""
        # 获取可交易的股票列表
        tradable_symbols = self.model_trainer.get_tradable_symbols()
        if not tradable_symbols:
            self.Debug("No tradable symbols after training, skipping rebalance")
            return

        # 使用预测引擎生成预测
        try:
            # 生成多时间跨度预测
            models = self.model_trainer.get_trained_models()
            predictions = self.prediction_engine.generate_multi_horizon_predictions(
                models, tradable_symbols
            )
            
            if not predictions:
                self.Debug("No valid predictions generated, using fallback strategy")
                self._rebalance_with_fallback_strategy()
                return
            
            # 验证预测结果
            validation_results = self.prediction_validator.validate_predictions(predictions)
            valid_predictions = {k: v for k, v in predictions.items() 
                               if validation_results.get(k, False)}
            
            # 记录预测汇总
            prediction_summary = self.prediction_validator.get_prediction_summary(predictions)
            self._log_prediction_summary(prediction_summary, validation_results)
            
            if not valid_predictions:
                self.Debug("No valid predictions after validation, using fallback strategy")
                self._rebalance_with_fallback_strategy()
                return
            
            # 计算预期收益
            expected_returns, valid_symbols = self.expected_return_calculator.calculate_expected_returns(
                valid_predictions
            )
            
            if not expected_returns:
                self.Debug("No valid expected returns, using fallback strategy")
                self._rebalance_with_fallback_strategy()
                return
            
            # 计算协方差矩阵
            covariance_matrix = self.covariance_calculator.calculate_covariance_matrix(
                valid_symbols
            )
            
            if covariance_matrix is None:
                self.Debug("Failed to calculate covariance matrix, using fallback strategy")
                self._rebalance_with_fallback_strategy()
                return
            
            # 优化投资组合
            expected_returns_array = np.array([expected_returns[s] for s in valid_symbols])
            self.Debug(f"Expected returns array: {expected_returns_array}")
            
            weights, final_valid_symbols = self.portfolio_optimizer.optimize_portfolio(
                expected_returns_array, covariance_matrix, valid_symbols
            )
            
            self.Debug(f"Calculated weights: {weights}")
            self.Debug(f"Final valid symbols: {final_valid_symbols}")
            
            # 应用风险管理限制
            risk_adjusted_returns, risk_filtered_symbols = self.risk_manager.apply_risk_controls(
                expected_returns, valid_symbols
            )
            
            # 重新计算权重（如果符号发生变化）
            if risk_filtered_symbols != valid_symbols:
                self.Debug(f"Risk management filtered symbols: {len(valid_symbols)} -> {len(risk_filtered_symbols)}")
                
                # 重新计算协方差矩阵
                covariance_matrix = self.covariance_calculator.calculate_covariance_matrix(
                    risk_filtered_symbols
                )
                
                if covariance_matrix is None:
                    self.Debug("Failed to recalculate covariance matrix after risk filtering")
                    self._rebalance_with_fallback_strategy()
                    return
                
                # 重新优化投资组合
                risk_adjusted_returns_array = np.array([risk_adjusted_returns[s] for s in risk_filtered_symbols])
                weights, final_valid_symbols = self.portfolio_optimizer.optimize_portfolio(
                    risk_adjusted_returns_array, covariance_matrix, risk_filtered_symbols
                )
            else:
                final_valid_symbols = risk_filtered_symbols
            
            # 设置持仓
            self.smart_rebalancer.execute_smart_rebalance(
                weights, final_valid_symbols
            )
            
        except Exception as e:
            self.Debug(f"Error in model-based rebalancing: {e}")
            self._rebalance_with_fallback_strategy()
    
    def _rebalance_with_fallback_strategy(self):
        """使用备用策略进行调仓"""
        try:
            fallback_strategy = self.config.TRAINING_CONFIG['fallback_strategy']
            
            if fallback_strategy == 'equal_weights':
                # 等权重策略
                valid_symbols = self.config.SYMBOLS
                n_symbols = len(valid_symbols)
                weights = [1.0 / n_symbols] * n_symbols
                
                self.smart_rebalancer.execute_smart_rebalance(weights, valid_symbols)
                self.Debug(f"Applied equal weights strategy to {n_symbols} symbols")
                
            elif fallback_strategy == 'momentum':
                # 动量策略
                self._apply_momentum_strategy()
                
            elif fallback_strategy == 'skip':
                # 跳过此次调仓
                self.Debug("Skipping rebalancing due to fallback strategy")
                
        except Exception as e:
            self.Debug(f"Error in fallback strategy: {e}")
    
    def _apply_momentum_strategy(self):
        """应用简单动量策略"""
        try:
            momentum_scores = {}
            
            for symbol in self.config.SYMBOLS:
                try:
                    # 获取过去20天的数据计算动量
                    history = self.History(symbol, 20, Resolution.Daily)
                    prices = [x.Close for x in history]
                    
                    if len(prices) >= 20:
                        momentum = (prices[-1] - prices[0]) / prices[0]
                        momentum_scores[symbol] = momentum
                    
                except Exception as e:
                    self.Debug(f"Error calculating momentum for {symbol}: {e}")
                    continue
            
            if momentum_scores:
                # 选择动量最强的股票进行等权重配置
                sorted_symbols = sorted(momentum_scores.keys(), 
                                      key=lambda x: momentum_scores[x], reverse=True)
                
                # 选择前50%的股票
                top_symbols = sorted_symbols[:max(1, len(sorted_symbols) // 2)]
                n_top = len(top_symbols)
                
                weights = [1.0 / n_top] * n_top
                self.smart_rebalancer.execute_smart_rebalance(weights, top_symbols)
                
                self.Debug(f"Applied momentum strategy to {n_top} top symbols: {top_symbols}")
            else:
                self.Debug("No momentum data available, keeping current positions")
                
        except Exception as e:
            self.Debug(f"Error in momentum strategy: {e}, keeping current positions")
        
    def OnData(self, data):
        """数据更新事件处理"""
        # 存储最新数据slice
        self.current_slice = data
        
        # 更新风险监控指标
        self._update_risk_monitoring()
        
        # 每日收盘时记录性能日志
        if self.Time.hour == 16:  # 美股收盘时间
            self._log_daily_performance()
    
    def _update_risk_monitoring(self):
        """更新风险监控指标"""
        try:
            # 更新回撤监控
            current_value = self.Portfolio.TotalPortfolioValue
            drawdown_metrics = self.drawdown_monitor.update_portfolio_value(current_value)
            
            # 计算日收益率
            if hasattr(self, 'previous_portfolio_value'):
                daily_return = (current_value - self.previous_portfolio_value) / self.previous_portfolio_value
                volatility_metrics = self.volatility_monitor.update_return(daily_return)
            else:
                volatility_metrics = {'volatility_alert': False}
            
            # 记录风险指标
            if drawdown_metrics.get('drawdown_alert', False):
                self.Debug(f"Drawdown alert: {drawdown_metrics['current_drawdown']:.3f}")
            
            if volatility_metrics.get('volatility_alert', False):
                self.Debug(f"Volatility alert: {volatility_metrics.get('recent_volatility', 0):.3f}")
            
            # 保存当前值用于下次计算
            self.previous_portfolio_value = current_value
                
        except Exception as e:
            self.Debug(f"Error in risk monitoring: {e}")
    
    def _log_daily_performance(self):
        """记录每日性能日志"""
        try:
            current_value = self.Portfolio.TotalPortfolioValue
            
            # 计算日收益率
            if hasattr(self, 'previous_portfolio_value') and self.previous_portfolio_value > 0:
                daily_return = (current_value - self.previous_portfolio_value) / self.previous_portfolio_value
                self.performance_metrics['daily_returns'].append(daily_return)
                self.Debug(f"Daily Performance - Return: {daily_return:.4f} ({daily_return*100:.2f}%)")
            
            # 记录持仓状况
            self._log_portfolio_status()
            
            # 计算累计统计
            if len(self.performance_metrics['daily_returns']) > 0:
                returns_array = np.array(self.performance_metrics['daily_returns'])
                cumulative_return = np.prod(1 + returns_array) - 1
                volatility = np.std(returns_array) * np.sqrt(252)
                sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
                
                self.Debug(f"Performance Summary:")
                self.Debug(f"  Portfolio Value: ${current_value:,.2f}")
                self.Debug(f"  Cumulative Return: {cumulative_return:.4f} ({cumulative_return*100:.2f}%)")
                self.Debug(f"  Annualized Volatility: {volatility:.4f}")
                self.Debug(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
            
        except Exception as e:
            self.Debug(f"Error in daily performance logging: {e}")
    
    def _log_portfolio_status(self):
        """记录当前投资组合状态"""
        try:
            total_value = self.Portfolio.TotalPortfolioValue
            invested_holdings = []
            
            for holding in self.Portfolio.Values:
                if holding.Invested:
                    weight = holding.HoldingsValue / total_value
                    invested_holdings.append({
                        'symbol': str(holding.Symbol),
                        'quantity': holding.Quantity,
                        'value': holding.HoldingsValue,
                        'weight': weight,
                        'price': holding.Price
                    })
            
            if invested_holdings:
                self.Debug(f"Current Holdings ({len(invested_holdings)} positions):")
                for holding in sorted(invested_holdings, key=lambda x: x['weight'], reverse=True):
                    self.Debug(f"  {holding['symbol']}: {holding['weight']:.3f} (${holding['value']:,.0f}, {holding['quantity']} shares @ ${holding['price']:.2f})")
            
            cash_ratio = self.Portfolio.Cash / total_value
            self.Debug(f"Cash Position: {cash_ratio:.3f} (${self.Portfolio.Cash:,.2f})")
            
        except Exception as e:
            self.Debug(f"Error in portfolio status logging: {e}")
    
    def _log_prediction_summary(self, prediction_summary, validation_results):
        """记录预测汇总信息"""
        try:
            if not prediction_summary:
                self.Debug("No prediction summary available")
                return
            
            total_symbols = prediction_summary.get('total_symbols', 0)
            valid_count = sum(1 for v in validation_results.values() if v)
            invalid_count = total_symbols - valid_count
            
            self.Debug(f"Prediction Summary:")
            self.Debug(f"  Total symbols predicted: {total_symbols}")
            self.Debug(f"  Valid predictions: {valid_count}")
            self.Debug(f"  Invalid predictions: {invalid_count}")
            self.Debug(f"  Success rate: {valid_count/total_symbols:.2%}" if total_symbols > 0 else "  Success rate: N/A")
            
            avg_confidence = prediction_summary.get('avg_confidence', 0)
            self.Debug(f"  Average confidence: {avg_confidence:.3f}")
            
            # 市场状态分布
            regime_dist = prediction_summary.get('regime_distribution', {})
            if regime_dist:
                self.Debug(f"  Market regime distribution: {regime_dist}")
            
            # 收益率分布
            return_dist = prediction_summary.get('return_distribution', {})
            if return_dist:
                mean_return = return_dist.get('mean', 0)
                positive_count = return_dist.get('positive_count', 0)
                negative_count = return_dist.get('negative_count', 0)
                self.Debug(f"  Expected returns - Mean: {mean_return:.4f}, Positive: {positive_count}, Negative: {negative_count}")
            
        except Exception as e:
            self.Debug(f"Error in prediction summary logging: {e}")
    
    def Rebalance(self):
        """定期调仓入口"""
        if self.IsWarmingUp:
            return
        
        rebalance_start_time = time.time()
        self.Debug(f"=== REBALANCE START at {self.Time} ===")
        
        # 记录调仓前状态
        current_value = self.Portfolio.TotalPortfolioValue
        self.Debug(f"Pre-rebalance Portfolio Value: ${current_value:,.2f}")
        
        # 检查是否需要重新训练
        if self._should_retrain():
            self.Debug("Retraining models...")
            training_success = self._perform_training()
            
            if not training_success:
                self.Debug("Training failed, proceeding with fallback strategy")
        
        # 执行调仓
        self._perform_rebalancing()
        
        # 更新性能指标
        self.performance_metrics['rebalance_count'] += 1
        
        # 记录调仓完成
        rebalance_time = time.time() - rebalance_start_time
        self.Debug(f"=== REBALANCE COMPLETED in {rebalance_time:.1f}s ===")
        self.Debug(f"Total rebalances: {self.performance_metrics['rebalance_count']}")
        
        # 记录调仓后状态
        post_value = self.Portfolio.TotalPortfolioValue
        self.Debug(f"Post-rebalance Portfolio Value: ${post_value:,.2f}")
        
        if current_value > 0:
            value_change = (post_value - current_value) / current_value
            self.Debug(f"Portfolio value change during rebalance: {value_change:.4f} ({value_change*100:.2f}%)")
