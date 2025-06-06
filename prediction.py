# 预测模块
from AlgorithmImports import *
import numpy as np
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
from config import AlgorithmConfig
from data_processing import DataProcessor
import time

class PredictionEngine:
    """多时间跨度预测引擎"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        self.data_processor = DataProcessor(algorithm_instance)
        
    def generate_multi_horizon_predictions(self, models, symbols):
        """生成多时间跨度预测"""
        predictions = {}
        
        if not TF_AVAILABLE:
            self.algorithm.Debug("TensorFlow not available, using simplified predictions")
            return self._generate_simple_predictions(symbols)
        
        for symbol in symbols:
            try:
                prediction_data = self._predict_single_symbol(symbol, models.get(symbol))
                if prediction_data:
                    predictions[symbol] = prediction_data
                    
            except Exception as e:
                self.algorithm.Debug(f"Error predicting {symbol}: {e}")
                continue
        
        self.algorithm.Debug(f"Generated predictions for {len(predictions)} symbols")
        return predictions
    
    def _generate_simple_predictions(self, symbols):
        """当TensorFlow不可用时的简化预测方法"""
        predictions = {}
        
        for symbol in symbols:
            try:
                # 获取历史数据
                history = self.algorithm.History(symbol, 60, Resolution.Daily)
                history_list = list(history)
                if len(history_list) < 30:
                    continue
                
                prices = np.array([x.Close for x in history_list])
                current_price = prices[-1]
                
                # 简单的动量和均值回归预测
                short_ma = np.mean(prices[-5:])   # 5日移动平均
                long_ma = np.mean(prices[-20:])   # 20日移动平均
                
                # 计算简单的动量信号
                momentum = (short_ma - long_ma) / long_ma
                
                # 计算波动率
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns)
                
                # 简单的预测逻辑
                base_return = momentum * 0.1  # 动量因子
                
                # 生成多时间跨度预测
                horizon_predictions = {}
                for horizon in self.config.PREDICTION_HORIZONS:
                    # 时间跨度衰减
                    time_decay = 0.95 ** (horizon - 1)
                    expected_return = base_return * time_decay
                    
                    predicted_price = current_price * (1 + expected_return)
                    
                    horizon_predictions[horizon] = {
                        'predicted_price': predicted_price,
                        'expected_return': expected_return,
                        'price_std': volatility * current_price,
                        'confidence_interval': {
                            'lower': predicted_price * (1 - volatility),
                            'upper': predicted_price * (1 + volatility)
                        }
                    }
                
                predictions[symbol] = {
                    'predictions': horizon_predictions,
                    'confidence': {
                        'uncertainty_scores': {h: 0.5 for h in self.config.PREDICTION_HORIZONS},
                        'overall_confidence': 0.5
                    },
                    'current_price': current_price,
                    'market_regime': 'neutral'
                }
                
            except Exception as e:
                self.algorithm.Debug(f"Error in simple prediction for {symbol}: {e}")
                continue
        
        return predictions
    
    def _predict_single_symbol(self, symbol, model_info):
        """为单个股票生成预测"""
        try:
            # 检查模型信息是否存在
            if model_info is None:
                self.algorithm.Debug(f"No model info found for {symbol}")
                return None
                
            model = model_info['model']
            effective_lookback = model_info['effective_lookback']
            
            # 检查是否存在对应的scaler
            if symbol not in self.data_processor.scalers:
                self.algorithm.Debug(f"No scaler found for {symbol}, skipping prediction")
                return None
            
            # 获取预测所需的历史数据
            required_data = effective_lookback + 100  # 额外数据用于特征计算
            prices = self.data_processor.get_historical_data(symbol, required_data)
            
            if prices is None or len(prices) < effective_lookback:
                return None
            
            # 创建特征矩阵
            feature_matrix = self.data_processor.create_feature_matrix(prices)
            
            # 使用训练时的缩放器
            scaled_data = self.data_processor.scale_data(feature_matrix, symbol, fit=False)
            
            # 准备输入序列
            X = scaled_data[-effective_lookback:].reshape(1, effective_lookback, -1)
            
            # Monte Carlo Dropout预测（用于不确定性量化）
            predictions_mc = self._monte_carlo_predictions(model, X)
            
            # 计算预测统计信息
            predictions_stats = self._calculate_prediction_statistics(predictions_mc)
            
            # 反向缩放预测结果
            current_price = prices[-1]
            scaled_predictions = self._inverse_scale_predictions(
                predictions_stats, symbol, current_price
            )
            
            # 计算置信度和趋势分析
            confidence_analysis = self._analyze_prediction_confidence(
                predictions_mc, scaled_predictions
            )
            
            return {
                'predictions': scaled_predictions,
                'confidence': confidence_analysis,
                'current_price': current_price,
                'market_regime': self.data_processor.calculate_market_regime(prices)
            }
            
        except Exception as e:
            self.algorithm.Debug(f"Error in single symbol prediction for {symbol}: {e}")
            return None
    
    def _monte_carlo_predictions(self, model, X, n_samples = None):
        """Monte Carlo Dropout预测获取不确定性"""
        if n_samples is None:
            n_samples = self.config.TRAINING_CONFIG['mc_dropout_samples']
        
        # 启用训练模式以激活dropout
        predictions_samples = []
        
        for _ in range(n_samples):
            # 在推理时启用dropout
            pred = model(X, training=True)
            
            # 如果是多输出模型，处理每个输出
            if isinstance(pred, list):
                sample_dict = {}
                for i, horizon in enumerate(self.config.PREDICTION_HORIZONS):
                    sample_dict[horizon] = pred[i].numpy().flatten()
                predictions_samples.append(sample_dict)
            else:
                predictions_samples.append({1: pred.numpy().flatten()})
        
        # 重新组织数据结构
        mc_predictions = {}
        for horizon in self.config.PREDICTION_HORIZONS:
            horizon_samples = [sample[horizon] for sample in predictions_samples if horizon in sample]
            if horizon_samples:
                mc_predictions[horizon] = np.array(horizon_samples)
        
        return mc_predictions
    
    def _calculate_prediction_statistics(self, mc_predictions):
        """计算预测统计信息"""
        stats = {}
        
        for horizon, samples in mc_predictions.items():
            if len(samples) > 0:
                mean_pred = np.mean(samples, axis=0)
                std_pred = np.std(samples, axis=0)
                
                # 计算置信区间
                confidence_level = 0.95
                alpha = 1 - confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                lower_bound = np.percentile(samples, lower_percentile, axis=0)
                upper_bound = np.percentile(samples, upper_percentile, axis=0)
                
                stats[horizon] = {
                    'mean': mean_pred[0] if len(mean_pred) > 0 else 0,
                    'std': std_pred[0] if len(std_pred) > 0 else 0,
                    'lower_bound': lower_bound[0] if len(lower_bound) > 0 else 0,
                    'upper_bound': upper_bound[0] if len(upper_bound) > 0 else 0,
                    'samples': samples
                }
        
        return stats
    
    def _inverse_scale_predictions(self, predictions_stats, symbol, 
                                 current_price):
        """反向缩放预测结果到实际价格"""
        scaled_predictions = {}
        
        for horizon, stats in predictions_stats.items():
            try:
                # 构造虚拟数据用于反向缩放
                dummy_data = np.zeros((1, self.data_processor.scalers[symbol].n_features_in_))
                dummy_data[0, 0] = stats['mean']  # 假设价格在第一列
                
                # 反向缩放
                unscaled = self.data_processor.scalers[symbol].inverse_transform(dummy_data)
                predicted_price = unscaled[0, 0]
                
                # 计算预期收益率
                expected_return = (predicted_price - current_price) / current_price
                
                # 处理置信区间
                dummy_lower = dummy_data.copy()
                dummy_lower[0, 0] = stats['lower_bound']
                dummy_upper = dummy_data.copy()
                dummy_upper[0, 0] = stats['upper_bound']
                
                lower_price = self.data_processor.scalers[symbol].inverse_transform(dummy_lower)[0, 0]
                upper_price = self.data_processor.scalers[symbol].inverse_transform(dummy_upper)[0, 0]
                
                scaled_predictions[horizon] = {
                    'predicted_price': predicted_price,
                    'expected_return': expected_return,
                    'price_std': stats['std'],
                    'confidence_interval': {
                        'lower': lower_price,
                        'upper': upper_price
                    }
                }
                
            except Exception as e:
                self.algorithm.Debug(f"Error inverse scaling for {symbol} horizon {horizon}: {e}")
                scaled_predictions[horizon] = {
                    'predicted_price': current_price,
                    'expected_return': 0.0,
                    'price_std': 0.0,
                    'confidence_interval': {'lower': current_price, 'upper': current_price}
                }
        
        return scaled_predictions
    
    def _analyze_prediction_confidence(self, mc_predictions, scaled_predictions):
        """分析预测置信度"""
        confidence_analysis = {}
        
        # 1. 不确定性评分
        uncertainty_scores = {}
        for horizon in self.config.PREDICTION_HORIZONS:
            if horizon in scaled_predictions:
                std = scaled_predictions[horizon]['price_std']
                predicted_price = scaled_predictions[horizon]['predicted_price']
                
                # 标准化的不确定性（变异系数）
                cv = std / abs(predicted_price) if predicted_price != 0 else 1.0
                uncertainty_score = max(0, 1 - cv)  # 越低的变异系数，置信度越高
                uncertainty_scores[horizon] = uncertainty_score
        
        # 2. 趋势一致性分析
        trend_consistency = self._analyze_trend_consistency(scaled_predictions)
        
        # 3. 预测范围合理性
        range_reasonableness = self._analyze_prediction_range(scaled_predictions)
        
        # 4. 综合置信度评分
        overall_confidence = self._calculate_overall_confidence(
            uncertainty_scores, trend_consistency, range_reasonableness
        )
        
        confidence_analysis = {
            'uncertainty_scores': uncertainty_scores,
            'trend_consistency': trend_consistency,
            'range_reasonableness': range_reasonableness,
            'overall_confidence': overall_confidence
        }
        
        return confidence_analysis
    
    def _analyze_trend_consistency(self, predictions):
        """分析趋势一致性"""
        horizons = sorted(predictions.keys())
        if len(horizons) < 2:
            return {'score': 0.5, 'direction': 'neutral'}
        
        returns = []
        for horizon in horizons:
            returns.append(predictions[horizon]['expected_return'])
        
        # 检查趋势方向一致性
        positive_count = sum(1 for r in returns if r > 0)
        negative_count = sum(1 for r in returns if r < 0)
        
        if positive_count == len(returns):
            direction = 'bullish'
            consistency_score = 1.0
        elif negative_count == len(returns):
            direction = 'bearish'  
            consistency_score = 1.0
        else:
            direction = 'mixed'
            consistency_score = max(positive_count, negative_count) / len(returns)
        
        # 检查预测幅度的递进性（短期预测应该较小）
        magnitude_consistency = self._check_magnitude_progression(returns, horizons)
        
        overall_score = (consistency_score + magnitude_consistency) / 2
        
        return {
            'score': overall_score,
            'direction': direction,
            'magnitude_consistency': magnitude_consistency
        }
    
    def _check_magnitude_progression(self, returns, horizons):
        """检查预测幅度的递进性"""
        if len(returns) < 2:
            return 1.0
        
        # 计算每日平均收益率
        daily_returns = [abs(ret) / horizon for ret, horizon in zip(returns, horizons)]
        
        # 理想情况下，日均收益率应该相对稳定
        std_daily_returns = np.std(daily_returns)
        mean_daily_returns = np.mean(daily_returns)
        
        if mean_daily_returns == 0:
            return 1.0
        
        cv = std_daily_returns / mean_daily_returns
        consistency_score = max(0, 1 - cv)  # 变异系数越小，一致性越好
        
        return consistency_score
    
    def _analyze_prediction_range(self, predictions):
        """分析预测范围的合理性"""
        range_scores = {}
        
        for horizon, pred in predictions.items():
            expected_return = abs(pred['expected_return'])
            
            # 基于时间跨度的合理性检查
            max_reasonable_daily_return = 0.1  # 10%日涨跌幅上限
            max_reasonable_return = max_reasonable_daily_return * horizon * 0.7  # 打个折扣
            
            if expected_return <= max_reasonable_return:
                range_scores[horizon] = 1.0
            else:
                # 超出合理范围则降低评分
                range_scores[horizon] = max(0, max_reasonable_return / expected_return)
        
        overall_range_score = np.mean(list(range_scores.values())) if range_scores else 0.5
        
        return {
            'individual_scores': range_scores,
            'overall_score': overall_range_score
        }
    
    def _calculate_overall_confidence(self, uncertainty_scores, 
                                    trend_consistency, 
                                    range_reasonableness):
        """计算综合置信度评分"""
        # 权重设置
        weights = {
            'uncertainty': 0.4,
            'trend_consistency': 0.3,
            'range_reasonableness': 0.3
        }
        
        # 计算加权平均
        uncertainty_avg = np.mean(list(uncertainty_scores.values())) if uncertainty_scores else 0.5
        trend_score = trend_consistency.get('score', 0.5)
        range_score = range_reasonableness.get('overall_score', 0.5)
        
        overall_confidence = (
            weights['uncertainty'] * uncertainty_avg +
            weights['trend_consistency'] * trend_score +
            weights['range_reasonableness'] * range_score
        )
        
        return overall_confidence

class ExpectedReturnCalculator:
    """预期收益计算器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        
    def calculate_expected_returns(self, predictions):
        """计算预期收益率"""
        expected_returns = {}
        valid_symbols = []
        
        for symbol, pred_data in predictions.items():
            try:
                # 计算多时间跨度加权收益率
                weighted_return = self._calculate_weighted_return(pred_data)
                
                # 应用置信度调整
                confidence_adjusted_return = self._apply_confidence_adjustment(
                    weighted_return, pred_data['confidence']
                )
                
                # 应用市场状态调整
                regime_adjusted_return = self._apply_regime_adjustment(
                    confidence_adjusted_return, pred_data['market_regime']
                )
                
                expected_returns[symbol] = regime_adjusted_return
                valid_symbols.append(symbol)
                
                self.algorithm.Debug(f"{symbol} expected return calculation:")
                self.algorithm.Debug(f"  Raw weighted: {weighted_return:.6f}")
                self.algorithm.Debug(f"  Confidence adjusted: {confidence_adjusted_return:.6f}")
                self.algorithm.Debug(f"  Final (regime adjusted): {regime_adjusted_return:.6f}")
                
            except Exception as e:
                self.algorithm.Debug(f"Error calculating expected return for {symbol}: {e}")
                continue
        
        return expected_returns, valid_symbols
    
    def _calculate_weighted_return(self, pred_data):
        """计算多时间跨度加权收益率"""
        weighted_return = 0.0
        total_weight = 0.0
        
        predictions = pred_data['predictions']
        
        for horizon in self.config.PREDICTION_HORIZONS:
            if horizon in predictions:
                horizon_return = predictions[horizon]['expected_return']
                weight = self.config.HORIZON_WEIGHTS.get(horizon, 0)
                
                weighted_return += horizon_return * weight
                total_weight += weight
        
        if total_weight > 0:
            weighted_return /= total_weight
        
        return weighted_return
    
    def _apply_confidence_adjustment(self, base_return, confidence):
        """应用置信度调整"""
        overall_confidence = confidence.get('overall_confidence', 0.5)
        
        # 低置信度时缩小预期收益
        confidence_factor = 0.5 + 0.5 * overall_confidence
        adjusted_return = base_return * confidence_factor
        
        return adjusted_return
    
    def _apply_regime_adjustment(self, base_return, market_regime):
        """应用市场状态调整"""
        regime_factors = {
            'high_volatility': 0.8,  # 高波动时保守
            'trending': 1.2,         # 趋势明确时积极
            'low_volatility': 1.0,   # 低波动时正常
            'neutral': 1.0,          # 中性时正常
            'unknown': 0.9           # 未知时略保守
        }
        
        factor = regime_factors.get(market_regime, 1.0)
        adjusted_return = base_return * factor
        
        return adjusted_return

class PredictionValidator:
    """预测验证器"""
    
    @staticmethod
    def validate_predictions(predictions):
        """验证预测结果的有效性"""
        validation_results = {}
        
        for symbol, pred_data in predictions.items():
            is_valid = True
            
            # 检查必要字段
            required_fields = ['predictions', 'confidence', 'current_price']
            for field in required_fields:
                if field not in pred_data:
                    is_valid = False
                    break
            
            if is_valid and 'predictions' in pred_data:
                # 检查预测值的有效性
                for horizon, pred in pred_data['predictions'].items():
                    expected_return = pred.get('expected_return', 0)
                    
                    # 检查是否为有效数值
                    if np.isnan(expected_return) or np.isinf(expected_return):
                        is_valid = False
                        break
                    
                    # 检查是否在合理范围内
                    if abs(expected_return) > 1.0:  # 100%的涨跌幅上限
                        is_valid = False
                        break
            
            validation_results[symbol] = is_valid
        
        return validation_results
    
    @staticmethod
    def get_prediction_summary(predictions):
        """获取预测汇总信息"""
        if not predictions:
            return {}
        
        summary = {
            'total_symbols': len(predictions),
            'avg_confidence': 0,
            'regime_distribution': {},
            'return_distribution': {}
        }
        
        confidences = []
        regimes = []
        returns = []
        
        for symbol, pred_data in predictions.items():
            # 置信度
            confidence = pred_data.get('confidence', {}).get('overall_confidence', 0)
            confidences.append(confidence)
            
            # 市场状态
            regime = pred_data.get('market_regime', 'unknown')
            regimes.append(regime)
            
            # 预期收益（使用1天期）
            predictions_dict = pred_data.get('predictions', {})
            if 1 in predictions_dict:
                expected_return = predictions_dict[1].get('expected_return', 0)
                returns.append(expected_return)
        
        # 计算统计信息
        if confidences:
            summary['avg_confidence'] = np.mean(confidences)
        
        if regimes:
            from collections import Counter
            regime_counts = Counter(regimes)
            summary['regime_distribution'] = dict(regime_counts)
        
        if returns:
            summary['return_distribution'] = {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'positive_count': sum(1 for r in returns if r > 0),
                'negative_count': sum(1 for r in returns if r < 0)
            }
        
        return summary 