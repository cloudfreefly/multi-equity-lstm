# 数据处理模块
from AlgorithmImports import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from config import AlgorithmConfig, TechnicalConfig
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    # 创建简单的技术指标替代函数
    class talib:
        @staticmethod
        def SMA(close, timeperiod):
            """简单移动平均"""
            return pd.Series(close).rolling(window=timeperiod).mean().values
        
        @staticmethod
        def EMA(close, timeperiod):
            """指数移动平均"""
            return pd.Series(close).ewm(span=timeperiod).mean().values
        
        @staticmethod
        def RSI(close, timeperiod=14):
            """相对强弱指数"""
            delta = pd.Series(close).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
            rs = gain / loss
            return (100 - (100 / (1 + rs))).values
        
        @staticmethod
        def BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2):
            """布林线"""
            sma = pd.Series(close).rolling(window=timeperiod).mean()
            std = pd.Series(close).rolling(window=timeperiod).std()
            upper = sma + (std * nbdevup)
            lower = sma - (std * nbdevdn)
            return upper.values, sma.values, lower.values
        
        @staticmethod
        def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
            """MACD指标"""
            exp1 = pd.Series(close).ewm(span=fastperiod).mean()
            exp2 = pd.Series(close).ewm(span=slowperiod).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=signalperiod).mean()
            histogram = macd - signal
            return macd.values, signal.values, histogram.values

class DataProcessor:
    """数据处理和预处理类"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.scalers = {}
        self.config = AlgorithmConfig()
        self.tech_config = TechnicalConfig()
        
    def get_historical_data(self, symbol, days):
        """获取历史价格数据"""
        try:
            history = self.algorithm.History(symbol, days, Resolution.Daily)
            
            # 将枚举对象转换为列表
            history_list = list(history)
            if len(history_list) == 0:
                self.algorithm.Debug(f"No historical data available for {symbol}")
                return None
            
            prices = np.array([x.Close for x in history_list])
            volumes = np.array([x.Volume for x in history_list])
            
            # 数据质量检查
            if self._validate_price_data(prices, symbol):
                return prices
            else:
                return None
                
        except Exception as e:
            self.algorithm.Debug(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def _validate_price_data(self, prices, symbol):
        """验证价格数据质量"""
        if len(prices) == 0:
            self.algorithm.Debug(f"Empty price data for {symbol}")
            return False
            
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            self.algorithm.Debug(f"Invalid price data (NaN/Inf) for {symbol}")
            return False
            
        if np.any(prices <= 0):
            self.algorithm.Debug(f"Non-positive prices found for {symbol}")
            return False
            
        # 检查异常波动
        returns = np.diff(prices) / prices[:-1]
        if np.any(np.abs(returns) > 0.5):  # 单日涨跌幅超过50%
            self.algorithm.Debug(f"Extreme price movements detected for {symbol}")
            return False
            
        return True
    
    def calculate_technical_indicators(self, prices, volumes = None):
        """计算技术指标"""
        indicators = {}
        
        try:
            # 价格相关指标
            indicators['sma_20'] = talib.SMA(prices, timeperiod=20)
            indicators['sma_50'] = talib.SMA(prices, timeperiod=50)
            indicators['ema_12'] = talib.EMA(prices, timeperiod=12)
            indicators['ema_26'] = talib.EMA(prices, timeperiod=26)
            
            # RSI
            indicators['rsi'] = talib.RSI(prices, timeperiod=self.tech_config.TECHNICAL_INDICATORS['rsi_period'])
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                prices,
                fastperiod=self.tech_config.TECHNICAL_INDICATORS['macd_fast'],
                slowperiod=self.tech_config.TECHNICAL_INDICATORS['macd_slow'],
                signalperiod=self.tech_config.TECHNICAL_INDICATORS['macd_signal']
            )
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_histogram'] = macd_hist
            
            # 布林带
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                prices,
                timeperiod=self.tech_config.TECHNICAL_INDICATORS['bollinger_period'],
                nbdevup=self.tech_config.TECHNICAL_INDICATORS['bollinger_std'],
                nbdevdn=self.tech_config.TECHNICAL_INDICATORS['bollinger_std']
            )
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            
            # ATR (平均真实波幅)
            high = prices  # 简化处理，实际应该用高价
            low = prices   # 简化处理，实际应该用低价
            indicators['atr'] = talib.ATR(high, low, prices, timeperiod=self.tech_config.TECHNICAL_INDICATORS['atr_period'])
            
            # 价格动量
            indicators['momentum'] = talib.MOM(prices, timeperiod=10)
            indicators['roc'] = talib.ROC(prices, timeperiod=10)
            
            # 如果有成交量数据
            if volumes is not None:
                indicators['volume_sma'] = talib.SMA(volumes, timeperiod=self.tech_config.TECHNICAL_INDICATORS['volume_ma_period'])
                # 成交量加权平均价格近似
                indicators['vwap_approx'] = np.cumsum(prices * volumes) / np.cumsum(volumes)
            
        except Exception as e:
            self.algorithm.Debug(f"Error calculating technical indicators: {e}")
            
        return indicators
    
    def create_feature_matrix(self, prices, volumes = None):
        """创建特征矩阵，包含价格和技术指标"""
        features = []
        
        # 价格特征
        features.append(prices.reshape(-1, 1))
        
        # 收益率特征
        returns = np.diff(prices) / prices[:-1]
        returns = np.concatenate([[0], returns])  # 第一个元素设为0
        features.append(returns.reshape(-1, 1))
        
        # 对数收益率
        log_returns = np.diff(np.log(prices))
        log_returns = np.concatenate([[0], log_returns])
        features.append(log_returns.reshape(-1, 1))
        
        # 波动率特征（滚动标准差）
        volatility = pd.Series(returns).rolling(window=20, min_periods=1).std().values
        features.append(volatility.reshape(-1, 1))
        
        # 技术指标
        indicators = self.calculate_technical_indicators(prices, volumes)
        for name, values in indicators.items():
            if values is not None and not np.all(np.isnan(values)):
                # 填充NaN值
                values_filled = pd.Series(values).fillna(method='bfill').fillna(method='ffill').values
                features.append(values_filled.reshape(-1, 1))
        
        # 合并所有特征
        feature_matrix = np.concatenate(features, axis=1)
        
        # 处理任何剩余的NaN
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        return feature_matrix
    
    def scale_data(self, data, symbol, fit = True):
        """数据缩放处理"""
        if symbol not in self.scalers and not fit:
            raise ValueError(f"No scaler found for {symbol} and fit=False")
        
        if fit:
            # 使用RobustScaler以更好地处理异常值
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(data)
            self.scalers[symbol] = scaler
        else:
            scaler = self.scalers[symbol]
            scaled_data = scaler.transform(data)
        
        return scaled_data
    
    def inverse_scale_predictions(self, predictions, symbol):
        """反向缩放预测结果"""
        if symbol not in self.scalers:
            raise ValueError(f"No scaler found for {symbol}")
        
        scaler = self.scalers[symbol]
        
        # 如果预测是多维的，需要构造对应的形状
        if predictions.ndim == 1:
            # 单一预测值
            dummy_data = np.zeros((len(predictions), scaler.n_features_in_))
            dummy_data[:, 0] = predictions  # 假设价格在第一列
            unscaled = scaler.inverse_transform(dummy_data)
            return unscaled[:, 0]
        else:
            return scaler.inverse_transform(predictions)
    
    def create_multi_horizon_sequences(self, data, seq_length, 
                                     horizons):
        """创建多时间跨度预测序列"""
        X = []
        y_dict = {h: [] for h in horizons}
        
        max_horizon = max(horizons)
        
        for i in range(len(data) - seq_length - max_horizon):
            # 输入序列
            X.append(data[i:(i + seq_length)])
            
            # 多个时间跨度的目标值
            for horizon in horizons:
                target_idx = i + seq_length + horizon - 1
                if target_idx < len(data):
                    y_dict[horizon].append(data[target_idx, 0])  # 假设价格在第一列
        
        X = np.array(X)
        y_arrays = {h: np.array(y_dict[h]) for h in horizons}
        
        return X, y_arrays
    
    def detect_outliers(self, data, method = 'iqr', 
                       threshold = 1.5):
        """检测异常值"""
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (data < lower_bound) | (data > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            outliers = z_scores > threshold
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return outliers
    
    def clean_data(self, data, symbol):
        """数据清洗"""
        cleaned_data = data.copy()
        
        # 检测异常值
        outliers = self.detect_outliers(cleaned_data[:, 0])  # 检测价格列的异常值
        
        if np.any(outliers):
            self.algorithm.Debug(f"Found {np.sum(outliers)} outliers in {symbol} data")
            
            # 使用线性插值替换异常值
            outlier_indices = np.where(outliers)[0]
            for idx in outlier_indices:
                if 0 < idx < len(cleaned_data) - 1:
                    cleaned_data[idx, 0] = (cleaned_data[idx-1, 0] + cleaned_data[idx+1, 0]) / 2
                elif idx == 0:
                    cleaned_data[idx, 0] = cleaned_data[idx+1, 0]
                else:
                    cleaned_data[idx, 0] = cleaned_data[idx-1, 0]
        
        return cleaned_data
    
    def calculate_market_regime(self, prices):
        """检测市场状态"""
        if len(prices) < 60:
            return "unknown"
        
        # 计算不同时间窗口的指标
        recent_prices = prices[-60:]  # 最近60天
        
        # 波动率
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # 年化波动率
        
        # 趋势强度
        trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        price_range = np.max(recent_prices) - np.min(recent_prices)
        trend_strength = abs(trend_slope) / (price_range / len(recent_prices))
        
        # 动量
        momentum = (recent_prices[-1] - recent_prices[-20]) / recent_prices[-20]
        
        # 判断市场状态
        if volatility > 0.25:
            regime = "high_volatility"
        elif abs(momentum) > 0.05 and trend_strength > 0.1:
            regime = "trending"
        elif volatility < 0.15 and abs(momentum) < 0.02:
            regime = "low_volatility"
        else:
            regime = "neutral"
        
        return regime

class DataValidator:
    """数据验证工具"""
    
    @staticmethod
    def validate_sequences(X, y):
        """验证序列数据的完整性"""
        if X.shape[0] == 0:
            return False
        
        # 检查所有时间跨度的目标值长度是否一致
        y_lengths = [len(targets) for targets in y.values()]
        if len(set(y_lengths)) > 1:
            return False
        
        # 检查X和y的样本数是否匹配
        if X.shape[0] != y_lengths[0]:
            return False
        
        return True
    
    @staticmethod
    def check_data_quality(data, symbol):
        """检查数据质量指标"""
        quality_metrics = {}
        
        # 缺失值比例
        quality_metrics['missing_ratio'] = np.sum(np.isnan(data)) / data.size
        
        # 异常值比例（使用IQR方法）
        if data.size > 0:
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            outliers = np.sum((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR))
            quality_metrics['outlier_ratio'] = outliers / data.size
        else:
            quality_metrics['outlier_ratio'] = 0
        
        # 数据连续性（检查是否有大的跳跃）
        if len(data) > 1:
            jumps = np.abs(np.diff(data)) / data[:-1]
            quality_metrics['max_jump'] = np.max(jumps)
            quality_metrics['jump_ratio'] = np.sum(jumps > 0.1) / len(jumps)
        else:
            quality_metrics['max_jump'] = 0
            quality_metrics['jump_ratio'] = 0
        
        return quality_metrics 