# region imports
from AlgorithmImports import *
# endregion
# 训练管理模块
import numpy as np
import time
import gc

try:
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, GlobalAveragePooling1D
    from tensorflow.keras.models import Model
    TF_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        from keras.layers import Input, Conv1D, LSTM, Dense, GlobalAveragePooling1D
        from keras.models import Model
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False

class TrainingManager:
    """模型训练管理器"""
    
    def __init__(self, algorithm):
        """初始化训练管理器"""
        self.algorithm = algorithm
        self.lstm_models = {}
        self.scalers = {}
        self.effective_lookbacks = {}
        self.tradable_symbols = []
    
    def should_retrain(self):
        """判断是否应该重新训练模型"""
        # 检查全局开关
        if not self.algorithm.config.TRAINING_CONFIG['enable_retraining']:
            return False
        
        # 检查是否有现有模型
        if not self.lstm_models:
            self.algorithm.Debug("No existing models found, training required")
            return True
        
        # 检查训练频率
        frequency = self.algorithm.config.TRAINING_CONFIG['retraining_frequency']
        
        if frequency == 'always':
            return True
        elif frequency == 'monthly':
            # 检查是否为月初（已经在调度中控制）
            return True
        elif frequency == 'weekly':
            # 每周重新训练
            current_day = self.algorithm.Time.weekday()
            return current_day == 0  # 周一
        elif frequency == 'quarterly':
            # 每季度重新训练
            current_month = self.algorithm.Time.month
            return current_month in [1, 4, 7, 10] and self.algorithm.Time.day <= 7
        elif frequency == 'never':
            return False
        
        # 默认情况
        return True
    
    def perform_training(self):
        """执行模型训练"""
        if not TF_AVAILABLE:
            self.algorithm.Debug("TensorFlow not available, skipping training")
            return False
            
        training_start_time = time.time()
        
        # 清理旧模型释放内存
        self._cleanup_old_models()
        
        # 重置训练状态
        self.lstm_models = {}
        self.scalers = {}
        self.effective_lookbacks = {}
        self.tradable_symbols = []

        successful_training = 0
        failed_training = 0

        for symbol in self.algorithm.config.SYMBOLS:
            try:
                if self._train_single_symbol(symbol, training_start_time):
                    successful_training += 1
                else:
                    failed_training += 1
                    
                # 检查总训练时间限制
                total_elapsed = time.time() - training_start_time
                if total_elapsed > self.algorithm.config.TRAINING_CONFIG['max_training_time']:
                    self.algorithm.Debug(f"Training time limit reached ({total_elapsed:.1f}s), stopping early with {len(self.tradable_symbols)} symbols")
                    break
                    
            except Exception as e:
                self.algorithm.Debug(f"Error training {symbol}: {e}")
                failed_training += 1
                continue

        self._log_training_summary(training_start_time, successful_training, failed_training)
        
        # 内存清理
        gc.collect()
        
        return len(self.tradable_symbols) > 0
    
    def _cleanup_old_models(self):
        """清理旧模型释放内存"""
        if hasattr(self.algorithm, 'lstm_models'):
            for model in self.algorithm.lstm_models.values():
                try:
                    del model
                except:
                    pass
        gc.collect()
    
    def _train_single_symbol(self, symbol, training_start_time):
        """训练单个股票的模型"""
        symbol_start_time = time.time()
        
        # 使用滚动窗口：获取训练窗口+lookback的数据
        required_data = self.algorithm.config.TRAINING_WINDOW + self.algorithm.config.LOOKBACK_DAYS + 1
        history = self.algorithm.History(symbol, required_data, Resolution.DAILY)
        history_list = list(history)
        prices = [x.Close for x in history_list]
        
        if len(prices) < required_data:
            self.algorithm.Debug(f"Not enough historical data for {symbol}. Found {len(prices)} data points, need {required_data}.")
            return False

        # 只使用最近的训练窗口数据进行训练
        training_prices = prices[-self.algorithm.config.TRAINING_WINDOW:]
        self.algorithm.Debug(f"Training {symbol} with latest {len(training_prices)} data points (6-month window)")
        
        # 数据预处理
        if not self._preprocess_training_data(symbol, training_prices):
            return False
        
        # 构建和训练模型
        if not self._build_and_train_model(symbol, symbol_start_time):
            return False
        
        return True
    
    def _preprocess_training_data(self, symbol, training_prices):
        """预处理训练数据"""
        # 打印价格统计信息
        prices_array = np.array(training_prices)
        self.algorithm.Debug(f"{symbol} training price stats ={prices_array.min():.2f}, max={prices_array.max():.2f}, mean={prices_array.mean():.2f}, std={prices_array.std():.2f}")
        
        # 检查价格数据有效性
        if np.any(np.isnan(prices_array)) or np.any(np.isinf(prices_array)):
            self.algorithm.Debug(f"Invalid price data for {symbol}, skipping")
            return False
        
        # 数据缩放
        prices_reshaped = prices_array.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_scaled = scaler.fit_transform(prices_reshaped)
        self.scalers[symbol] = scaler
        
        # 使用更小的lookback窗口，适应较少的训练数据
        effective_lookback = min(self.algorithm.config.LOOKBACK_DAYS, len(training_prices) // 3)
        X, y = self.create_sequences(prices_scaled, effective_lookback)

        if len(X) < 10:  # 确保有足够的训练样本
            self.algorithm.Debug(f"Not enough training samples for {symbol}. Got {len(X)} samples.")
            return False

        # 调试：打印训练数据统计
        self.algorithm.Debug(f"{symbol} training data - X shape: {X.shape}, y shape: {y.shape}")
        self.algorithm.Debug(f"{symbol} X stats ={X.min():.4f}, max={X.max():.4f}, mean={X.mean():.4f}")
        self.algorithm.Debug(f"{symbol} y stats ={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}")
        
        # 保存训练数据和lookback
        setattr(self, f'_{symbol}_X', X)
        setattr(self, f'_{symbol}_y', y)
        self.effective_lookbacks[symbol] = effective_lookback
        
        return True
    
    def _build_and_train_model(self, symbol, symbol_start_time):
        """构建和训练模型"""
        X = getattr(self, f'_{symbol}_X')
        y = getattr(self, f'_{symbol}_y')
        
        # 构建CNN+LSTM+Attention模型（优化版本，减少参数量）
        input_layer = Input(shape=(X.shape[1], 1))
        x = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(input_layer)
        x = LSTM(units=32, return_sequences=True)(x)
        
        # 简化的attention机制
        attention_weights = tf.keras.layers.Dense(1, activation='tanh')(x)
        attention_weights = tf.keras.layers.Softmax(axis=1)(attention_weights)
        attended = tf.keras.layers.Multiply()([x, attention_weights])
        x = tf.keras.layers.GlobalAveragePooling1D()(attended)
        
        output = Dense(1)(x)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # 训练模型并记录loss
        self.algorithm.Debug(f"Starting training for {symbol} with {self.algorithm.config.MODEL_CONFIG['epochs']} epochs...")
        history_fit = model.fit(X, y, epochs=self.algorithm.config.MODEL_CONFIG['epochs'], batch_size=8, verbose=0)
        
        final_loss = history_fit.history['loss'][-1]
        initial_loss = history_fit.history['loss'][0]
        symbol_time = time.time() - symbol_start_time
        
        self.algorithm.Debug(f"{symbol} training completed in {symbol_time:.1f}s - Initial loss: {initial_loss:.6f}, Final loss: {final_loss:.6f}")

        # 验证训练效果
        if np.isnan(final_loss) or np.isinf(final_loss):
            self.algorithm.Debug(f"Invalid training result for {symbol}, discarding")
            return False

        # 保存模型和相关信息
        self.lstm_models[symbol] = model
        self.tradable_symbols.append(symbol)
        
        # 清理临时训练数据
        delattr(self, f'_{symbol}_X')
        delattr(self, f'_{symbol}_y')
        
        return True
    
    def create_sequences(self, data, seq_length):
        """将价格序列拆分成(X, y)训练对"""
        X, y = [], []
        for i in range(len(data) - seq_length - 1):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    def _log_training_summary(self, training_start_time, successful_training, failed_training):
        """记录训练总结"""
        total_training_time = time.time() - training_start_time
        self.algorithm.Debug(f"CNN+LSTM+Attention training completed for {self.tradable_symbols}")
        self.algorithm.Debug(f"Training used {self.algorithm.config.TRAINING_WINDOW} days of data ({self.algorithm.config.TRAINING_WINDOW/21:.1f} months)")
        self.algorithm.Debug(f"Training results: {successful_training} successful, {failed_training} failed")
        self.algorithm.Debug(f"Total tradable symbols after training: {len(self.tradable_symbols)}")
        self.algorithm.Debug(f"Total training time: {total_training_time:.1f} seconds")
        
        # 调试：打印每个股票的effective_lookback
        for symbol in self.tradable_symbols:
            self.algorithm.Debug(f"{symbol} effective_lookback: {self.effective_lookbacks[symbol]}")
    
    def get_model_for_symbol(self, symbol):
        """获取指定股票的模型"""
        return self.lstm_models.get(symbol)
    
    def get_scaler_for_symbol(self, symbol):
        """获取指定股票的缩放器"""
        return self.scalers.get(symbol)
    
    def get_effective_lookback_for_symbol(self, symbol):
        """获取指定股票的有效lookback"""
        return self.effective_lookbacks.get(symbol)
    
    def update_algorithm_models(self):
        """更新算法的模型引用"""
        self.algorithm.lstm_models = self.lstm_models
        self.algorithm.scalers = self.scalers
        self.algorithm.effective_lookbacks = self.effective_lookbacks
        self.algorithm.tradable_symbols = self.tradable_symbols 