# 模型训练模块
from AlgorithmImports import *
import numpy as np
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization,
                             MultiHeadAttention, GlobalAveragePooling1D, Concatenate,
                             LayerNormalization, Add)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
except ImportError:
    # Fallback for older QuantConnect environments
    try:
        import tensorflow as tf
        from keras.models import Model
        from keras.layers import (Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization,
                                 MultiHeadAttention, GlobalAveragePooling1D, Concatenate,
                                 LayerNormalization, Add)
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from keras.optimizers import Adam
        from keras.regularizers import l2
    except ImportError:
        pass  # Will handle TensorFlow unavailability in runtime

import time
import gc
# 简化类型注解以兼容QuantConnect云端
from config import AlgorithmConfig
from data_processing import DataProcessor, DataValidator

class MultiHorizonModel:
    """多时间跨度预测模型"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        self.models = {}
        self.training_history = {}
        self.feature_importance = {}
        
    def build_model(self, input_shape, horizons):
        """构建多时间跨度CNN+LSTM+Attention模型"""
        
        inputs = Input(shape=input_shape, name='main_input')
        
        # 多尺度CNN层
        conv_outputs = []
        for i, (filters, kernel_size) in enumerate(zip(
            self.config.MODEL_CONFIG['conv_filters'],
            self.config.MODEL_CONFIG['conv_kernels']
        )):
            conv = Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                kernel_regularizer=l2(0.001),
                name=f'conv1d_{i+1}'
            )(inputs)
            conv = BatchNormalization(name=f'bn_conv_{i+1}')(conv)
            conv = Dropout(self.config.MODEL_CONFIG['dropout_rate'], name=f'dropout_conv_{i+1}')(conv)
            conv_outputs.append(conv)
        
        # 合并多尺度特征
        if len(conv_outputs) > 1:
            merged_conv = Concatenate(axis=-1, name='conv_concat')(conv_outputs)
        else:
            merged_conv = conv_outputs[0]
        
        # 多层LSTM
        lstm_output = merged_conv
        for i, units in enumerate(self.config.MODEL_CONFIG['lstm_units']):
            return_sequences = (i < len(self.config.MODEL_CONFIG['lstm_units']) - 1) or True  # 最后一层也返回序列用于attention
            lstm_output = LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.config.MODEL_CONFIG['dropout_rate'],
                recurrent_dropout=self.config.MODEL_CONFIG['dropout_rate'],
                kernel_regularizer=l2(0.001),
                name=f'lstm_{i+1}'
            )(lstm_output)
            lstm_output = BatchNormalization(name=f'bn_lstm_{i+1}')(lstm_output)
        
        # Multi-Head Attention机制
        attention_output = MultiHeadAttention(
            num_heads=self.config.MODEL_CONFIG['attention_heads'],
            key_dim=lstm_output.shape[-1] // self.config.MODEL_CONFIG['attention_heads'],
            dropout=self.config.MODEL_CONFIG['dropout_rate'],
            name='multi_head_attention'
        )(lstm_output, lstm_output)
        
        # 残差连接
        attention_output = Add(name='residual_connection')([lstm_output, attention_output])
        attention_output = LayerNormalization(name='layer_norm')(attention_output)
        
        # 全局平均池化
        pooled_output = GlobalAveragePooling1D(name='global_avg_pool')(attention_output)
        
        # 共享特征层
        shared_features = Dense(
            128, 
            activation='relu', 
            kernel_regularizer=l2(0.001),
            name='shared_dense'
        )(pooled_output)
        shared_features = Dropout(self.config.MODEL_CONFIG['dropout_rate'], name='shared_dropout')(shared_features)
        
        # 为每个时间跨度创建专门的输出头
        outputs = []
        output_names = []
        for horizon in horizons:
            # 时间跨度特定的特征层
            horizon_features = Dense(
                64, 
                activation='relu',
                kernel_regularizer=l2(0.001),
                name=f'horizon_{horizon}_dense'
            )(shared_features)
            horizon_features = Dropout(self.config.MODEL_CONFIG['dropout_rate'], name=f'horizon_{horizon}_dropout')(horizon_features)
            
            # 输出层
            output = Dense(1, name=f'output_{horizon}d')(horizon_features)
            outputs.append(output)
            output_names.append(f'output_{horizon}d')
        
        # 创建模型
        model = Model(inputs=inputs, outputs=outputs, name='multi_horizon_model')
        
        # 编译模型 - 为不同时间跨度设置不同的损失权重
        losses = {name: 'mse' for name in output_names}
        loss_weights = {name: self.config.HORIZON_WEIGHTS[horizons[i]] for i, name in enumerate(output_names)}
        
        # 为每个输出指定metrics
        metrics_dict = {name: ['mae'] for name in output_names}
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics_dict
        )
        
        return model
    
    def prepare_training_data(self, data_processor, symbol, 
                            horizons):
        """准备训练数据"""
        try:
            # 获取历史数据
            required_data = self.config.TRAINING_WINDOW + self.config.LOOKBACK_DAYS + max(horizons) + 50
            prices = data_processor.get_historical_data(symbol, required_data)
            
            if prices is None or len(prices) < required_data:
                self.algorithm.Debug(f"Insufficient data for {symbol}")
                return None
            
            # 创建特征矩阵
            feature_matrix = data_processor.create_feature_matrix(prices)
            
            # 数据清洗
            cleaned_data = data_processor.clean_data(feature_matrix, symbol)
            
            # 数据缩放
            scaled_data = data_processor.scale_data(cleaned_data, symbol, fit=True)
            
            # 只使用最近的训练窗口数据
            training_data = scaled_data[-self.config.TRAINING_WINDOW:]
            
            # 计算有效的lookback长度
            effective_lookback = min(self.config.LOOKBACK_DAYS, len(training_data) // 3)
            
            # 创建多时间跨度序列
            X, y_dict = data_processor.create_multi_horizon_sequences(
                training_data, effective_lookback, horizons
            )
            
            # 验证数据
            if not DataValidator.validate_sequences(X, y_dict):
                self.algorithm.Debug(f"Invalid sequences for {symbol}")
                return None
            
            if X.shape[0] < 20:  # 确保有足够的训练样本
                self.algorithm.Debug(f"Too few training sequences for {symbol}: {X.shape[0]}")
                return None
            
            return X, y_dict, effective_lookback
            
        except Exception as e:
            self.algorithm.Debug(f"Error preparing training data for {symbol}: {e}")
            return None
    
    def train_model(self, symbol, data_processor):
        """训练单个股票的模型"""
        start_time = time.time()
        
        try:
            # 准备训练数据
            training_data = self.prepare_training_data(
                data_processor, symbol, self.config.PREDICTION_HORIZONS
            )
            
            if training_data is None:
                return False
            
            X, y_dict, effective_lookback = training_data
            
            # 数据统计
            self.algorithm.Debug(f"Training {symbol}: X shape={X.shape}, effective_lookback={effective_lookback}")
            for horizon, y in y_dict.items():
                self.algorithm.Debug(f"  {horizon}d targets: {len(y)} samples, range=[{y.min():.4f}, {y.max():.4f}]")
            
            # 构建模型
            model = self.build_model(
                input_shape=(X.shape[1], X.shape[2]),
                horizons=self.config.PREDICTION_HORIZONS
            )
            
            # 准备目标数据 - 转换为字典格式
            y_train = {f'output_{h}d': y_dict[h] for h in self.config.PREDICTION_HORIZONS}
            
            # 分割训练和验证数据
            val_ratio = self.config.TRAINING_CONFIG['model_validation_ratio']
            split_idx = int(len(X) * (1 - val_ratio))
            
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train_dict = {key: values[:split_idx] for key, values in y_train.items()}
            y_val_dict = {key: values[split_idx:] for key, values in y_train.items()}
            
            # 设置回调
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.MODEL_CONFIG['patience'],
                    restore_best_weights=True,
                    verbose=0
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,
                    patience=3,
                    min_lr=1e-6,
                    verbose=0
                )
            ]
            
            # 训练模型
            history = model.fit(
                X_train, y_train_dict,
                validation_data=(X_val, y_val_dict),
                epochs=self.config.MODEL_CONFIG['epochs'],
                batch_size=self.config.MODEL_CONFIG['batch_size'],
                callbacks=callbacks,
                verbose=0
            )
            
            # 验证训练结果
            final_loss = min(history.history['val_loss'])
            initial_loss = history.history['val_loss'][0]
            
            if np.isnan(final_loss) or np.isinf(final_loss):
                self.algorithm.Debug(f"Invalid training result for {symbol}")
                return False
            
            # 保存模型和相关信息
            self.models[symbol] = {
                'model': model,
                'effective_lookback': effective_lookback,
                'training_samples': X.shape[0],
                'feature_dim': X.shape[2]
            }
            
            self.training_history[symbol] = {
                'history': history.history,
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'improvement': (initial_loss - final_loss) / initial_loss,
                'training_time': time.time() - start_time
            }
            
            # 记录训练结果
            training_time = time.time() - start_time
            improvement = (initial_loss - final_loss) / initial_loss * 100
            
            self.algorithm.Debug(f"{symbol} training completed:")
            self.algorithm.Debug(f"  Time: {training_time:.1f}s")
            self.algorithm.Debug(f"  Loss: {initial_loss:.6f} -> {final_loss:.6f} ({improvement:.1f}% improvement)")
            self.algorithm.Debug(f"  Samples: {X.shape[0]}, Features: {X.shape[2]}")
            
            return True
            
        except Exception as e:
            self.algorithm.Debug(f"Error training model for {symbol}: {e}")
            return False
    
    def get_model_summary(self, symbol):
        """获取模型摘要信息"""
        if symbol not in self.models:
            return {}
        
        model_info = self.models[symbol]
        history_info = self.training_history.get(symbol, {})
        
        return {
            'effective_lookback': model_info['effective_lookback'],
            'training_samples': model_info['training_samples'],
            'feature_dim': model_info['feature_dim'],
            'final_loss': history_info.get('final_loss', 0),
            'improvement': history_info.get('improvement', 0),
            'training_time': history_info.get('training_time', 0)
        }

class ModelTrainer:
    """模型训练管理器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        self.data_processor = DataProcessor(algorithm_instance)
        self.multi_horizon_model = MultiHorizonModel(algorithm_instance)
        
        # 重要：添加scaler管理
        self.scalers = {}
        
    def train_all_models(self):
        """训练所有股票的模型"""
        training_start_time = time.time()
        self.algorithm.Debug("Starting multi-horizon model training")
        
        # 清理旧模型释放内存
        self._cleanup_models()
        
        successful_symbols = []
        failed_symbols = []
        
        for symbol in self.config.SYMBOLS:
            try:
                # 检查训练时间限制
                elapsed_time = time.time() - training_start_time
                if elapsed_time > self.config.TRAINING_CONFIG['max_training_time']:
                    self.algorithm.Debug(f"Training time limit reached ({elapsed_time:.1f}s), stopping early")
                    break
                
                # 训练模型
                success = self.multi_horizon_model.train_model(symbol, self.data_processor)
                
                if success:
                    successful_symbols.append(symbol)
                    self.algorithm.Debug(f"✓ {symbol} training successful")
                else:
                    failed_symbols.append(symbol)
                    self.algorithm.Debug(f"✗ {symbol} training failed")
                
                # 定期内存清理
                if len(successful_symbols) % self.config.TRAINING_CONFIG['memory_cleanup_interval'] == 0:
                    gc.collect()
                
            except Exception as e:
                self.algorithm.Debug(f"Critical error training {symbol}: {e}")
                failed_symbols.append(symbol)
                continue
        
        # 训练总结
        total_time = time.time() - training_start_time
        self.algorithm.Debug(f"Multi-horizon training completed:")
        self.algorithm.Debug(f"  Total time: {total_time:.1f}s")
        self.algorithm.Debug(f"  Successful: {len(successful_symbols)}")
        self.algorithm.Debug(f"  Failed: {len(failed_symbols)}")
        self.algorithm.Debug(f"  Success rate: {len(successful_symbols)/(len(successful_symbols)+len(failed_symbols))*100:.1f}%")
        
        # 重要：收集并同步scalers到算法和data_processor
        self._collect_and_sync_scalers(successful_symbols)
        
        # 返回模型字典和成功训练的股票列表
        return self.multi_horizon_model.models, successful_symbols
    
    def _cleanup_models(self):
        """清理旧模型释放内存"""
        if hasattr(self.multi_horizon_model, 'models'):
            for symbol, model_info in self.multi_horizon_model.models.items():
                try:
                    if 'model' in model_info:
                        del model_info['model']
                except:
                    pass
        
        self.multi_horizon_model.models = {}
        self.multi_horizon_model.training_history = {}
        gc.collect()
        
        self.algorithm.Debug("Model cleanup completed")
    
    def get_trained_models(self):
        """获取训练好的模型字典"""
        return self.multi_horizon_model.models
    
    def get_tradable_symbols(self):
        """获取可交易的股票列表"""
        return list(self.multi_horizon_model.models.keys())
    
    def get_training_statistics(self):
        """获取训练统计信息"""
        stats = {
            'total_models': len(self.multi_horizon_model.models),
            'avg_training_time': 0,
            'avg_improvement': 0,
            'best_model': None,
            'worst_model': None
        }
        
        if not self.multi_horizon_model.training_history:
            return stats
        
        # 计算平均指标
        training_times = []
        improvements = []
        
        for symbol, history in self.multi_horizon_model.training_history.items():
            training_times.append(history.get('training_time', 0))
            improvements.append(history.get('improvement', 0))
        
        if training_times:
            stats['avg_training_time'] = np.mean(training_times)
            stats['avg_improvement'] = np.mean(improvements)
            
            # 找到最佳和最差模型
            best_idx = np.argmax(improvements)
            worst_idx = np.argmin(improvements)
            
            symbols = list(self.multi_horizon_model.training_history.keys())
            stats['best_model'] = symbols[best_idx]
            stats['worst_model'] = symbols[worst_idx]
        
        return stats
    
    def validate_model_quality(self, symbol):
        """验证模型质量"""
        if symbol not in self.multi_horizon_model.models:
            return {'quality_score': 0.0}
        
        history = self.multi_horizon_model.training_history.get(symbol, {})
        
        # 计算质量分数
        improvement = history.get('improvement', 0)
        final_loss = history.get('final_loss', 1.0)
        
        # 质量分数基于多个因素
        quality_score = 0.0
        
        # 1. 损失改善程度 (0-40分)
        if improvement > 0:
            quality_score += min(improvement * 40, 40)
        
        # 2. 最终损失水平 (0-30分)
        if final_loss < 0.1:
            quality_score += 30
        elif final_loss < 0.2:
            quality_score += 20
        elif final_loss < 0.5:
            quality_score += 10
        
        # 3. 训练样本数 (0-20分)
        training_samples = self.multi_horizon_model.models[symbol]['training_samples']
        if training_samples > 100:
            quality_score += 20
        elif training_samples > 50:
            quality_score += 15
        elif training_samples > 20:
            quality_score += 10
        
        # 4. 特征维度 (0-10分)
        feature_dim = self.multi_horizon_model.models[symbol]['feature_dim']
        if feature_dim > 10:
            quality_score += 10
        elif feature_dim > 5:
            quality_score += 5
        
        return {
            'quality_score': quality_score,
            'improvement': improvement,
            'final_loss': final_loss,
            'training_samples': training_samples,
            'feature_dim': feature_dim
        }
    
    def _collect_and_sync_scalers(self, successful_symbols):
        """收集训练过程中的scalers并同步到算法"""
        # 从data_processor收集scalers
        self.scalers = {}
        for symbol in successful_symbols:
            if symbol in self.data_processor.scalers:
                self.scalers[symbol] = self.data_processor.scalers[symbol]
        
        # 同步到算法的data_processor
        self.algorithm.data_processor.scalers.update(self.scalers)
        
        # 记录日志
        self.algorithm.Debug(f"Collected {len(self.scalers)} scalers for symbols: {list(self.scalers.keys())}")
        self.algorithm.Debug(f"Algorithm data_processor now has {len(self.algorithm.data_processor.scalers)} scalers") 