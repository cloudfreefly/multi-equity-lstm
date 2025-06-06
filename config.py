# region imports
from AlgorithmImports import *
# endregion
# 算法配置文件
class AlgorithmConfig:
    """算法主要配置参数"""
    # 时间设置
    START_DATE = (2021, 1, 1)
    END_DATE = (2025, 1, 31)
    INITIAL_CASH = 100000
    
    # 股票池配置
    SYMBOLS = [
        "SPY", "AVGO", "AAPL", "INTC", "NVDA", "AMZN", 
        "LLY", "MSFT", "GOOG", "META", "TSLA", "NFLX", "GLD"
    ]
    
    # 数据和训练配置
    LOOKBACK_DAYS = 252
    MIN_HISTORY_DAYS = 250
    MAX_HISTORY_DAYS = 500
    TRAINING_WINDOW = 126  # 6个月的交易日数据
    WARMUP_DAYS = 400
    
    # 多时间跨度预测配置
    PREDICTION_HORIZONS = [1, 5, 10]  # 预测时间跨度（天）
    HORIZON_WEIGHTS = {1: 0.5, 5: 0.3, 10: 0.2}  # 各时间跨度的权重
    
    # 模型配置
    MODEL_CONFIG = {
        'conv_filters': [16, 32, 64],  # 多尺度CNN滤波器
        'conv_kernels': [3, 5, 7],     # 多尺度CNN核大小
        'lstm_units': [64, 32],        # 多层LSTM单元数
        'attention_heads': 4,          # 注意力头数
        'dropout_rate': 0.2,           # Dropout率
        'batch_size': 16,
        'epochs': 20,
        'patience': 5                  # 早停耐心参数
    }
    
    # 投资组合优化配置
    PORTFOLIO_CONFIG = {
        'min_weight': 0.01,           # 最小权重（1%）
        'max_weight': 0.25,           # 最大权重（25%）
        'weight_threshold': 0.01,     # 权重筛选阈值
        'rebalance_tolerance': 0.005, # 调仓容忍度（0.5%）
        'cash_buffer': 0.02,          # 现金缓冲（2%）
        'sector_max_weight': 0.4,     # 单行业最大权重
        'concentration_limit': 0.6    # 集中度限制
    }
    
    # 风险管理配置（统一配置）
    RISK_CONFIG = {
        'enable_risk_management': True, # 是否启用风险管理
        'max_single_position_weight': 0.25, # 单个持仓最大权重
        'max_drawdown': 0.15,         # 最大回撤限制
        'var_confidence': 0.05,       # VaR置信水平
        'volatility_threshold': 0.3,   # 波动率阈值
        'correlation_threshold': 0.8,  # 相关性阈值
        'liquidity_min_volume': 1000000  # 最小日成交量
    }
    
    # 训练控制配置
    TRAINING_CONFIG = {
        'enable_retraining': True,     # 是否启用重新训练
        'retraining_frequency': 'monthly',  # 重新训练频率：'monthly', 'weekly', 'quarterly'
        'max_training_time': 480,      # 最大训练时间（秒）
        'memory_cleanup_interval': 5,  # 内存清理间隔
        'model_validation_ratio': 0.2,  # 验证集比例
        'early_stopping_delta': 0.001,  # 早停最小改善
        'mc_dropout_samples': 10,     # Monte Carlo Dropout采样数
        'fallback_strategy': 'equal_weights'  # 无模型时的备用策略：'equal_weights', 'momentum', 'skip'
    }
    
    # 预测配置
    PREDICTION_CONFIG = {
        'confidence_threshold': 0.6,  # 置信度阈值
        'trend_consistency_weight': 0.3, # 趋势一致性权重
        'uncertainty_penalty': 0.2,   # 不确定性惩罚
        'volatility_adjustment': True, # 是否进行波动率调整
        'regime_detection': True      # 是否启用市场状态检测
    }
    
    # 日志配置（引用LoggingConfig）
    LOGGING_CONFIG = {
        'enable_daily_reports': True,
        'log_portfolio_overview': True,
        'log_holdings_details': True,
        'log_performance_analysis': True,
        'log_volume_analysis': True,
        'log_prediction_analysis': True,
        'max_holdings_display': 10,
        'max_volume_display': 5
    }

class TechnicalConfig:
    """技术指标和信号配置"""
    # 技术指标参数
    TECHNICAL_INDICATORS = {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bollinger_period': 20,
        'bollinger_std': 2,
        'atr_period': 14,
        'volume_ma_period': 20
    }
    
    # 市场状态检测参数
    MARKET_REGIME = {
        'volatility_lookback': 60,
        'trend_lookback': 40,
        'momentum_lookback': 20,
        'correlation_lookback': 30
    }

class LoggingConfig:
    """日志和调试配置"""
    DEBUG_LEVEL = {
        'training': True,
        'prediction': True,
        'portfolio': True,
        'risk': True,
        'performance': True
    }
    
    LOG_INTERVALS = {
        'training_progress': 5,  # 每5个epoch记录一次
        'memory_usage': 10,      # 每10次操作记录内存使用
        'portfolio_status': 1,   # 每次调仓记录组合状态
        'daily_report_frequency': 1  # 每日报告频率
    }
    
    # 每日报告配置
    DAILY_REPORT_CONFIG = {
        'enable_portfolio_overview': True,
        'enable_holdings_details': True,
        'enable_performance_analysis': True,
        'enable_volume_analysis': True,
        'enable_prediction_analysis': True,
        'max_holdings_display': 10,
        'max_volume_display': 5
    } 