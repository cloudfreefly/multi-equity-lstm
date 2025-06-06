# Multi-Horizon LSTM Trading Algorithm

🚀 **基于深度学习的多时间跨度股票交易算法**

## 📋 项目简介

这是一个先进的量化交易算法，使用多时间跨度LSTM神经网络进行股票价格预测和投资组合优化。该算法集成了现代投资组合理论、风险管理和不确定性量化技术，能够在QuantConnect云平台上运行。

## ✨ 核心特性

### 🧠 深度学习模型
- **多时间跨度预测**: 同时预测1天、5天、10天的价格变化
- **CNN+LSTM+Attention架构**: 提取时空特征和长期依赖关系
- **Monte Carlo Dropout**: 不确定性量化，评估预测可信度
- **动态特征工程**: 自动计算技术指标和市场状态

### 📈 投资组合优化
- **现代投资组合理论**: 基于均值-方差优化
- **多目标优化**: 平衡收益、风险和流动性
- **动态权重调整**: 根据市场状态自适应调整
- **交易成本考虑**: 最小化换手率和交易费用

### 🛡️ 风险管理
- **实时风险监控**: 追踪回撤、波动率等关键指标
- **动态风险限制**: 根据市场状态调整风险敞口
- **流动性管理**: 确保充足的流动性缓冲
- **止损保护**: 多层次止损机制

### 📊 智能调仓
- **条件触发**: 基于时间、市场状态、风险水平
- **最小化冲击**: 考虑市场冲击成本的智能执行
- **流动性优化**: 优先选择高流动性股票
- **成本控制**: 最小化交易成本和税务影响

## 🏗️ 系统架构

```
Multi-Horizon LSTM Trading Algorithm
├── 🧠 模型训练层 (Model Training)
│   ├── 数据处理 (data_processing.py)
│   ├── 模型训练 (model_training.py)
│   └── 训练管理 (training_manager.py)
├── 🔮 预测引擎 (Prediction Engine)
│   ├── 多时间跨度预测 (prediction.py)
│   └── 不确定性量化
├── 💼 投资组合优化 (Portfolio Optimization)
│   ├── 现代投资组合理论 (portfolio_optimization.py)
│   └── 动态权重分配
├── 🛡️ 风险管理 (Risk Management)
│   ├── 实时风险监控 (risk_management.py)
│   └── 动态风险控制
├── ⚖️ 智能调仓 (Smart Rebalancing)
│   ├── 调仓决策 (rebalancing_manager.py)
│   └── 交易执行优化
└── 📈 主控制器 (main.py)
    ├── 算法协调
    └── 事件处理
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- QuantConnect CLI
- TensorFlow 2.x
- NumPy, Pandas, Scikit-learn

### 安装部署

#### 1. 克隆项目
```bash
git clone https://github.com/yourusername/multi-equity-lstm.git
cd multi-equity-lstm
```

#### 2. 安装依赖
```bash
pip install -r requirements.txt
```

#### 3. 配置QuantConnect
```bash
# 安装QuantConnect CLI
pip install lean

# 配置API密钥（可选，用于云端部署）
lean login
```

#### 4. 本地测试
```bash
# 本地回测
lean backtest

# 实时模拟
lean live deploy
```

#### 5. 云端部署
```bash
# 上传到QuantConnect云平台
lean cloud push

# 启动云端算法
lean cloud deploy
```

### 配置说明

#### config.py - 主要配置
```python
# 交易股票列表
SYMBOLS = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM']

# 预测时间跨度
PREDICTION_HORIZONS = [1, 5, 10]  # 天数

# 训练参数
TRAINING_CONFIG = {
    'lookback_period': 60,          # 历史数据窗口
    'max_training_time': 900,       # 最大训练时间(秒)
    'batch_size': 32,               # 批次大小
    'epochs': 50,                   # 训练轮数
    'learning_rate': 0.001,         # 学习率
    'dropout_rate': 0.2,            # Dropout率
    'mc_dropout_samples': 100       # Monte Carlo样本数
}

# 投资组合配置
PORTFOLIO_CONFIG = {
    'risk_aversion': 2.0,           # 风险厌恶系数
    'max_weight': 0.4,              # 单股票最大权重
    'min_weight': 0.05,             # 单股票最小权重
    'transaction_cost': 0.001       # 交易成本
}
```

#### lean.json - QuantConnect配置
```json
{
    "algorithm-type-name": "MultiHorizonTradingAlgorithm",
    "algorithm-language": "Python",
    "algorithm-location": "main.py"
}
```

## 📊 使用示例

### 基本回测
```python
from AlgorithmImports import *
from main import MultiHorizonTradingAlgorithm

# 在QuantConnect平台直接运行main.py
# 或使用CLI进行本地回测
```

### 自定义配置
```python
# 修改config.py中的参数
SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
PREDICTION_HORIZONS = [1, 3, 7, 14]

# 调整风险管理参数
RISK_MANAGEMENT_CONFIG = {
    'max_drawdown': 0.15,           # 最大回撤15%
    'stop_loss': 0.05,              # 止损5%
    'volatility_threshold': 0.25    # 波动率阈值
}
```

## 📈 性能监控

### 日志输出
算法会自动输出以下性能指标：

#### 每日性能日志
- 📊 **收益率**: 日收益率、累计收益率
- 📈 **风险指标**: 夏普比率、最大回撤、波动率
- 💰 **投资组合价值**: 总价值、现金头寸

#### 预测性能日志
- 🎯 **预测准确率**: 各时间跨度的预测成功率
- 🔮 **置信度分析**: 预测置信度分布
- 🌊 **市场状态**: 检测到的市场状态

#### 交易日志
- 🔄 **调仓记录**: 调仓前后的投资组合变化
- 💵 **交易汇总**: 买入、卖出、手续费统计
- ⚖️ **权重分布**: 各股票的权重分配

### 关键指标

| 指标类别 | 监控指标 | 目标值 |
|---------|----------|--------|
| 收益性能 | 年化收益率 | > 12% |
| 风险控制 | 最大回撤 | < 15% |
| 风险调整收益 | 夏普比率 | > 1.5 |
| 预测质量 | 预测准确率 | > 55% |
| 交易效率 | 年换手率 | < 200% |

## 🔧 技术细节

### 模型架构
```
输入层 (60天 × 特征数)
    ↓
CNN层 (提取局部模式)
    ↓  
LSTM层 (捕获时序依赖)
    ↓
Attention层 (聚焦重要信息)
    ↓
全连接层
    ↓
输出层 (1天、5天、10天预测)
```

### CNN+LSTM+Attention模型详细架构

#### 🎯 多尺度CNN特征提取
```python
# 多尺度卷积滤波器
conv_filters = [16, 32, 64]     # 滤波器数量
conv_kernels = [3, 5, 7]        # 多尺度核大小
```

| 层级 | 滤波器 | 核大小 | 作用 |
|------|--------|--------|------|
| **Conv1D-1** | 16个 | 3×1 | 短期局部模式 |
| **Conv1D-2** | 32个 | 5×1 | 中期趋势模式 |
| **Conv1D-3** | 64个 | 7×1 | 长期结构模式 |

#### 🔄 多层LSTM时序建模
```python
# 双层LSTM架构
lstm_units = [64, 32]           # LSTM单元数
```

| 层级 | 单元数 | 序列输出 | 作用 |
|------|--------|----------|------|
| **LSTM-1** | 64个 | 保持序列 | 提取长期依赖 |
| **LSTM-2** | 32个 | 最终输出 | 序列特征压缩 |

#### ⚡ Multi-Head Attention机制
```python
# 多头注意力配置
num_heads = 4                   # 注意力头数
attention_dim = 32              # 注意力维度
```

### 🎯 多时间跨度策略逻辑

#### 第一步：多时间跨度预测生成

**预测输出结构**：
```python
predictions[symbol] = {
    'predictions': {
        1: {'expected_return': 0.015, 'confidence': 0.85},   # 1天预测
        5: {'expected_return': 0.045, 'confidence': 0.78},   # 5天预测  
        10: {'expected_return': 0.075, 'confidence': 0.72}   # 10天预测
    },
    'market_regime': 'trending',
    'overall_confidence': 0.78
}
```

#### 第二步：时间跨度加权聚合

**权重配置**：
```python
HORIZON_WEIGHTS = {
    1: 0.5,    # 短期预测权重50% - 精准度优势
    5: 0.3,    # 中期预测权重30% - 趋势捕获  
    10: 0.2    # 长期预测权重20% - 方向指引
}
```

**加权收益计算**：
```python
weighted_return = (
    return_1d × 0.5 +
    return_5d × 0.3 + 
    return_10d × 0.2
)
```

#### 第三步：不确定性量化调整

**Monte Carlo Dropout置信度**：
```python
confidence_factor = 0.5 + 0.5 × overall_confidence
# 置信度范围：[0.5, 1.0]
```

**市场状态调整**：
```python
regime_factors = {
    'trending': 1.2,        # 趋势市场：增强信号
    'volatile': 0.8,        # 高波动：降低敞口
    'low_vol': 1.0,         # 低波动：正常权重
    'neutral': 1.0          # 中性市场：基准权重
}
```

#### 第四步：最终期望收益计算

```python
final_expected_return = (
    weighted_return × 
    confidence_factor × 
    regime_factor × 
    momentum_factor
)
```

#### 第五步：投资组合优化

**三种优化策略**：

1. **均值-方差优化** (默认)
   ```python
   # 最大化风险调整收益
   max: (μ'w - λ/2 × w'Σw)
   ```

2. **风险平价**
   ```python
   # 等风险贡献
   RC_i = w_i × (Σw)_i = 1/n × σ_p²
   ```

3. **最大分散化**
   ```python
   # 最大化分散化比率
   max: (w'σ) / √(w'Σw)
   ```

#### 第六步：约束条件应用

**权重约束**：
```python
constraints = {
    'min_weight': 0.01,         # 最小权重1%
    'max_weight': 0.25,         # 最大权重25%
    'sum_weights': 1.0,         # 权重和为1
    'no_short': True            # 禁止做空
}
```

**风险约束**：
```python
risk_constraints = {
    'max_portfolio_volatility': 0.20,    # 最大组合波动率20%
    'max_sector_exposure': 0.40,         # 最大行业敞口40%
    'min_diversification': 0.60          # 最小分散化度60%
}
```

### 📊 技术指标体系

#### 基础价格特征（4个）
| 指标 | 计算方法 | 作用 |
|------|----------|------|
| **原始价格** | Close Price | 基础价格信息 |
| **收益率** | (P_t - P_{t-1}) / P_{t-1} | 日收益率变化 |
| **对数收益率** | ln(P_t) - ln(P_{t-1}) | 正态分布特性 |
| **波动率** | 20日滚动标准差 | 风险测量 |

#### 趋势追踪指标（4个）
| 指标 | 参数 | 作用 |
|------|------|------|
| **SMA_20** | 20日简单移动平均 | 短期趋势 |
| **SMA_50** | 50日简单移动平均 | 中期趋势 |
| **EMA_12** | 12日指数移动平均 | 快速趋势 |
| **EMA_26** | 26日指数移动平均 | 慢速趋势 |

#### 动量指标（5个）
| 指标 | 参数 | 计算公式 | 作用 |
|------|------|----------|------|
| **RSI** | 14日 | 相对强弱指数 | 超买超卖 |
| **MACD** | (12,26,9) | 快慢线差值 | 趋势转换 |
| **MACD Signal** | 9日 | MACD平滑线 | 买卖信号 |
| **MACD Histogram** | MACD-Signal | 柱状图 | 动量变化 |
| **ROC** | 14日 | 变化率 | 价格动量 |

#### 波动率指标（4个）
| 指标 | 参数 | 作用 |
|------|------|------|
| **布林带上轨** | (20,2) | 压力位判断 |
| **布林带下轨** | (20,2) | 支撑位判断 |
| **布林带宽度** | (上轨-下轨)/中轨 | 波动率测量 |
| **ATR** | 14日平均真实波幅 | 实际波动幅度 |

#### 成交量指标（3个）
| 指标 | 计算方法 | 作用 |
|------|----------|------|
| **成交量** | 原始交易量 | 流动性指标 |
| **成交量SMA** | 20日平均成交量 | 成交量趋势 |
| **成交量比率** | 当日量/平均量 | 异常成交识别 |

#### 市场结构指标（4个）
| 指标 | 计算方法 | 作用 |
|------|----------|------|
| **High-Low价差** | (高-低)/收盘 | 日内波动 |
| **收盘相对位置** | (收盘-低)/(高-低) | 强弱势判断 |
| **价格通道位置** | 在20日通道中的位置 | 相对价格水平 |
| **涨跌幅排名** | 相对SPY的表现 | 相对强弱 |

### 🛡️ 备用策略机制

#### 备用策略触发条件

1. **模型训练失败** - 没有可交易股票
2. **预测生成失败** - LSTM无法生成预测  
3. **预测验证失败** - 置信度过低
4. **期望收益计算失败** - 数据质量问题
5. **协方差矩阵失败** - 历史数据不足
6. **优化求解失败** - 数值计算问题
7. **权重验证失败** - 约束条件冲突
8. **紧急风险触发** - 超出风险限制

#### 备用策略类型

**动量策略** (当前配置)：
```python
# 选择过去20日表现最好的50%股票
# 基于动量因子进行权重分配
momentum_scores = returns_20d.rank(ascending=False)
selected_symbols = momentum_scores.head(n//2)
```

**等权重策略**：
```python
# 所有股票等权重分配
equal_weights = 1.0 / len(symbols)
```

### 风险模型
- **VaR计算**: 历史模拟法和参数化方法
- **压力测试**: 模拟极端市场情况
- **相关性监控**: 实时跟踪股票间相关性

## 🚨 风险提示

⚠️ **重要声明**
- 本算法仅供学习和研究使用
- 量化交易存在市场风险，过往业绩不代表未来表现
- 实盘交易前请充分测试和验证
- 建议咨询专业的投资顾问

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发环境设置
```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest tests/

# 代码格式化
black src/
flake8 src/
```

### 提交规范
- Feature: 新功能开发
- Fix: Bug修复
- Docs: 文档更新
- Style: 代码格式调整
- Refactor: 代码重构

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 📞 联系方式

- **项目主页**: https://github.com/yourusername/multi-equity-lstm
- **问题反馈**: https://github.com/yourusername/multi-equity-lstm/issues
- **讨论区**: https://github.com/yourusername/multi-equity-lstm/discussions

## 🙏 致谢

感谢以下开源项目和社区：
- [QuantConnect](https://www.quantconnect.com/) - 量化交易平台
- [TensorFlow](https://tensorflow.org/) - 深度学习框架
- [Scikit-learn](https://scikit-learn.org/) - 机器学习库
- [Pandas](https://pandas.pydata.org/) - 数据分析库

---

⭐ **如果这个项目对您有帮助，请给个Star支持一下！** 