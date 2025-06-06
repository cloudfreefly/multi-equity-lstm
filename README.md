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

### 特征工程
- **价格特征**: 收益率、对数收益率
- **技术指标**: SMA、EMA、RSI、MACD、布林带
- **波动率指标**: 滚动标准差、ATR
- **市场状态**: 趋势强度、动量指标

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