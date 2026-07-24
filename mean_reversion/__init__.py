"""均值回归算法包

不依赖市面封装指标（布林带/RSI/MACD），从价格结构本身定义"偏离"与"能量衰竭"。

模块：
  signal_residual.py  信号A：滚动回归残差法
  signal_energy.py    信号B：能量衰竭法
  fuser.py            信号融合

用法：
  from mean_reversion.fuser import compute_signal
  result = compute_signal(df)
"""

from .signal_residual import compute_residual_signal, compute_reversion_debt
from .signal_energy import compute_energy_signal
from .fuser import compute_signal, SignalResult

__all__ = ["compute_residual_signal", "compute_reversion_debt",
           "compute_energy_signal", "compute_signal", "SignalResult"]
