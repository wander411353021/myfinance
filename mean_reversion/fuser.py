"""信号融合模块

融合规则：

                  能量衰竭强 (≥4)    能量衰竭中 (3)    能量衰竭弱 (<3)
残差强回归 (z≤-2)     ★★★ 强烈买入     ★★☆ 买入        ★☆☆ 关注
残差弱回归 (-2<z≤-1.5)  ★★☆ 买入        ★★☆ 买入        ☆☆☆ 不操作
残差正常 (z>-1.5)     ☆☆☆ 不操作       ☆☆☆ 不操作       ☆☆☆ 不操作
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from .signal_residual import compute_residual_signal
from .signal_energy import compute_energy_signal


@dataclass
class SignalResult:
    """最终信号结果。"""
    code: str = ""
    date: str = ""
    signal: str = "neutral"         # "buy" | "neutral" | "sell"
    confidence: int = 0             # 1-5: 1弱 2关注 3买入 4强买入 5强烈买入
    z_residual: float = 0.0         # 残差偏离度
    energy_score: int = 0           # 能量衰竭评分 0-5
    reg_slope: float = 0.0          # 回归斜率
    drop_energy: float = 0.0        # 下行能量
    volume_ratio: float = 0.0       # 量比
    details: dict = field(default_factory=dict)


def compute_signal(
    df: pd.DataFrame,
    code: str = "",
    reg_window: int = 60,
    energy_window: int = 10,
    **kwargs
) -> SignalResult:
    """对一只股票的日线 DataFrame 计算均值回归信号。

    Parameters
    ----------
    df : pd.DataFrame
        必须含 columns: close, high, low, volume
        （推荐从 tdx_quant.get_daily_kline_from_tdx 获得）
    code : str
        股票代码，仅用于结果标识。
    reg_window : int
        回归窗口，默认 60。
    energy_window : int
        能量窗口，默认 10。

    Returns
    -------
    SignalResult
    """
    result = SignalResult(code=code)

    if df is None or len(df) < max(reg_window, 21):
        return result

    if "date" in df.columns and len(df) > 0:
        if hasattr(df["date"].iloc[-1], "strftime"):
            result.date = df["date"].iloc[-1].strftime("%Y-%m-%d")
        else:
            result.date = str(df["date"].iloc[-1])

    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.float64)

    # 信号A：残差回归
    res = compute_residual_signal(closes, reg_window=reg_window)
    result.z_residual = res["z_residual"]
    result.reg_slope = res["a"]
    resid_level = res["level"]

    # 信号B：能量衰竭
    ene = compute_energy_signal(closes, volumes, highs, lows,
                                energy_window=energy_window)
    result.energy_score = ene["energy_score"]
    result.drop_energy = ene["drop_energy"]

    # 量比
    if len(volumes) >= 21:
        ma20_vol = np.mean(volumes[-21:-1])
        result.volume_ratio = round(float(volumes[-1] / ma20_vol), 4) if ma20_vol > 0 else 0

    # 详情
    result.details = {
        "reg_a": res["a"],
        "reg_b": res["b"],
        "residual_std": res["residual_std"],
        "predicted": res["predicted"],
        "decelerating": ene["decelerating"],
        "volume_shrink": ene["volume_shrink"],
        "stalled": ene["stalled"],
        "near_low": ene["near_low"],
    }

    # ---- 融合 ----
    energy_score = ene["energy_score"]

    # 从融合矩阵推导 confidence
    if resid_level >= 3 and energy_score >= 4:
        # 残差强回归 + 能量强衰竭 → 强烈买入
        result.signal = "buy"
        result.confidence = 5
    elif resid_level >= 3 and energy_score >= 3:
        result.signal = "buy"
        result.confidence = 3
    elif resid_level >= 3 and energy_score >= 2:
        result.signal = "buy"
        result.confidence = 2
    elif resid_level >= 1 and energy_score >= 3:
        result.signal = "buy"
        result.confidence = 3
    elif resid_level >= 1 and energy_score >= 2:
        result.signal = "buy"
        result.confidence = 2
    elif resid_level >= 1 and energy_score >= 4:
        result.signal = "buy"
        result.confidence = 3
    elif resid_level <= -3:
        # 强向下回归
        result.signal = "sell"
        result.confidence = min(5, abs(resid_level) + energy_score // 2)
    else:
        result.signal = "neutral"
        result.confidence = 0

    return result
