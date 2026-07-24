"""信号融合模块

基础融合规则（正常状态）：

                  能量衰竭强 (>=4)    能量衰竭中 (3)    能量衰竭弱 (<3)
残差强回归 (z<=-2)     ★★★ 强烈买入     ★★☆ 买入        ★☆☆ 关注
残差弱回归 (-2<z<=-1.5)  ★★☆ 买入        ★★☆ 买入        ☆☆☆ 不操作
残差正常 (z>-1.5)     ☆☆☆ 不操作       ☆☆☆ 不操作       ☆☆☆ 不操作

负债期调整（上方透支量 >= overhang_min）：
  - 买入阈值从 z <= -1.5 收紧为 z <= -3.0
  - z in (-3.0, -1.5] → 被负债压制，不触发买入
  - 负债期内反弹到回归线上方 → fake_bounce = True
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .signal_residual import compute_residual_signal, compute_reversion_debt
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
    overhang: float = 0.0           # 上方透支量
    in_debt: bool = False           # 是否在负债期
    debt_remaining: int = 0         # 剩余负债天数
    fake_bounce: bool = False       # 今天是否假反弹
    details: dict = field(default_factory=dict)


def compute_signal(
    df: pd.DataFrame,
    code: str = "",
    reg_window: int = 120,
    energy_window: int = 10,
    **kwargs
) -> SignalResult:
    """对一只股票的日线 DataFrame 计算均值回归信号。

    Parameters
    ----------
    df : pd.DataFrame
        必须含 columns: close, high, low, volume
    code : str
        股票代码，仅用于结果标识。
    reg_window : int
        滚动回归窗口（交易日数），默认 120。
        调大(180-250) → 曲线更平滑，判断大级别趋势偏离。
        调小(40-60)  → 更敏感，适合短线回归。
    energy_window : int
        能量衰竭统计窗口（交易日数），默认 10。
        调大(15-20) → 统计更多天数，能量衰竭判断更慢但更稳。
        调小(5-7)   → 反应更快，信号更早但噪声更多。

    其他参数通过 **kwargs 传入：
      overhang_min=0.15
        进入负债期的 overhang 阈值。
        调大(0.3+) → 更不容易进入负债期（放松惩罚）。
        调小(0.05) → 更容易进入负债期（更严格）。
      debt_z_buy=-3.0
        负债期内的买入 z 阈值。
        调大(-2.5) → 负债期买入条件放松。
        调小(-3.5) → 负债期买入条件更严。

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

    # ---- 信号A：残差回归 ----
    res = compute_residual_signal(closes, reg_window=reg_window)
    result.z_residual = res["z_residual"]
    result.reg_slope = res["a"]
    resid_level = res["level"]

    # ---- 上方透支量 / 负债期 ----
    debt = compute_reversion_debt(closes, reg_window=reg_window)
    result.overhang = debt["overhang"]
    result.in_debt = debt["in_debt"]
    result.debt_remaining = debt["debt_remaining"]
    result.fake_bounce = debt["fake_bounce"]

    # ---- 信号B：能量衰竭 ----
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

    # ---- 融合（考虑负债期） ----
    energy_score = ene["energy_score"]
    overhang_min = kwargs.get("overhang_min", 0.15)

    # 负债期内：调整有效买入级别
    if debt["in_debt"] and res["z_residual"] > -3.0:
        # 虽然有偏离 (z <= -1.5) 但不到 -3.0 → 负债压制，不触发
        effective_level = 0
    elif debt["in_debt"] and res["z_residual"] <= -3.0:
        # 负债内但偏离极其严重 (z <= -3.0) → 仍算强回归
        effective_level = 3
    else:
        effective_level = resid_level

    # 从融合矩阵推导 confidence
    if effective_level >= 3 and energy_score >= 4:
        result.signal = "buy"
        result.confidence = 5
    elif effective_level >= 3 and energy_score >= 3:
        result.signal = "buy"
        result.confidence = 3
    elif effective_level >= 3 and energy_score >= 2:
        result.signal = "buy"
        result.confidence = 2
    elif effective_level >= 1 and energy_score >= 3:
        result.signal = "buy"
        result.confidence = 3
    elif effective_level >= 1 and energy_score >= 2:
        result.signal = "buy"
        result.confidence = 2
    elif effective_level >= 1 and energy_score >= 4:
        result.signal = "buy"
        result.confidence = 3
    elif resid_level <= -3:
        result.signal = "sell"
        result.confidence = min(5, abs(resid_level) + energy_score // 2)
    else:
        result.signal = "neutral"
        result.confidence = 0

    return result
