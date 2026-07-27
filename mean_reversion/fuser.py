"""信号融合模块（连续衰减版）

融合规则：
  1. 残差回归给出 z_residual（已使用对数回归，消除复利偏差）
  2. Overhang 连续衰减，平滑调整买入阈值：
       adjusted_weak_buy  = z_weak_buy  - overhang * overhang_penalty
       adjusted_strong_buy = z_strong_buy - overhang * overhang_penalty
  3. 能量衰竭评分作为辅助确认

负债期移除说明（从二分改为连续）：
  旧版：in_debt + debt_remaining + fake_bounce（硬切换）
  新版：overhang 每日指数衰减，阈值连续滑动（无硬切换）
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
    z_residual: float = 0.0         # 残差偏离度（经对数回归计算）
    energy_score: int = 0           # 能量衰竭评分 0-5
    reg_slope: float = 0.0          # 回归斜率（log空间）
    drop_energy: float = 0.0        # 下行能量
    volume_ratio: float = 0.0       # 量比
    overhang: float = 0.0           # 上方透支量（连续衰减，非二分）
    downtrend_guard: bool = False   # 是否触发下跌趋势防护
    adjusted_z_buy: float = -1.5    # 经 overhang 调整后的实际买入阈值
    stop_loss_pct: float = 0.0      # 建议止损比例
    position_pct: float = 0.0       # 建议仓位上限 (0~1)
    details: dict = field(default_factory=dict)


def fuse_signal(
    effective_level: int,
    energy_score: int,
    resid_level: int,
) -> tuple:
    """纯函数：由有效回归级别 / 能量评分推导 (signal, confidence)。

    Returns (signal_str, confidence_int)
    """
    if effective_level >= 3:
        if energy_score >= 4:
            return "buy", 5
        elif energy_score >= 3:
            return "buy", 4
        elif energy_score >= 1:
            return "neutral", 2
        else:
            return "neutral", 1
    elif effective_level >= 1:
        if energy_score >= 3:
            return "buy", 3
        elif energy_score >= 2:
            return "neutral", 2
        else:
            return "neutral", 1
    elif resid_level <= -3:
        return "sell", min(5, 3 + energy_score // 2)
    return "neutral", 0


def compute_signal(
    df: pd.DataFrame,
    code: str = "",
    reg_window: int = 120,
    energy_window: int = 10,
    **kwargs
) -> SignalResult:
    """对一只股票的日线 DataFrame 计算均值回归信号。

    默认使用对数回归 + 连续 overhang 衰减。

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
        能量衰竭统计窗口，默认 10。
        调大(15-20) → 更稳定，调小(5-7) → 更敏感。

    其他参数通过 **kwargs 传入：
      overhang_forget=0.99
        每日 overhang 衰减系数。
        调小(0.98) → 遗忘更快，透支更快消化。
        调大(0.995) → 遗忘更慢。
      overhang_penalty=0.5
        overhang 对买入阈值的惩罚斜率。
        调大(1.0) → 每个 unit overhang 将阈值收紧 1σ。
        调小(0.2) → 惩罚更轻。
      overhang_min=0.15
        overhang 最低生效阈值。
      max_below_bars=15
        下跌趋势防护：连续低于趋势线达到该天数 → 压制弱买点。
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

    # ---- 参数读取 ----
    overhang_forget = kwargs.get("overhang_forget", 0.98)
    overhang_penalty = kwargs.get("overhang_penalty", 0.3)
    overhang_min = kwargs.get("overhang_min", 0.15)
    max_below_bars = kwargs.get("max_below_bars", 15)

    # ---- 信号A：对数残差回归 (use_log=True) ----
    res = compute_residual_signal(closes, reg_window=reg_window, use_log=True)
    result.z_residual = res["z_residual"]
    result.reg_slope = res["a"]
    resid_level = res["level"]

    # ---- Overhang 连续衰减 ----
    debt = compute_reversion_debt(
        closes, reg_window=reg_window,
        use_log=True,
        overhang_forget=overhang_forget,
        overhang_min=overhang_min,
        max_overhang=kwargs.get("max_overhang", 5.0),
    )
    result.overhang = debt["overhang"]

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
        "not_new_low": ene["not_new_low"],
        "hammer_like": ene["hammer_like"],
    }

    # ---- 连续阈值调整 ----
    # overhang 越大 → adjusted_z_buy 越负 → 买入要求越严
    adjusted_weak = -1.5 - debt["overhang"] * overhang_penalty
    adjusted_strong = -2.0 - debt["overhang"] * overhang_penalty
    result.adjusted_z_buy = round(float(adjusted_weak), 4)

    if res["z_residual"] <= adjusted_strong:
        effective_level = 3
    elif res["z_residual"] <= adjusted_weak:
        effective_level = 1
    else:
        effective_level = 0

    # 下跌趋势防护
    downtrend_guard = (
        max_below_bars > 0
        and debt.get("consec_below", 0) >= max_below_bars
        and effective_level == 1
        and not (ene["hammer_like"] or ene["not_new_low"])
    )
    if downtrend_guard:
        effective_level = 0

    result.downtrend_guard = downtrend_guard

    # ---- 融合 ----
    result.signal, result.confidence = fuse_signal(
        effective_level, ene["energy_score"], resid_level
    )

    # 交易纪律建议
    result.stop_loss_pct = kwargs.get("stop_loss_pct", -0.08)
    result.position_pct = kwargs.get("position_pct", 0.3)

    return result
