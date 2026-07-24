"""信号A：滚动回归残差法

核心思想：股价围绕自身短期趋势线回归，先拟合趋势再计算偏离。

算法：
  1. 对最近 N 日做线性回归: price = a * t + b
  2. 残差 = 实际价格 - 回归预测价格
  3. 标准化残差 z_residual = 残差 / 残差标准差
  4. z_residual 越负 → 价格越低于趋势线 → 向上回归预期

参数调优：
  reg_window=120 (默认):
    - 大窗口 → 曲线平滑，适合判断中长期趋势
    - 调大(180-250) → 更平滑，反应更慢，适合大级别判断
    - 调小(40-60) → 更敏感，反应更快，适合短线回归
  z 阈值:
    - |z|>2.0: 强信号（约95%置信区间）
    - |z|>1.5: 弱信号（约85%置信区间）
    - 调大(2.5+) → 信号更少但更可靠
    - 调小(1.0) → 信号更多但噪声增加
"""

import numpy as np


def compute_residual_signal(
    closes: np.ndarray,
    reg_window: int = 120,
    z_strong_buy: float = -2.0,
    z_weak_buy: float = -1.5,
    z_weak_sell: float = 1.5,
    z_strong_sell: float = 2.0,
) -> dict:
    """计算滚动回归残差信号。

    Parameters
    ----------
    closes : np.ndarray
        收盘价序列（从旧到新），至少 reg_window 个元素。
    reg_window : int
        回归窗口（交易日数）。
        默认 120（约半年），曲线平滑。
        调大 → 更平滑、反应更慢，适合大级别趋势偏离。
        调小(60) → 更敏感、曲线略抖动，适合短线回归。
    z_strong_buy, z_weak_buy : float
        买入阈值（负值）。
        resudial 低于此值时触发买入。
        绝对值越大 → 要求偏离越极端 → 信号越少但越可靠。
    z_weak_sell, z_strong_sell : float
        卖出阈值（正值）。
        resudial 高于此值时触发卖出注意。
        绝对值越大 → 要求偏离越极端 → 信号越少。

    Returns
    -------
    dict:
        z_residual : float  标准化残差（当日）
        a          : float  回归斜率（趋势方向）
        b          : float  回归截距
        residual_std : float  残差标准差
        predicted  : float  回归预测价格（当日）
        level      : int    信号级别: -3强卖 -1弱卖 0正常 +1弱买 +3强买
    """
    if len(closes) < reg_window:
        return {"z_residual": 0, "a": 0, "b": 0,
                "residual_std": 0, "predicted": closes[-1] if len(closes) else 0,
                "level": 0}

    prices = closes[-reg_window:].astype(np.float64)
    t = np.arange(reg_window, dtype=np.float64)

    # 线性回归: price = a * t + b
    # 使用最小二乘公式
    n = reg_window
    sum_t = np.sum(t)
    sum_p = np.sum(prices)
    sum_tt = np.sum(t * t)
    sum_tp = np.sum(t * prices)

    a = (n * sum_tp - sum_t * sum_p) / (n * sum_tt - sum_t * sum_t)
    b = (sum_p - a * sum_t) / n

    # 残差
    predicted_all = a * t + b
    residuals = prices - predicted_all
    residual_std = np.std(residuals, ddof=1)  # 样本标准差

    # 当日残差
    current_residual = residuals[-1]
    z_residual = current_residual / residual_std if residual_std > 1e-10 else 0.0

    # 信号级别判定
    if z_residual <= z_strong_buy:
        level = 3   # 强向上回归
    elif z_residual <= z_weak_buy:
        level = 1   # 弱向上回归
    elif z_residual >= z_strong_sell:
        level = -3  # 强向下回归
    elif z_residual >= z_weak_sell:
        level = -1  # 弱向下回归
    else:
        level = 0   # 正常

    return {
        "z_residual": round(float(z_residual), 4),
        "a": round(float(a), 6),
        "b": round(float(b), 4),
        "residual_std": round(float(residual_std), 4),
        "predicted": round(float(predicted_all[-1]), 4),
        "level": level,
    }


def compute_rolling_regression(closes: np.ndarray, window: int = 120):
    """计算全量滚动回归预测值。

    对第 i 天，用 closes[i-window+1 : i+1] 做线性回归，返回第 i 天的预测值。
    前 window-1 天为 NaN。

    用于在 K 线图上叠加平滑回归线。

    Parameters
    ----------
    closes : np.ndarray  收盘价序列（从旧到新）
    window : int         回归窗口，默认 120。

    Returns
    -------
    preds : np.ndarray   每日回归预测值（前 window-1 个为 NaN）
    slopes : np.ndarray  每日回归斜率（前 window-1 个为 NaN）
    """
    preds, slopes = _rolling_regression_full(closes, window)
    return preds, slopes


def _rolling_regression_full(closes: np.ndarray, window: int):
    """对全量 closes 做滚动回归，返回每日预测值和斜率。"""
    n = len(closes)
    preds = np.full(n, np.nan)
    slopes = np.full(n, np.nan)

    t = np.arange(window, dtype=np.float64)
    sum_t = np.sum(t)
    sum_tt = np.sum(t * t)
    denom = window * sum_tt - sum_t * sum_t

    for i in range(window - 1, n):
        y = closes[i - window + 1: i + 1].astype(np.float64)
        sum_y = np.sum(y)
        sum_ty = np.sum(t * y)
        a = (window * sum_ty - sum_t * sum_y) / denom
        b = (sum_y - a * sum_t) / window
        preds[i] = a * (window - 1) + b
        slopes[i] = a

    return preds, slopes


def compute_reversion_debt(
    closes: np.ndarray,
    reg_window: int = 60,
    debt_multiplier: float = 50.0,
    min_debt_bars: int = 10,
    overhang_min: float = 0.15,
) -> dict:
    """计算上方透支量 (overhang) 与回归负债 (reversion debt)。

    涨得越高 → overhang 越大 → 负债期越长 → 买入阈值越紧。
    防止在大涨之后过早抄底。

    核心逻辑：
      - 价格在回归线上方时，累加每日超出比例 → overhang
      - 价格跌破回归线时，若 overhang > 阈值 → 进入负债期
      - 负债期内：买入阈值从 z=-1.5 收紧到 z=-3.0
      - 负债期内反弹到回归线上方 → fake_bounce = True
      - 负债期长度 = overhang × debt_multiplier (最小 min_debt_bars 天)
      - 负债期结束后自动重置

    Parameters
    ----------
    closes : np.ndarray  收盘价序列（从旧到新）
    reg_window : int     滚动回归窗口，默认 120。
    debt_multiplier : float
        每 unit overhang 对应负债天数，默认 50。
        调大(80-100) → 负债期更长，更保守。
        调小(20-30)  → 负债期更短，恢复更快。
    min_debt_bars : int
        最小负债天数，默认 10。
        防止 overhang 刚过阈值时负债期过短。
    overhang_min : float
        进入负债的 overhang 阈值，默认 0.15 (15%)。
        调大(0.3+) → 需要更大透支才惩罚，较宽松。
        调小(0.05) → 少量透支即惩罚，较严格。

    Returns
    -------
    dict:
        overhang       : float  当前累积透支量
        in_debt        : bool   是否在负债期
        debt_remaining : int    剩余负债天数（0 = 不在期或已到期）
        fake_bounce    : bool   今天是否假反弹（负债期内突破回归线）
    """
    n = len(closes)
    if n < reg_window:
        return {"overhang": 0.0, "in_debt": False,
                "debt_remaining": 0, "fake_bounce": False}

    preds, _ = _rolling_regression_full(closes, reg_window)

    cum_overhang = 0.0
    debt_count = 0      # 倒计时 (0=不在负债期)
    daily_fake = False

    for i in range(1, n):
        if np.isnan(preds[i]):
            continue

        if closes[i] > preds[i]:
            # ── 价格在回归线上方 ──
            if debt_count > 0:
                # 负债期内反弹 → 标记假突破
                daily_fake = True
            else:
                # 正常累积 overhang
                excess = (closes[i] - preds[i]) / preds[i]
                cum_overhang += max(excess, 0)
        else:
            # ── 价格在回归线下方 ──
            daily_fake = False

            if cum_overhang >= overhang_min and debt_count == 0:
                # 刚跌破 → 进入负债期
                debt_count = int(cum_overhang * debt_multiplier)
                debt_count = max(debt_count, min_debt_bars)

            if debt_count > 0:
                debt_count -= 1
                if debt_count == 0:
                    # 负债期结束 → 重置
                    cum_overhang = 0.0

    # 当前状态
    return {
        "overhang": round(float(cum_overhang), 4),
        "in_debt": debt_count > 0,
        "debt_remaining": debt_count,
        "fake_bounce": daily_fake,
    }
