"""信号A：滚动回归残差法

核心思想：股价围绕自身短期趋势线回归，先拟合趋势再计算偏离。

算法：
  1. 对最近 N 日做对数线性回归: log(price) = a * t + b
     （对数回归消除复利偏差，大牛股不会误判为"持续超买"）
  2. 残差 = log(实际价格) - log(回归预测价格)
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
    use_log: bool = True,
    robust_std: bool = True,
) -> dict:
    """计算滚动回归残差信号（支持对数回归）。

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
    use_log : bool
        是否使用对数回归。True=log(price)回归，消除复利偏差。
        默认 True。设为 False 则使用绝对价格线性回归。
    robust_std : bool
        是否使用稳健标准差（剔除当日点再计算 std）。

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
    if use_log:
        prices = np.log(np.maximum(prices, 1e-8))
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
    # 稳健 std：剔除当日点，避免当日极值把标准差撑大、稀释 z_residual
    if robust_std and len(residuals) > 3:
        residual_std = np.std(residuals[:-1], ddof=1)
    else:
        residual_std = np.std(residuals, ddof=1)

    # 当日残差
    current_residual = residuals[-1]
    z_residual = current_residual / residual_std if residual_std > 1e-10 else 0.0

    # 回归预测价（还原到价格空间用于展示）
    predicted_price = np.exp(predicted_all[-1]) if use_log else predicted_all[-1]

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
        "predicted": round(float(predicted_price), 4),
        "level": level,
    }


def compute_rolling_regression(closes: np.ndarray, window: int = 120, use_log: bool = True):
    """计算全量滚动回归预测值。

    对第 i 天，用 closes[i-window+1 : i+1] 做线性回归，返回第 i 天的预测值。
    前 window-1 天为 NaN。

    用于在 K 线图上叠加平滑回归线。

    Parameters
    ----------
    closes : np.ndarray  收盘价序列（从旧到新）
    window : int         回归窗口，默认 120。
    use_log : bool       是否使用对数回归，默认 True。

    Returns
    -------
    preds : np.ndarray   每日回归预测值（前 window-1 个为 NaN）
    slopes : np.ndarray  每日回归斜率（前 window-1 个为 NaN）
    """
    preds, slopes = _rolling_regression_full(closes, window, use_log=use_log)
    return preds, slopes


def _rolling_regression_full(closes: np.ndarray, window: int, use_log: bool = True):
    """对全量 closes 做滚动回归，返回每日预测值和斜率。

    use_log=True 时在 log 空间做回归，返回的 preds 为 exp(回归值)（价格空间）。
    """
    n = len(closes)
    preds = np.full(n, np.nan)
    slopes = np.full(n, np.nan)

    t = np.arange(window, dtype=np.float64)
    sum_t = np.sum(t)
    sum_tt = np.sum(t * t)
    denom = window * sum_tt - sum_t * sum_t

    for i in range(window - 1, n):
        y = closes[i - window + 1: i + 1].astype(np.float64)
        if use_log:
            y = np.log(np.maximum(y, 1e-8))
        sum_y = np.sum(y)
        sum_ty = np.sum(t * y)
        a = (window * sum_ty - sum_t * sum_y) / denom
        b = (sum_y - a * sum_t) / window
        pred = a * (window - 1) + b
        preds[i] = np.exp(pred) if use_log else pred
        slopes[i] = a

    return preds, slopes


def compute_reversion_debt(
    closes: np.ndarray,
    reg_window: int = 120,
    use_log: bool = True,
    overhang_forget: float = 0.98,
    overhang_min: float = 0.15,
    max_overhang: float = 5.0,
) -> dict:
    """计算上方透支量 (overhang) — 连续衰减版。

    相比旧版（二分负债期），核心改进：
      - overhang 每天按 overhang_forget 指数遗忘 → 时间自然消化透支
      - 没有负债期/非负债期的硬切换 → 连续平滑
      - 涨得越高 → overhang 增长越快 → 阈值自动越紧
      - 盘得越久 → overhang 衰减越多 → 阈值慢慢恢复

    Parameters
    ----------
    closes : np.ndarray  收盘价序列（从旧到新）
    reg_window : int     滚动回归窗口，默认 120。
    use_log : bool       是否使用对数回归，默认 True。
    overhang_forget : float
        每日 overhang 衰减系数，默认 0.98。
        调小(0.95) → 遗忘更快，透支更快消化。
        调大(0.99) → 遗忘更慢，透支影响更久。
        半衰期 ≈ ln(0.5)/ln(forget)：
          0.98 → ~34 天
          0.95 → ~14 天
          0.99 → ~69 天
    overhang_min : float
        最低生效阈值，低于此值认为无影响。默认 0.15 (15%)。
        调大(0.3) → 更宽松，小透支不处罚。
        调小(0.05) → 更严格。
    max_overhang : float
        overhang 上限，默认 5.0。
        防止单次大涨造成过度累积。
        调大(10.0) → 允许更大透支积累。
        调小(2.0)  → 上限更低，阈值恢复更快。

    Returns
    -------
    dict:
        overhang     : float  当前透支量（连续值，已指数衰减，有上限）
        consec_below : int    连续低于趋势线的天数
    """
    n = len(closes)
    if n < reg_window:
        return {"overhang": 0.0, "consec_below": 0}

    preds, _ = _rolling_regression_full(closes, reg_window, use_log=use_log)

    cum_overhang = 0.0
    consec_below = 0

    for i in range(1, n):
        if np.isnan(preds[i]):
            continue

        # 每日衰减（无论上方还是下方，时间都在消化透支）
        cum_overhang *= overhang_forget

        if closes[i] > preds[i]:
            # 上方：累积透支（有上限 cap）
            excess = (closes[i] - preds[i]) / preds[i]
            cum_overhang += max(excess, 0)
            cum_overhang = min(cum_overhang, max_overhang)  # cap
            consec_below = 0
        else:
            # 下方：透支自然衰减（forget 已在上面应用）
            consec_below += 1

    if cum_overhang < overhang_min:
        cum_overhang = 0.0

    return {
        "overhang": round(float(cum_overhang), 4),
        "consec_below": int(consec_below),
    }
