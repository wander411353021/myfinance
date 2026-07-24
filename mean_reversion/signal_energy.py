"""信号B：能量衰竭法

核心思想：连续同向大幅波动消耗"势能"——下跌力度减弱 + 成交萎缩 → 卖盘衰竭 → 即将向上回归。

评分规则 (0-5分):
  - +1: 下行能量 > 阈值
  - +1: 下跌减速（后半段能量 < 前半段 × 0.5）
  - +1: 缩量确认（成交量 < 20日均量 × 0.8）
  - +1: 最近一日几乎不跌（跌幅 < 0.5%）
  - +1: 收盘接近最低（无下影线，确认卖压持续但推不动了）
"""

import numpy as np


def compute_energy_signal(
    closes: np.ndarray,
    volumes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    energy_window: int = 10,
    drop_threshold: float = 0.005,
    e_min: float = 0.03,
    volume_ratio: float = 0.8,
) -> dict:
    """计算能量衰竭信号。

    5 维评分：下行能量 + 减速 + 缩量 + 跌不动 + 收盘近低。

    Parameters
    ----------
    closes : np.ndarray  收盘价（从旧到新）
    volumes : np.ndarray 成交量（手，从旧到新）
    highs : np.ndarray   最高价
    lows : np.ndarray    最低价
    energy_window : int
        统计下行能量的窗口（交易日数），默认 10。
        调大(15-20) → 统计更多天数，需更长下跌才触发，信号更稳。
        调小(5-7)   → 更敏感，短期下跌即可触发。
    drop_threshold : float
        有效下跌阈值，默认 0.005 (0.5%)。
        调大(0.01) → 只统计跌幅 >1% 的阴线，忽略小跌。
        调小(0.002)→ 微跌也算在内，更敏感。
    e_min : float
        最小下行能量阈值，默认 0.03（约 3 个 1% 阴线）。
        调大(0.05+) → 需要更大下跌才确认，更严格。
        调小(0.01)  → 少量下跌即可触发。
    volume_ratio : float
        成交量萎缩比例阈值，默认 0.8（当日量 < 20日均量 × 0.8）。
        调大(1.0)  → 缩量条件放松（正常量也可）。
        调小(0.5)  → 需极度缩量才确认，更严格。

    Returns
    -------
    dict:
        energy_score : int   衰竭评分 (0-5)
        drop_energy  : float 下行能量
        decelerating : bool  是否减速
        volume_shrink : bool 是否缩量
        stalled      : bool  最近一日是否跌不动
        near_low     : bool  收盘是否接近最低
        level        : int   信号级别: -1卖出 0正常 +1买入 +2强买
    """
    min_required = max(energy_window + 1, 21)  # 至少 21 日（20日均量）
    if len(closes) < min_required:
        return {"energy_score": 0, "drop_energy": 0,
                "decelerating": False, "volume_shrink": False,
                "stalled": False, "near_low": False, "level": 0}

    # --- 下行能量 ---
    prices = closes[-(energy_window + 1):].astype(np.float64)
    daily_drops = -np.diff(prices) / prices[:-1]  # 正数 = 下跌
    daily_drops = np.maximum(daily_drops, 0)       # 只取下跌
    # 只算有效下跌
    effective_drops = np.where(daily_drops >= drop_threshold, daily_drops, 0)
    drop_energy = float(np.sum(effective_drops))

    # --- 下跌减速 ---
    half = energy_window // 2
    front_half = np.sum(effective_drops[:half])
    back_half = np.sum(effective_drops[half:])
    decelerating = (front_half > 0) and (back_half < front_half * 0.5)

    # --- 缩量确认 ---
    vol = volumes[-(energy_window + 1):].astype(np.float64)
    ma20_vol = np.mean(volumes[-21:-1]) if len(volumes) >= 21 else np.mean(vol)
    last_vol = vol[-1]
    volume_shrink = last_vol < ma20_vol * volume_ratio

    # --- 最近一日是否跌不动 ---
    last_drop = daily_drops[-1]
    stalled = last_drop < drop_threshold  # 最后一跌小于 0.5%

    # --- 收盘接近最低（无下影线或下影线很短） ---
    last_high = float(highs[-1])
    last_low = float(lows[-1])
    last_close = float(closes[-1])
    total_range = last_high - last_low
    if total_range > 1e-10:
        lower_shadow_ratio = (last_close - last_low) / total_range
        near_low = lower_shadow_ratio < 0.2  # 收盘在底部20%内
    else:
        near_low = False

    # --- 评分 ---
    score = 0
    if drop_energy >= e_min:
        score += 1
    if decelerating:
        score += 1
    if volume_shrink:
        score += 1
    if stalled:
        score += 1
    if near_low:
        score += 1

    # 信号级别
    if score >= 4:
        level = 2   # 强买入
    elif score >= 3:
        level = 1   # 买入
    elif score >= 2 and (decelerating and stalled):
        level = 1   # 虽有减速但能量不足
    else:
        level = 0   # 不操作

    return {
        "energy_score": score,
        "drop_energy": round(drop_energy, 4),
        "decelerating": bool(decelerating),
        "volume_shrink": bool(volume_shrink),
        "stalled": bool(stalled),
        "near_low": bool(near_low),
        "level": level,
    }
