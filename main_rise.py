"""主升浪无未来函数检测（detect_main_rise）。

定义：主升浪 = 中周期(reg120 对数回归)斜率显著向上 且 价格位于长周期(reg250)
      趋势线上方 且 段内回撤受控(不深跌) 的持续上行段。

设计要点（采纳反馈：大周期 reg250 斜率在拐点处滞后/噪声大，不可靠）：
  - 趋势方向与「加速」只用 reg120 斜率（中周期，灵敏准确）；
  - reg250 只用其「位置」(close 在 reg250 上方)，不碰其斜率 → 位置比斜率稳；
  - 段内回撤用价格高点回撤约束，深跌即退出主升浪状态，直接压误检；
  - 量能仅做软标签，不硬过滤（参考 golden_pit 教训：硬过滤信号 -84% 胜率不升）。

输出：每日 in_main_rise(bool 数组)、score(0~1)、segments[(s,e)]。
全部仅用第 i 天及之前数据，无未来函数。
"""

import numpy as np
from mean_reversion.signal_residual import compute_rolling_regression


def detect_main_rise(closes, highs=None, lows=None, opens=None, volumes=None,
                     reg250=None, reg120=None,
                     reg_win_long=250, reg_win_mid=120,
                     slope_annual_min=0.20, pos_gate=0.97,
                     max_dd=0.25, min_len=20, vol_confirm=1.0, vol_win=20):
    """主升浪检测（无未来函数）。

    Parameters
    ----------
    closes : np.ndarray           收盘价序列（旧→新）
    reg250 : np.ndarray|None      长周期回归线（位置用）；None 则内部算
    reg120 : np.ndarray|None      预留（未用于斜率，斜率内部由 closes 算）
    reg_win_long : int            长周期窗口，默认 250
    reg_win_mid  : int            中周期窗口，默认 120（方向/加速）
    slope_annual_min : float      中周期年化涨幅下限（主升浪门槛），默认 20%
    pos_gate : float              close 相对 reg250 的位置门控，默认 1.0（线上方）
    max_dd : float                段内最大回撤上限（超则退出主升），默认 15%
    min_len : int                 主升浪最短交易日数（过滤脉冲），默认 20

    Returns
    -------
    in_main_rise : np.ndarray(bool)  每日是否处于主升浪
    score : np.ndarray(float)        0~1 强度（斜率超额 0.7 + 回撤控制 0.3）
    segments : list[(s,e)]           主升浪段（已按 min_len 过滤）
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    if reg250 is None:
        reg250, _ = compute_rolling_regression(closes, reg_win_long)
    _, slopes120 = compute_rolling_regression(closes, reg_win_mid)
    # 对数斜率 a(每日) → 年化涨幅 ≈ exp(a*250)-1
    annual120 = np.exp(slopes120 * 250.0) - 1.0
    vol_base = None
    if volumes is not None and vol_confirm > 1.0:
        vol = np.asarray(volumes, dtype=float)
        vol_base = np.full(n, np.nan)
        for i in range(vol_win, n):
            vol_base[i] = vol[i - vol_win:i].mean()

    in_state = np.zeros(n, dtype=bool)
    score = np.zeros(n, dtype=float)
    segments = []
    state = False
    seg_start = 0
    peak = 0.0

    for i in range(n):
        ready = np.isfinite(reg250[i]) and np.isfinite(slopes120[i])
        if not ready:
            if state:
                state = False
                segments.append((seg_start, i - 1))
            continue

        up = (annual120[i] > slope_annual_min) and (closes[i] > reg250[i] * pos_gate)
        if vol_base is not None:
            up = up and np.isfinite(vol_base[i]) and vol[i] > vol_base[i] * vol_confirm

        if state:
            peak = max(peak, closes[i])
            dd = (peak - closes[i]) / peak if peak > 0 else 0.0
            if (dd > max_dd) or (not up):
                state = False
                segments.append((seg_start, i - 1))
            else:
                in_state[i] = True
                s_slope = min(1.0, max(0.0, annual120[i] / slope_annual_min))
                s_dd = max(0.0, 1.0 - dd / max(max_dd, 1e-9))
                score[i] = min(1.0, 0.7 * s_slope + 0.3 * s_dd)

        if not state:
            if up:
                state = True
                seg_start = i
                peak = closes[i]
                in_state[i] = True
                s_slope = min(1.0, max(0.0, annual120[i] / slope_annual_min))
                score[i] = min(1.0, 0.7 * s_slope + 0.3 * 1.0)

    if state:
        segments.append((seg_start, n - 1))

    segments = [(s, e) for (s, e) in segments if e - s + 1 >= min_len]
    in_state = np.zeros(n, dtype=bool)
    for s, e in segments:
        in_state[s:e + 1] = True
    return in_state, score, segments
