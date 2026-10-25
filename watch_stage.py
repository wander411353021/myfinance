# -*- coding: utf-8 -*-
"""观察池阶段第一版（2026-10-25）。

只做“阶段标签”，不产生买入信号、不评分。
严格因果：第 t 天状态只使用 <=t 的收盘/高低/量/换手/B/A。

阶段:
  S0 压制     : 价格在参考成本线 B 下方（默认至少 60 日）
  S1 小碎步上B: 首次收盘站上 B，且 B 的 20 日斜率仍 <=0
  S2 放量上A  : 连续 2 日收盘站上主成本 A，且量比>=1.5 或 5 日累计换手>=10%
  S3 B转正    : S2 之后，B 的 20 日斜率转正
  S4 缩量回踩 : S2 后 30 日内，缩量（量比<0.7）回踩 A 或 B 并收盘守住
  S5 再确认   : S4 之后收盘突破 S2 以来最高价，且量比>=1.2

注意:
  - “放量后必然回踩”不是未来已知信息；S4 必须等回踩实际发生才标记。
  - 本模块仅用于观察池/画图，不参与 v3/v3b/early 或买卖信号。
"""
import numpy as np

STAGE_NAMES = {
    0: 'S0 压制',
    1: 'S1 上B',
    2: 'S2 放量上A',
    3: 'S3 B转正',
    4: 'S4 缩量回踩',
    5: 'S5 再确认',
}

STAGE_COLORS = {
    0: '#90A4AE',
    1: '#F9A825',
    2: '#E53935',
    3: '#8E24AA',
    4: '#1E88E5',
    5: '#43A047',
}


def _rolling_vol_ratio(volume, short=5, long=20):
    v = np.asarray(volume, dtype=float)
    n = len(v)
    out = np.full(n, np.nan)
    for t in range(long - 1, n):
        s = v[t - short + 1:t + 1]
        l = v[t - long + 1:t + 1]
        if np.isfinite(s).all() and np.isfinite(l).all() and l.mean() > 0:
            out[t] = s.mean() / l.mean()
    return out


def _rolling_turn_sum(turnover, window=5):
    t = np.asarray(turnover, dtype=float)
    n = len(t)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        s = t[i - window + 1:i + 1]
        if np.isfinite(s).all():
            out[i] = s.sum()
    return out


def compute_watch_stages(closes, highs, lows, B, A, volume=None, turnover=None,
                         suppress_days=60, slope_window=20,
                         a_vol_ratio=1.5, a_turn_sum=0.10,
                         pullback_max_days=30, pullback_vol_ratio=0.7,
                         support_tol=0.02, invalid_days=2):
    """返回长度 n 的 stage 数组（0..5）。所有输入按时间升序。"""
    c = np.asarray(closes, dtype=float)
    h = np.asarray(highs, dtype=float)
    lo = np.asarray(lows, dtype=float)
    b = np.asarray(B, dtype=float)
    a = np.asarray(A, dtype=float)
    n = len(c)
    if not (len(h) == len(lo) == len(b) == len(a) == n):
        raise ValueError('closes/highs/lows/B/A 长度必须一致')

    vr = (_rolling_vol_ratio(volume) if volume is not None
          else np.full(n, np.nan))
    t5 = (_rolling_turn_sum(turnover, 5) if turnover is not None
          else np.full(n, np.nan))

    slope = np.full(n, np.nan)
    w = max(1, int(slope_window))
    for t in range(w, n):
        if np.isfinite(b[t]) and np.isfinite(b[t - w]) and b[t - w] > 0:
            slope[t] = b[t] / b[t - w] - 1.0

    stages = np.zeros(n, dtype=int)
    st = 0
    suppress = 0
    above_a_cnt = 0
    below_b_cnt = 0
    s2_idx = None
    s2_high = np.nan

    for t in range(n):
        if not (np.isfinite(c[t]) and np.isfinite(b[t]) and np.isfinite(a[t])):
            stages[t] = st
            continue

        close_above_b = c[t] > b[t]
        if close_above_b:
            # 首次上 B：之前必须已经在 B 下方压制足够久，且 B 仍向下/走平
            if (st == 0 and suppress >= suppress_days
                    and np.isfinite(slope[t]) and slope[t] <= 0):
                st = 1
            suppress = 0
            below_b_cnt = 0
        else:
            suppress += 1
            below_b_cnt += 1

        # 失效：进入阶段后连续跌回 B 下方
        if st > 0 and below_b_cnt >= invalid_days:
            st = 0
            s2_idx = None
            s2_high = np.nan
            above_a_cnt = 0

        if st >= 1:
            if c[t] > a[t]:
                above_a_cnt += 1
            else:
                above_a_cnt = 0

            # S2: 连续 2 日站上 A + 量能/换手确认
            if (st == 1 and above_a_cnt >= 2
                    and ((np.isfinite(vr[t]) and vr[t] >= a_vol_ratio)
                         or (np.isfinite(t5[t]) and t5[t] >= a_turn_sum))):
                st = 2
                s2_idx = t
                s2_high = h[t] if np.isfinite(h[t]) else np.nan

            # S3: B 的 20 日斜率转正
            if st == 2 and np.isfinite(slope[t]) and slope[t] > 0:
                st = 3

            # S4: S2 后缩量回踩 A/B，并收盘守住
            if (st in (2, 3) and s2_idx is not None
                    and 0 < t - s2_idx <= pullback_max_days
                    and np.isfinite(vr[t]) and vr[t] < pullback_vol_ratio):
                support = None
                if (np.isfinite(lo[t]) and np.isfinite(a[t])
                        and lo[t] <= a[t] * (1 + support_tol)
                        and c[t] >= a[t] * (1 - support_tol)):
                    support = a[t]
                elif (np.isfinite(lo[t]) and np.isfinite(b[t])
                        and lo[t] <= b[t] * (1 + support_tol)
                        and c[t] >= b[t] * (1 - support_tol)):
                    support = b[t]
                if support is not None:
                    st = 4

            # S5: 回踩后重新放量突破 S2 以来高点
            if (st == 4 and np.isfinite(s2_high)
                    and c[t] > s2_high
                    and np.isfinite(vr[t]) and vr[t] >= 1.2):
                st = 5

            # 放在 S5 判断之后：避免当天 high 提前并入，挡住当天的突破
            if st >= 2 and np.isfinite(h[t]):
                if not np.isfinite(s2_high):
                    s2_high = h[t]
                else:
                    s2_high = max(s2_high, h[t])

        stages[t] = st

    return stages
