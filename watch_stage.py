# -*- coding: utf-8 -*-
"""观察池阶段第一版（2026-10-25）+ S6 过热出池（2026-10-25 polo4111 补齐）。

只做“阶段标签”，不产生买入信号、不评分。
严格因果：第 t 天状态只使用 <=t 的收盘/高低/量/换手/B/A。

阶段:
  S0 压制     : 价格在参考成本线 B 下方（默认至少 60 日）
  S1 小碎步上B: 首次收盘站上 B，且 B 的 20 日斜率仍 <=0
  S2 放量上A  : 连续 2 日收盘站上主成本 A，且量比>=1.5 或 5 日累计换手>=10%
  S3 B转正    : S2 之后，B 的 20 日斜率转正
  S4 缩量回踩 : S3 后 30 日内，缩量（量比<0.7）回踩 A 或 B 并收盘守住
  S5 再确认   : S4 之后收盘突破 S2 以来最高价，且量比>=1.2
  S6 过热出池 : S3/S4/S5 阶段 close 偏离主成本 A >= overheat_dist(默认40%) → 出池,
                不再产生观察信号(涨幅过大不敢买); 回落跌破参考线 B 后回 S0 重新观察

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
    6: 'S6 过热出池',
}

STAGE_COLORS = {
    0: '#90A4AE',
    1: '#F9A825',
    2: '#E53935',
    3: '#8E24AA',
    4: '#1E88E5',
    5: '#43A047',
    6: '#FF7043',
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
                         support_tol=0.02, invalid_days=2,
                         allow_s1_rearm=True, overheat_dist=0.40, a_max_dev=0.25):
    """返回长度 n 的 stage 数组（0..6）。所有输入按时间升序。

    a_max_dev: S2 触发时 close 偏离主成本 A 的上限(默认25%) —
               S2 语义是"低位放量站上主成本", 高位横盘(close 已远超A)放量不算启动。
    """
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
    above_b_cnt = 0
    s1_rearm = False
    above_a_cnt = 0
    below_b_cnt = 0
    s2_idx = None
    s3_idx = None
    s2_high = np.nan

    for t in range(n):
        if not (np.isfinite(c[t]) and np.isfinite(b[t]) and np.isfinite(a[t])):
            stages[t] = st
            continue

        close_above_b = c[t] > b[t]
        if close_above_b:
            above_b_cnt += 1
            # 首次上 B：此前压制足够久；或 S1 失败后重新站上 B 连续 2 天
            if st == 0 and np.isfinite(slope[t]) and slope[t] <= 0:
                if (suppress >= suppress_days
                        or (allow_s1_rearm and s1_rearm and above_b_cnt >= 2)):
                    st = 1
                    s1_rearm = False
            suppress = 0
            below_b_cnt = 0
        else:
            above_b_cnt = 0
            suppress += 1
            below_b_cnt += 1

        # 失效：进入阶段后连续跌回 B 下方；仅 S1 失败后允许重武装
        if st > 0 and below_b_cnt >= invalid_days:
            _failed_stage = st
            st = 0
            s1_rearm = (_failed_stage == 1)
            above_b_cnt = 0
            s2_idx = None
            s3_idx = None
            s2_high = np.nan
            above_a_cnt = 0

        if st >= 1:
            if c[t] > a[t]:
                above_a_cnt += 1
            else:
                above_a_cnt = 0

            # S2: 连续 2 日站上 A + 量能/换手确认, 且站上时偏离 A 不超上限(低位启动, 高位横盘不算)
            if (st == 1 and above_a_cnt >= 2
                    and ((np.isfinite(vr[t]) and vr[t] >= a_vol_ratio)
                         or (np.isfinite(t5[t]) and t5[t] >= a_turn_sum))
                    and np.isfinite(a[t]) and a[t] > 0
                    and c[t] / a[t] - 1.0 <= a_max_dev):
                st = 2
                s2_idx = t
                s3_idx = None
                s2_high = h[t] if np.isfinite(h[t]) else np.nan

            # S3: B 的 20 日斜率转正
            if (st == 2 and s2_idx is not None and t > s2_idx
                    and np.isfinite(slope[t]) and slope[t] > 0):
                st = 3
                s3_idx = t

            # S4: 必须在 S3（B 转正）之后缩量回踩 A/B
            if (st == 3 and s3_idx is not None
                    and 0 < t - s3_idx <= pullback_max_days
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

            # S6: 过热出池 — S3/S4/S5 阶段 close 偏离主成本A >= overheat_dist
            #     (涨幅过大不敢买, 出池后不再产生观察信号; 回落跌破B由失效判定回S0)
            if (st in (3, 4, 5) and np.isfinite(a[t]) and a[t] > 0
                    and c[t] / a[t] - 1.0 >= overheat_dist):
                st = 6

            # 只在 S2/S3（回踩前）更新；S4 后冻结，S5 用回踩前高点判断
            if st in (2, 3) and np.isfinite(h[t]):
                if not np.isfinite(s2_high):
                    s2_high = h[t]
                else:
                    s2_high = max(s2_high, h[t])

        stages[t] = st

    return stages
