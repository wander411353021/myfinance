# -*- coding: utf-8 -*-
"""回踩坑探索:三种回踩形态识别 + 20/60日持有回测对比。

A. reg 回踩:上涨趋势(reg250 斜率>0)中 close 回踩 reg250 附近(浅坑,区别于黄金坑 z<-1.5)
B. MA 回踩:多头趋势中 close 回踩 20/60 日均线企稳
C. 成本区回踩:回踩前期放量堆成本区上轨

买入 = 企稳确认日;收益 = 20/60 日持有。
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression

def reg_slope_up(reg, i, look=20):
    """reg 在 i 日是否上行(20日前以来)"""
    if i < look:
        return False
    return np.isfinite(reg[i]) and np.isfinite(reg[i - look]) and reg[i] > reg[i - look]

def detect_pullback_reg(closes, reg250, touch_lo=0.95, touch_hi=1.03, bounce=0.02, max_z=0.0, min_days=3):
    """A. reg 回踩坑:趋势向上中回踩 reg250 附近,企稳(反弹 bounce)后确认。

    返回 [(回踩低点idx, 确认买入idx)]。
    无未来函数:只用 i 及之前。
    """
    c = np.asarray(closes, dtype=float)
    n = len(c)
    out = []
    i = 40
    while i < n - 10:
        if not reg_slope_up(reg250, i):
            i += 1
            continue
        r = reg250[i]
        if not np.isfinite(r) or r <= 0:
            i += 1
            continue
        # 触及 reg 附近:close 最低下探到 [reg*lo, reg*hi]
        ratio = c[i] / r
        # 低点检测:近 min_days 天最低
        lo = np.min(c[max(0, i - min_days + 1):i + 1])
        if touch_lo <= lo / r <= touch_hi and ratio >= 1.0:
            # 企稳:后续 5 天内反弹 >= bounce 或站上 reg
            for j in range(i + 1, min(i + 6, n - 1)):
                if c[j] / lo - 1 >= bounce or c[j] >= r:
                    out.append((i, j))
                    i = j + 1
                    break
            else:
                i += 1
            continue
        i += 1
    return out

def detect_pullback_ma(closes, ma, touch=0.03, bounce=0.02):
    """B. MA 回踩:多头(close>ma 且 ma 上行)中回踩 MA(±touch),重新站上确认。

    返回 [(回踩低点idx, 确认idx)]。
    """
    c = np.asarray(closes, dtype=float)
    m = np.asarray(ma, dtype=float)
    n = len(c)
    out = []
    i = 5
    while i < n - 10:
        if i < 3 or not np.isfinite(m[i]) or m[i] <= 0:
            i += 1
            continue
        ma_up = m[i] > m[i - 3]
        if not ma_up or c[i] <= m[i]:
            i += 1
            continue
        lo = np.min(c[max(0, i - 2):i + 1])
        if abs(lo / m[i] - 1) <= touch:
            for j in range(i + 1, min(i + 6, n - 1)):
                if c[j] >= m[j] and c[j] / lo - 1 >= bounce:
                    out.append((i, j))
                    i = j + 1
                    break
            else:
                i += 1
            continue
        i += 1
    return out

def detect_pullback_cost(closes, volumes, cost_hi=1.05, bounce=0.02):
    """C. 成本区回踩:回踩前期放量堆成本区上轨(堆内均价×cost_hi)。

    返回 [(回踩低点idx, 确认idx)]。成本区 = 最近的高位放量堆(HIGH)的均价区。
    """
    c = np.asarray(closes, dtype=float)
    n = len(c)
    vcl = pr.detect_volume_clusters(closes, volumes)
    out = []
    i = 30
    while i < n - 10:
        # 找 i 之前最近的一个 HIGH 堆(至少 10 天前)
        zone = None
        for s, e, kd, dr, pk, vr in vcl:
            if kd == 'HIGH' and e < i - 10 and s > i - 300:
                zone = (s, e)
        if zone is None:
            i += 1
            continue
        s0, e0 = zone
        cost = np.mean(c[s0:e0 + 1])  # 堆内均价 = 成本区
        hi = cost * cost_hi
        lo = np.min(c[max(0, i - 2):i + 1])
        if lo <= hi * 1.03 and lo >= cost * 0.95:
            for j in range(i + 1, min(i + 6, n - 1)):
                if c[j] / lo - 1 >= bounce:
                    out.append((i, j))
                    i = j + 1
                    break
            else:
                i += 1
            continue
        i += 1
    return out

def run_pool(pool_file='stock_pool_300.txt', end='20260827'):
    pool = []
    with open(pool_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                pool.append(line.split(',')[0].strip())
    res = {'reg': [], 'ma20': [], 'ma60': [], 'cost': []}
    for k, code in enumerate(pool):
        fcode = code if code[:2] in ('sh', 'sz') else ('sh' if code[0] in '69' else 'sz') + code
        df = pr._load_df(fcode, end)
        if df is None or len(df) < 400:
            continue
        c = df['close'].values.astype(float)
        v = df['volume'].values.astype(float)
        n = len(c)
        reg250, _ = compute_rolling_regression(c, window=250, use_log=True)
        ma20 = pd.Series(c).rolling(20).mean().values
        ma60 = pd.Series(c).rolling(60).mean().values
        for name, hits in [('reg', detect_pullback_reg(c, reg250)),
                           ('ma20', detect_pullback_ma(c, ma20)),
                           ('ma60', detect_pullback_ma(c, ma60)),
                           ('cost', detect_pullback_cost(c, v))]:
            for lo_i, buy_i in hits:
                if buy_i + 60 >= n:
                    continue
                res[name].append([c[buy_i + 20] / c[buy_i] - 1, c[buy_i + 60] / c[buy_i] - 1])
        if (k + 1) % 100 == 0:
            print(f'  进度 {k+1}', flush=True)
    print(f'\n股票池 {pool_file}, 股票 {len(pool)} 只')
    for name, rows in res.items():
        R = np.array(rows)
        if len(R) < 15:
            print(f'  {name:<6}: n={len(R):>4} (样本少)')
            continue
        print(f'  {name:<6}: n={len(R):>4}  20日胜率={np.mean(R[:,0]>=0):5.1%} 均值={np.mean(R[:,0]):+6.1%} | '
              f'60日胜率={np.mean(R[:,1]>=0):5.1%} 均值={np.mean(R[:,1]):+6.1%}')

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='stock_pool_300.txt')
    args = ap.parse_args()
    run_pool(args.pool)
