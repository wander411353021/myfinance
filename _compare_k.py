# -*- coding: utf-8 -*-
"""k 标定对比: k=0.05(reasonix) vs k=0.025(真实换手率标定), 3板块指数级分档验证"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from chip_panel import cost_series
from index_v10_plot import fetch_index

BLOCKS = [('人工智能', '880948'), ('芯片', '880952'), ('存储芯片', '880973')]
END = '20261009'

def depth_series(idx_df, k):
    V = idx_df['volume'].values.astype(float)
    C = idx_df['close'].values.astype(float)
    H = idx_df['high'].values.astype(float)
    L = idx_df['low'].values.astype(float)
    ma20 = np.convolve(V, np.ones(20)/20, mode='valid')
    ma20 = np.concatenate([np.full(19, np.nan), ma20])
    ratio = np.where(ma20 > 0, V/ma20, 1.0)
    turn = (k * ratio).clip(0.005, 0.30)
    turn[:20] = k
    c35 = cost_series(H, L, V, turn, 0.35)
    return C / c35 - 1  # 偏离COST35

def fwd(idx_df, h):
    C = idx_df['close'].values.astype(float)
    out = np.full(len(C), np.nan)
    for i in range(len(C) - h):
        out[i] = C[i+h] / C[i] - 1
    return out

def bucket(s, f20, f60, buckets):
    rows = []
    for lo, hi, lab in buckets:
        m = (s >= lo) & (s < hi)
        if m.sum() == 0:
            rows.append((lab, 0, np.nan, np.nan, np.nan, np.nan)); continue
        rows.append((lab, int(m.sum()), f20[m].mean(), (f20[m] > 0).mean(), f60[m].mean(), (f60[m] > 0).mean()))
    return rows

BUCKETS = [(-1.0, -0.08, '深破<-8%'), (-0.08, -0.03, '中度'), (-0.03, 0.0, '轻度'),
           (0.0, 0.08, '微高'), (0.08, 2.0, '高位')]

for name, ic in BLOCKS:
    idx = fetch_index('sh%s' % ic, END, datalen=2000, anchor=True)
    f20, f60 = fwd(idx, 20), fwd(idx, 60)
    print('\n== %s ==' % name)
    for k in (0.05, 0.025):
        s = depth_series(idx, k)
        print('  k=%.3f:' % k)
        for lab, n, m20, w20, m60, w60 in bucket(s, f20, f60, BUCKETS):
            print('    %-8s n=%4d  20日%+6.2f%%/%4.1f%%  60日%+6.2f%%/%4.1f%%' % (lab, n, m20*100, w20*100, m60*100, w60*100))
