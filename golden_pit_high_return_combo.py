# -*- coding: utf-8 -*-
"""高收益坑分层验证: 低位深坑(距250高点)+短坑+前几次出坑 组合的收益分布。"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression

def main():
    pool = [l.split(',')[0].strip() for l in open('stock_pool_1000.txt', encoding='utf-8') if l.strip()]
    rows = []
    for k, sym in enumerate(pool):
        f = os.path.join('.cache_kline', f'{sym}.npy')
        if not os.path.exists(f): continue
        d = np.load(f, allow_pickle=True).item()
        c = d['close'].astype(float)
        nn = len(c)
        reg, _ = compute_rolling_regression(c, window=250, use_log=True)
        pits = [p for p in pr.detect_golden_pit_v2(c, reg) if p[2] is not None]
        pits.sort(key=lambda p: p[2])
        for idx, (s, b, lch) in enumerate(pits):
            if lch + 20 >= nn: continue
            plen = b - s + 1
            hi250 = np.max(c[max(0, b-250):b+1])
            dd = c[b] / hi250 - 1
            cnt = sum(1 for ps, pb, pl in pits[:idx] if lch - pl <= 90) + 1
            r20 = c[lch+20]/c[lch]-1
            rows.append([dd, plen, cnt, r20])
        if (k+1) % 500 == 0: print(f'  进度 {k+1}', flush=True)
    R = np.array(rows)
    def show(label, m):
        M = R[m]
        if len(M) < 15:
            print(f'  {label:<42} n={len(M):>4} 样本少'); return
        hi = np.mean(M[:, 3] > 0.15)
        print(f'  {label:<42} n={len(M):>5} 胜率={np.mean(M[:,3]>=0):5.1%} 均值={np.mean(M[:,3]):+6.1%} 中位={np.median(M[:,3]):+6.1%} 高收益(>15%)={hi:5.1%}')
    print(f'\n=== 高收益坑分层 (1000池 {len(R)} 坑, 20日) ===')
    show('基线(全部)', np.ones(len(R), bool))
    show('坑底距250高点 < -30%(低位)', R[:, 0] < -0.30)
    show('坑底距250高点 < -40%', R[:, 0] < -0.40)
    show('坑底距250高点 < -30% + 坑长<10', (R[:, 0] < -0.30) & (R[:, 1] < 10))
    show('坑底距250高点 < -30% + 坑长<10 + 第1-2次', (R[:, 0] < -0.30) & (R[:, 1] < 10) & (R[:, 2] <= 2))
    show('坑底距250高点 < -40% + 坑长<10 + 第1-2次', (R[:, 0] < -0.40) & (R[:, 1] < 10) & (R[:, 2] <= 2))
    show('坑长<10 + 第1-2次(不含位置)', (R[:, 1] < 10) & (R[:, 2] <= 2))
    show('第1-2次(对比)', R[:, 2] <= 2)

if __name__ == '__main__':
    main()
