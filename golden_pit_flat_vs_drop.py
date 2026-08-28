# -*- coding: utf-8 -*-
"""验证: 低位横盘(z深但没创新低) vs 下跌坑(创新低) 的胜率差异。

创新低判据(因果): 坑底 close[b] < 进坑前60日最低(或进坑价)。横盘=不创新低。
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression

def main():
    pool = [l.split(',')[0].strip() for l in open('stock_pool_1000.txt', encoding='utf-8') if l.strip()]
    rows = []  # [是否创新低, 坑内跌幅, r20]
    for k, sym in enumerate(pool):
        f = os.path.join('.cache_kline', f'{sym}.npy')
        if not os.path.exists(f): continue
        d = np.load(f, allow_pickle=True).item()
        c = d['close'].astype(float)
        nn = len(c)
        reg, _ = compute_rolling_regression(c, window=250, use_log=True)
        pits = [p for p in pr.detect_golden_pit_v2(c, reg) if p[2] is not None]
        for s, b, lch in pits:
            if lch + 20 >= nn or b < 60: continue
            pre_low = np.min(c[b-60:b])          # 坑底前60日最低
            new_low = c[b] < pre_low             # 是否创新低
            in_drop = c[s] / c[b] - 1            # 进坑价→坑底跌幅
            r20 = c[lch+20]/c[lch]-1
            rows.append([new_low, in_drop, r20])
        if (k+1) % 500 == 0: print(f'  进度 {k+1}', flush=True)
    R = np.array(rows)
    def show(label, m):
        M = R[m]
        if len(M) < 15:
            print(f'  {label:<34} n={len(M):>4} 样本少'); return
        print(f'  {label:<34} n={len(M):>5}  胜率={np.mean(M[:,2]>=0):5.1%} 均值={np.mean(M[:,2]):+6.1%} 高收益(>15%)={np.mean(M[:,2]>0.15):5.1%}')
    print(f'\n=== 横盘坑 vs 下跌坑 (1000池 {len(R)} 坑) ===')
    show('全部', np.ones(len(R), bool))
    show('创新低(真下跌坑)', R[:, 0] == 1)
    show('未创新低(低位横盘)', R[:, 0] == 0)
    print('\n--- 坑内跌幅分层 ---')
    show('坑内跌幅>20%(真下跌)', R[:, 1] > 0.20)
    show('坑内跌幅10-20%', (R[:, 1] > 0.10) & (R[:, 1] <= 0.20))
    show('坑内跌幅<10%(横盘)', R[:, 1] <= 0.10)

if __name__ == '__main__':
    main()
