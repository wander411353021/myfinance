# -*- coding: utf-8 -*-
"""信号量-胜率权衡网格: 位置阈值 × 坑长 × 出坑次数, 找胜率65%+样本最大的档。"""
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
            rows.append([dd, plen, cnt, c[lch+20]/c[lch]-1])
        if (k+1) % 500 == 0: print(f'  进度 {k+1}', flush=True)
    R = np.array(rows)
    print(f'\n=== 信号量网格 (1000池 {len(R)} 坑, 20日) ===')
    print(f'{"条件":<46}{"n":>6}{"胜率":>8}{"均值":>8}{"高收益>15%":>10}')
    print(f'{"基线(全部)":<46}{len(R):>6}{np.mean(R[:,3]>=0):>7.1%}{np.mean(R[:,3]):>+7.1%}{np.mean(R[:,3]>0.15):>9.1%}')
    for dd_th in (-0.25, -0.30, -0.35, -0.40):
        for plen_th in (999, 15, 10):
            for cnt_th in (99, 3, 2):
                m = (R[:, 0] < dd_th) & (R[:, 1] < plen_th) & (R[:, 2] <= cnt_th)
                if np.sum(m) < 30: continue
                tag = f'距高点<{dd_th*100:.0f}% + 坑长<{plen_th if plen_th<999 else "∞"} + 第{min(cnt_th,99)}次'
                print(f'{tag:<46}{np.sum(m):>6}{np.mean(R[m][:,3]>=0):>7.1%}{np.mean(R[m][:,3]):>+7.1%}{np.mean(R[m][:,3]>0.15):>9.1%}')

if __name__ == '__main__':
    main()
