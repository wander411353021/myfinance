# -*- coding: utf-8 -*-
"""第1-2次出坑 × 坑长 × reg 方向 组合(全部因果, 1000池)。"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression

def main():
    pool = [l.split(',')[0].strip() for l in open('stock_pool_1000.txt', encoding='utf-8') if l.strip()]
    rows = []  # [出坑次数1-2, plen, reg_up, r20, r60]
    for k, sym in enumerate(pool):
        f = os.path.join('.cache_kline', f'{sym}.npy')
        if not os.path.exists(f): continue
        d = np.load(f, allow_pickle=True).item()
        c = d['close'].astype(float); nn = len(c)
        reg, _ = compute_rolling_regression(c, window=250, use_log=True)
        pits = [p for p in pr.detect_golden_pit_v2(c, reg) if p[2] is not None]
        pits.sort(key=lambda p: p[2])
        for idx, (s, b, lch) in enumerate(pits):
            if lch + 60 >= nn or lch < 20: continue
            cnt = sum(1 for ps, pb, pl in pits[:idx] if lch - pl <= 90) + 1
            early = cnt <= 2
            plen = b - s + 1
            reg_up = np.isfinite(reg[lch]) and np.isfinite(reg[lch-20]) and reg[lch] > reg[lch-20]
            rows.append([early, plen, reg_up, c[lch+20]/c[lch]-1, c[lch+60]/c[lch]-1])
        if (k+1) % 500 == 0: print(f'  进度 {k+1}', flush=True)
    R = np.array(rows)
    def show(label, m):
        M = R[m]
        if len(M) < 15:
            print(f'  {label:<36} n={len(M):>4} 样本少'); return
        print(f'  {label:<36} n={len(M):>4}  20日胜率={np.mean(M[:,3]>=0):5.1%} 均值={np.mean(M[:,3]):+6.1%} | 60日胜率={np.mean(M[:,4]>=0):5.1%}')
    print(f'\n=== 第1-2次出坑组合 (1000池 {len(R)} 坑) ===')
    show('全部(基线)', np.ones(len(R), bool))
    show('第1-2次出坑', R[:, 0] == 1)
    show('第1-2次 + 坑长>=6', (R[:, 0] == 1) & (R[:, 1] >= 6))
    show('第1-2次 + reg上行', (R[:, 0] == 1) & (R[:, 2] == 1))
    show('第1-2次 + 坑长>=6 + reg上行', (R[:, 0] == 1) & (R[:, 1] >= 6) & (R[:, 2] == 1))
    show('第1-2次 + (坑长>=6 或 reg上行)', (R[:, 0] == 1) & ((R[:, 1] >= 6) | (R[:, 2] == 1)))
    show('坑长>=6(对比)', R[:, 1] >= 6)

if __name__ == '__main__':
    main()
