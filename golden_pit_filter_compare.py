# -*- coding: utf-8 -*-
"""过滤方式对比: 坑长 / 坑底距reg绝对深度 / 组合。目标: 清抖动微坑, 尽量保胜率。"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression

def apply_filter(c, reg, pits, mode):
    out = []
    for s, b, lch in pits:
        if lch is None:
            out.append((s, b, lch)); continue
        plen = b - s + 1
        dd_reg = c[b] / reg[b] - 1
        if mode == 'len3' and plen < 3: continue
        if mode == 'reg8' and dd_reg > -0.08: continue
        if mode == 'reg10' and dd_reg > -0.10: continue
        if mode == 'len3_reg8' and (plen < 3 or dd_reg > -0.08): continue
        if mode == 'len2_reg8' and (plen < 2 or dd_reg > -0.08): continue
        out.append((s, b, lch))
    return out

def main():
    pool = [l.split(',')[0].strip() for l in open('stock_pool_1000.txt', encoding='utf-8') if l.strip()]
    for mode in ('none', 'len3', 'reg8', 'reg10', 'len3_reg8', 'len2_reg8'):
        rows = []
        for k, sym in enumerate(pool):
            f = os.path.join('.cache_kline', f'{sym}.npy')
            if not os.path.exists(f): continue
            d = np.load(f, allow_pickle=True).item()
            cc = d['close'].astype(float); nn = len(cc)
            rg, _ = compute_rolling_regression(cc, window=250, use_log=True)
            pits = pr.detect_golden_pit_v3(cc, rg)
            pits = apply_filter(cc, rg, pits, mode)
            for s, b, lch in pits:
                if lch is None or lch + 60 >= nn: continue
                rows.append([cc[lch+20]/cc[lch]-1, cc[lch+60]/cc[lch]-1])
            if (k+1) % 500 == 0: print(f'  {mode} 进度 {k+1}', flush=True)
        R = np.array(rows)
        print(f'  {mode:<12}: n={len(R):>5}  20日胜率={np.mean(R[:,0]>=0):5.1%} 均值={np.mean(R[:,0]):+6.1%} | 60日胜率={np.mean(R[:,1]>=0):5.1%}')

if __name__ == '__main__':
    main()
