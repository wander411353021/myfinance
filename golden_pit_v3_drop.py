# -*- coding: utf-8 -*-
"""v3 + 坑内跌幅门槛(进坑价→坑底>=X%): 排除低波动股浅微坑。全池验证。"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression

def scan(c, reg, min_drop=0.0):
    """v3 逻辑 + 坑内跌幅>=min_drop 过滤(坑结算后检查)。"""
    pits = pr.detect_golden_pit_v3(c, reg)
    if min_drop <= 0:
        return pits
    out = []
    for s, b, lch in pits:
        if lch is None:
            out.append((s, b, lch))
            continue
        in_drop = c[s] / c[b] - 1
        if in_drop >= min_drop:
            out.append((s, b, lch))
    return out

def main():
    pool = [l.split(',')[0].strip() for l in open('stock_pool_1000.txt', encoding='utf-8') if l.strip()]
    for md in (0.0, 0.06, 0.08, 0.10, 0.12):
        rows = []
        for k, sym in enumerate(pool):
            f = os.path.join('.cache_kline', f'{sym}.npy')
            if not os.path.exists(f): continue
            d = np.load(f, allow_pickle=True).item()
            cc = d['close'].astype(float); nn = len(cc)
            rg, _ = compute_rolling_regression(cc, window=250, use_log=True)
            for s, b, lch in scan(cc, rg, md):
                if lch is None or lch + 60 >= nn: continue
                rows.append([cc[lch+20]/cc[lch]-1, cc[lch+60]/cc[lch]-1])
            if (k+1) % 500 == 0: print(f'  min_drop={md} 进度 {k+1}', flush=True)
        R = np.array(rows)
        print(f'  min_drop={md:.0%}: n={len(R):>5}  20日胜率={np.mean(R[:,0]>=0):5.1%} 均值={np.mean(R[:,0]):+6.1%} | 60日胜率={np.mean(R[:,1]>=0):5.1%}')

if __name__ == '__main__':
    main()
