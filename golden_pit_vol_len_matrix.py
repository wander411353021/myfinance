# -*- coding: utf-8 -*-
"""验证: 波动率 × 坑长 矩阵——低波动股是否需要更长坑长(洗盘时间)。

波动率 = 坑前60日残差std / reg(相对波动率)。
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression

def main():
    pool = [l.split(',')[0].strip() for l in open('stock_pool_1000.txt', encoding='utf-8') if l.strip()]
    rows = []  # [波动率, 坑长, r20]
    for k, sym in enumerate(pool):
        f = os.path.join('.cache_kline', f'{sym}.npy')
        if not os.path.exists(f): continue
        d = np.load(f, allow_pickle=True).item()
        cc = d['close'].astype(float); nn = len(cc)
        rg, _ = compute_rolling_regression(cc, window=250, use_log=True)
        resid = cc - rg
        for s, b, lch in pr.detect_golden_pit_v3(cc, rg):
            if lch is None or lch + 20 >= nn or s < 60: continue
            seg = resid[s-60:s]
            seg = seg[np.isfinite(seg)]
            vol = np.std(seg) / rg[s] if rg[s] > 0 and len(seg) >= 30 else np.nan  # 相对波动率
            plen = b - s + 1
            rows.append([vol, plen, cc[lch+20]/cc[lch]-1])
        if (k+1) % 500 == 0: print(f'  进度 {k+1}', flush=True)
    R = np.array(rows)
    R = R[np.isfinite(R[:, 0])]
    vol_med = np.median(R[:, 0])
    print(f'\n=== 波动率 × 坑长 胜率矩阵 (1000池 {len(R)} 坑) ===')
    print(f'波动率中位数: {vol_med:.4f}')
    print(f'{"分组":<30}{"n":>6}{"胜率":>8}{"均值":>8}')
    groups = [
        ('低波动+短坑(3-4)', (R[:,0]<vol_med) & (R[:,1]<=4)),
        ('低波动+中坑(5-8)', (R[:,0]<vol_med) & (R[:,1]>4) & (R[:,1]<=8)),
        ('低波动+长坑(9+)', (R[:,0]<vol_med) & (R[:,1]>8)),
        ('高波动+短坑(3-4)', (R[:,0]>=vol_med) & (R[:,1]<=4)),
        ('高波动+中坑(5-8)', (R[:,0]>=vol_med) & (R[:,1]>4) & (R[:,1]<=8)),
        ('高波动+长坑(9+)', (R[:,0]>=vol_med) & (R[:,1]>8)),
    ]
    for name, m in groups:
        M = R[m]
        if len(M) < 15:
            print(f'  {name:<30}{len(M):>6}  样本少')
            continue
        print(f'  {name:<30}{len(M):>6}{np.mean(M[:,2]>=0):>7.1%}{np.mean(M[:,2]):>+7.1%}')
    # 低波动内: 坑长门槛扫描
    print('\n--- 低波动组内坑长门槛 ---')
    lo = R[:, 0] < vol_med
    for ml in (3, 5, 8, 10):
        m = lo & (R[:, 1] >= ml)
        M = R[m]
        if len(M) >= 15:
            print(f'  低波动 坑长>={ml:<3}: n={len(M):>5} 胜率={np.mean(M[:,2]>=0):5.1%} 均值={np.mean(M[:,2]):+6.1%}')
    print('--- 高波动组内坑长门槛 ---')
    hi = R[:, 0] >= vol_med
    for ml in (3, 5, 8):
        m = hi & (R[:, 1] >= ml)
        M = R[m]
        if len(M) >= 15:
            print(f'  高波动 坑长>={ml:<3}: n={len(M):>5} 胜率={np.mean(M[:,2]>=0):5.1%} 均值={np.mean(M[:,2]):+6.1%}')

if __name__ == '__main__':
    main()
