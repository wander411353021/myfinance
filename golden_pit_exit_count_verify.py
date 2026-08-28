# -*- coding: utf-8 -*-
"""因果特征验证: 坑内已假出坑次数能否预测"最后一次出坑"。

对每个出坑信号, 统计它之前 90 天内已出坑次数(同大坑内, 出坑日之前已知)。
若第 N 次出坑胜率显著高(接近事后 65.5%), 则可因果使用。
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression

def main():
    pool = [l.split(',')[0].strip() for l in open('stock_pool_1000.txt', encoding='utf-8') if l.strip()]
    rows = []  # [坑内出坑次数(1-based), r20, r60, 是否最后出坑(事后)]
    for k, sym in enumerate(pool):
        f = os.path.join('.cache_kline', f'{sym}.npy')
        if not os.path.exists(f): continue
        d = np.load(f, allow_pickle=True).item()
        c = d['close'].astype(float); nn = len(c)
        reg, _ = compute_rolling_regression(c, window=250, use_log=True)
        pits = [p for p in pr.detect_golden_pit_v2(c, reg) if p[2] is not None]
        pits.sort(key=lambda p: p[2])
        for idx, (s, b, lch) in enumerate(pits):
            if lch + 60 >= nn: continue
            # 之前 90 天内的出坑次数(因果)
            cnt = sum(1 for ps, pb, pl in pits[:idx] if lch - pl <= 90)
            cnt1 = cnt + 1  # 本次是第几次
            nxt = pits[idx + 1][2] if idx + 1 < len(pits) else None
            last = (nxt is None) or (nxt - lch) > 30
            r20 = c[lch + 20] / c[lch] - 1
            r60 = c[lch + 60] / c[lch] - 1
            rows.append([cnt1, r20, r60, last])
        if (k + 1) % 500 == 0: print(f'  进度 {k+1}', flush=True)
    R = np.array(rows)
    print(f'\n=== 因果特征: 坑内第几次出坑 (1000池 {len(R)} 坑) ===')
    for n_ in (1, 2, 3, 4, 5):
        m = R[:, 0] == n_
        if np.sum(m) < 15:
            print(f'  第{n_}次出坑: n={np.sum(m):>4} 样本少'); continue
        print(f'  第{n_}次出坑: n={np.sum(m):>4}  20日胜率={np.mean(R[m][:,1]>=0):5.1%} 均值={np.mean(R[m][:,1]):+6.1%} | '
              f'60日胜率={np.mean(R[m][:,2]>=0):5.1%} | 事后"最后出坑"占比={np.mean(R[m][:,3]):.0%}')
    m = R[:, 0] >= 3
    print(f'  第3次+: n={np.sum(m):>4}  20日胜率={np.mean(R[m][:,1]>=0):5.1%} 均值={np.mean(R[m][:,1]):+6.1%}')
    print(f'\n  事后"最后出坑"占比随次数: ', end='')
    for n_ in (1, 2, 3, 4, 5):
        m = R[:, 0] == n_
        if np.sum(m) >= 15:
            print(f'{n_}次:{np.mean(R[m][:,3]):.0%} ', end='')
    print()

if __name__ == '__main__':
    main()
