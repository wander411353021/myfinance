# -*- coding: utf-8 -*-
"""对比 reg60/reg120/reg250 作为黄金坑方向过滤的胜率(1000池588信号)。"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression

def main(end='20260827'):
    pool = [l.split(',')[0].strip() for l in open('stock_pool_1000.txt', encoding='utf-8') if l.strip()]
    rows = []
    for k, code in enumerate(pool):
        fcode = code if code[:2] in ('sh', 'sz') else ('sh' if code[0] in '69' else 'sz') + code
        df = pr._load_df(fcode, end)
        if df is None or len(df) < 400:
            continue
        c = df['close'].values.astype(float)
        v = df['volume'].values.astype(float)
        n = len(c)
        regs = {}
        for w in (60, 120, 250):
            r, _ = compute_rolling_regression(c, window=w, use_log=True)
            regs[w] = r
        pits = pr.detect_golden_pit(c, regs[250])
        vcl = pr.detect_volume_clusters(c, v)
        for s, b, lch in pits:
            if lch is None or lch - b > 5 or b - s + 1 < 8 or lch + 60 >= n or lch < 20:
                continue
            # 各 reg 窗口的方向(20日斜率)
            dirs = {}
            for w in (60, 120, 250):
                r = regs[w]
                dirs[w] = (np.isfinite(r[lch]) and np.isfinite(r[lch-20]) and r[lch] > r[lch-20])
            pile = False
            for ss, ee, kd, dr, pk, vr in vcl:
                if kd == 'HIGH' and lch < ss <= lch + 7 and pk >= 5.0:
                    pile = True
                    break
            rows.append([dirs[60], dirs[120], dirs[250], pile, c[lch+20]/c[lch]-1, c[lch+60]/c[lch]-1])
        if (k + 1) % 200 == 0:
            print(f'  进度 {k+1}/{len(pool)}', flush=True)
    R = np.array(rows)
    print(f'\n黄金坑信号: {len(R)}')

    def show(label, m):
        M = R[m]
        if len(M) < 15:
            print(f'  {label:<32} n={len(M):>4} 样本少'); return
        print(f'  {label:<32} n={len(M):>4}  20日胜率={np.mean(M[:,4]>=0):5.1%} 均值={np.mean(M[:,4]):+6.1%} | '
              f'60日胜率={np.mean(M[:,5]>=0):5.1%} 均值={np.mean(M[:,5]):+6.1%}')

    print('\n--- 各 reg 窗口方向过滤(20日斜率) ---')
    show('基线(全部)', np.ones(len(R), bool))
    show('reg60 上行', R[:, 0] == 1)
    show('reg120 上行', R[:, 1] == 1)
    show('reg250 上行', R[:, 2] == 1)
    show('reg60 下行', R[:, 0] == 0)
    show('reg120 下行', R[:, 1] == 0)
    show('reg250 下行', R[:, 2] == 0)
    print('\n--- 组合:两个窗口一致 ---')
    show('reg120+250 都上行', (R[:, 1] == 1) & (R[:, 2] == 1))
    show('reg60+120+250 都上行', (R[:, 0] == 1) & (R[:, 1] == 1) & (R[:, 2] == 1))
    show('reg120+250 都下行', (R[:, 1] == 0) & (R[:, 2] == 0))
    print('\n--- 方向 × 放量堆(reg120) ---')
    show('reg120上行+放量堆', (R[:, 1] == 1) & (R[:, 3] == 1))
    show('reg120下行+放量堆', (R[:, 1] == 0) & (R[:, 3] == 1))

if __name__ == '__main__':
    main()
