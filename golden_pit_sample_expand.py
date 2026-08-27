# -*- coding: utf-8 -*-
"""扩展样本方案:reg250 上行样本少,量化"放宽方向 OR 放量堆"的胜率/样本权衡(1000池)。"""
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
        reg250, _ = compute_rolling_regression(c, window=250, use_log=True)
        pits = pr.detect_golden_pit(c, reg250)
        vcl = pr.detect_volume_clusters(c, v)
        for s, b, lch in pits:
            if lch is None or lch - b > 5 or b - s + 1 < 8 or lch + 60 >= n or lch < 20:
                continue
            if not (np.isfinite(reg250[lch]) and np.isfinite(reg250[lch-20])):
                continue
            slope = reg250[lch] / reg250[lch-20] - 1  # 20日 reg 斜率
            pile = False
            for ss, ee, kd, dr, pk, vr in vcl:
                if kd == 'HIGH' and lch < ss <= lch + 7 and pk >= 5.0:
                    pile = True
                    break
            rows.append([slope, pile, c[lch+20]/c[lch]-1, c[lch+60]/c[lch]-1])
        if (k + 1) % 200 == 0:
            print(f'  进度 {k+1}/{len(pool)}', flush=True)
    R = np.array(rows)
    up = R[:, 0] > 0
    pile = R[:, 1] == 1
    print(f'\n黄金坑信号: {len(R)}')

    def show(label, m):
        M = R[m]
        if len(M) < 15:
            print(f'  {label:<40} n={len(M):>4} 样本少'); return
        print(f'  {label:<40} n={len(M):>4} ({len(M)/len(R):5.1%})  20日胜率={np.mean(M[:,2]>=0):5.1%} 均值={np.mean(M[:,2]):+6.1%} | 60日胜率={np.mean(M[:,3]>=0):5.1%}')

    print('\n--- 原方案(严格) ---')
    show('A. reg250 上行(原)', up)
    show('B. 下行+放量堆(原)', (~up) & pile)
    print('\n--- 扩展方案 ---')
    show('C. 上行 OR 放量堆(并集)', up | pile)
    show('D. 上行 或 下行+放量堆', up | ((~up) & pile))
    show('E. 斜率>-1% 或 放量堆', (R[:,0] > -0.01) | pile)
    show('F. 斜率>-2% 或 放量堆', (R[:,0] > -0.02) | pile)
    show('G. 全部(基线)', np.ones(len(R), bool))
    print('\n--- 斜率阈值敏感度(不放量堆) ---')
    for th in (0.0, -0.005, -0.01, -0.02, -0.03):
        m = R[:, 0] > th
        show(f'斜率>{th:.3f}', m)

if __name__ == '__main__':
    main()
