# -*- coding: utf-8 -*-
"""验证:黄金坑 + reg 上行(reg250 斜率>0)的组合增益。

分组:全部 / reg上行 / reg下行 / reg上行+放量堆 / reg下行+放量堆
收益:20日 / 60日。1000 池。
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression

def main(pool_file='stock_pool_1000.txt', end='20260827'):
    pool = [l.split(',')[0].strip() for l in open(pool_file, encoding='utf-8') if l.strip()]
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
            if lch is None or lch - b > 5 or b - s + 1 < 8:
                continue
            if lch + 60 >= n or lch < 20 or not np.isfinite(reg250[lch]) or not np.isfinite(reg250[lch - 20]):
                continue
            reg_up = reg250[lch] > reg250[lch - 20]
            pile = False
            for ss, ee, kd, dr, pk, vr in vcl:
                if kd == 'HIGH' and lch < ss <= lch + 7 and pk >= 5.0:
                    pile = True
                    break
            rows.append([reg_up, pile, c[lch+20]/c[lch]-1, c[lch+60]/c[lch]-1])
        if (k + 1) % 200 == 0:
            print(f'  进度 {k+1}/{len(pool)}', flush=True)
    R = np.array(rows)
    print(f'\n黄金坑信号(1000池): {len(R)}')

    def show(label, m):
        M = R[m]
        if len(M) < 15:
            print(f'  {label:<28} n={len(M):>4} 样本少'); return
        print(f'  {label:<28} n={len(M):>4}  20日胜率={np.mean(M[:,2]>=0):5.1%} 均值={np.mean(M[:,2]):+6.1%} | '
              f'60日胜率={np.mean(M[:,3]>=0):5.1%} 均值={np.mean(M[:,3]):+6.1%}')

    print('\n--- 全部 / reg 方向 ---')
    show('全部信号(基线)', np.ones(len(R), bool))
    show('reg 上行', R[:, 0] == 1)
    show('reg 下行', R[:, 0] == 0)
    print('\n--- reg × 放量堆 ---')
    show('reg上行 + 放量堆', (R[:, 0] == 1) & (R[:, 1] == 1))
    show('reg上行 无放量堆', (R[:, 0] == 1) & (R[:, 1] == 0))
    show('reg下行 + 放量堆', (R[:, 0] == 0) & (R[:, 1] == 1))
    show('reg下行 无放量堆', (R[:, 0] == 0) & (R[:, 1] == 0))

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='stock_pool_1000.txt')
    args = ap.parse_args()
    main(args.pool)
