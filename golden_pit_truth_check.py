# -*- coding: utf-8 -*-
"""修复后黄金坑真实数据复现(1000池, 因果算法):
分组: reg250 上行/下行, 坑长分层, 放量堆可执行口径(确认后 lch+7 买入)
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression

CACHE = '.cache_kline'
POOL = 'stock_pool_1000.txt'

def load(sym):
    f = os.path.join(CACHE, f'{sym}.npy')
    return np.load(f, allow_pickle=True).item() if os.path.exists(f) else None

def main():
    pool = [l.split(',')[0].strip() for l in open(POOL, encoding='utf-8') if l.strip()]
    rows = []  # [reg_up, pit_len, pile(可执行), r20(出坑日买), r20_pile(lch+7买), r60]
    for k, sym in enumerate(pool):
        d = load(sym)
        if d is None:
            continue
        c = d['close'].astype(float)
        v = d['vol'].astype(float)
        n = len(c)
        reg, _ = compute_rolling_regression(c, window=250, use_log=True)
        pits = pr.detect_golden_pit(c, reg)
        vcl = pr.detect_volume_clusters(c, v)
        for s, b, lch in pits:
            if lch is None or lch - b > 5:  # 规则F: 快启动<=5
                continue
            if lch + 60 >= n or lch < 20:
                continue
            if not (np.isfinite(reg[lch]) and np.isfinite(reg[lch - 20])):
                continue
            reg_up = reg[lch] > reg[lch - 20]
            plen = b - s + 1
            # 放量堆: 出坑后7天内 HIGH 堆峰值>=5(可执行口径: lch+7 买入)
            pile = False
            for ss, ee, kd, dr, pk, vr in vcl:
                if kd == 'HIGH' and lch < ss <= lch + 7 and pk >= 5.0:
                    pile = True
                    break
            r20 = c[lch + 20] / c[lch] - 1
            r20p = c[lch + 27] / c[lch + 7] - 1 if pile and lch + 27 < n else np.nan
            r60 = c[lch + 60] / c[lch] - 1
            rows.append([reg_up, plen, pile, r20, r20p, r60])
        if (k + 1) % 200 == 0:
            print(f'  进度 {k+1}/{len(pool)}', flush=True)
    R = np.array(rows, dtype=float)
    print(f'\n规则F信号(修复后, 1000池): {len(R)}')

    def show(label, m):
        M = R[m]
        if len(M) < 15:
            print(f'  {label:<36} n={len(M):>4} 样本少'); return
        print(f'  {label:<36} n={len(M):>4}  20日胜率={np.mean(M[:,3]>=0):5.1%} 均值={np.mean(M[:,3]):+6.1%} | '
              f'60日胜率={np.mean(M[:,5]>=0):5.1%} 均值={np.mean(M[:,5]):+6.1%}')

    print('\n--- 基线 / reg 方向 ---')
    show('全部(基线)', np.ones(len(R), bool))
    show('reg250 上行', R[:, 0] == 1)
    show('reg250 下行', R[:, 0] == 0)
    print('\n--- 坑长分层 ---')
    show('坑长 0-5(短坑)', R[:, 1] <= 5)
    show('坑长 6-20', (R[:, 1] > 5) & (R[:, 1] <= 20))
    show('坑长 21-40(长坑)', (R[:, 1] > 20) & (R[:, 1] <= 40))
    show('坑长 >40', R[:, 1] > 40)
    print('\n--- 放量堆(可执行口径) ---')
    show('有放量堆(出坑日买)', R[:, 2] == 1)
    show('无放量堆(出坑日买)', R[:, 2] == 0)
    PM = R[R[:, 2] == 1]
    if len(PM) >= 15:
        print(f'  {"放量堆确认后 lch+7 买入":<36} n={len(PM):>4}  20日胜率={np.mean(PM[:,4]>=0):5.1%} 均值={np.mean(PM[:,4]):+6.1%}')
    print('\n--- reg × 坑长 ---')
    show('reg上行+坑长>=8', (R[:, 0] == 1) & (R[:, 1] >= 8))
    show('reg下行+坑长>=8', (R[:, 0] == 0) & (R[:, 1] >= 8))
    print('\n--- 分年度 ---')
    for y in range(2023, 2027):
        pass
    # 年度: 用 ts 不可得, 跳过(信号日期在缓存中无日期列, 只留胜率汇总)

if __name__ == '__main__':
    main()
