# -*- coding: utf-8 -*-
"""组合增强:坑长 / reg方向 / 放量堆延迟买入(lch+7) 的组合胜率。全部因果。"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression
from golden_pit_v2_rewrite import detect_golden_pit_v2

def main():
    pool = [l.split(',')[0].strip() for l in open('stock_pool_1000.txt', encoding='utf-8') if l.strip()]
    rows = []  # [r20, plen, reg_up, pile, r20_pile(lch+7买)]
    for k, sym in enumerate(pool):
        f = os.path.join('.cache_kline', f'{sym}.npy')
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True).item()
        c = d['close'].astype(float)
        v = d['vol'].astype(float)
        n = len(c)
        reg, _ = compute_rolling_regression(c, window=250, use_log=True)
        pits = detect_golden_pit_v2(c, reg)
        vcl = pr.detect_volume_clusters(c, v)
        for s, b, lch in pits:
            if lch is None or lch - b > 5 or lch + 20 >= n or lch < 20:
                continue
            r20 = c[lch + 20] / c[lch] - 1
            plen = b - s + 1
            reg_up = np.isfinite(reg[lch]) and np.isfinite(reg[lch - 20]) and reg[lch] > reg[lch - 20]
            pile = False
            for ss, ee, kd, dr, pk, vr in vcl:
                if kd == 'HIGH' and lch < ss <= lch + 7 and pk >= 5.0:
                    pile = True
                    break
            r20p = c[lch + 27] / c[lch + 7] - 1 if pile and lch + 27 < n else np.nan
            rows.append([r20, plen, reg_up, pile, r20p])
        if (k + 1) % 300 == 0:
            print(f'  进度 {k+1}/{len(pool)}', flush=True)
    R = np.array(rows, dtype=float)
    pile = R[:, 3] == 1
    long = R[:, 1] >= 6
    up = R[:, 2] == 1

    def show(label, m, col=0):
        M = R[m]
        MM = M[:, col]
        MM = MM[np.isfinite(MM)]
        if len(MM) < 15:
            print(f'  {label:<38} n={len(MM):>4} 样本少'); return
        print(f'  {label:<38} n={len(MM):>4}  20日胜率={np.mean(MM>=0):5.1%} 均值={np.mean(MM):+6.1%}')

    print(f'\n=== 组合增强(1000池, v2, 快启动<=5) ===')
    show('基线(全部, 出坑日买)', np.ones(len(R), bool))
    show('放量堆确认(lch+7买)', pile, 4)
    show('坑长>=6', long)
    show('reg上行', up)
    show('坑长>=6 + reg上行', long & up)
    show('坑长>=6 + 放量堆(lch+7买)', long & pile, 4)
    show('reg上行 + 放量堆(lch+7买)', up & pile, 4)
    show('长坑/放量堆/reg上行 任一', long | pile | up)
    show('  (出坑日买)', long | pile | up, 0)
    # 权重: 长坑用出坑日买, 短坑有放量堆用 lch+7
    r_hybrid = []
    for r, pl, pu, pi, rp in rows:
        if pl >= 6:
            r_hybrid.append(r)
        elif pi:
            r_hybrid.append(rp)
    r_hybrid = np.array([x for x in r_hybrid if np.isfinite(x)])
    if len(r_hybrid) >= 15:
        print(f'  {"混合:长坑出坑买/短坑放量堆延迟买":<38} n={len(r_hybrid):>4}  20日胜率={np.mean(r_hybrid>=0):5.1%} 均值={np.mean(r_hybrid):+6.1%}')

if __name__ == '__main__':
    main()
