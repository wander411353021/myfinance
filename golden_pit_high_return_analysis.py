# -*- coding: utf-8 -*-
"""高收益坑特征分析: 20日收益>15% 的坑 vs 普通坑, 对比各因果特征。"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression

def main():
    pool = [l.split(',')[0].strip() for l in open('stock_pool_1000.txt', encoding='utf-8') if l.strip()]
    feats = []  # 每个坑一行特征 + r20
    for k, sym in enumerate(pool):
        f = os.path.join('.cache_kline', f'{sym}.npy')
        if not os.path.exists(f): continue
        d = np.load(f, allow_pickle=True).item()
        c = d['close'].astype(float)
        v = d['vol'].astype(float)
        nn = len(c)
        reg, _ = compute_rolling_regression(c, window=250, use_log=True)
        resid = c - reg
        rstd = np.full(nn, np.nan)
        for i in range(59, nn):
            rstd[i] = np.std(resid[i-59:i+1])
        pits = [p for p in pr.detect_golden_pit_v2(c, reg) if p[2] is not None]
        pits.sort(key=lambda p: p[2])
        for idx, (s, b, lch) in enumerate(pits):
            if lch + 20 >= nn or lch < 60: continue
            r20 = c[lch + 20] / c[lch] - 1
            # --- 特征(全部因果, <=lch 可得) ---
            plen = b - s + 1
            zmin = np.min(resid[s:b+1] / rstd[s:b+1]) if rstd[s] > 0 else -9
            dd_reg = c[b] / reg[b] - 1  # 坑底相对 reg
            hi250 = np.max(c[max(0, b-250):b+1])
            dd_hi = c[b] / hi250 - 1    # 坑底相对 250 日高点
            pre60 = c[s] / c[max(0, s-60)] - 1  # 坑前 60 日涨幅(位置高低)
            reg_up = (reg[lch] / reg[max(0, lch-20)] - 1) if lch >= 20 else 0  # reg 20日斜率
            vp = np.mean(v[max(0, lch-5):lch])
            vol_r = v[lch] / vp if vp > 0 else 1  # 出坑日量比
            exit_g = c[lch] / c[lch-1] - 1 if lch >= 1 else 0  # 出坑日涨幅
            cnt = sum(1 for ps, pb, pl in pits[:idx] if lch - pl <= 90) + 1  # 第几次出坑
            launch = lch - b
            feats.append([plen, zmin, dd_reg, dd_hi, pre60, reg_up, vol_r, exit_g, cnt, launch, c[lch], r20])
        if (k+1) % 500 == 0: print(f'  进度 {k+1}', flush=True)
    F = np.array(feats)
    names = ['坑长', 'z最低', '坑底距reg', '坑底距250高点', '坑前60日涨幅', 'reg20日斜率',
             '出坑日量比', '出坑日涨幅', '第几次出坑', '快启动', '出坑价', 'r20']
    print(f'\n=== 高收益坑特征分析 (1000池 {len(F)} 坑) ===')
    hi = F[:, -1] > 0.15
    mid = (F[:, -1] >= 0) & (F[:, -1] <= 0.15)
    lo = F[:, -1] < 0
    print(f'高收益(r20>15%): {np.sum(hi)} ({np.mean(hi):.0%}) | 小赚(0~15%): {np.sum(mid)} | 亏损(<0): {np.sum(lo)}')
    print(f'\n{"特征":<14}{"高收益组均值":>14}{"小赚组均值":>14}{"亏损组均值":>14}{"高vs亏":>10}')
    for i, nm in enumerate(names[:-1]):
        h, m, l = np.mean(F[hi, i]), np.mean(F[mid, i]), np.mean(F[lo, i])
        print(f'{nm:<14}{h:>14.3f}{m:>14.3f}{l:>14.3f}{h-l:>+10.3f}')

if __name__ == '__main__':
    main()
