# -*- coding: utf-8 -*-
"""克服 z 缺陷的候选算法对比(全池 1000, 20日持有, 全部因果):

A. v2 现版: z<-1.5(滚动std)
B. z<-1.5 + 创新低(坑底 < 坑前60日最低)
C. z<-1.5 + 坑内跌幅>=10%(进坑价→坑底)
D. 纯跌幅位置(不用z): 距250日高点<-25% + 坑内跌幅>=12% + 创新低
E. z_MAD: z = resid / (1.4826*MAD(resid,60))  稳健 std
F. z<-1.5 + 创新低 + 坑内跌幅>=10% (组合)
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression

def rolling_std(x, w):
    out = np.full(len(x), np.nan)
    for i in range(w-1, len(x)):
        out[i] = np.std(x[i-w+1:i+1])
    return out

def rolling_mad(x, w):
    out = np.full(len(x), np.nan)
    for i in range(w-1, len(x)):
        med = np.median(x[i-w+1:i+1])
        out[i] = np.median(np.abs(x[i-w+1:i+1] - med))
    return out

def scan_pits(c, reg, mode):
    """通用坑扫描: 返回 [(s, b, lch)] (因果, 出坑即结算)。"""
    n = len(c)
    resid = c - reg
    if mode in ('A', 'B', 'C', 'F'):
        rstd = rolling_std(resid, 60)
        z = np.full(n, np.nan)
        for i in range(59, n):
            z[i] = resid[i] / rstd[i] if rstd[i] > 0 else 0.0
    elif mode == 'E':
        rm = rolling_mad(resid, 60)
        z = np.full(n, np.nan)
        for i in range(59, n):
            z[i] = resid[i] / (1.4826 * rm[i]) if rm[i] > 0 else 0.0
    gate = reg * 0.9
    out = []
    i = 59
    while i < n:
        if not (np.isfinite(z[i]) and z[i] < -1.5):
            i += 1
            continue
        pre_std = np.std(resid[i-60:i]) if i >= 60 and np.std(resid[i-60:i]) > 0 else (rstd[i] if 'rstd' in dir() else 1.0)
        s = i; b = i; bv = c[i]; lch = None
        # 坑内条件
        def in_pit(j):
            zj = resid[j] / pre_std if pre_std > 0 else 0.0
            if not (np.isfinite(zj) and zj < -1.5):
                return False
            if mode in ('B', 'F'):
                # 创新低: 当日 close < 前60日最低(当日可知)
                if j >= 60 and c[j] >= np.min(c[j-60:j]):
                    return False
            if mode in ('C', 'F'):
                # 坑内跌幅: 进坑后累计跌幅>=10%(由 s 与当日比较)
                if c[j] / c[s] - 1 > -0.10:
                    return False
            return True
        j = i
        while j < n:
            if in_pit(j):
                if c[j] < bv: bv = c[j]; b = j
            else:
                zj = resid[j] / pre_std if pre_std > 0 else 0.0
                if zj >= -1.5 and c[j] >= gate[j] and c[j] > bv * 1.02:
                    lch = j
                    break
            j += 1
        if lch is not None:
            out.append((s, b, lch))
            i = lch + 1
        else:
            i = j + 1
    return out

def scan_d(c, reg):
    """D: 纯跌幅位置, 不用 z。坑 = 距250高点<-25% + 坑内跌幅>=12% + 创新低。"""
    n = len(c)
    out = []
    i = 60
    while i < n:
        hi250 = np.max(c[max(0, i-250):i+1])
        if c[i] / hi250 - 1 > -0.25:
            i += 1
            continue
        s = i; b = i; bv = c[i]; lch = None
        j = i
        while j < n:
            if c[j] < bv: bv = c[j]; b = j
            if c[j] / c[s] - 1 < -0.12:  # 坑内已跌>=12%
                # 找坑底后出坑
                for k in range(b+1, n):
                    if c[k] >= reg[k]*0.9 and c[k] > c[b]*1.02:
                        lch = k
                        break
                break
            j += 1
        if lch is not None:
            out.append((s, b, lch))
            i = lch + 1
        else:
            i = j + 1
    return out

def main():
    pool = [l.split(',')[0].strip() for l in open('stock_pool_1000.txt', encoding='utf-8') if l.strip()]
    for mode in ('A', 'B', 'C', 'D', 'E', 'F'):
        rows = []
        for k, sym in enumerate(pool):
            f = os.path.join('.cache_kline', f'{sym}.npy')
            if not os.path.exists(f): continue
            d = np.load(f, allow_pickle=True).item()
            cc = d['close'].astype(float); nn = len(cc)
            rg, _ = compute_rolling_regression(cc, window=250, use_log=True)
            pits = scan_d(cc, rg) if mode == 'D' else scan_pits(cc, rg, mode)
            for s, b, lch in pits:
                if lch is None or lch + 20 >= nn: continue
                rows.append(cc[lch+20]/cc[lch]-1)
            if (k+1) % 500 == 0: print(f'  {mode} 进度 {k+1}', flush=True)
        R = np.array(rows)
        print(f'  {mode}: n={len(R):>5}  20日胜率={np.mean(R>=0):5.1%} 均值={np.mean(R):+6.1%} 高收益(>15%)={np.mean(R>0.15):5.1%}')

if __name__ == '__main__':
    main()
