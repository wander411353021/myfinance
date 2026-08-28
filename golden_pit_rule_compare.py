# -*- coding: utf-8 -*-
"""黄金坑规则放宽对比（reasonix 定版口径）：
用 panic_reversal.detect_golden_pit（z<-1.5）+ 20日持有，
对比不同 快启动(launch) / 坑长(pit_len) 阈值的样本量、胜率、均值、分年度。
复用 .cache_kline 缓存。用法:
  python3 golden_pit_rule_compare.py stock_pool_300.txt 300
"""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression

WORKDIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(WORKDIR, '.cache_kline')


def load(symbol):
    f = os.path.join(CACHE, f'{symbol}.npy')
    if os.path.exists(f):
        return np.load(f, allow_pickle=True).item()
    return None


def collect(symbol):
    d = load(symbol)
    if d is None:
        return []
    c = d['close']; h = d['high']; l = d['low']; v = d['vol']; ts = d['ts']
    n = len(c)
    reg250, _ = compute_rolling_regression(c, window=250, use_log=True)
    pits = pr.detect_golden_pit(c, reg250)
    out = []
    for s, b, lch in pits:
        if lch is None or lch + 20 >= n:
            continue
        out.append({
            'symbol': symbol,
            'launch': int(lch - b), 'pit_len': int(b - s + 1),
            'gain20': float(c[lch + 20] / c[lch] - 1.0),
            'year': int(pd.to_datetime(int(ts[lch]), unit='s').year),
        })
    return out


def main():
    pool_file = sys.argv[1] if len(sys.argv) > 1 else 'stock_pool_300.txt'
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    stocks = [l.strip().split(',')[0] for l in open(os.path.join(WORKDIR, pool_file)) if l.strip()][:top]
    print(f'扫描 {len(stocks)} 只 ...', flush=True)
    rows = []
    t0 = time.time()
    for i, sym in enumerate(stocks):
        try:
            rows += collect(sym)
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(stocks)} 信号{len(rows)} {time.time()-t0:.0f}s', flush=True)
    print(f'完成, 共 {len(rows)} 个已出坑信号(z<-1.5)', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv('/tmp/gp_rule_compare.csv', index=False)

    def rule(launch_max, len_min):
        return df[(df['launch'] <= launch_max) & (df['pit_len'] >= len_min)]

    print('\n=== 规则对比 (20日持有, reasonix口径) ===')
    tests = [
        ('A 现版  l<=5 + len>=8', 5, 8),
        ('B       l<=5 + len>=5', 5, 5),
        ('C       l<=5 + len>=4', 5, 4),
        ('D       l<=8 + len>=8', 8, 8),
        ('E       l<=8 + len>=5', 8, 5),
        ('F 深坑全收 l<=5', 5, 1),
        ('G 深坑 l<=8', 8, 1),
        ('H 全收', 99999, 1),
    ]
    for lab, lm, ln in tests:
        g = rule(lm, ln)
        if len(g):
            print(f'  {lab}: n={len(g)} 胜率{(g["gain20"]>0).mean()*100:.1f}% '
                  f'均值{g["gain20"].mean()*100:+.2f}% 中位{np.median(g["gain20"])*100:+.2f}%')

    print('\n=== 分年度 (A vs B vs F) ===')
    for lab, lm, ln in [('A', 5, 8), ('B', 5, 5), ('F', 5, 1)]:
        g = rule(lm, ln)
        parts = []
        for yr in [2024, 2025, 2026]:
            gy = g[g['year'] == yr]
            if len(gy):
                parts.append(f'{yr}:n{len(gy)}/{(gy["gain20"]>0).mean()*100:.0f}%')
        print(f'  {lab}: ' + ' '.join(parts))

    # 新增样本质量
    A = rule(5, 8); B = rule(5, 5); F = rule(5, 1)
    nb = B[~B.index.isin(A.index)]
    nf = F[~F.index.isin(B.index)]
    print('\n=== 新增样本质量 ===')
    if len(nb):
        print(f'  B新增(坑长5-7) {len(nb)}个: 胜率{(nb["gain20"]>0).mean()*100:.1f}% 均值{nb["gain20"].mean()*100:+.2f}%')
    if len(nf):
        print(f'  F新增(坑长<5) {len(nf)}个: 胜率{(nf["gain20"]>0).mean()*100:.1f}% 均值{nf["gain20"].mean()*100:+.2f}%')


if __name__ == '__main__':
    main()
