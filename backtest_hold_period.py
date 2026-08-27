# -*- coding: utf-8 -*-
"""
持有期优化验证 — 对比不同持有期在全周期及2026年的表现
假设：60天持有期在弱市太长，黄金坑动能约20天
"""
import sys
sys.path.insert(0, '.')
from golden_pit_v2_backtest import *
import numpy as np
import time

stocks = []
with open('stock_pool_1000.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            code, name = line.split(',', 1)
            stocks.append((code, name))

print(f"股票池: {len(stocks)}只")
print("=" * 80)

all_signals = []
success_count = 0
fail_count = 0
start_time = time.time()

for idx, (symbol, name) in enumerate(stocks):
    if (idx + 1) % 200 == 0:
        print(f"  进度: {idx+1}/{len(stocks)} ({time.time()-start_time:.0f}s) 信号{len(all_signals)}")

    df = fetch_kline_sina(symbol, datalen=1023)
    if df is None or len(df) < 310:
        fail_count += 1
        continue

    closes = df['close'].values.astype(float)
    dates = df['date'].values
    n = len(closes)

    reg250, _ = compute_rolling_regression(closes, window=250)
    pits_v1 = detect_golden_pit_v1(closes, reg250)

    for pit in pits_v1:
        s, b, lch = pit
        if lch is None or lch + 60 >= n:
            continue
        if lch - b > 5:
            continue
        if b - s + 1 < 8:
            continue
        buy_px = closes[lch]
        buy_date = pd.Timestamp(dates[lch]).strftime('%Y-%m-%d')
        buy_year = pd.Timestamp(dates[lch]).year

        rets_hold = {}
        for hd in [10, 20, 30, 45, 60]:
            rets_hold[hd] = closes[lch+hd] / buy_px - 1

        all_signals.append({
            'buy_date': buy_date, 'buy_year': buy_year,
            'rets': rets_hold,
        })

    success_count += 1

elapsed = time.time() - start_time
print(f"\n完成: {success_count}只成功, 失败{fail_count}, 耗时{elapsed:.0f}s")
print(f"信号总数: {len(all_signals)}")

def calc_stats(rets):
    if not rets:
        return (0, 0, 0, 0)
    return (len(rets), sum(1 for r in rets if r > 0)/len(rets), np.mean(rets), np.median(rets))

print("\n" + "=" * 90)
print("全周期（2022-2026）不同持有期对比")
print("=" * 90)
print(f"{'持有期':<10} {'信号数':>6} {'胜率':>8} {'均值':>8} {'中位':>8} {'累计(简单乘)':>14}")
print("-" * 65)
for hd in [10, 20, 30, 45, 60]:
    rets = [s['rets'][hd] for s in all_signals]
    n, wr, mean, med = calc_stats(rets)
    cum = np.prod([1+r for r in rets])
    print(f"{hd}天{'':<6} {n:>6} {wr:>7.1%} {mean:>7.1%} {med:>7.1%} {cum:>13.2f}x")

print("\n" + "=" * 90)
print("按年份 × 持有期 胜率矩阵")
print("=" * 90)
years = sorted(set(s['buy_year'] for s in all_signals))
print(f"{'年份':<8}", end='')
for hd in [10, 20, 30, 60]:
    print(f" {'%2dd'%hd:>10}", end='')
print()
print("-" * 55)
for year in years:
    grp = [s for s in all_signals if s['buy_year'] == year]
    print(f"{year:<8}", end='')
    for hd in [10, 20, 30, 60]:
        rets = [s['rets'][hd] for s in grp]
        n, wr, mean, med = calc_stats(rets)
        print(f" {wr:>9.1%}", end='')
    print()

print("\n按年份 × 持有期 均值矩阵")
print("=" * 90)
print(f"{'年份':<8}", end='')
for hd in [10, 20, 30, 60]:
    print(f" {'%2dd'%hd:>10}", end='')
print()
print("-" * 55)
for year in years:
    grp = [s for s in all_signals if s['buy_year'] == year]
    print(f"{year:<8}", end='')
    for hd in [10, 20, 30, 60]:
        rets = [s['rets'][hd] for s in grp]
        n, wr, mean, med = calc_stats(rets)
        print(f" {mean:>9.1%}", end='')
    print()

print("\n" + "=" * 90)
print("回测完成")
print("=" * 90)
