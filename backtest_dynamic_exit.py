# -*- coding: utf-8 -*-
"""
动态离场策略回测 — 20天底仓 + 峰值回撤离场
策略定义：
  1. 买入（出坑日 lch 收盘）
  2. 至少持有20天（底仓期，不卖出）
  3. 20天后每天检查：若收盘价较持仓期间最高峰值回撤 >= 阈值(如8%)，则离场
  4. 若始终未触发，最长持有 max_hold 天（如60天）后离场
对比：纯20天、纯60天、动态离场（多阈值）
"""
import sys
sys.path.insert(0, '.')
from golden_pit_v2_backtest import *
import numpy as np
import time
import random

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
    if (idx + 1) % 150 == 0:
        elapsed = time.time() - start_time
        print(f"  进度: {idx+1}/{len(stocks)} ({elapsed:.0f}s) 信号{len(all_signals)} 成功{success_count}")

    df = fetch_kline_sina(symbol, datalen=1023)
    if df is None or len(df) < 310:
        fail_count += 1
        # 节流：失败后稍等再继续，避免风控
        time.sleep(random.uniform(0.05, 0.15))
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

        # 模拟动态离场
        peak = buy_px
        exit_ret = None
        exit_day = None
        for d in range(1, 61):
            if lch + d >= n:
                exit_ret = closes[n-1] / buy_px - 1
                exit_day = d
                break
            c = closes[lch + d]
            if c > peak:
                peak = c
            if d >= 20:  # 底仓期结束，开始跟踪
                dd = 1 - c / peak
                if dd >= 0.08:  # 8%回撤触发
                    exit_ret = c / buy_px - 1
                    exit_day = d
                    break
        if exit_ret is None:
            exit_ret = closes[min(lch+60, n-1)] / buy_px - 1
            exit_day = 60

        # 纯20天、纯60天对照
        ret20 = closes[lch+20] / buy_px - 1
        ret60 = closes[min(lch+60, n-1)] / buy_px - 1

        # 其他回撤阈值对照（8%已算，算5%/10%/12%）
        rets_dd = {}
        for dd_thr in [0.05, 0.10, 0.12, 0.15]:
            pk = buy_px
            r = None
            for d in range(1, 61):
                if lch + d >= n:
                    r = closes[n-1] / buy_px - 1
                    break
                c = closes[lch+d]
                if c > pk:
                    pk = c
                if d >= 20 and (1 - c/pk) >= dd_thr:
                    r = c / buy_px - 1
                    break
            if r is None:
                r = closes[min(lch+60, n-1)] / buy_px - 1
            rets_dd[dd_thr] = r

        all_signals.append({
            'buy_date': buy_date, 'buy_year': buy_year,
            'ret20': ret20, 'ret60': ret60,
            'ret_dyn8': exit_ret, 'exit_day': exit_day,
            'ret_dd5': rets_dd[0.05], 'ret_dd10': rets_dd[0.10],
            'ret_dd12': rets_dd[0.12], 'ret_dd15': rets_dd[0.15],
        })

    success_count += 1
    # 节流
    time.sleep(random.uniform(0.02, 0.08))

elapsed = time.time() - start_time
print(f"\n完成: {success_count}只成功, 失败{fail_count}, 耗时{elapsed:.0f}s")
print(f"信号总数: {len(all_signals)}")

# 统计
def calc_stats(rets):
    if not rets:
        return (0, 0, 0, 0)
    return (len(rets), sum(1 for r in rets if r > 0)/len(rets), np.mean(rets), np.median(rets))

print("\n" + "=" * 90)
print("全周期策略对比")
print("=" * 90)
strategies = [
    ('纯20天', lambda s: s['ret20'], None),
    ('纯60天', lambda s: s['ret60'], None),
    ('20天底仓+5%回撤离场', lambda s: s['ret_dd5'], None),
    ('20天底仓+8%回撤离场', lambda s: s['ret_dyn8'], None),
    ('20天底仓+10%回撤离场', lambda s: s['ret_dd10'], None),
    ('20天底仓+12%回撤离场', lambda s: s['ret_dd12'], None),
    ('20天底仓+15%回撤离场', lambda s: s['ret_dd15'], None),
]
print(f"{'策略':<26} {'信号数':>6} {'胜率':>8} {'均值':>8} {'中位':>8} {'平均持有':>10}")
print("-" * 75)
for label, fn, _ in strategies:
    rets = [fn(s) for s in all_signals]
    n, wr, mean, med = calc_stats(rets)
    if '底仓' in label:
        avg_exit = np.mean([s['exit_day'] for s in all_signals]) if '8%' in label else '-'
        print(f"{label:<26} {n:>6} {wr:>7.1%} {mean:>7.1%} {med:>7.1%} {avg_exit:>10}")
    else:
        print(f"{label:<26} {n:>6} {wr:>7.1%} {mean:>7.1%} {med:>7.1%} {'-':>10}")

# 按年份 × 策略
print("\n" + "=" * 90)
print("按年份 × 策略 胜率矩阵")
print("=" * 90)
years = sorted(set(s['buy_year'] for s in all_signals))
print(f"{'年份':<8}", end='')
for label, fn, _ in strategies:
    print(f" {label[:12]:>14}", end='')
print()
print("-" * (8 + 15 * len(strategies)))
for year in years:
    grp = [s for s in all_signals if s['buy_year'] == year]
    print(f"{year:<8}", end='')
    for label, fn, _ in strategies:
        rets = [fn(s) for s in grp]
        n, wr, mean, med = calc_stats(rets)
        print(f" {wr:>13.1%}", end='')
    print()

print("\n按年份 × 策略 均值矩阵")
print("=" * 90)
print(f"{'年份':<8}", end='')
for label, fn, _ in strategies:
    print(f" {label[:12]:>14}", end='')
print()
print("-" * (8 + 15 * len(strategies)))
for year in years:
    grp = [s for s in all_signals if s['buy_year'] == year]
    print(f"{year:<8}", end='')
    for label, fn, _ in strategies:
        rets = [fn(s) for s in grp]
        n, wr, mean, med = calc_stats(rets)
        print(f" {mean:>13.1%}", end='')
    print()

# 8%离场的离场日分布
print("\n" + "=" * 90)
print("8%回撤离场策略 — 离场时机分布")
print("=" * 90)
exit_days = [s['exit_day'] for s in all_signals]
print(f"平均离场: {np.mean(exit_days):.1f}天, 中位: {np.median(exit_days):.0f}天")
print(f"持有20天(触发/上限): {sum(1 for d in exit_days if d==20)}个")
print(f"持有21-40天: {sum(1 for d in exit_days if 21<=d<=40)}个")
print(f"持有41-60天: {sum(1 for d in exit_days if 41<=d<=60)}个")

print("\n" + "=" * 90)
print("回测完成")
print("=" * 90)
