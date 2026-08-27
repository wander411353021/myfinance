# -*- coding: utf-8 -*-
"""
最终组合验证 — 大盘状态 × 持有期（穿越牛熊）
组合逻辑：
  大盘多头(bull) → 持有60天吃趋势（2024模式）
  大盘空头(bear) → 持有20天快进快出（2026保护）
对比：纯20天、纯60天、评分组合、大盘动态、大盘动态+评分
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

# 沪深300 + 60日线
df_hs300 = fetch_kline_sina('sh000300', datalen=1023)
hs300_close = df_hs300['close'].values.astype(float)
hs300_dates = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in df_hs300['date'].values]
hs300_ma60 = pd.Series(hs300_close).rolling(60).mean().values
hs300_date_idx = {d: i for i, d in enumerate(hs300_dates)}

def mkt_state(date_str):
    i = hs300_date_idx.get(date_str)
    if i is None or not np.isfinite(hs300_ma60[i]):
        return None
    return 'bull' if hs300_close[i] > hs300_ma60[i] else 'bear'

def hs300_ret(date_str, days):
    i = hs300_date_idx.get(date_str)
    if i is None or i < days:
        return None
    return hs300_close[i] / hs300_close[i-days] - 1

# 评分函数
def score_signal(closes, volumes, reg250, s, b, lch, rs20):
    score = 0.0
    vol_pit = np.mean(volumes[s:b+1])
    vol_pre = np.mean(volumes[max(0, s-20):s]) if s >= 20 else vol_pit
    if vol_pre > 0:
        shrink = vol_pit / vol_pre
        if shrink <= 0.3: score += 25
        elif shrink <= 0.7: score += 25 - (shrink-0.3)/0.4*15
        elif shrink <= 1.0: score += 10 - (shrink-0.7)/0.3*10
    vol_lch = volumes[lch]
    vol_lch_pre = np.mean(volumes[max(0, lch-5):lch]) if lch >= 5 else vol_lch
    if vol_lch_pre > 0:
        fill = vol_lch / vol_lch_pre
        if fill >= 2.0: score += 25
        elif fill >= 1.2: score += 10 + (fill-1.2)/0.8*15
        elif fill >= 0.8: score += 5 + (fill-0.8)/0.4*5
    pre_high = np.max(closes[max(0, b-60):b]) if b >= 60 else closes[max(0,b-20):b+1].max()
    if pre_high > 0:
        depth = 1 - closes[b] / pre_high
        if 0.15 <= depth <= 0.50: score += 15
        elif 0.05 <= depth < 0.15: score += 8
        elif depth > 0.50: score += 5
    if rs20 is not None:
        if rs20 > 0.10: score += 15
        elif rs20 > 0.05: score += 12
        elif rs20 > 0: score += 8
        elif rs20 > -0.05: score += 4
    if b >= 20 and np.isfinite(reg250[b]) and np.isfinite(reg250[b-20]):
        slope = reg250[b]/reg250[b-20]-1
        if slope > 0.02: score += 10
        elif slope > 0: score += 6
        elif slope > -0.02: score += 3
    vol_bottom = np.mean(volumes[max(0,b-2):b+3])
    vol_pre2 = np.mean(volumes[max(0,s-20):s]) if s >= 20 else vol_bottom
    if vol_pre2 > 0:
        bs = vol_bottom/vol_pre2
        if bs < 0.5: score += 10
        elif bs < 0.8: score += 6
        elif bs < 1.0: score += 3
    return score

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
        time.sleep(random.uniform(0.05, 0.15))
        continue

    closes = df['close'].values.astype(float)
    volumes = df['volume'].values.astype(float)
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
        state = mkt_state(buy_date)

        rs20 = None
        if b >= 20:
            stock_ret20 = closes[b] / closes[b-20] - 1
            mkt_ret20 = hs300_ret(buy_date, 20)
            if mkt_ret20 is not None:
                rs20 = stock_ret20 - mkt_ret20

        score = score_signal(closes, volumes, reg250, s, b, lch, rs20)

        all_signals.append({
            'buy_date': buy_date, 'buy_year': buy_year, 'score': score,
            'state': state,
            'ret20': closes[lch+20]/buy_px - 1,
            'ret60': closes[lch+60]/buy_px - 1,
        })

    success_count += 1
    time.sleep(random.uniform(0.02, 0.08))

elapsed = time.time() - start_time
print(f"\n完成: {success_count}只成功, 失败{fail_count}, 耗时{elapsed:.0f}s")
print(f"信号总数: {len(all_signals)}")

def calc_stats(rets):
    if not rets:
        return (0, 0, 0, 0)
    return (len(rets), sum(1 for r in rets if r > 0)/len(rets), np.mean(rets), np.median(rets))

def mk_strategies():
    return {
        '纯20天': lambda s: s['ret20'],
        '纯60天': lambda s: s['ret60'],
        '组合评分(45分档)': lambda s: s['ret60'] if s['score'] >= 45 else s['ret20'],
        '大盘动态(多头60/空头20)': lambda s: s['ret60'] if s['state'] == 'bull' else s['ret20'],
        '大盘+评分(多头60高/空头20)': lambda s: s['ret60'] if (s['state'] == 'bull' and s['score'] >= 40) else s['ret20'],
    }

strategies = mk_strategies()

# 大盘状态分布
from collections import Counter
print(f"\n大盘状态分布: {dict(Counter(s['state'] or 'unknown' for s in all_signals))}")

print("\n" + "=" * 90)
print("最终方案对比（全周期）")
print("=" * 90)
print(f"{'策略':<30} {'信号数':>6} {'胜率':>8} {'均值':>8} {'中位':>8}")
print("-" * 70)
for label, fn in strategies.items():
    rets = [fn(s) for s in all_signals]
    n, wr, mean, med = calc_stats(rets)
    print(f"{label:<30} {n:>6} {wr:>7.1%} {mean:>7.1%} {med:>7.1%}")

print("\n" + "=" * 90)
print("按年份 × 策略 胜率矩阵")
print("=" * 90)
years = sorted(set(s['buy_year'] for s in all_signals))
print(f"{'年份':<8}", end='')
for label, fn in strategies.items():
    print(f" {label[:14]:>16}", end='')
print()
print("-" * (8 + 17 * len(strategies)))
for year in years:
    grp = [s for s in all_signals if s['buy_year'] == year]
    print(f"{year:<8}", end='')
    for label, fn in strategies.items():
        rets = [fn(s) for s in grp]
        n, wr, mean, med = calc_stats(rets)
        print(f" {wr:>15.1%}", end='')
    print()

print("\n按年份 × 策略 均值矩阵")
print("=" * 90)
print(f"{'年份':<8}", end='')
for label, fn in strategies.items():
    print(f" {label[:14]:>16}", end='')
print()
print("-" * (8 + 17 * len(strategies)))
for year in years:
    grp = [s for s in all_signals if s['buy_year'] == year]
    print(f"{year:<8}", end='')
    for label, fn in strategies.items():
        rets = [fn(s) for s in grp]
        n, wr, mean, med = calc_stats(rets)
        print(f" {mean:>15.1%}", end='')
    print()

# 大盘动态策略在大盘多头/空头下的明细
print("\n" + "=" * 90)
print("大盘动态策略明细（多头60天 / 空头20天）")
print("=" * 90)
for st in ['bull', 'bear']:
    grp = [s for s in all_signals if s['state'] == st]
    print(f"\n--- 大盘{st}状态（{len(grp)}信号）---")
    print(f"  20天: ", end='')
    n, wr, mean, med = calc_stats([s['ret20'] for s in grp])
    print(f"胜率{wr:.1%} 均值{mean:.1%} | 60天: ", end='')
    n, wr, mean, med = calc_stats([s['ret60'] for s in grp])
    print(f"胜率{wr:.1%} 均值{mean:.1%}")

print("\n" + "=" * 90)
print("回测完成")
print("=" * 90)
