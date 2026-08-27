# -*- coding: utf-8 -*-
"""
2026年专项分析 — 为什么胜率只有33%？
1. 2026年大盘环境（沪深300走势）
2. 2026年信号按评分分组表现
3. 2026年按持有期敏感性（10/20/30/45/60天）
4. 2026年按买入月份分布
5. 成功 vs 失败信号特征对比
6. 不同评分层在2026年的表现（高分是否在弱市也有效）
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

# 1. 沪深300 2026年走势
df_hs300 = fetch_kline_sina('sh000300', datalen=1023)
hs300_close = df_hs300['close'].values.astype(float)
hs300_dates = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in df_hs300['date'].values]
hs300_ma60 = pd.Series(hs300_close).rolling(60).mean().values

print("\n" + "=" * 90)
print("1. 沪深300 2026年环境")
print("=" * 90)
idx_2026 = [i for i, d in enumerate(hs300_dates) if d >= '2026-01-01']
if idx_2026:
    start = idx_2026[0]
    print(f"  2026-01-01 收盘: {hs300_close[start]:.1f}")
    # 每月收盘
    cur_month = None
    for i in idx_2026:
        m = hs300_dates[i][:7]
        if m != cur_month:
            if cur_month:
                prev_idx = idx_2026[idx_2026.index(i) - 1] if i != idx_2026[0] else start
                print(f"  {hs300_dates[i][:7]} 收盘: {hs300_close[i]:.1f}  月末vs2025末: {(hs300_close[i]/hs300_close[start]-1)*100:+.1f}%")
            cur_month = m
    # 60日线状态
    bull_days = sum(1 for i in idx_2026 if np.isfinite(hs300_ma60[i]) and hs300_close[i] > hs300_ma60[i])
    print(f"  2026年内60日线多头天数: {bull_days}/{len(idx_2026)} ({bull_days/len(idx_2026)*100:.0f}%)")

# 2-6. 全市场回测，收集2026信号详细数据
all_signals_2026 = []
success_count = 0
start_time = time.time()

for idx, (symbol, name) in enumerate(stocks):
    if (idx + 1) % 200 == 0:
        print(f"  进度: {idx+1}/{len(stocks)} ({time.time()-start_time:.0f}s)")

    df = fetch_kline_sina(symbol, datalen=1023)
    if df is None or len(df) < 310:
        continue

    closes = df['close'].values.astype(float)
    volumes = df['volume'].values.astype(float)
    dates = df['date'].values
    n = len(closes)
    buy_dates = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in dates]

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
        buy_date = buy_dates[lch]
        if not buy_date.startswith('2026'):
            continue

        buy_px = closes[lch]
        # 多持有期
        rets_hold = {}
        for hd in [10, 20, 30, 45, 60]:
            if lch + hd < n:
                rets_hold[hd] = closes[lch+hd] / buy_px - 1
            else:
                rets_hold[hd] = closes[n-1] / buy_px - 1

        # 评分（简化：用6维度）
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
        if b >= 20:
            slope = reg250[b]/reg250[b-20]-1 if np.isfinite(reg250[b]) and np.isfinite(reg250[b-20]) else 0
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

        # 大盘60日状态
        mkt_i = hs300_date_idx_2026.get(buy_date) if 'hs300_date_idx_2026' in globals() else None

        all_signals_2026.append({
            'symbol': symbol, 'name': name, 'buy_date': buy_date,
            'buy_month': buy_date[:7], 'score': score,
            'depth': 1 - closes[b]/pre_high if pre_high > 0 else 0,
            'rets': rets_hold, 'ret60': rets_hold[60],
        })
    success_count += 1

print(f"\n完成: {success_count}只, 2026年信号{len(all_signals_2026)}个")

def stats(rets):
    if not rets:
        return (0, 0, 0)
    return (len(rets), sum(1 for r in rets if r > 0)/len(rets), np.mean(rets))

# 2. 按评分分组
print("\n" + "=" * 90)
print("2. 2026年信号按评分分组（持有60天）")
print("=" * 90)
print(f"{'评分区间':<12} {'信号数':>6} {'胜率':>8} {'均值':>8} {'中位':>8}")
print("-" * 55)
for lo, hi, label in [(0,30,'0-30分'), (30,40,'30-40分'), (40,50,'40-50分'), (50,100,'50分+')]:
    grp = [s for s in all_signals_2026 if lo <= s['score'] < hi]
    if grp:
        n, wr, mean = stats([s['ret60'] for s in grp])
        print(f"{label:<12} {n:>6} {wr:>7.1%} {mean:>7.1%} {np.median([s['ret60'] for s in grp]):>7.1%}")

# 3. 按持有期
print("\n" + "=" * 90)
print("3. 2026年按持有期敏感性")
print("=" * 90)
print(f"{'持有期':<10} {'信号数':>6} {'胜率':>8} {'均值':>8} {'中位':>8}")
print("-" * 50)
for hd in [10, 20, 30, 45, 60]:
    rets = [s['rets'][hd] for s in all_signals_2026]
    n, wr, mean = stats(rets)
    print(f"{hd}天{'':<6} {n:>6} {wr:>7.1%} {mean:>7.1%} {np.median(rets):>7.1%}")

# 4. 按买入月份
print("\n" + "=" * 90)
print("4. 2026年按买入月份分布")
print("=" * 90)
print(f"{'月份':<10} {'信号数':>6} {'胜率':>8} {'均值':>8} {'中位':>8}")
print("-" * 50)
by_month = {}
for s in all_signals_2026:
    by_month.setdefault(s['buy_month'], []).append(s['ret60'])
for m in sorted(by_month.keys()):
    rets = by_month[m]
    n, wr, mean = stats(rets)
    print(f"{m:<10} {n:>6} {wr:>7.1%} {mean:>7.1%} {np.median(rets):>7.1%}")

# 5. 成功 vs 失败特征对比
print("\n" + "=" * 90)
print("5. 2026年成功 vs 失败信号特征对比")
print("=" * 90)
succ = [s for s in all_signals_2026 if s['ret60'] > 0]
fail = [s for s in all_signals_2026 if s['ret60'] <= 0]
print(f"\n成功: {len(succ)}个, 失败: {len(fail)}个")
print(f"{'特征':<15} {'成功组':>12} {'失败组':>12}")
print("-" * 45)
for key, label in [('score', '平均评分'), ('depth', '平均坑深')]:
    sv = np.mean([s[key] for s in succ]) if succ else 0
    fv = np.mean([s[key] for s in fail]) if fail else 0
    print(f"{label:<15} {sv:>12.1f} {fv:>12.1f}")

# 6. 成功失败的时间分布
print(f"\n--- 成功/失败的时间分布 ---")
print(f"{'月份':<10} {'成功数':>6} {'失败数':>6} {'胜率':>8}")
print("-" * 40)
for m in sorted(by_month.keys()):
    s_cnt = sum(1 for s in all_signals_2026 if s['buy_month']==m and s['ret60']>0)
    f_cnt = sum(1 for s in all_signals_2026 if s['buy_month']==m and s['ret60']<=0)
    total = s_cnt + f_cnt
    print(f"{m:<10} {s_cnt:>6} {f_cnt:>6} {s_cnt/total:>7.1%}" if total else f"{m:<10} {0:>6} {0:>6}")

# 7. 高分(40+)在2026年的表现 vs 全历史
print("\n" + "=" * 90)
print("7. 2026年 40分以上信号的表现")
print("=" * 90)
grp40 = [s for s in all_signals_2026 if s['score'] >= 40]
if grp40:
    n, wr, mean = stats([s['ret60'] for s in grp40])
    print(f"  40分+信号: {n}个, 胜率{wr:.1%}, 均值{mean:.1%}")
    print(f"  中位: {np.median([s['ret60'] for s in grp40]):.1%}")
else:
    print("  2026年无40分+信号")

print("\n" + "=" * 90)
print("分析完成")
print("=" * 90)
