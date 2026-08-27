# -*- coding: utf-8 -*-
"""
300只A股批量回测 — 验证黄金坑策略在全市场的真实胜率
"""
import sys
sys.path.insert(0, '.')
from golden_pit_v2_backtest import *
import numpy as np
import time

# 读取股票池
stocks = []
with open('stock_pool_300.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            code, name = line.split(',', 1)
            stocks.append((code, name))

print(f"股票池: {len(stocks)}只")
print(f"策略: v1黄金坑(250日回归) + 快启动≤5天 + 坑长≥8 + 固定60天持有")
print(f"时间: 2022-06 ~ 2026-08 (约1023个交易日)")
print("=" * 80)

# 批量回测
all_results = []          # 所有交易结果
by_year = {}              # 按买入年份统计
by_market = {}            # 按市场统计
success_count = 0
fail_count = 0
pit_count_total = 0
signal_count_total = 0

start_time = time.time()

for idx, (symbol, name) in enumerate(stocks):
    if (idx + 1) % 30 == 0:
        elapsed = time.time() - start_time
        print(f"  进度: {idx+1}/{len(stocks)} ({elapsed:.0f}s) "
              f"成功{success_count} 失败{fail_count} "
              f"坑{pit_count_total} 信号{signal_count_total}")

    df = fetch_kline_sina(symbol, datalen=1023)
    if df is None or len(df) < 310:
        fail_count += 1
        continue

    closes = df['close'].values.astype(float)
    volumes = df['volume'].values.astype(float)
    dates = df['date'].values
    n = len(closes)

    # v1黄金坑检测
    reg250, _ = compute_rolling_regression(closes, window=250)
    pits_v1 = detect_golden_pit_v1(closes, reg250)
    pit_count_total += len(pits_v1)

    # 市场分类
    if symbol.startswith('sh60'):
        market = '沪市主板'
    elif symbol.startswith('sz00'):
        market = '深市主板'
    elif symbol.startswith('sz30'):
        market = '创业板'
    elif symbol.startswith('sh688'):
        market = '科创板'
    else:
        market = '其他'

    # 快启动+坑长过滤 + 60天持有
    for pit in pits_v1:
        s, b, lch = pit
        if lch is None or lch + 60 >= n:
            continue
        if lch - b > 5:  # 快启动
            continue
        if b - s + 1 < 8:  # 坑长
            continue
        buy_px = closes[lch]
        sell_px = closes[min(lch + 60, n - 1)]
        ret = sell_px / buy_px - 1
        buy_year = pd.Timestamp(dates[lch]).year
        result = {
            'symbol': symbol, 'name': name, 'market': market,
            'buy_year': buy_year, 'buy_idx': lch, 'ret': ret,
            'pit_len': b - s + 1, 'launch_days': lch - b,
        }
        all_results.append(result)
        signal_count_total += 1

        # 按年份
        if buy_year not in by_year:
            by_year[buy_year] = []
        by_year[buy_year].append(ret)

        # 按市场
        if market not in by_market:
            by_market[market] = []
        by_market[market].append(ret)

    success_count += 1

elapsed = time.time() - start_time
print(f"\n完成: {success_count}只成功, {fail_count}只失败, 耗时{elapsed:.0f}s")
print(f"检测到坑总数: {pit_count_total}, 有效信号(快启动+坑长+60天): {signal_count_total}")

# ============================================================
# 汇总统计
# ============================================================
print("\n" + "=" * 80)
print("全市场汇总（300只A股，快启动≤5天+坑长≥8+60天持有）")
print("=" * 80)

if all_results:
    rets = [r['ret'] for r in all_results]
    wr = sum(1 for r in rets if r > 0) / len(rets)
    print(f"\n总信号数: {len(rets)}")
    print(f"胜率: {wr:.1%}")
    print(f"平均收益: {np.mean(rets):.1%}")
    print(f"中位收益: {np.median(rets):.1%}")
    print(f"最大收益: {np.max(rets):.1%}")
    print(f"最小收益: {np.min(rets):.1%}")
    print(f"盈利>30%占比: {sum(1 for r in rets if r > 0.3)/len(rets):.1%}")
    print(f"亏损>15%占比: {sum(1 for r in rets if r < -0.15)/len(rets):.1%}")

# 按年份
print(f"\n--- 按买入年份 ---")
print(f"{'年份':<8} {'信号数':>6} {'胜率':>8} {'均值':>8} {'中位':>8}")
print("-" * 45)
for year in sorted(by_year.keys()):
    rets = by_year[year]
    wr = sum(1 for r in rets if r > 0) / len(rets)
    print(f"{year:<8} {len(rets):>6} {wr:>7.1%} {np.mean(rets):>7.1%} {np.median(rets):>7.1%}")

# 按市场
print(f"\n--- 按市场板块 ---")
print(f"{'板块':<10} {'信号数':>6} {'胜率':>8} {'均值':>8} {'中位':>8}")
print("-" * 50)
for market in sorted(by_market.keys()):
    rets = by_market[market]
    wr = sum(1 for r in rets if r > 0) / len(rets)
    print(f"{market:<10} {len(rets):>6} {wr:>7.1%} {np.mean(rets):>7.1%} {np.median(rets):>7.1%}")

# 按收益分布
print(f"\n--- 收益分布 ---")
bins = [(-1, -0.3, '亏>30%'), (-0.3, -0.15, '亏15-30%'),
        (-0.15, 0, '亏0-15%'), (0, 0.15, '盈0-15%'),
        (0.15, 0.3, '盈15-30%'), (0.3, 1, '盈>30%')]
total = len(all_results)
for lo, hi, label in bins:
    cnt = sum(1 for r in all_results if lo <= r['ret'] < hi)
    print(f"  {label:<10}: {cnt:>4} ({cnt/total:.1%})")

# 胜率最高的股票TOP10
print(f"\n--- 胜率最高股票TOP10 (至少2个信号) ---")
stock_stats = {}
for r in all_results:
    key = (r['symbol'], r['name'])
    if key not in stock_stats:
        stock_stats[key] = []
    stock_stats[key].append(r['ret'])
top = [(k, v) for k, v in stock_stats.items() if len(v) >= 2]
top.sort(key=lambda x: (sum(1 for r in x[1] if r > 0) / len(x[1]), np.mean(x[1])), reverse=True)
for (sym, name), rets in top[:10]:
    wr = sum(1 for r in rets if r > 0) / len(rets)
    print(f"  {name}({sym}): n={len(rets)} 胜率={wr:.0%} 均值={np.mean(rets):.1%}")

print("\n" + "=" * 80)
print("回测完成")
print("=" * 80)
