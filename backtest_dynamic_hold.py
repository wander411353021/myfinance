# -*- coding: utf-8 -*-
"""
动态持有期回测 — 大盘多头60天持有 vs 空头20天持有
对比：纯20天、纯60天、动态
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

# 沪深300 + 60日线
df_hs300 = fetch_kline_sina('sh000300', datalen=1023)
hs300_close = df_hs300['close'].values.astype(float)
hs300_dates = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in df_hs300['date'].values]
hs300_ma60 = pd.Series(hs300_close).rolling(60).mean().values
hs300_date_idx = {d: i for i, d in enumerate(hs300_dates)}

def mkt_state(date_str):
    """返回 'bull'(多头) / 'bear'(空头) / None"""
    i = hs300_date_idx.get(date_str)
    if i is None or not np.isfinite(hs300_ma60[i]):
        return None
    return 'bull' if hs300_close[i] > hs300_ma60[i] else 'bear'

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
        state = mkt_state(buy_date)

        # 各种持有期的收益
        rets = {}
        for hd in [20, 60]:
            rets[hd] = closes[lch+hd] / buy_px - 1
        # 动态：多头60天，空头20天
        if state == 'bull':
            dyn_hd = 60
        elif state == 'bear':
            dyn_hd = 20
        else:
            dyn_hd = 30
        rets['dyn'] = closes[lch+dyn_hd] / buy_px - 1
        rets['dyn_hd'] = dyn_hd

        all_signals.append({
            'buy_date': buy_date, 'buy_year': buy_year, 'mkt_state': state,
            'rets': rets,
        })

    success_count += 1

elapsed = time.time() - start_time
print(f"\n完成: {success_count}只成功, 失败{fail_count}, 耗时{elapsed:.0f}s")
print(f"信号总数: {len(all_signals)}")

# 统计
def calc_stats(rets):
    if not rets:
        return (0, 0, 0, 0)
    return (len(rets), sum(1 for r in rets if r > 0)/len(rets), np.mean(rets), np.median(rets))

# 大盘状态分布
states = {}
for s in all_signals:
    st = s['mkt_state'] or 'unknown'
    states.setdefault(st, 0)
    states[st] += 1
print(f"\n信号大盘状态分布: {states}")

print("\n" + "=" * 90)
print("全周期对比：纯20天 vs 纯60天 vs 动态")
print("=" * 90)
print(f"{'策略':<20} {'信号数':>6} {'胜率':>8} {'均值':>8} {'中位':>8} {'平均持有':>10}")
print("-" * 70)
configs = [
    ('纯20天', lambda s: s['rets'][20], 20),
    ('纯60天', lambda s: s['rets'][60], 60),
    ('动态(多头60/空头20)', lambda s: s['rets']['dyn'], '动态'),
]
for label, fn, hd in configs:
    rets = [fn(s) for s in all_signals]
    n, wr, mean, med = calc_stats(rets)
    if hd == '动态':
        avg_hd = np.mean([s['rets']['dyn_hd'] for s in all_signals])
        print(f"{label:<20} {n:>6} {wr:>7.1%} {mean:>7.1%} {med:>7.1%} {avg_hd:>8.0f}天")
    else:
        print(f"{label:<20} {n:>6} {wr:>7.1%} {mean:>7.1%} {med:>7.1%} {hd:>8}天")

print("\n" + "=" * 90)
print("按年份对比")
print("=" * 90)
years = sorted(set(s['buy_year'] for s in all_signals))
print(f"{'年份':<8} {'20天胜率':>10} {'20天均值':>10} | {'60天胜率':>10} {'60天均值':>10} | {'动态胜率':>10} {'动态均值':>10}")
print("-" * 85)
for year in years:
    grp = [s for s in all_signals if s['buy_year'] == year]
    r20 = [s['rets'][20] for s in grp]
    r60 = [s['rets'][60] for s in grp]
    rd = [s['rets']['dyn'] for s in grp]
    n20, w20, m20, _ = calc_stats(r20)
    n60, w60, m60, _ = calc_stats(r60)
    nd, wd, md, _ = calc_stats(rd)
    print(f"{year:<8} {w20:>9.1%} {m20:>9.1%} | {w60:>9.1%} {m60:>9.1%} | {wd:>9.1%} {md:>9.1%}")

print("\n" + "=" * 90)
print("按大盘状态 × 持有期 表现（验证动态逻辑）")
print("=" * 90)
for st in ['bull', 'bear']:
    grp = [s for s in all_signals if s['mkt_state'] == st]
    print(f"\n--- 大盘{st}状态（信号{len(grp)}个）---")
    print(f"{'持有期':<10} {'信号数':>6} {'胜率':>8} {'均值':>8} {'中位':>8}")
    for hd in [20, 60]:
        rets = [s['rets'][hd] for s in grp]
        n, wr, mean, med = calc_stats(rets)
        print(f"{hd}天{'':<6} {n:>6} {wr:>7.1%} {mean:>7.1%} {med:>7.1%}")

print("\n" + "=" * 90)
print("回测完成")
print("=" * 90)
