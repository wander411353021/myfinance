# -*- coding: utf-8 -*-
"""
300只A股回测 + 多均线窗口大盘过滤对比
测试 20/60/90/120/250 日均线作为沪深300趋势门控的效果
"""
import sys
sys.path.insert(0, '.')
from golden_pit_v2_backtest import *
import numpy as np
import time

# 读取300只股票池
stocks = []
with open('stock_pool_300.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            code, name = line.split(',', 1)
            stocks.append((code, name))

print(f"股票池: {len(stocks)}只")
print(f"策略: v1黄金坑(250日回归) + 快启动≤5天 + 坑长≥8 + 60天持有")
print(f"对比: 沪深300在 N 日均线之上才开仓 (N=20/60/90/120/250)")
print("=" * 80)

# 1. 获取沪深300大盘数据
print("获取沪深300大盘数据...")
df_hs300 = fetch_kline_sina('sh000300', datalen=1023)
if df_hs300 is None:
    print("ERROR: 无法获取沪深300数据")
    sys.exit(1)

hs300_close = df_hs300['close'].values.astype(float)
hs300_dates = df_hs300['date'].values

# 预计算各均线窗口的多头字典
MA_WINDOWS = [20, 60, 90, 120, 250]
bull_dicts = {}
for win in MA_WINDOWS:
    ma = pd.Series(hs300_close).rolling(win).mean().values
    bull = {}
    for i in range(len(hs300_dates)):
        d = pd.Timestamp(hs300_dates[i]).strftime('%Y-%m-%d')
        if np.isfinite(ma[i]):
            bull[d] = bool(hs300_close[i] > ma[i])
        else:
            bull[d] = None
    bull_dicts[win] = bull
    true_cnt = sum(1 for v in bull.values() if v is True)
    false_cnt = sum(1 for v in bull.values() if v is False)
    print(f"  {win}日均线: 多头{true_cnt}天, 空头{false_cnt}天")

print()

# 2. 批量回测（一次性收集所有信号，再分别应用各均线过滤）
all_signals = []  # 所有原始信号
success_count = 0
fail_count = 0
start_time = time.time()

for idx, (symbol, name) in enumerate(stocks):
    if (idx + 1) % 50 == 0:
        elapsed = time.time() - start_time
        print(f"  进度: {idx+1}/{len(stocks)} ({elapsed:.0f}s) 成功{success_count} 失败{fail_count} 信号{len(all_signals)}")

    df = fetch_kline_sina(symbol, datalen=1023)
    if df is None or len(df) < 310:
        fail_count += 1
        continue

    closes = df['close'].values.astype(float)
    volumes = df['volume'].values.astype(float)
    dates = df['date'].values
    n = len(closes)

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
        sell_px = closes[min(lch + 60, n - 1)]
        ret = sell_px / buy_px - 1
        buy_date = pd.Timestamp(dates[lch]).strftime('%Y-%m-%d')
        buy_year = pd.Timestamp(dates[lch]).year

        all_signals.append({
            'symbol': symbol, 'name': name, 'market': market,
            'buy_date': buy_date, 'buy_year': buy_year, 'ret': ret,
        })

    success_count += 1

elapsed = time.time() - start_time
print(f"\n完成: {success_count}只成功, {fail_count}只失败, 耗时{elapsed:.0f}s")
print(f"原始信号总数: {len(all_signals)}")

# 3. 统计函数
def calc_stats(signals):
    if not signals:
        return {'n': 0, 'wr': 0, 'mean': 0, 'median': 0, 'max': 0, 'min': 0}
    rets = [s['ret'] for s in signals]
    return {
        'n': len(rets),
        'wr': sum(1 for r in rets if r > 0) / len(rets),
        'mean': np.mean(rets),
        'median': np.median(rets),
        'max': np.max(rets),
        'min': np.min(rets),
    }

# 4. 各均线窗口过滤效果对比
print("\n" + "=" * 90)
print("各均线窗口大盘过滤效果对比（沪深300收盘价 > N日均线才开仓）")
print("=" * 90)
print(f"\n{'均线窗口':<10} {'信号数':>6} {'保留率':>8} {'胜率':>8} {'胜率变化':>10} "
      f"{'均值':>8} {'均值变化':>10} {'中位':>8}")
print("-" * 90)

s_raw = calc_stats(all_signals)
print(f"{'无过滤':<10} {s_raw['n']:>6} {'100.0%':>8} {s_raw['wr']:>7.1%} {'基准':>10} "
      f"{s_raw['mean']:>7.1%} {'基准':>10} {s_raw['median']:>7.1%}")

best_win = None
best_wr = 0
for win in MA_WINDOWS:
    bull = bull_dicts[win]
    filtered = [s for s in all_signals if bull.get(s['buy_date'], None) is True]
    s = calc_stats(filtered)
    wr_diff = (s['wr'] - s_raw['wr']) * 100
    mean_diff = (s['mean'] - s_raw['mean']) * 100
    keep_rate = s['n'] / s_raw['n'] * 100 if s_raw['n'] > 0 else 0
    print(f"{win}日{'':<7} {s['n']:>6} {keep_rate:>7.1f}% {s['wr']:>7.1%} "
          f"{'+' if wr_diff>=0 else ''}{wr_diff:>8.1f}pp {s['mean']:>7.1%} "
          f"{'+' if mean_diff>=0 else ''}{mean_diff:>8.1f}pp {s['median']:>7.1%}")
    if s['n'] >= 20 and s['wr'] > best_wr:
        best_wr = s['wr']
        best_win = win

if best_win:
    print(f"\n→ 最优均线窗口: {best_win}日 (胜率{best_wr:.1%}, 信号数≥20)")

# 5. 最优窗口按年份详细分析
if best_win:
    print(f"\n" + "=" * 90)
    print(f"最优窗口({best_win}日均线)按年份详细分析")
    print("=" * 90)
    bull = bull_dicts[best_win]
    filtered = [s for s in all_signals if bull.get(s['buy_date'], None) is True]

    by_year_raw = {}
    by_year_fil = {}
    for s in all_signals:
        y = s['buy_year']
        by_year_raw.setdefault(y, []).append(s['ret'])
    for s in filtered:
        y = s['buy_year']
        by_year_fil.setdefault(y, []).append(s['ret'])

    print(f"\n{'年份':<8} {'原始n':>6} {'原始胜率':>10} {'原始均值':>10} | "
          f"{'过滤n':>6} {'过滤胜率':>10} {'过滤均值':>10} | {'胜率变化':>10}")
    print("-" * 95)
    for year in sorted(by_year_raw.keys()):
        raw_rets = by_year_raw.get(year, [])
        fil_rets = by_year_fil.get(year, [])
        raw_wr = sum(1 for r in raw_rets if r > 0) / len(raw_rets) if raw_rets else 0
        fil_wr = sum(1 for r in fil_rets if r > 0) / len(fil_rets) if fil_rets else 0
        raw_mean = np.mean(raw_rets) if raw_rets else 0
        fil_mean = np.mean(fil_rets) if fil_rets else 0
        wr_diff = (fil_wr - raw_wr) * 100 if raw_rets and fil_rets else 0
        print(f"{year:<8} {len(raw_rets):>6} {raw_wr:>9.1%} {raw_mean:>9.1%} | "
              f"{len(fil_rets):>6} {fil_wr:>9.1%} {fil_mean:>9.1%} | "
              f"{'+' if wr_diff>=0 else ''}{wr_diff:>9.1f}pp")

    # 按板块
    print(f"\n--- 按市场板块 ---")
    by_mkt_raw = {}
    by_mkt_fil = {}
    for s in all_signals:
        by_mkt_raw.setdefault(s['market'], []).append(s['ret'])
    for s in filtered:
        by_mkt_fil.setdefault(s['market'], []).append(s['ret'])
    print(f"{'板块':<10} {'原始n':>6} {'原始胜率':>10} {'原始均值':>10} | {'过滤n':>6} {'过滤胜率':>10} {'过滤均值':>10}")
    print("-" * 80)
    for mkt in sorted(by_mkt_raw.keys()):
        raw_rets = by_mkt_raw.get(mkt, [])
        fil_rets = by_mkt_fil.get(mkt, [])
        raw_wr = sum(1 for r in raw_rets if r > 0) / len(raw_rets) if raw_rets else 0
        fil_wr = sum(1 for r in fil_rets if r > 0) / len(fil_rets) if fil_rets else 0
        raw_mean = np.mean(raw_rets) if raw_rets else 0
        fil_mean = np.mean(fil_rets) if fil_rets else 0
        print(f"{mkt:<10} {len(raw_rets):>6} {raw_wr:>9.1%} {raw_mean:>9.1%} | "
              f"{len(fil_rets):>6} {fil_wr:>9.1%} {fil_mean:>9.1%}")

# 6. 被过滤信号分析（用最优窗口）
if best_win:
    print(f"\n--- 被{best_win}日均线过滤掉的信号分析 ---")
    bull = bull_dicts[best_win]
    filtered_out = [s for s in all_signals if bull.get(s['buy_date'], None) is False]
    if filtered_out:
        rets = [s['ret'] for s in filtered_out]
        wr = sum(1 for r in rets if r > 0) / len(rets)
        print(f"  被过滤信号数: {len(filtered_out)}个")
        print(f"  被过滤信号胜率: {wr:.1%}")
        print(f"  被过滤信号均值: {np.mean(rets):.1%}")
        print(f"  被过滤信号中位: {np.median(rets):.1%}")

print("\n" + "=" * 90)
print("回测完成")
print("=" * 90)
