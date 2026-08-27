# -*- coding: utf-8 -*-
"""
300只A股回测 — 大盘过滤 + 个股独立走势豁免（分层过滤）
核心思路：不平杀，大盘空头时保留独立强势股的黄金坑

方案对比：
  A. 无过滤（基准）
  B. 大盘60日线硬过滤（之前的做法，会误杀独立强势股）
  C. 分层-个股60日线豁免：大盘空头时，若个股出坑日收盘 > 自身60日线 → 保留
  D. 分层-个股250日回归线豁免：大盘空头时，若个股出坑日收盘 > 自身250日回归线 → 保留
  E. 分层-相对强度豁免：大盘空头时，若个股近20日涨幅 > 沪深300近20日涨幅（跑赢大盘）→ 保留
"""
import sys
sys.path.insert(0, '.')
from golden_pit_v2_backtest import *
import numpy as np
import time

stocks = []
with open('stock_pool_300.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            code, name = line.split(',', 1)
            stocks.append((code, name))

print(f"股票池: {len(stocks)}只")
print("策略: v1黄金坑 + 快启动≤5天 + 坑长≥8 + 60天持有")
print("=" * 80)

# 1. 沪深300大盘数据 + 60日线 + 20日动量
print("获取沪深300大盘数据...")
df_hs300 = fetch_kline_sina('sh000300', datalen=1023)
hs300_close = df_hs300['close'].values.astype(float)
hs300_dates = df_hs300['date'].values
hs300_ma60 = pd.Series(hs300_close).rolling(60).mean().values
hs300_dates_str = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in hs300_dates]
hs300_date_idx = {d: i for i, d in enumerate(hs300_dates_str)}

# 大盘60日多头字典
mkt_bull60 = {}
for i, d in enumerate(hs300_dates_str):
    if np.isfinite(hs300_ma60[i]):
        mkt_bull60[d] = bool(hs300_close[i] > hs300_ma60[i])
    else:
        mkt_bull60[d] = None

# 大盘20日收益（用于相对强度）
def hs300_ret20(date_str):
    i = hs300_date_idx.get(date_str)
    if i is None or i < 20:
        return None
    return hs300_close[i] / hs300_close[i-20] - 1

print("  沪深300 60日线多头天数:", sum(1 for v in mkt_bull60.values() if v is True))
print()

# 2. 收集所有信号（附带个股独立强度指标）
all_signals = []
success_count = 0
fail_count = 0
start_time = time.time()

for idx, (symbol, name) in enumerate(stocks):
    if (idx + 1) % 50 == 0:
        elapsed = time.time() - start_time
        print(f"  进度: {idx+1}/{len(stocks)} ({elapsed:.0f}s) 信号{len(all_signals)}")

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

    # 个股自身指标
    reg250, _ = compute_rolling_regression(closes, window=250)
    stock_ma60 = pd.Series(closes).rolling(60).mean().values

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

        # 个股独立强度指标
        above_ma60 = bool(closes[lch] > stock_ma60[lch]) if np.isfinite(stock_ma60[lch]) else False
        above_reg250 = bool(closes[lch] > reg250[lch]) if np.isfinite(reg250[lch]) else False
        stock_ret20 = closes[lch] / closes[lch-20] - 1 if lch >= 20 else None
        mkt_ret20 = hs300_ret20(buy_date)
        beat_mkt = False
        if stock_ret20 is not None and mkt_ret20 is not None:
            beat_mkt = bool(stock_ret20 > mkt_ret20)

        all_signals.append({
            'symbol': symbol, 'name': name, 'market': market,
            'buy_date': buy_date, 'buy_year': buy_year, 'ret': ret,
            'above_ma60': above_ma60,
            'above_reg250': above_reg250,
            'beat_mkt': beat_mkt,
            'stock_ret20': stock_ret20,
            'mkt_ret20': mkt_ret20,
        })

    success_count += 1

elapsed = time.time() - start_time
print(f"\n完成: {success_count}只成功, 失败{fail_count}, 耗时{elapsed:.0f}s")
print(f"原始信号: {len(all_signals)}")

# 3. 统计函数
def calc_stats(signals, label=''):
    if not signals:
        return {'label': label, 'n': 0, 'wr': 0, 'mean': 0, 'median': 0, 'max': 0, 'min': 0}
    rets = [s['ret'] for s in signals]
    return {
        'label': label, 'n': len(rets),
        'wr': sum(1 for r in rets if r > 0) / len(rets),
        'mean': np.mean(rets), 'median': np.median(rets),
        'max': np.max(rets), 'min': np.min(rets),
    }

# 4. 各策略
def strategy_c(sig):
    """C: 大盘60日多头全保留；空头时仅保留个股>自身60日线"""
    b = mkt_bull60.get(sig['buy_date'], None)
    if b is True:
        return True
    if b is False:
        return sig['above_ma60']
    return True  # 数据不足不拦

def strategy_d(sig):
    """D: 大盘60日多头全保留；空头时仅保留个股>自身250日回归线"""
    b = mkt_bull60.get(sig['buy_date'], None)
    if b is True:
        return True
    if b is False:
        return sig['above_reg250']
    return True

def strategy_e(sig):
    """E: 大盘60日多头全保留；空头时仅保留跑赢大盘(20日)的个股"""
    b = mkt_bull60.get(sig['buy_date'], None)
    if b is True:
        return True
    if b is False:
        return sig['beat_mkt']
    return True

strategies = {
    'A: 无过滤(基准)': lambda s: True,
    'B: 大盘60日硬过滤': lambda s: mkt_bull60.get(s['buy_date'], None) is True,
    'C: 分层-个股60日线豁免': strategy_c,
    'D: 分层-个股250日回归线豁免': strategy_d,
    'E: 分层-跑赢大盘豁免': strategy_e,
}

print("\n" + "=" * 90)
print("策略对比（300只A股，60天持有）")
print("=" * 90)
print(f"\n{'策略':<28} {'信号数':>6} {'保留率':>8} {'胜率':>8} {'胜率变化':>10} {'均值':>8} {'均值变化':>10} {'中位':>8}")
print("-" * 100)

s_raw = calc_stats(all_signals, 'A')
print(f"{'A: 无过滤(基准)':<28} {s_raw['n']:>6} {'100.0%':>8} {s_raw['wr']:>7.1%} {'基准':>10} "
      f"{s_raw['mean']:>7.1%} {'基准':>10} {s_raw['median']:>7.1%}")

for label in ['B: 大盘60日硬过滤', 'C: 分层-个股60日线豁免', 'D: 分层-个股250日回归线豁免', 'E: 分层-跑赢大盘豁免']:
    fn = strategies[label]
    filtered = [s for s in all_signals if fn(s)]
    s = calc_stats(filtered, label)
    wr_diff = (s['wr'] - s_raw['wr']) * 100
    mean_diff = (s['mean'] - s_raw['mean']) * 100
    keep = s['n'] / s_raw['n'] * 100 if s_raw['n'] else 0
    print(f"{label:<28} {s['n']:>6} {keep:>7.1f}% {s['wr']:>7.1%} "
          f"{'+' if wr_diff>=0 else ''}{wr_diff:>8.1f}pp {s['mean']:>7.1%} "
          f"{'+' if mean_diff>=0 else ''}{mean_diff:>8.1f}pp {s['median']:>7.1%}")

# 5. 最优策略按年份分析
print("\n" + "=" * 90)
print("最优策略按年份分析")
print("=" * 90)

# 找出胜率最高且信号数>=50的
best_label = None
best_s = None
for label in ['C: 分层-个股60日线豁免', 'D: 分层-个股250日回归线豁免', 'E: 分层-跑赢大盘豁免']:
    fn = strategies[label]
    filtered = [s for s in all_signals if fn(s)]
    s = calc_stats(filtered, label)
    if s['n'] >= 50 and (best_s is None or s['wr'] > best_s['wr']):
        best_s = s
        best_label = label

if best_label:
    print(f"\n最优策略: {best_label}")
    fn = strategies[best_label]
    filtered = [s for s in all_signals if fn(s)]

    by_year_raw = {}
    by_year_fil = {}
    for s in all_signals:
        by_year_raw.setdefault(s['buy_year'], []).append(s['ret'])
    for s in filtered:
        by_year_fil.setdefault(s['buy_year'], []).append(s['ret'])

    print(f"\n{'年份':<8} {'原始n':>6} {'原始胜率':>10} {'原始均值':>10} | {'过滤n':>6} {'过滤胜率':>10} {'过滤均值':>10} | {'胜率变化':>10}")
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

    # 被过滤信号分析
    filtered_out = [s for s in all_signals if not fn(s)]
    print(f"\n--- 被过滤信号分析（{best_label}）---")
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
