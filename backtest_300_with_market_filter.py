# -*- coding: utf-8 -*-
"""
300只A股回测 + 沪深300大盘趋势过滤对比
过滤规则：个股出坑日买入时，沪深300收盘价必须在250日均线之上（多头市场）
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
print(f"大盘过滤: 沪深300收盘价 > 250日均线（多头市场才开仓）")
print("=" * 80)

# 1. 获取沪深300大盘数据（新浪接口）
print("获取沪深300大盘数据...")
df_hs300 = fetch_kline_sina('sh000300', datalen=1023)
if df_hs300 is None:
    print("ERROR: 无法获取沪深300数据")
    sys.exit(1)

hs300_close = df_hs300['close'].values.astype(float)
hs300_dates = df_hs300['date'].values
hs300_ma250 = pd.Series(hs300_close).rolling(250).mean().values

# 构建日期→是否多头的映射
hs300_bull = {}
for i in range(len(hs300_dates)):
    d = pd.Timestamp(hs300_dates[i]).strftime('%Y-%m-%d')
    if np.isfinite(hs300_ma250[i]):
        hs300_bull[d] = bool(hs300_close[i] > hs300_ma250[i])
    else:
        hs300_bull[d] = None  # 数据不足，无法判断

bull_days = sum(1 for v in hs300_bull.values() if v is True)
bear_days = sum(1 for v in hs300_bull.values() if v is False)
unknown_days = sum(1 for v in hs300_bull.values() if v is None)
print(f"  沪深300: {len(hs300_dates)}天, 多头{bull_days}天, 空头{bear_days}天, 数据不足{unknown_days}天")
print(f"  最新: {hs300_dates[-1]} 收盘={hs300_close[-1]:.2f} MA250={hs300_ma250[-1]:.2f} "
      f"{'多头' if hs300_close[-1] > hs300_ma250[-1] else '空头'}")
print()

# 2. 批量回测
all_results_raw = []       # 无大盘过滤
all_results_filtered = []  # 有大盘过滤
by_year_raw = {}
by_year_filtered = {}
by_market_raw = {}
by_market_filtered = {}

success_count = 0
fail_count = 0
start_time = time.time()

for idx, (symbol, name) in enumerate(stocks):
    if (idx + 1) % 50 == 0:
        elapsed = time.time() - start_time
        print(f"  进度: {idx+1}/{len(stocks)} ({elapsed:.0f}s) "
              f"成功{success_count} 失败{fail_count} "
              f"原始信号{len(all_results_raw)} 过滤后{len(all_results_filtered)}")

    df = fetch_kline_sina(symbol, datalen=1023)
    if df is None or len(df) < 310:
        fail_count += 1
        continue

    closes = df['close'].values.astype(float)
    volumes = df['volume'].values.astype(float)
    dates = df['date'].values
    n = len(closes)

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

    # v1黄金坑检测
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

        result = {
            'symbol': symbol, 'name': name, 'market': market,
            'buy_date': buy_date, 'buy_year': buy_year, 'ret': ret,
        }

        # 无过滤
        all_results_raw.append(result)
        if buy_year not in by_year_raw:
            by_year_raw[buy_year] = []
        by_year_raw[buy_year].append(ret)
        if market not in by_market_raw:
            by_market_raw[market] = []
        by_market_raw[market].append(ret)

        # 大盘过滤
        is_bull = hs300_bull.get(buy_date, None)
        if is_bull:  # True=多头市场，None/False都不开仓
            all_results_filtered.append(result)
            if buy_year not in by_year_filtered:
                by_year_filtered[buy_year] = []
            by_year_filtered[buy_year].append(ret)
            if market not in by_market_filtered:
                by_market_filtered[market] = []
            by_market_filtered[market].append(ret)

    success_count += 1

elapsed = time.time() - start_time
print(f"\n完成: {success_count}只成功, {fail_count}只失败, 耗时{elapsed:.0f}s")
print(f"原始信号: {len(all_results_raw)}个, 大盘过滤后: {len(all_results_filtered)}个")
print(f"过滤掉信号: {len(all_results_raw) - len(all_results_filtered)}个 "
      f"({(len(all_results_raw)-len(all_results_filtered))/len(all_results_raw)*100:.1f}%)")

# 3. 汇总对比
def calc_stats(results):
    if not results:
        return {'n': 0, 'wr': 0, 'mean': 0, 'median': 0, 'max': 0, 'min': 0}
    rets = [r['ret'] for r in results] if isinstance(results[0], dict) else results
    return {
        'n': len(rets),
        'wr': sum(1 for r in rets if r > 0) / len(rets),
        'mean': np.mean(rets),
        'median': np.median(rets),
        'max': np.max(rets),
        'min': np.min(rets),
    }

print("\n" + "=" * 80)
print("整体对比（300只A股，快启动+坑长+60天持有）")
print("=" * 80)
s_raw = calc_stats(all_results_raw)
s_fil = calc_stats(all_results_filtered)
print(f"\n{'指标':<12} {'无大盘过滤':>12} {'有大盘过滤':>12} {'变化':>12}")
print("-" * 55)
print(f"{'信号数':<12} {s_raw['n']:>12} {s_fil['n']:>12} {s_fil['n']-s_raw['n']:>+12}")
print(f"{'胜率':<12} {s_raw['wr']:>11.1%} {s_fil['wr']:>11.1%} {(s_fil['wr']-s_raw['wr'])*100:>+10.1f}pp")
print(f"{'平均收益':<12} {s_raw['mean']:>11.1%} {s_fil['mean']:>11.1%} {(s_fil['mean']-s_raw['mean'])*100:>+10.1f}pp")
print(f"{'中位收益':<12} {s_raw['median']:>11.1%} {s_fil['median']:>11.1%} {(s_fil['median']-s_raw['median'])*100:>+10.1f}pp")
print(f"{'最大收益':<12} {s_raw['max']:>11.1%} {s_fil['max']:>11.1%}")
print(f"{'最小收益':<12} {s_raw['min']:>11.1%} {s_fil['min']:>11.1%}")

# 4. 按年份对比
print(f"\n--- 按买入年份对比 ---")
print(f"{'年份':<8} {'原始n':>6} {'原始胜率':>10} {'原始均值':>10} | {'过滤n':>6} {'过滤胜率':>10} {'过滤均值':>10} | {'胜率变化':>10}")
print("-" * 95)
all_years = sorted(set(list(by_year_raw.keys()) + list(by_year_filtered.keys())))
for year in all_years:
    raw_rets = by_year_raw.get(year, [])
    fil_rets = by_year_filtered.get(year, [])
    raw_wr = sum(1 for r in raw_rets if r > 0) / len(raw_rets) if raw_rets else 0
    fil_wr = sum(1 for r in fil_rets if r > 0) / len(fil_rets) if fil_rets else 0
    raw_mean = np.mean(raw_rets) if raw_rets else 0
    fil_mean = np.mean(fil_rets) if fil_rets else 0
    wr_diff = (fil_wr - raw_wr) * 100 if raw_rets and fil_rets else 0
    print(f"{year:<8} {len(raw_rets):>6} {raw_wr:>9.1%} {raw_mean:>9.1%} | "
          f"{len(fil_rets):>6} {fil_wr:>9.1%} {fil_mean:>9.1%} | {wr_diff:>+9.1f}pp")

# 5. 按板块对比
print(f"\n--- 按市场板块对比 ---")
print(f"{'板块':<10} {'原始n':>6} {'原始胜率':>10} {'原始均值':>10} | {'过滤n':>6} {'过滤胜率':>10} {'过滤均值':>10}")
print("-" * 80)
all_markets = sorted(set(list(by_market_raw.keys()) + list(by_market_filtered.keys())))
for market in all_markets:
    raw_rets = by_market_raw.get(market, [])
    fil_rets = by_market_filtered.get(market, [])
    raw_wr = sum(1 for r in raw_rets if r > 0) / len(raw_rets) if raw_rets else 0
    fil_wr = sum(1 for r in fil_rets if r > 0) / len(fil_rets) if fil_rets else 0
    raw_mean = np.mean(raw_rets) if raw_rets else 0
    fil_mean = np.mean(fil_rets) if fil_rets else 0
    print(f"{market:<10} {len(raw_rets):>6} {raw_wr:>9.1%} {raw_mean:>9.1%} | "
          f"{len(fil_rets):>6} {fil_wr:>9.1%} {fil_mean:>9.1%}")

# 6. 被过滤掉的信号分析（空头市场中的信号）
print(f"\n--- 被大盘过滤掉的信号分析（沪深300在250日线下方时的信号）---")
filtered_out = [r for r in all_results_raw if r not in all_results_filtered]
# 注意：上面的判断不准确，因为result是dict，不能直接比较
# 重新计算被过滤的信号
filtered_out_rets = []
for r in all_results_raw:
    is_bull = hs300_bull.get(r['buy_date'], None)
    if is_bull is not None and not is_bull:
        filtered_out_rets.append(r['ret'])
if filtered_out_rets:
    wr = sum(1 for r in filtered_out_rets if r > 0) / len(filtered_out_rets)
    print(f"  被过滤信号数: {len(filtered_out_rets)}个")
    print(f"  被过滤信号胜率: {wr:.1%}（这些信号如果开仓会拉低整体胜率）")
    print(f"  被过滤信号均值: {np.mean(filtered_out_rets):.1%}")
    print(f"  被过滤信号中位: {np.median(filtered_out_rets):.1%}")

# 7. 结论
print("\n" + "=" * 80)
print("结论")
print("=" * 80)
wr_improve = (s_fil['wr'] - s_raw['wr']) * 100
mean_improve = (s_fil['mean'] - s_raw['mean']) * 100
print(f"大盘过滤(沪深300>250日线)效果:")
print(f"  信号数: {s_raw['n']} → {s_fil['n']} (减少{s_raw['n']-s_fil['n']}个, {(s_raw['n']-s_fil['n'])/s_raw['n']*100:.1f}%)")
print(f"  胜率: {s_raw['wr']:.1%} → {s_fil['wr']:.1%} ({'+' if wr_improve>=0 else ''}{wr_improve:.1f}pp)")
print(f"  均值: {s_raw['mean']:.1%} → {s_fil['mean']:.1%} ({'+' if mean_improve>=0 else ''}{mean_improve:.1f}pp)")
if wr_improve > 0:
    print(f"  → 大盘过滤有效，胜率提升{wr_improve:.1f}个百分点")
else:
    print(f"  → 大盘过滤在当前回测中未提升胜率，需进一步分析")
print("\n" + "=" * 80)
print("回测完成")
print("=" * 80)
