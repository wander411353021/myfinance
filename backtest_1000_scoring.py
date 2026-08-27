# -*- coding: utf-8 -*-
"""
1000只A股大规模回测 — 黄金坑信号质量评分排序验证
包含：
  1. 全量1000只整体表现
  2. 评分分位数组验证
  3. 分数阈值累积分析
  4. 子集稳定性检验（将1000只随机分成2组各500只，验证评分结论是否一致）
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
print("策略: v1黄金坑 + 快启动≤5天 + 坑长≥8 + 60天持有")
print("=" * 80)

# 1. 沪深300（用于相对强度）
df_hs300 = fetch_kline_sina('sh000300', datalen=1023)
hs300_close = df_hs300['close'].values.astype(float)
hs300_dates = df_hs300['date'].values
hs300_date_idx = {pd.Timestamp(d).strftime('%Y-%m-%d'): i for i, d in enumerate(hs300_dates)}
def hs300_ret(date_str, days):
    i = hs300_date_idx.get(date_str)
    if i is None or i < days:
        return None
    return hs300_close[i] / hs300_close[i-days] - 1

# 2. 评分函数（与300只回测一致）
def score_signal(closes, volumes, reg250, s, b, lch, rs20):
    n = len(closes)
    score = 0.0

    # 1. 缩量挖坑 (25分)
    vol_pit = np.mean(volumes[s:b+1])
    vol_pre = np.mean(volumes[max(0, s-20):s]) if s >= 20 else vol_pit
    if vol_pre > 0:
        shrink = vol_pit / vol_pre
        if shrink <= 0.3:
            score += 25
        elif shrink <= 0.7:
            score += 25 - (shrink - 0.3) / 0.4 * 15
        elif shrink <= 1.0:
            score += 10 - (shrink - 0.7) / 0.3 * 10
        else:
            score += 0

    # 2. 放量出坑 (25分)
    vol_lch = volumes[lch]
    vol_lch_pre = np.mean(volumes[max(0, lch-5):lch]) if lch >= 5 else vol_lch
    if vol_lch_pre > 0:
        fill = vol_lch / vol_lch_pre
        if fill >= 2.0:
            score += 25
        elif fill >= 1.2:
            score += 10 + (fill - 1.2) / 0.8 * 15
        elif fill >= 0.8:
            score += 5 + (fill - 0.8) / 0.4 * 5
        else:
            score += 0

    # 3. 坑深适度 (15分)
    pre_high = np.max(closes[max(0, b-60):b]) if b >= 60 else closes[max(0,b-20):b+1].max()
    if pre_high > 0:
        depth = 1 - closes[b] / pre_high
        if 0.15 <= depth <= 0.50:
            score += 15
        elif 0.05 <= depth < 0.15:
            score += 8
        elif depth > 0.50:
            score += 5
        else:
            score += 0

    # 4. 个股相对强度 (15分)
    if rs20 is not None:
        if rs20 > 0.10:
            score += 15
        elif rs20 > 0.05:
            score += 12
        elif rs20 > 0:
            score += 8
        elif rs20 > -0.05:
            score += 4
        else:
            score += 0

    # 5. 回归线方向 (10分)
    if b >= 20 and np.isfinite(reg250[b]) and np.isfinite(reg250[b-20]):
        slope = reg250[b] / reg250[b-20] - 1
        if slope > 0.02:
            score += 10
        elif slope > 0:
            score += 6
        elif slope > -0.02:
            score += 3
        else:
            score += 0

    # 6. 坑底缩量企稳 (10分)
    vol_bottom = np.mean(volumes[max(0, b-2):b+3])
    vol_pre2 = np.mean(volumes[max(0, s-20):s]) if s >= 20 else vol_bottom
    if vol_pre2 > 0:
        bottom_shrink = vol_bottom / vol_pre2
        if bottom_shrink < 0.5:
            score += 10
        elif bottom_shrink < 0.8:
            score += 6
        elif bottom_shrink < 1.0:
            score += 3
        else:
            score += 0

    return score

# 3. 批量回测 + 评分
all_signals = []
success_count = 0
fail_count = 0
start_time = time.time()

for idx, (symbol, name) in enumerate(stocks):
    if (idx + 1) % 100 == 0:
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

        # 个股相对强度（20日）
        rs20 = None
        if b >= 20:
            stock_ret20 = closes[b] / closes[b-20] - 1
            mkt_ret20 = hs300_ret(buy_date, 20)
            if mkt_ret20 is not None:
                rs20 = stock_ret20 - mkt_ret20

        score = score_signal(closes, volumes, reg250, s, b, lch, rs20)

        all_signals.append({
            'symbol': symbol, 'name': name, 'market': market,
            'buy_date': buy_date, 'buy_year': buy_year, 'ret': ret,
            'score': score, 'rs20': rs20,
        })

    success_count += 1

elapsed = time.time() - start_time
print(f"\n完成: {success_count}只成功, 失败{fail_count}, 耗时{elapsed:.0f}s")
print(f"信号总数: {len(all_signals)}")

# 4. 统计函数
def calc_stats(signals):
    if not signals:
        return {'n': 0, 'wr': 0, 'mean': 0, 'median': 0, 'max': 0, 'min': 0}
    rets = [s['ret'] for s in signals]
    return {'n': len(rets), 'wr': sum(1 for r in rets if r > 0)/len(rets),
            'mean': np.mean(rets), 'median': np.median(rets),
            'max': np.max(rets), 'min': np.min(rets)}

# 5. 整体表现
s_all = calc_stats(all_signals)
print("\n" + "=" * 90)
print(f"整体表现（1000只A股，{len(all_signals)}个信号，60天持有）")
print("=" * 90)
print(f"\n信号数: {s_all['n']}")
print(f"胜率: {s_all['wr']:.1%}")
print(f"平均收益: {s_all['mean']:.1%}")
print(f"中位收益: {s_all['median']:.1%}")
print(f"最大收益: {s_all['max']:.1%}")
print(f"最小收益: {s_all['min']:.1%}")

# 按年份
by_year = {}
for s in all_signals:
    by_year.setdefault(s['buy_year'], []).append(s['ret'])
print(f"\n--- 按买入年份 ---")
print(f"{'年份':<8} {'信号数':>6} {'胜率':>8} {'均值':>8} {'中位':>8}")
print("-" * 45)
for year in sorted(by_year.keys()):
    rets = by_year[year]
    wr = sum(1 for r in rets if r > 0) / len(rets)
    print(f"{year:<8} {len(rets):>6} {wr:>7.1%} {np.mean(rets):>7.1%} {np.median(rets):>7.1%}")

# 按板块
by_market = {}
for s in all_signals:
    by_market.setdefault(s['market'], []).append(s['ret'])
print(f"\n--- 按市场板块 ---")
print(f"{'板块':<10} {'信号数':>6} {'胜率':>8} {'均值':>8} {'中位':>8}")
print("-" * 50)
for market in sorted(by_market.keys()):
    rets = by_market[market]
    wr = sum(1 for r in rets if r > 0) / len(rets)
    print(f"{market:<10} {len(rets):>6} {wr:>7.1%} {np.mean(rets):>7.1%} {np.median(rets):>7.1%}")

# 6. 评分分位数验证
print("\n" + "=" * 90)
print("评分分位数组表现（验证：高分是否更优）")
print("=" * 90)
scores = [s['score'] for s in all_signals]
qs = np.percentile(scores, [0, 20, 40, 60, 80, 100])
print(f"\n评分范围: {min(scores):.0f}-{max(scores):.0f}, 均值{np.mean(scores):.1f}, 中位{np.median(scores):.1f}")
print(f"{'分位数组':<12} {'评分范围':<12} {'信号数':>6} {'胜率':>8} {'均值':>8} {'中位':>8}")
print("-" * 70)
for i in range(5):
    lo, hi = qs[i], qs[i+1]
    if i < 4:
        grp = [s for s in all_signals if lo <= s['score'] < hi]
    else:
        grp = [s for s in all_signals if lo <= s['score'] <= hi]
    st = calc_stats(grp)
    print(f"Q{5-i} (Top{(5-i)*20}%)".ljust(12) + f" {lo:.0f}-{hi:.0f}".ljust(12) +
          f" {st['n']:>6} {st['wr']:>7.1%} {st['mean']:>7.1%} {st['median']:>7.1%}")

# 7. 分数阈值累积
print("\n" + "=" * 90)
print("分数阈值累积分析")
print("=" * 90)
print(f"{'阈值':<8} {'信号数':>6} {'保留率':>8} {'胜率':>8} {'胜率变化':>10} {'均值':>8} {'均值变化':>10} {'中位':>8}")
print("-" * 85)
print(f"{'无阈值':<8} {s_all['n']:>6} {'100.0%':>8} {s_all['wr']:>7.1%} {'基准':>10} "
      f"{s_all['mean']:>7.1%} {'基准':>10} {s_all['median']:>7.1%}")
for thr in [25, 30, 35, 40, 45, 50, 55, 60]:
    grp = [s for s in all_signals if s['score'] >= thr]
    if len(grp) < 20:
        break
    st = calc_stats(grp)
    wr_diff = (st['wr'] - s_all['wr']) * 100
    mean_diff = (st['mean'] - s_all['mean']) * 100
    keep = st['n'] / s_all['n'] * 100
    print(f"{thr}分以上{'':<4} {st['n']:>6} {keep:>7.1f}% {st['wr']:>7.1%} "
          f"{'+' if wr_diff>=0 else ''}{wr_diff:>8.1f}pp {st['mean']:>7.1%} "
          f"{'+' if mean_diff>=0 else ''}{mean_diff:>8.1f}pp {st['median']:>7.1%}")

# 8. 子集稳定性检验
print("\n" + "=" * 90)
print("子集稳定性检验（1000只随机分2组，验证评分结论一致性）")
print("=" * 90)
np.random.seed(7)
symbols = list(set(s['symbol'] for s in all_signals))
np.random.shuffle(symbols)
half = len(symbols) // 2
set_a = set(symbols[:half])
set_b = set(symbols[half:])

sig_a = [s for s in all_signals if s['symbol'] in set_a]
sig_b = [s for s in all_signals if s['symbol'] in set_b]

for grp_name, grp in [('子集A (~500只)', sig_a), ('子集B (~500只)', sig_b)]:
    sg_all = calc_stats(grp)
    print(f"\n--- {grp_name} (信号{sg_all['n']}个) ---")
    print(f"  整体: 胜率{sg_all['wr']:.1%} 均值{sg_all['mean']:.1%}")
    grp_scores = [s['score'] for s in grp]
    grp_qs = np.percentile(grp_scores, [0, 20, 40, 60, 80, 100])
    print(f"  分位数组:")
    for i in range(5):
        lo, hi = grp_qs[i], grp_qs[i+1]
        if i < 4:
            g = [s for s in grp if lo <= s['score'] < hi]
        else:
            g = [s for s in grp if lo <= s['score'] <= hi]
        st = calc_stats(g)
        print(f"    Q{5-i}({lo:.0f}-{hi:.0f}分): n={st['n']:>4} 胜率={st['wr']:.1%} 均值={st['mean']:.1%}")

print("\n" + "=" * 90)
print("回测完成")
print("=" * 90)
