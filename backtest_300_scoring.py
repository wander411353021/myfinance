# -*- coding: utf-8 -*-
"""
黄金坑信号质量评分排序 — 回测验证
对每个黄金坑信号做多维度质量评分，验证高分信号是否真的更优，
并找出可执行的最优分数阈值。
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

# 2. 评分函数
def score_signal(closes, volumes, reg250, s, b, lch):
    """
    黄金坑信号多维度质量评分 (0-100)
    维度：
      1. 缩量挖坑(25分): 坑内均量 vs 坑前20日均量，缩量越充分越高
      2. 放量出坑(25分): 出坑日量 vs 出坑前5日均量，放量越猛越高
      3. 坑深适度(15分): 坑底相对坑前60日最高跌幅15%-50%最佳
      4. 个股相对强度(15分): 个股20日涨幅 - 大盘20日涨幅
      5. 回归线方向(10分): 250日回归线向上=上涨中继概率高
      6. 坑底缩量企稳(10分): 坑底附近3日均量 vs 坑前，缩量=洗盘充分
    """
    n = len(closes)
    score = 0.0

    # 1. 缩量挖坑 (25分)
    vol_pit = np.mean(volumes[s:b+1])
    vol_pre = np.mean(volumes[max(0, s-20):s]) if s >= 20 else vol_pit
    if vol_pre > 0:
        shrink = vol_pit / vol_pre
        # shrink 越小越好，0.3-0.7 最佳区间
        if shrink <= 0.3:
            score += 25
        elif shrink <= 0.7:
            score += 25 - (shrink - 0.3) / 0.4 * 15
        elif shrink <= 1.0:
            score += 10 - (shrink - 0.7) / 0.3 * 10
        else:
            score += max(0, 0)  # 放量挖坑=恐慌砸盘，不加分

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
            score += 0  # 出坑无量=弱反弹

    # 3. 坑深适度 (15分)
    pre_high = np.max(closes[max(0, b-60):b]) if b >= 60 else closes[max(0,b-20):b+1].max()
    if pre_high > 0:
        depth = 1 - closes[b] / pre_high
        if 0.15 <= depth <= 0.50:
            score += 15
        elif 0.05 <= depth < 0.15:
            score += 8
        elif depth > 0.50:
            score += 5  # 太深可能是基本面恶化
        else:
            score += 0  # 太浅=假坑

    # 4. 个股相对强度 (15分)
    if b >= 20:
        stock_ret20 = closes[b] / closes[b-20] - 1
        mkt_ret20 = hs300_ret(pd.Timestamp(dates_lch_cache[b]).strftime('%Y-%m-%d'), 20) if False else None
        # 用当前函数无法访问dates，改为在外部传入
        score += 0  # 占位，实际在外部计算

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
            score += 0  # 回归线向下=下跌趋势中的坑

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

    return score, shrink, fill, depth


# 3. 批量回测 + 评分
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

        # 完整评分
        score, shrink, fill, depth = score_signal(closes, volumes, reg250, s, b, lch)

        # 补上RS维度（15分）
        rs_score = 0
        if rs20 is not None:
            if rs20 > 0.10:
                rs_score = 15
            elif rs20 > 0.05:
                rs_score = 12
            elif rs20 > 0:
                rs_score = 8
            elif rs20 > -0.05:
                rs_score = 4
            else:
                rs_score = 0
        score += rs_score

        all_signals.append({
            'symbol': symbol, 'name': name, 'market': market,
            'buy_date': buy_date, 'buy_year': buy_year, 'ret': ret,
            'score': score, 'shrink': shrink, 'fill': fill, 'depth': depth, 'rs20': rs20,
        })

    success_count += 1

elapsed = time.time() - start_time
print(f"\n完成: {success_count}只成功, 失败{fail_count}, 耗时{elapsed:.0f}s")
print(f"信号总数: {len(all_signals)}")

# 4. 评分分布分析
scores = [s['score'] for s in all_signals]
print(f"\n评分分布: min={min(scores):.0f}, max={max(scores):.0f}, 均值={np.mean(scores):.1f}, 中位={np.median(scores):.1f}")

# 5. 按分数分位数组验证
def calc_stats(signals):
    if not signals:
        return {'n': 0, 'wr': 0, 'mean': 0, 'median': 0}
    rets = [s['ret'] for s in signals]
    return {'n': len(rets), 'wr': sum(1 for r in rets if r > 0)/len(rets),
            'mean': np.mean(rets), 'median': np.median(rets)}

print("\n" + "=" * 90)
print("按评分分位数组的表现")
print("=" * 90)
quantiles = [0, 20, 40, 60, 80, 100]
qs = np.percentile(scores, quantiles)
print(f"{'分位数组':<12} {'评分范围':<12} {'信号数':>6} {'胜率':>8} {'均值':>8} {'中位':>8}")
print("-" * 70)
groups = []
for i in range(5):
    lo, hi = qs[i], qs[i+1]
    grp = [s for s in all_signals if lo <= s['score'] < hi] if i < 4 else [s for s in all_signals if lo <= s['score'] <= hi]
    groups.append(grp)
    st = calc_stats(grp)
    print(f"Q{5-i} (Top{(5-i)*20}%)".ljust(12) + f" {lo:.0f}-{hi:.0f}".ljust(12) +
          f" {st['n']:>6} {st['wr']:>7.1%} {st['mean']:>7.1%} {st['median']:>7.1%}")

# 6. 按分数阈值分析（累积）
print("\n" + "=" * 90)
print("按分数阈值（累积：只交易分数>=阈值的信号）")
print("=" * 90)
print(f"{'阈值':<8} {'信号数':>6} {'保留率':>8} {'胜率':>8} {'胜率变化':>10} {'均值':>8} {'均值变化':>10} {'中位':>8}")
print("-" * 85)
s_all = calc_stats(all_signals)
print(f"{'无阈值':<8} {s_all['n']:>6} {'100.0%':>8} {s_all['wr']:>7.1%} {'基准':>10} "
      f"{s_all['mean']:>7.1%} {'基准':>10} {s_all['median']:>7.1%}")

for thr in [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]:
    grp = [s for s in all_signals if s['score'] >= thr]
    if len(grp) < 10:
        break
    st = calc_stats(grp)
    wr_diff = (st['wr'] - s_all['wr']) * 100
    mean_diff = (st['mean'] - s_all['mean']) * 100
    keep = st['n'] / s_all['n'] * 100
    print(f"{thr}分以上{'':<4} {st['n']:>6} {keep:>7.1f}% {st['wr']:>7.1%} "
          f"{'+' if wr_diff>=0 else ''}{wr_diff:>8.1f}pp {st['mean']:>7.1%} "
          f"{'+' if mean_diff>=0 else ''}{mean_diff:>8.1f}pp {st['median']:>7.1%}")

# 7. 最优阈值确定
print("\n" + "=" * 90)
print("最优阈值筛选")
print("=" * 90)
best_thr = None
best_score_val = 0
for thr in [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]:
    grp = [s for s in all_signals if s['score'] >= thr]
    if len(grp) < 20:
        continue
    st = calc_stats(grp)
    # 综合评分：胜率提升 + 均值提升，同时惩罚信号数过少
    composite = (st['wr'] - s_all['wr']) * 100 * 0.7 + (st['mean'] - s_all['mean']) * 100 * 0.3
    if composite > best_score_val:
        best_score_val = composite
        best_thr = thr
        best_grp = grp
        best_st = st

if best_thr:
    print(f"\n最优阈值: {best_thr}分以上")
    st = best_st
    print(f"  信号数: {st['n']}个 (保留率{st['n']/s_all['n']*100:.1f}%)")
    print(f"  胜率: {st['wr']:.1%} (基准{s_all['wr']:.1%}, {'+' if st['wr']>s_all['wr'] else ''}{(st['wr']-s_all['wr'])*100:.1f}pp)")
    print(f"  均值: {st['mean']:.1%} (基准{s_all['mean']:.1%})")
    print(f"  中位: {st['median']:.1%}")

    # 最优阈值按年份
    print(f"\n--- 最优阈值({best_thr}分以上)按年份 ---")
    by_year_raw = {}
    by_year_fil = {}
    for s in all_signals:
        by_year_raw.setdefault(s['buy_year'], []).append(s['ret'])
    for s in best_grp:
        by_year_fil.setdefault(s['buy_year'], []).append(s['ret'])
    print(f"{'年份':<8} {'原始n':>6} {'原始胜率':>10} {'原始均值':>10} | {'过滤n':>6} {'过滤胜率':>10} {'过滤均值':>10} | {'胜率变化':>10}")
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
    print(f"\n--- 最优阈值({best_thr}分以上)按板块 ---")
    by_mkt_raw = {}
    by_mkt_fil = {}
    for s in all_signals:
        by_mkt_raw.setdefault(s['market'], []).append(s['ret'])
    for s in best_grp:
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

print("\n" + "=" * 90)
print("回测完成")
print("=" * 90)
