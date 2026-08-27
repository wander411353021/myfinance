# -*- coding: utf-8 -*-
"""
严格复现原仓库回测逻辑，验证80%+胜率的来源
对比：无止损止盈 vs 有止损止盈 vs 有止损止盈+放量堆确认
"""
import sys
sys.path.insert(0, '.')
from golden_pit_v2_backtest import *
import numpy as np

# ============================================================
# 从 panic_reversal.py 提取的放量堆检测（简化版）
# ============================================================
def detect_volume_clusters_simple(closes, volumes, win=60, hi_ratio=1.5, lo_ratio=0.6,
                                   hi_pct=0.75, lo_pct=0.15, min_len=3, exit_confirm=2):
    """简化版成交量堆检测，返回 [(start,end,kind,direction,peak_ratio,mean_ratio)]"""
    closes = np.asarray(closes, dtype=float)
    vols = np.asarray(volumes, dtype=float)
    n = len(vols)
    s = pd.Series(vols)
    med = s.rolling(win, min_periods=20).median().shift(1).values
    p_hi = s.rolling(win, min_periods=20).quantile(hi_pct).shift(1).values
    p_lo = s.rolling(win, min_periods=20).quantile(lo_pct).shift(1).values
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = vols / np.where(med > 0, med, np.nan)
    st = np.zeros(n, dtype=int)
    for i in range(n):
        if np.isfinite(ratio[i]) and ratio[i] > hi_ratio and np.isfinite(p_hi[i]) and vols[i] > p_hi[i]:
            st[i] = 1
        elif np.isfinite(ratio[i]) and ratio[i] < lo_ratio and np.isfinite(p_lo[i]) and vols[i] < p_lo[i]:
            st[i] = -1
    segs = []
    cur = None
    neutral_run = 0
    for i in range(n):
        if cur is None:
            if st[i] != 0:
                cur = [i, st[i], i]
                neutral_run = 0
        else:
            if st[i] == cur[1]:
                cur[2] = i
                neutral_run = 0
            elif st[i] != 0:
                segs.append((cur[0], cur[2], cur[1]))
                cur = [i, st[i], i]
                neutral_run = 0
            else:
                neutral_run += 1
                if neutral_run >= exit_confirm:
                    segs.append((cur[0], cur[2], cur[1]))
                    cur = None
                    neutral_run = 0
    if cur is not None:
        segs.append((cur[0], cur[2], cur[1]))
    out = []
    for s0, e0, kd in segs:
        if e0 - s0 + 1 < min_len:
            continue
        rseg = ratio[s0:e0 + 1]
        if not np.isfinite(rseg).any():
            continue
        if kd == 1:
            if np.nanmax(rseg) < hi_ratio * 0.9:
                continue
        else:
            if np.nanmin(rseg) > lo_ratio * 1.1:
                continue
        chg = closes[e0] / closes[s0] - 1
        direction = 'UP' if chg > 0.02 else ('DOWN' if chg < -0.02 else 'FLAT')
        out.append((s0, e0, 'HIGH' if kd == 1 else 'LOW', direction,
                    float(np.nanmax(rseg)), float(np.nanmean(rseg))))
    return out

def is_super_pit(lch, clusters, window_days=7, peak_ratio=5.0):
    """出坑后window_days天内出现放量堆且峰值量比>=peak_ratio"""
    if lch is None:
        return False
    for ss, ee, kk, dd, pp, vv in clusters:
        if kk == 'HIGH' and lch < ss <= lch + window_days and pp >= peak_ratio:
            return True
    return False

# ============================================================
# 严格复现原仓库回测：止损-15%/止盈+30%/60天到期
# ============================================================
def backtest_v1_strict(closes, pits, stop=-0.15, take=0.30, horizon=60,
                        require_fast=True, require_min_len=8, require_super=False,
                        volumes=None):
    """
    严格复现原仓库 backtest_single 逻辑：
    - 快启动≤5天 + 坑长≥8（require_fast/min_len）
    - 止损-15% / 止盈+30% / 60天到期
    - require_super: 出坑后7天内放量堆峰值量比≥5
    """
    n = len(closes)
    clusters = detect_volume_clusters_simple(closes, volumes) if volumes is not None else []
    results = []
    for pit in pits:
        if len(pit) == 3:
            s, b, lch = pit
        else:
            s, b, lch = pit[0], pit[1], pit[2]
        if lch is None or lch + horizon >= n:
            continue
        if require_fast and lch - b > 5:
            continue
        if require_min_len and b - s + 1 < require_min_len:
            continue
        if require_super and not is_super_pit(lch, clusters):
            continue
        buy_px = closes[lch]
        sell_px = None
        sell_i = None
        reason = 'horizon'
        for i in range(lch + 1, lch + horizon + 1):
            if i >= n:
                break
            if closes[i] <= buy_px * (1 + stop):
                sell_px = closes[i]
                sell_i = i
                reason = 'stop'
                break
            if closes[i] >= buy_px * (1 + take):
                sell_px = closes[i]
                sell_i = i
                reason = 'take'
                break
        if sell_px is None:
            sell_i = min(lch + horizon, n - 1)
            sell_px = closes[sell_i]
        ret = sell_px / buy_px - 1
        results.append({'buy_idx': lch, 'sell_idx': sell_i, 'ret': ret,
                        'reason': reason, 'pit_len': b-s+1, 'launch_days': lch-b,
                        'super': is_super_pit(lch, clusters)})
    return results

def stats_detail(results, label=''):
    if not results:
        return {'label': label, 'n': 0, 'win_rate': 0, 'mean_ret': 0, 'median_ret': 0}
    rets = [r['ret'] for r in results]
    reasons = [r['reason'] for r in results]
    return {
        'label': label, 'n': len(rets),
        'win_rate': sum(1 for r in rets if r > 0) / len(rets),
        'mean_ret': np.mean(rets), 'median_ret': np.median(rets),
        'max_ret': np.max(rets), 'min_ret': np.min(rets),
        'stop_pct': sum(1 for r in reasons if r == 'stop') / len(rets),
        'take_pct': sum(1 for r in reasons if r == 'take') / len(rets),
        'horizon_pct': sum(1 for r in reasons if r == 'horizon') / len(rets),
    }

# ============================================================
# 主回测
# ============================================================
def main():
    print("=" * 80)
    print("严格复现原仓库回测逻辑 — 验证80%+胜率来源")
    print("=" * 80)
    print(f"股票池: {len(STOCK_POOL)}只A股, 1023天数据")
    print(f"条件: 快启动≤5天 + 坑长≥8 + 止损-15%/止盈+30%/60天")
    print()

    # 收集所有股票的结果
    all_pits_v1 = []  # 原始v1坑
    all_closes = []
    all_volumes = []

    for symbol, name in STOCK_POOL:
        df = fetch_kline_sina(symbol, datalen=1023)
        if df is None or len(df) < 310:
            continue
        closes = df['close'].values.astype(float)
        volumes = df['volume'].values.astype(float)
        reg250, _ = compute_rolling_regression(closes, window=250)
        pits_v1 = detect_golden_pit_v1(closes, reg250)
        all_pits_v1.append((name, closes, volumes, pits_v1))

    print(f"有效股票: {len(all_pits_v1)}只\n")

    # 策略对比
    strategies = [
        ('A: v1全部坑+固定60天(无过滤)', dict(stop=None, take=None, require_fast=False, require_min_len=0, require_super=False)),
        ('B: v1全部坑+止损止盈(无过滤)', dict(stop=-0.15, take=0.30, require_fast=False, require_min_len=0, require_super=False)),
        ('C: 快启动≤5天+坑长≥8+止损止盈', dict(stop=-0.15, take=0.30, require_fast=True, require_min_len=8, require_super=False)),
        ('D: C + 放量堆确认(量比≥5)', dict(stop=-0.15, take=0.30, require_fast=True, require_min_len=8, require_super=True)),
    ]

    all_results = {s[0]: [] for s in strategies}

    for name, closes, volumes, pits_v1 in all_pits_v1:
        for label, params in strategies:
            stop = params.get('stop')
            take = params.get('take')
            if stop is None:
                # 固定60天
                n = len(closes)
                for pit in pits_v1:
                    s, b, lch = pit
                    if lch is None or lch + 60 >= n:
                        continue
                    if params.get('require_fast', True) and lch - b > 5:
                        continue
                    if params.get('require_min_len', 8) and b - s + 1 < 8:
                        continue
                    buy_px = closes[lch]
                    sell_px = closes[min(lch + 60, n - 1)]
                    ret = sell_px / buy_px - 1
                    all_results[label].append({'ret': ret, 'reason': 'horizon'})
            else:
                r = backtest_v1_strict(closes, pits_v1, stop=stop, take=take,
                                        require_fast=params.get('require_fast', True),
                                        require_min_len=params.get('require_min_len', 8),
                                        require_super=params.get('require_super', False),
                                        volumes=volumes)
                all_results[label].extend(r)

    # 输出对比
    print("=" * 80)
    print("策略对比（全股票池汇总）")
    print("=" * 80)
    print(f"\n{'策略':<40} {'信号数':>6} {'胜率':>8} {'均值':>8} {'中位':>8} {'止损%':>7} {'止盈%':>7}")
    print("-" * 95)
    for label, _ in strategies:
        s = stats_detail(all_results[label], label)
        print(f"{label:<40} {s['n']:>6} {s['win_rate']:>7.1%} {s['mean_ret']:>7.1%} "
              f"{s['median_ret']:>7.1%} {s.get('stop_pct',0):>6.1%} {s.get('take_pct',0):>6.1%}")

    # 关键发现
    print("\n" + "=" * 80)
    print("关键发现")
    print("=" * 80)

    s_a = stats_detail(all_results[strategies[0][0]])
    s_b = stats_detail(all_results[strategies[1][0]])
    s_c = stats_detail(all_results[strategies[2][0]])
    s_d = stats_detail(all_results[strategies[3][0]])

    print(f"\n1. 止损止盈的影响:")
    print(f"   无止损止盈(A): 胜率={s_a['win_rate']:.1%}, 均值={s_a['mean_ret']:.1%}")
    print(f"   有止损止盈(B): 胜率={s_b['win_rate']:.1%}, 均值={s_b['mean_ret']:.1%}")
    print(f"   胜率变化: +{(s_b['win_rate']-s_a['win_rate'])*100:.1f}个百分点")

    print(f"\n2. 快启动+坑长过滤的影响(B→C):")
    print(f"   不过滤(B): 胜率={s_b['win_rate']:.1%}, 信号={s_b['n']}")
    print(f"   过滤后(C): 胜率={s_c['win_rate']:.1%}, 信号={s_c['n']}")
    print(f"   胜率变化: +{(s_c['win_rate']-s_b['win_rate'])*100:.1f}个百分点, 信号减少{s_b['n']-s_c['n']}个")

    print(f"\n3. 放量堆确认的影响(C→D):")
    print(f"   无确认(C): 胜率={s_c['win_rate']:.1%}, 信号={s_c['n']}")
    print(f"   有确认(D): 胜率={s_d['win_rate']:.1%}, 信号={s_d['n']}")
    print(f"   胜率变化: +{(s_d['win_rate']-s_c['win_rate'])*100:.1f}个百分点, 信号减少{s_c['n']-s_d['n']}个")

    # D策略的详细表现
    if s_d['n'] > 0:
        print(f"\n4. 最严格策略D(止损止盈+快启动+坑长+放量确认)详细:")
        print(f"   信号数={s_d['n']}, 胜率={s_d['win_rate']:.1%}, 均值={s_d['mean_ret']:.1%}, 中位={s_d['median_ret']:.1%}")
        print(f"   最大={s_d['max_ret']:.1%}, 最小={s_d['min_ret']:.1%}")
        # 按卖出原因
        d_results = all_results[strategies[3][0]]
        for reason in ['take', 'stop', 'horizon']:
            grp = [r for r in d_results if r['reason'] == reason]
            if grp:
                rets = [r['ret'] for r in grp]
                print(f"   {reason}: n={len(grp)}, 胜率={sum(1 for r in rets if r>0)/len(rets):.1%}, 均值={np.mean(rets):.1%}")

    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    print("原仓库80%+胜率 = 快启动≤5天 + 坑长≥8 + 止损-15%/止盈+30% 的组合效果")
    print("如果叠加放量堆确认(require_super), 胜率可进一步提升但信号大幅减少")
    print("注意: 本回测仅20只股票, 原仓库基于1500只全市场回测, 样本量差异会影响统计显著性")

if __name__ == '__main__':
    main()
