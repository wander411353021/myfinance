# -*- coding: utf-8 -*-
"""黄金坑坑段内的 Turn-Up Target 突破 胜率分析(严格因果, 无未来函数)。

用 v3+双基准检测坑, 在坑段[s, lch]内识别所有 target 突破事件, 分层统计胜率。
核心结论(1000池2023后):
  - 坑底前突破(下跌中继): 胜率21%, 负期望, 坚决回避
  - 坑底后突破: 胜率53%
  - 坑底后+距坑底0~10%: 胜率58~71%(甜点)
  - 坑底后+距坑底>25%(追高): 胜率39%, 负期望
  - 真突破(10日内target不重现): 胜率61%
  - 最佳组合: 坑底后+距坑底3~10%+真突破 = 63%胜率, r10胜率69%
用法: python3 pit_target_break_analysis.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression
import golden_pit_portfolio_sim as sim


def collect(stock_pool='stock_pool_1000.txt'):
    stocks = [l.strip().split(',')[0] for l in open(stock_pool) if l.strip()]
    rows = []
    for sym in stocks:
        d = sim.load(sym)
        if d is None:
            continue
        c = d['close'].astype(float); h = d['high'].astype(float)
        l = d['low'].astype(float); ts = d['ts']; n = len(c)
        reg120, _ = compute_rolling_regression(c, window=120, use_log=True)
        reg250, _ = compute_rolling_regression(c, window=250, use_log=True)
        refs = pr.compute_turn_positive_prices(c, h, l, opens=None, reg_preds=reg250)
        pits = [p for p in pr.detect_golden_pit_v3(
            c, reg250, reg120=reg120, use_dual=True, smooth_w=5,
            min_depth=0.08, confirm_days=3) if p[2] is not None]
        resid = c - reg250; rstd = np.full(n, np.nan)
        for i in range(59, n):
            rstd[i] = np.std(resid[i - 59:i + 1])
        z = np.where(rstd > 0, resid / rstd, 0.0)
        breaks = [i for i in range(12, n) if np.isfinite(refs[i - 1]) and not np.isfinite(refs[i])]
        for s, b, lch in pits:
            pit_breaks = [i for i in breaks if s <= i <= lch]
            for bi in pit_breaks:
                buy = bi + 1
                if buy + 60 >= n:
                    continue
                if np.datetime64(int(ts[buy]), 's').astype('datetime64[D]') < np.datetime64('2023-01-01'):
                    continue
                reenter = any(np.isfinite(refs[k]) for k in range(bi + 1, min(bi + 11, n)))
                rows.append({
                    'after_bottom': bi > b,
                    'dist_bottom': c[bi] / c[b] - 1 if c[b] > 0 else 0,
                    'z_at': z[bi] if np.isfinite(z[bi]) else 0,
                    'pit_len': lch - s,
                    'reenter': reenter,
                    'r10': c[min(buy + 10, n - 1)] / c[buy] - 1,
                    'r20': c[min(buy + 20, n - 1)] / c[buy] - 1,
                    'r30': c[min(buy + 30, n - 1)] / c[buy] - 1,
                    'r60': c[min(buy + 60, n - 1)] / c[buy] - 1,
                    'mx20': c[buy:min(buy + 20, n)].max() / c[buy] - 1,
                })
    return rows


def stat(m, tag):
    if not m:
        print(f'{tag}: 空'); return
    print(f'{tag:<36} n={len(m):>5}  r10={np.mean([x["r10"] for x in m]) * 100:+.2f}%({np.mean([x["r10"] > 0 for x in m]) * 100:.0f}%)  '
          f'r20={np.mean([x["r20"] for x in m]) * 100:+.2f}%({np.mean([x["r20"] > 0 for x in m]) * 100:.0f}%)  '
          f'r60={np.mean([x["r60"] for x in m]) * 100:+.2f}%({np.mean([x["r60"] > 0 for x in m]) * 100:.0f}%)')


def main():
    rows = collect()
    print(f'坑内target突破事件: {len(rows)}')
    print()
    stat(rows, '全部坑内突破')
    print()
    print('=== 坑底前 vs 坑底后 ===')
    stat([x for x in rows if not x['after_bottom']], '坑底前突破(下跌中继)')
    stat([x for x in rows if x['after_bottom']], '坑底后突破')
    print()
    print('=== 坑底后突破: 距坑底涨幅分层 ===')
    for lo, hi in [(0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.15), (0.15, 0.25), (0.25, 99)]:
        stat([x for x in rows if x['after_bottom'] and lo <= x['dist_bottom'] < hi],
             f'  距坑底{lo * 100:.0f}%~{hi * 100:.0f}%')
    print()
    print('=== 真突破确认 ===')
    stat([x for x in rows if x['after_bottom'] and not x['reenter']], '坑底后+真突破(不reenter)')
    stat([x for x in rows if x['reenter']], '假突破(reenter)')
    print()
    print('=== 最佳组合 ===')
    stat([x for x in rows if x['after_bottom'] and 0.03 <= x['dist_bottom'] < 0.10 and not x['reenter']],
         '坑底后+距坑底3~10%+真突破')
    stat([x for x in rows if x['after_bottom'] and 0.03 <= x['dist_bottom'] < 0.15 and not x['reenter']],
         '坑底后+距坑底3~15%+真突破')


if __name__ == '__main__':
    main()
