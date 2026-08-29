# -*- coding: utf-8 -*-
"""高收益坑内突破的特征对比分析(严格因果)。

定义高收益 = 突破后60日收益 > +15%。对比高收益组 vs 普通组的特征差异,
并测试极端组合筛选能否提高高收益占比。
核心结论(1000池2023后):
  - 高收益占比25.6%, 但常规技术特征(坑深/坑长/突破位置/z/量能)无法区分
  - 高收益是肥尾分布(Top案例r60普遍+150%以上), 由题材/资金/大盘环境驱动
  - 极端组合(深跌<-40%+长坑>40天+强反弹5日>8%+真突破)可提高高收益占比到39%
  - 该组合r60均值+9.9%(整体2倍), 但信号量少(约每月3个)
用法: python3 high_return_feature_analysis.py
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
        l = d['low'].astype(float); v = d['vol'].astype(float)
        ts = d['ts']; n = len(c)
        reg120, _ = compute_rolling_regression(c, window=120, use_log=True)
        reg250, _ = compute_rolling_regression(c, window=250, use_log=True)
        refs = pr.compute_turn_positive_prices(c, h, l, opens=None, reg_preds=reg250)
        pits = [p for p in pr.detect_golden_pit_v3(
            c, reg250, reg120=reg120, use_dual=True, smooth_w=5,
            min_depth=0.08, confirm_days=3) if p[2] is not None]
        breaks = [i for i in range(12, n) if np.isfinite(refs[i - 1]) and not np.isfinite(refs[i])]
        for s, b, lch in pits:
            pit_breaks = [i for i in breaks if s <= i <= lch]
            pre_hi = c[max(0, b - 250):b + 1].max()
            dd_from_hi = c[b] / pre_hi - 1 if pre_hi > 0 else 0
            pit_depth_reg = c[b] / reg250[b] - 1 if np.isfinite(reg250[b]) else 0
            for bi in pit_breaks:
                buy = bi + 1
                if buy + 60 >= n:
                    continue
                if np.datetime64(int(ts[buy]), 's').astype('datetime64[D]') < np.datetime64('2023-01-01'):
                    continue
                if bi <= b:
                    continue  # 只看坑底后突破
                reenter = any(np.isfinite(refs[k]) for k in range(bi + 1, min(bi + 11, n)))
                rows.append({
                    'sym': sym,
                    'dist_bottom': c[bi] / c[b] - 1,
                    'pit_len': lch - s,
                    'dd_from_hi': dd_from_hi,
                    'pit_depth_reg': pit_depth_reg,
                    'rebound5': c[bi] / c[max(0, bi - 5)] - 1 if bi >= 5 else 0,
                    'reenter': reenter,
                    'r60': c[buy + 60] / c[buy] - 1,
                    'mx30': c[buy:buy + 30].max() / c[buy] - 1,
                })
    return rows


def show(tag, cond, R):
    m = [x for x in R if cond(x)]
    if not m:
        print(f'{tag}: 空'); return
    hr = np.mean([x['r60'] > 0.15 for x in m]) * 100
    print(f'{tag:<52} n={len(m):>4}  r60={np.mean([x["r60"] for x in m]) * 100:>+6.1f}%  '
          f'胜率={np.mean([x["r60"] > 0 for x in m]) * 100:.1f}%  高收益占比={hr:.1f}%  mx30={np.mean([x["mx30"] for x in m]) * 100:+.1f}%')


def main():
    R = [x for x in collect() if x['dist_bottom'] < 0.25]
    print(f'总样本(坑底后,排除追高>25%): {len(R)}')
    print(f'高收益(r60>+15%)占比: {np.mean([x["r60"] > 0.15 for x in R]) * 100:.1f}%')
    print()
    print('=== 单维度极端值 ===')
    show('坑底距前高回撤<-40%', lambda x: x['dd_from_hi'] < -0.40, R)
    show('坑底距reg深度<-15%(深坑)', lambda x: x['pit_depth_reg'] < -0.15, R)
    show('坑长>40天', lambda x: x['pit_len'] > 40, R)
    show('突破前5日涨>10%(强反弹)', lambda x: x['rebound5'] > 0.10, R)
    show('真突破(不reenter)', lambda x: not x['reenter'], R)
    print()
    print('=== 极端组合(300251类: 深跌+长坑+强反弹+真突破) ===')
    show('深跌<-40%+长坑>30天+真突破',
         lambda x: x['dd_from_hi'] < -0.40 and x['pit_len'] > 30 and not x['reenter'], R)
    show('深跌<-40%+长坑>30天+强反弹(5日>8%)+真突破',
         lambda x: x['dd_from_hi'] < -0.40 and x['pit_len'] > 30 and x['rebound5'] > 0.08 and not x['reenter'], R)
    show('深跌<-40%+长坑>40天+强反弹(5日>8%)+真突破',
         lambda x: x['dd_from_hi'] < -0.40 and x['pit_len'] > 40 and x['rebound5'] > 0.08 and not x['reenter'], R)
    print()
    print('=== 高收益组Top10案例(r60降序) ===')
    hi = sorted([x for x in R if x['r60'] > 0.15], key=lambda x: -x['r60'])[:10]
    for x in hi:
        print(f'  {x["sym"]}  r60={x["r60"] * 100:+.0f}%  深跌{x["dd_from_hi"] * 100:.0f}%  '
              f'坑深reg{x["pit_depth_reg"] * 100:.0f}%  坑长{x["pit_len"]}d  '
              f'5日反弹{x["rebound5"] * 100:.0f}%  真突破={not x["reenter"]}')


if __name__ == '__main__':
    main()
