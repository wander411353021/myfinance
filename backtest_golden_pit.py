# -*- coding: utf-8 -*-
"""黄金坑体系回测 + 可视化(买卖点 + 收益曲线)。

策略(已验证 93% 超高信号):
  买入:黄金坑 + 快启动(≤5天) + 出坑日买入(出坑 = 收复门控线且 >坑底×1.02)
  # 规则F(2026-08-28): 坑长≥8约束已移除——短坑V反胜率更高
  加仓(可选):出坑后 7 天内放量堆峰值量比≥5
  卖出规则(可配):止损 / 止盈 / 到期(交易日)

用法:
    from backtest_golden_pit import backtest_single, plot_backtest
    trades, equity, dates = backtest_single('sz300437', '20251230')
    plot_backtest(df, trades, equity, dates, name='300437')
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression


def backtest_single(code, end_date, stop=-0.15, take=0.30, horizon=60,
                    require_super=False, start_capital=1.0):
    """单只回测:黄金坑+快启动信号,出坑日买入,规则卖出。

    返回 (trades, equity, eq_dates):
      trades: [{code, buy_date, buy_price, sell_date, sell_price, ret, super, reason}]
      equity: 累计收益曲线(逐交易日,等权多笔)
    """
    df = pr._load_df(code, end_date)
    if df is None or len(df) < 300:
        return [], None, None
    c = df['close'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    v = df['volume'].values.astype(float)
    dates = df['date'].dt.strftime('%Y%m%d').values
    n = len(c)
    reg250, _ = compute_rolling_regression(c, window=250, use_log=True)
    pits = pr.detect_golden_pit(c, reg250)
    cl = pr.detect_volume_clusters(c, v)

    trades = []
    for s, b, lch in pits:
        if lch is None or lch + horizon >= n:
            continue
        if lch - b > 5:
            continue  # 快启动(坑底后5天内收复)
        # 规则F(2026-08-28): 坑长≥8约束移除。短坑V反胜率更高(见 golden_pit_rule_compare.py)
        # 超高:出坑后7天内放量堆峰值≥5
        post_peak = max([pp for ss, ee, kk, dd, pp, vv in cl
                         if kk == 'HIGH' and lch < ss <= lch + 7] or [0])
        super_ok = post_peak >= 5
        if require_super and not super_ok:
            continue
        # 买入:出坑日
        buy_px = c[lch]
        sell_px = None
        sell_i = None
        reason = 'horizon'
        for i in range(lch + 1, lch + horizon + 1):
            if i >= n:
                break
            if c[i] <= buy_px * (1 + stop):
                sell_px = c[i]
                sell_i = i
                reason = 'stop'
                break
            if c[i] >= buy_px * (1 + take):
                sell_px = c[i]
                sell_i = i
                reason = 'take'
                break
        if sell_px is None:
            sell_i = min(lch + horizon, n - 1)
            sell_px = c[sell_i]
        ret = sell_px / buy_px - 1
        trades.append({
            'code': code, 'buy_date': dates[lch], 'buy_price': buy_px,
            'sell_date': dates[sell_i], 'sell_price': sell_px, 'ret': ret,
            'super': super_ok, 'reason': reason,
        })

    # 收益曲线:按买入日排序,等权资金逐笔累乘
    trades.sort(key=lambda t: t['buy_date'])
    if not trades:
        return trades, None, None
    eq = np.ones(n)
    cap = start_capital
    for t in trades:
        bi = np.where(dates == t['buy_date'])[0]
        si = np.where(dates == t['sell_date'])[0]
        if len(bi) == 0 or len(si) == 0:
            continue
        bi, si = bi[0], si[0]
        # 资金从买入日投入,卖出日结算
        eq[bi:] = eq[bi:] * (1 + t['ret']) / (1 + t['ret']) * (1 + t['ret']) if False else eq[bi:]
        eq[bi:] = eq[bi:] * (1 + t['ret']) if si >= bi else eq
    eq = eq / eq[0]
    return trades, eq, dates


def plot_backtest(df, trades, equity, eq_dates, name='', save_path=None,
                  tail_days=400, show_super_only=False):
    """可视化:上=K线+买卖点,下=收益曲线。"""
    c = df['close'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    o = df['open'].values.astype(float)
    dates = df['date'].dt.strftime('%Y%m%d').values
    n = len(c)
    off = max(0, n - tail_days)
    x = np.arange(n - off)
    dd = pd.to_datetime(dates[off:])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True,
                                   gridspec_kw={'height_ratios': [3.2, 1.0]})
    fig.suptitle(f'{name} 黄金坑回测 (买入=出坑日, 止损{stop_pct if False else ""}止盈/到期)', fontsize=13, fontweight='bold')

    # K线
    for i in range(n - off):
        color = '#E53935' if c[off + i] >= o[off + i] else '#2E7D32'
        ax1.plot([i, i], [l[off + i], h[off + i]], color=color, linewidth=0.6)
        ax1.add_patch(plt.Rectangle((i - 0.3, min(o[off + i], c[off + i])), 0.6,
                                    abs(c[off + i] - o[off + i]) or 1e-6, color=color))
    # 买卖点
    for t in trades:
        bi = np.where(dates == t['buy_date'])[0]
        si = np.where(dates == t['sell_date'])[0]
        if len(bi) == 0 or len(si) == 0:
            continue
        bi, si = bi[0], si[0]
        if bi < off or bi >= n:
            continue
        if show_super_only and not t['super']:
            continue
        bx, sx = bi - off, si - off
        if sx < 0:
            continue
        bx = max(0, bx)
        color = '#FF8F00' if t['super'] else '#1976D2'
        ax1.plot(bx, t['buy_price'], '^', color=color, markersize=12, zorder=8)
        ax1.plot(min(sx, n - off - 1), t['sell_price'], 'v', color='#7B1FA2', markersize=12, zorder=8)
        ax1.annotate(f"{t['ret']:+.0%}" + ('★' if t['super'] else ''),
                     (min(sx, n - off - 1), t['sell_price']), textcoords='offset points',
                     xytext=(2, 8), fontsize=7, color='#7B1FA2')
    ax1.set_ylabel('Price'); ax1.grid(True, alpha=0.2)

    # 收益曲线
    if equity is not None and eq_dates is not None:
        eq_win = equity[off:]
        ax2.plot(x, eq_win, color='#1565C0', linewidth=1.8)
        ax2.axhline(1.0, color='#999', linewidth=0.8, linestyle='--')
        ax2.fill_between(x, 1.0, eq_win, where=(eq_win >= 1.0), color='#E53935', alpha=0.25)
        ax2.fill_between(x, 1.0, eq_win, where=(eq_win < 1.0), color='#2E7D32', alpha=0.25)
        ax2.set_ylabel('Equity')
        ax2.grid(True, alpha=0.2)
        final = eq_win[-1]
        ax2.set_title(f'收益曲线 期末={final - 1:+.1%}', fontsize=9, loc='left')

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=110)
        plt.close(fig)
        return save_path
    return fig
