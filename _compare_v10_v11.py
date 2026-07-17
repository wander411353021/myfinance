"""
V10 vs V11 对比回测 — 多头 only。

V10: 原始逐层突破信号 + 朴素策略（空仓收到 +1 入场，收到 -1 出场，无止损）。
V11: 信号经假突破过滤(成交量/ATR) + 持仓状态机(ATR 移动止损)。

指标（基于逐日盯市权益曲线，更贴近实盘）:
  trades / win% / avgRet / totalRet / avgHold(bar) / maxDD / stops(V11)
"""
import sys, os
sys.path.insert(0, r'E:\chip_analyzer_ui\new_algo')

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from price_segmenter_v10 import CausalIncrementalPriceSegmenter as Seg10, compute_buy_sell_signals as cbs10
from price_segmenter_v11 import (CausalIncrementalPriceSegmenter as Seg11,
                                  compute_buy_sell_signals as cbs11,
                                  compute_trades_v11)
from juejing import get_stock_klines_from_juejing

WORK = r'E:\chip_analyzer_ui\new_algo'
RESULT = os.path.join(WORK, 'result')
CACHE = os.path.join(RESULT, '_cache')
os.makedirs(CACHE, exist_ok=True)

# ── V11 可调参数（迭代时改这里） ──
V11_PARAMS = dict(vol_confirm_mult=1.5, atr_mult=1.0, stop_mult=3.0, risk_pct=0.10)

STOCKS = [
    ('000066', '20241228', 250),
    ('300437', '20210910', 150),
    ('688387', '20250915', 150),
]


def load(code, end):
    p = os.path.join(CACHE, f'{code}_{end}.pkl')
    if os.path.exists(p):
        return pd.read_pickle(p)
    df = get_stock_klines_from_juejing(code, end, fqt=1)
    df = df[df['close'] > 0].reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df.to_pickle(p)
    print(f"  fetched {code}: {len(df)} rows")
    return df


def backtest_naive(bs_signal, close):
    """V10 朴素策略：空仓+买入信号入场，卖出信号出场，无止损。"""
    n = len(close); pos = 0; ep = 0.0; ei = 0; trades = []
    for t in range(n):
        if pos == 0 and bs_signal[t] > 0:
            pos = 1; ep = close[t]; ei = t
        elif pos == 1 and bs_signal[t] < 0:
            pos = 0; trades.append((ei, t, ep, close[t], 'SIGNAL'))
    if pos == 1:
        trades.append((ei, n - 1, ep, close[-1], 'EOD'))
    return trades


def daily_equity(trades, close):
    n = len(close); pos = np.zeros(n)
    for ei, xi, ep, xp, _ in trades:
        pos[ei:xi + 1] = 1
    eq = np.ones(n)
    for t in range(1, n):
        if pos[t] == 1:
            eq[t] = eq[t - 1] * (1 + (close[t] - close[t - 1]) / close[t - 1])
        else:
            eq[t] = eq[t - 1]
    return eq


def metrics(trades, close):
    if not trades:
        return dict(n=0, win=0, winrate=0.0, avg=0.0, total=0.0, avg_hold=0.0, maxdd=0.0, stops=0)
    rets = [(x[3] - x[2]) / x[2] for x in trades]
    wins = sum(1 for r in rets if r > 0)
    holds = [x[1] - x[0] for x in trades]
    stops = sum(1 for x in trades if x[4] == 'STOP')
    eq = daily_equity(trades, close)
    total = eq[-1] - 1
    peak = np.maximum.accumulate(eq)
    maxdd = float(np.min(eq / peak - 1))
    return dict(n=len(trades), win=wins, winrate=wins / len(trades),
                avg=float(np.mean(rets)), total=float(total),
                avg_hold=float(np.mean(holds)), maxdd=maxdd, stops=stops)


def plot_compare(code, df, t10, t11, close):
    fig, ax = plt.subplots(2, 1, figsize=(20, 10), height_ratios=[3, 1], sharex=True)
    x = np.arange(len(close))
    ax[0].plot(x, close, color='#2C2C2A', linewidth=1.0, label='close')
    for ei, xi, ep, xp, _ in t10:
        ax[0].plot(ei, ep, '^', ms=7, color='#42A5F5', zorder=5)
        ax[0].plot(xi, xp, 'v', ms=7, color='#FF7043', zorder=5)
    for ei, xi, ep, xp, _ in t11:
        ax[0].plot(ei, ep, '^', ms=10, markerfacecolor='none', markeredgecolor='#0D47A1', markeredgewidth=1.6, zorder=6)
        ax[0].plot(xi, xp, 'v', ms=10, markerfacecolor='none', markeredgecolor='#B71C1C', markeredgewidth=1.6, zorder=6)
    ax[0].set_title(f'{code}  V10 naive (solid) vs V11 (outline)', fontsize=12, fontweight='bold')
    ax[0].legend(handles=[
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#42A5F5', markersize=7, label='V10 buy'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#FF7043', markersize=7, label='V10 sell'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='none', markeredgecolor='#0D47A1', markersize=9, label='V11 buy'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='none', markeredgecolor='#B71C1C', markersize=9, label='V11 sell'),
    ], loc='upper left', fontsize=8, ncol=4)
    ax[0].grid(True, alpha=0.25)
    eq10 = daily_equity(t10, close); eq11 = daily_equity(t11, close)
    ax[1].plot(x, eq10, color='#42A5F5', linewidth=1.2, label='V10 naive')
    ax[1].plot(x, eq11, color='#0D47A1', linewidth=1.2, label='V11')
    ax[1].axhline(1, color='gray', linewidth=0.5, alpha=0.5)
    ax[1].set_title('Equity curve (long-only, marked-to-market)', fontsize=10)
    ax[1].legend(loc='upper left', fontsize=8); ax[1].grid(True, alpha=0.25)
    plt.tight_layout()
    save = os.path.join(RESULT, f'{code}_compare_v10_v11.png')
    plt.savefig(save, dpi=130, bbox_inches='tight'); plt.close()
    return save


def main():
    print(f"V11 params: {V11_PARAMS}")
    print(f"{'code':<10}{'rows':>6}{'strat':<12}{'trades':>7}{'win%':>7}{'avgRet':>9}{'totRet':>9}{'hold':>7}{'maxDD':>9}{'stops':>7}")
    rows = []
    for code, end, tail in STOCKS:
        df = load(code, end)
        close = df['close'].values
        # V10
        r10 = Seg10().segment(close, df['volume'].values, high=df['high'].values,
                              low=df['low'].values, opn=df['open'].values)
        bs10, _, _, _ = cbs10(df, r10)
        t10 = backtest_naive(bs10, close)
        # V11
        r11 = Seg11().segment(close, df['volume'].values, high=df['high'].values,
                              low=df['low'].values, opn=df['open'].values)
        bs11, _ = cbs11(df, r11, vol_confirm_mult=V11_PARAMS['vol_confirm_mult'],
                        atr_mult=V11_PARAMS['atr_mult'])
        pos11, act11, t11 = compute_trades_v11(df, r11, bs11, _,
                                              stop_mult=V11_PARAMS['stop_mult'], risk_pct=V11_PARAMS['risk_pct'])
        m10 = metrics(t10, close); m11 = metrics(t11, close)
        print(f"{code:<10}{len(df):>6}{'V10':<12}{m10['n']:>7}{m10['winrate']*100:>6.1f}%{m10['avg']*100:>8.2f}%{m10['total']*100:>8.2f}%{m10['avg_hold']:>7.1f}{m10['maxdd']*100:>8.2f}%{'-':>7}")
        print(f"{'':<10}{'':>6}{'V11':<12}{m11['n']:>7}{m11['winrate']*100:>6.1f}%{m11['avg']*100:>8.2f}%{m11['total']*100:>8.2f}%{m11['avg_hold']:>7.1f}{m11['maxdd']*100:>8.2f}%{m11['stops']:>7}")
        rows.append((code, m10, m11))
        save = plot_compare(code, df, t10, t11, close)
        print(f"  chart -> {save}")
    # 汇总 CSV
    csv = os.path.join(RESULT, 'v10_v11_compare.csv')
    with open(csv, 'w') as f:
        f.write('code,strat,trades,winrate,avg_ret,total_ret,avg_hold,max_dd,stops\n')
        for code, m10, m11 in rows:
            f.write(f"{code},V10,{m10['n']},{m10['winrate']:.4f},{m10['avg']:.4f},{m10['total']:.4f},{m10['avg_hold']:.2f},{m10['maxdd']:.4f},0\n")
            f.write(f"{code},V11,{m11['n']},{m11['winrate']:.4f},{m11['avg']:.4f},{m11['total']:.4f},{m11['avg_hold']:.2f},{m11['maxdd']:.4f},{m11['stops']}\n")
    print(f"\nsummary -> {csv}")


if __name__ == '__main__':
    main()
