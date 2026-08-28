# -*- coding: utf-8 -*-
"""规则F黄金坑画图：从 .cache_kline 加载数据，规则F(z<-1.5+快启动≤5)检测，
标注 2026 年黄金坑买卖点 + 收益曲线。用法:
  python3 golden_pit_plot_ruleF.py sh688337 2026  # 画指定股、指定年份信号
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden_pit_rule_compare as rc
from golden_pit_v2_backtest import compute_rolling_regression
import panic_reversal as pr


def load_df(symbol):
    d = rc.load(symbol)
    if d is None:
        return None
    df = pd.DataFrame({
        'date': pd.to_datetime(d['ts'].astype(np.int64), unit='s'),
        'open': d['high'].copy(),   # 缓存无open,占位
        'high': d['high'], 'low': d['low'],
        'close': d['close'], 'volume': d['vol'],
    })
    return df


def detect_ruleF(df, horizon=20):
    """规则F: z<-1.5 + 快启动(lch-b<=5) + 出坑确认, 出坑日买入, 持有horizon日"""
    c = df['close'].values.astype(float)
    v = df['volume'].values.astype(float)
    n = len(c)
    dates = df['date'].dt.strftime('%Y%m%d').values
    reg250, _ = compute_rolling_regression(c, window=250, use_log=True)
    pits = pr.detect_golden_pit(c, reg250)
    trades = []
    for s, b, lch in pits:
        if lch is None or lch + horizon >= n:
            continue
        if lch - b > 5:   # 快启动(规则F核心)
            continue
        ret = c[lch + horizon] / c[lch] - 1.0
        trades.append({
            'code': symbol, 'buy_date': dates[lch], 'buy_price': c[lch],
            'sell_date': dates[lch + horizon], 'sell_price': c[lch + horizon],
            'ret': ret, 'super': False, 'pit_len': b - s + 1, 'launch': lch - b,
            'z_deep': None,
        })
    return trades, reg250, pits


def plot(symbol, year=None, save_path=None):
    df = load_df(symbol)
    if df is None:
        print(f'{symbol} 无缓存数据'); return None
    trades, reg250, pits = detect_ruleF(df)
    c = df['close'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    dates = df['date'].dt.strftime('%Y%m%d').values
    n = len(c)

    # 过滤年份
    show = [t for t in trades if year is None or t['buy_date'][:4] == str(year)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [3.2, 1.0]})
    # 只画最近 600 天; x 轴用 matplotlib 日期数字(不能用整数索引,否则 DateFormatter 会当 1970 起算)
    off = max(0, n - 600)
    x = mdates.date2num(df['date'].values[off:])

    # K线 + reg250
    for i in range(n - off):
        color = '#E53935' if c[off + i] >= df['open'].values[off + i] else '#2E7D32'
        ax1.plot([i, i], [l[off + i], h[off + i]], color=color, linewidth=0.5)
        ax1.add_patch(plt.Rectangle((i - 0.3, min(df['open'].values[off + i], c[off + i])), 0.6,
                                    abs(c[off + i] - df['open'].values[off + i]) or 1e-6, color=color))
    ax1.plot(x, reg250[off:], color='#1565C0', linewidth=1.4, label='reg250')

    # 标注坑区间(最近一年内出坑的)
    today = dates[-1]
    for t in show:
        bi = np.where(dates == t['buy_date'])[0][0]
        if bi < off:
            continue
        bx = bi - off
        # 出坑日竖线 + 买入标记
        ax1.axvline(bx, color='#FF6F00', linestyle='--', alpha=0.5, linewidth=1)
        ax1.plot(bx, t['buy_price'], '^', color='#FF6F00', markersize=13, zorder=8)
        ax1.annotate(f"买入 {t['buy_date']}  {t['ret']*100:+.1f}%"
                     f"\n坑长{t['pit_len']}d/启动{t['launch']}d",
                     (bx, t['buy_price']), textcoords='offset points',
                     xytext=(6, -22), fontsize=9, fontweight='bold', color='#FF6F00',
                     bbox=dict(boxstyle='round,pad=0.3', fc='#FFF3E0', ec='#FF6F00', alpha=0.9))
    ax1.plot([], [], '^', color='#FF6F00', markersize=13, label='规则F买入点(20日持有)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_title(f'{symbol} 黄金坑(规则F) {year or "全部"}信号  '
                  f'买入=出坑日, 20日持有', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Price'); ax1.grid(True, alpha=0.25)

    # 收益曲线(全部trades, 不按年份)
    if trades:
        eq = np.ones(n)
        for t in trades:
            bi = np.where(dates == t['buy_date'])[0]
            si = np.where(dates == t['sell_date'])[0]
            if len(bi) and len(si) and si[0] >= bi[0]:
                eq[bi[0]:] = eq[bi[0]:] * (1 + t['ret'])
        eq = eq / eq[0]
        ax2.plot(x, eq[off:], color='#1565C0', linewidth=1.8)
        ax2.axhline(1.0, color='#999', linewidth=0.8, linestyle='--')
        ax2.fill_between(x, 1.0, eq[off:], where=(eq[off:] >= 1.0), color='#E53935', alpha=0.25)
        ax2.fill_between(x, 1.0, eq[off:], where=(eq[off:] < 1.0), color='#2E7D32', alpha=0.25)
        ax2.set_title(f'规则F全信号累计收益曲线 期末={eq[-1]-1:+.1%}', fontsize=10, loc='left')

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=110)
        plt.close(fig)
        return save_path
    return fig


if __name__ == '__main__':
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'sh688337'
    year = sys.argv[2] if len(sys.argv) > 2 else '2026'
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result',
                       f'golden_pit_ruleF_{symbol}_{year}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    r = plot(symbol, year, save_path=out)
    print('已生成:', out)
    if r:
        print('信号:')
        trades, _, _ = detect_ruleF(load_df(symbol))
        for t in trades:
            if t['buy_date'][:4] == str(year):
                print(f"  {t['buy_date']} 买入{t['buy_price']:.2f} "
                      f"坑长{t['pit_len']}d 快启动{t['launch']}d 20日{t['ret']*100:+.1f}%")
