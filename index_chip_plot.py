# -*- coding: utf-8 -*-
"""概念指数筹码带画图(方案A: 换手率=量比代理, 临时脚本不入库)。

用法: python3 index_chip_plot.py <code> <tail> [end]
  code: 带交易所前缀的指数代码, 如 sh881479 / sh880001 / sz399001
  tail: 显示最后N天(默认150)
  end : anchor日期, 默认最新

换手率代理(方案A): turn = TURN_BASE * volume / MA20(volume), TURN_BASE=0.05
(fengwo 用 turn 驱动筹码衰减; 个股日换手 0.01~0.1, 指数无量纲, 用量比×基准校准)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from chip_panel import cost_series, chip_density_grid, CHIP_CMAP, CHIP_VMIN
from eltdx import Client
from eltdx_compat import get_bars

TURN_BASE = 0.05  # 量比基准(中位≈0.05=5%换手, 接近中换手个股)
NGRID = 1200
PS = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]


def fetch_index(code, end, datalen=800, anchor=False):
    """拉指数日线(返回升序; 无未来: 信号日只用≤当日)。anchor=True 锚定历史拉全量, 默认只拉最近 datalen 根。"""
    cli = Client(timeout=8.0)
    try:
        bars = get_bars(cli, code, 'day', datalen, end if anchor else None)
        rows = []
        for b in bars:
            rows.append(dict(date=b.time.date(), open=b.open, close=b.close,
                             high=b.high, low=b.low,
                             volume=b.volume_lots if b.volume_lots else 0.0,
                             amount=b.amount if b.amount else 0.0))
        df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
        # 只保留≤end(服务器可能多给)
        if end:
            df = df[df['date'].astype(str) <= end]
        return df
    finally:
        cli.close()



def plot_chip_band(df, tail=150, out=None, code='', vmax=3.0):
    """筹码带画图(方案A 量比代理换手率): K线 + 70%集中区色带(COST15-85) + COST50中线。
    返回输出路径; out=None 时不落盘(仅返回 fig 关闭前的摘要字符串)。
    无未来: 全部只用当日及以前数据。"""
    if df is None or len(df) < 300:
        return None
    H = df['high'].values.astype(float)
    L = df['low'].values.astype(float)
    C = df['close'].values.astype(float)
    V = df['volume'].values.astype(float)

    ma20 = np.convolve(V, np.ones(20) / 20, mode='valid')
    ma20 = np.concatenate([np.full(19, np.nan), ma20])
    ratio = np.where(ma20 > 0, V / ma20, 1.0)
    turn = (TURN_BASE * ratio).clip(0.005, 0.30)
    turn[:20] = TURN_BASE

    costs = {p: cost_series(H, L, V, turn, p) for p in PS}
    grid, dens = chip_density_grid(H, L, V, turn, ngrid=NGRID, nwin=tail, ext=0.2)

    N = min(tail, len(df))
    xs = np.arange(N) + 0.5
    dates = [str(d)[:10] for d in df['date'].values[-N:]]
    closes = C[-N:]

    fig, ax = plt.subplots(figsize=(15, 6.5))
    gmid = 0.5 * (grid[:-1] + grid[1:])
    dens = dens[:, -N:]
    for k in range(N):
        lo_b, hi_b = costs[0.15][-N + k], costs[0.85][-N + k]
        sel = (gmid >= lo_b) & (gmid <= hi_b)
        idx = np.where(sel)[0]
        if len(idx) == 0:
            continue
        pk = float(dens[idx, k].max())
        if pk <= 0:
            continue
        strength = min(pk / vmax, 1.0)
        for jj, j in enumerate(idx):
            rel = dens[jj, k] / pk
            t = (rel * strength) ** 0.85
            if t < CHIP_VMIN:
                continue
            ax.add_patch(plt.Rectangle((k, grid[j]), 1.0, grid[j + 1] - grid[j],
                                       facecolor=CHIP_CMAP(t), edgecolor='none',
                                       alpha=1.0))
    ax.plot(xs, closes, color='#101018', lw=1.8, label='指数收盘', zorder=6)
    mid = costs[0.45] + (costs[0.55] - costs[0.45]) / 2
    ax.plot(xs, mid[-N:], color='#006400', lw=1.5, label='COST50中线', zorder=6)
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, facecolor='#ff6a00', label='70%集中区 COST15-85'))
    ax.set_xlim(0, N)
    ylo, yhi = min(L[-N:]), max(H[-N:])
    pad = 0.02 * (yhi - ylo)
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_ylabel('指数点位')
    ax.set_title('概念指数筹码带 [方案A: turn=0.05×量比(volume/MA20), 网格%d] %s 末%d日' % (NGRID, code, N),
                 fontsize=10, loc='left')
    ax.legend(loc='upper left', fontsize=8)
    tick_idx = list(range(0, N, 30))
    ax.set_xticks([x + 0.5 for x in tick_idx])
    ax.set_xticklabels([dates[i] for i in tick_idx], rotation=45, fontsize=7)

    if out:
        plt.tight_layout()
        plt.savefig(out, dpi=100)
        plt.close()
    else:
        fig.canvas.draw()
        plt.close(fig)
    c50 = 0.5 * (costs[0.45][-1] + costs[0.55][-1])
    summary = dict(
        c15=float(costs[0.15][-1]), c35=float(costs[0.35][-1]),
        c50=float(c50), c75=float(costs[0.75][-1]), c85=float(costs[0.85][-1]),
        close=float(C[-1]), turn_med=float(np.nanmedian(turn[-N:])),
        dates=(dates[0], dates[-1]))
    return summary

def main():
    code = sys.argv[1] if len(sys.argv) > 1 else 'sh881479'
    tail = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    end = sys.argv[3] if len(sys.argv) > 3 else '20261008'
    df = fetch_index(code, end, datalen=1600, anchor=True)
    if df is None or len(df) < 300:
        print('数据不足:', code, len(df) if df is not None else 0)
        return
    s = plot_chip_band(df, tail=tail, out='result/index_chip_%s_%d.png' % (code.replace('/', '_'), tail), code=code)
    if not s:
        return
    print('日期范围: %s ~ %s' % s['dates'])
    print('换手代理 turn 中位: %.3f' % s['turn_med'])
    print('当前 COST15/35/50/75/85: %.2f/%.2f/%.2f/%.2f/%.2f' % (s['c15'], s['c35'], s['c50'], s['c75'], s['c85']))
