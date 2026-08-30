# -*- coding: utf-8 -*-
"""黄金坑v4双因子画图: 路径A(标准超跌)黄色, 路径B(贴近reg急跌)粉色
用法: python3 plot_v4_dual.py <code> <end_date> <days> [smooth_w]
示例: python3 plot_v4_dual.py 600865 2025-12-31 300 10
"""
import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, '.')
from mean_reversion.signal_residual import compute_rolling_regression
from tdx_quant import get_daily_kline_from_tdx
from golden_pit_v4 import detect_golden_pit_v4

def main():
    code = sys.argv[1] if len(sys.argv) > 1 else '600865'
    end = sys.argv[2] if len(sys.argv) > 2 else '2025-12-31'
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    smooth_w = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    df = get_daily_kline_from_tdx(code, end, datalen=1200)
    c = df['close'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    o = df['open'].values.astype(float)
    dates = df['date'].values.astype('datetime64[D]')
    n = len(c)
    reg120, _ = compute_rolling_regression(c, window=120, use_log=True)
    reg250, _ = compute_rolling_regression(c, window=250, use_log=True)
    def sm(a, w):
        k = np.ones(w) / w
        return np.convolve(a, k, mode='same')
    reg120s = sm(reg120, smooth_w)
    reg250s = sm(reg250, smooth_w)
    pits = detect_golden_pit_v4(c, reg250s)
    print(f'{code} 截至{end}: {n}根K线, v4双因子坑={len(pits)}个, reg平滑={smooth_w}日')
    s_idx = max(0, n - N)
    c2 = c[s_idx:]; h2 = h[s_idx:]; l2 = l[s_idx:]; o2 = o[s_idx:]
    d2 = dates[s_idx:]; r120 = reg120s[s_idx:]; r250 = reg250s[s_idx:]
    xd = np.arange(len(c2))
    fig, ax = plt.subplots(figsize=(20, 9))
    for i in range(len(c2)):
        color = '#d32f2f' if c2[i] >= o2[i] else '#388e3c'
        ax.plot([xd[i], xd[i]], [l2[i], h2[i]], color=color, linewidth=0.8, zorder=2)
        ax.plot([xd[i]-0.3, xd[i]+0.3], [o2[i], o2[i]], color=color, linewidth=1.2, zorder=2)
        ax.plot([xd[i]-0.3, xd[i]+0.3], [c2[i], c2[i]], color=color, linewidth=1.2, zorder=2)
    ax.plot(xd, r120, color='#1976d2', linewidth=1.3, zorder=4)
    ax.plot(xd, r250, color='#7b1fa2', linewidth=1.6, zorder=4)
    for s, b, lch, ptype in pits:
        if s < s_idx or lch < s_idx:
            continue
        ss = s - s_idx; bb = b - s_idx; ll = lch - s_idx
        if ss >= len(c2) or ll >= len(c2):
            continue
        fc = '#fff59d' if ptype == 'A' else '#f8bbd0'
        ec = '#f9a825' if ptype == 'A' else '#c2185b'
        ax.axvspan(ss - 0.5, ll + 0.5, alpha=0.35, color=fc, zorder=1)
        ax.plot(bb, c2[bb], marker='v', color=ec, markersize=11, zorder=6)
        ax.plot(ll, c2[ll], marker='*', color=ec, markersize=14, zorder=6)
        ax.text(bb, c2[bb] * 0.985, f'{ptype}坑', ha='center', va='top', fontsize=8, color=ec, fontweight='bold')
    legend_elems = [
        Line2D([0], [0], color='#1976d2', lw=1.3, label=f'reg120(平滑{smooth_w}日)'),
        Line2D([0], [0], color='#7b1fa2', lw=1.6, label=f'reg250(平滑{smooth_w}日)'),
        Line2D([0], [0], marker='v', color='#f9a825', linestyle='None', markersize=9, label='路径A坑底(标准超跌)'),
        Line2D([0], [0], marker='*', color='#f9a825', linestyle='None', markersize=11, label='路径A出坑'),
        Line2D([0], [0], marker='v', color='#c2185b', linestyle='None', markersize=9, label='路径B坑底(贴近reg急跌)'),
        Line2D([0], [0], marker='*', color='#c2185b', linestyle='None', markersize=11, label='路径B出坑'),
    ]
    ax.legend(handles=legend_elems, loc='upper left', fontsize=9, framealpha=0.9)
    tick_step = max(1, len(c2) // 15)
    ax.set_xticks(xd[::tick_step])
    ax.set_xticklabels([str(d2[i])[:10] for i in range(0, len(c2), tick_step)], rotation=45, ha='right', fontsize=8)
    ax.set_title(f'{code} 截至{end} 近{N}日 | V10+黄金坑v4双因子(路径A标准超跌/路径B贴近reg急跌) | reg平滑{smooth_w}日', fontsize=13, fontweight='bold')
    ax.set_ylabel('价格', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, len(c2))
    plt.tight_layout()
    import os
    os.makedirs('result', exist_ok=True)
    outp = f'result/{code}_v4_dual_{end}.png'
    plt.savefig(outp, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'保存: {outp}')

if __name__ == '__main__':
    main()
