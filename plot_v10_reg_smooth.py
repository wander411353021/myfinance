# -*- coding: utf-8 -*-
"""V10 画图 + 黄金坑v3(双基准) + reg平滑。

对 reg120/reg250 做滚动平均平滑后再画图和坑检测, 减少reg毛刺。
用法: python3 plot_v10_reg_smooth.py <code> <end_date> <tail_days> [smooth_w]
示例: python3 plot_v10_reg_smooth.py 688099 2025-11-01 150 10
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression
from tdx_quant import get_daily_kline_from_tdx
from price_segmenter_v10 import (CausalIncrementalPriceSegmenter,
                                  compute_buy_sell_signals, plot_price_segmentation_v10)


def smooth_reg(reg, w=10):
    """滚动平均平滑, w=1不平滑。对NaN区间保持原值。"""
    if w <= 1:
        return reg
    n = len(reg); sm = np.full(n, np.nan)
    for i in range(w - 1, n):
        seg = reg[i - w + 1:i + 1]
        if np.all(np.isfinite(seg)):
            sm[i] = np.mean(seg)
    return np.where(np.isfinite(sm), sm, reg)


def draw(symbol, end_date, tail_days=150, smooth_w=10, out_path=None):
    df = get_daily_kline_from_tdx(symbol, end_date, datalen=max(1200, tail_days + 400))
    if df is None or len(df) < 100:
        print(f'{symbol}: 数据不足'); return None
    c = df['close'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    o = df['open'].values.astype(float)
    v = df['volume'].values.astype(float)
    n = len(c)

    # 价格分段 + 买卖信号
    seg = CausalIncrementalPriceSegmenter(lookback=15, min_reversal_pct=0.02,
                                           confirm_bars=3, same_type_merge_gap=20)
    result = seg.segment(c, v, high=h, low=l, opn=o)
    bs_signal, bs_reason, bs_strength, all_levels = compute_buy_sell_signals(
        df, result, dur_horizon=120, touch_norm=3)

    # 原始reg + 平滑
    reg120_raw, _ = compute_rolling_regression(c, window=120, use_log=True)
    reg250_raw, _ = compute_rolling_regression(c, window=250, use_log=True)
    reg120 = smooth_reg(reg120_raw, smooth_w)
    reg250 = smooth_reg(reg250_raw, smooth_w)

    # 黄金坑v3(双基准) 用平滑后reg, smooth_w=1(不再二次平滑)
    pits = [p for p in pr.detect_golden_pit_v3(
        c, reg250, reg120=reg120, use_dual=True, smooth_w=1,
        min_depth=0.08, confirm_days=3) if p[2] is not None]
    print(f'{symbol} 截至{end_date}: {n}根K线, 黄金坑v3={len(pits)}个, reg平滑={smooth_w}日')

    # v10画图 (monkey-patch阻止plt.close, 以便后续标注黄金坑)
    _orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_price_segmentation_v10(
        df, result, bs_signal, bs_reason,
        tail_days=tail_days, name=f'{symbol} 截至{end_date} (reg平滑{smooth_w}日)',
        save_path=None, bs_strength=bs_strength, all_levels=all_levels,
        reg_preds=reg120, reg_preds_long=reg250,
        hide_ma=True, reg_win=120, reg_win_long=250, panic_info=None)
    plt.close = _orig_close

    # 在ax0上标注黄金坑
    fig = plt.gcf(); ax0 = fig.axes[0]
    offset = n - tail_days
    for s, b, lch in pits:
        if lch < offset - 2 or s > n - 1:
            continue
        xs = max(0, s - offset); xb = b - offset; xl = lch - offset
        ax0.axvspan(xs - 0.5, xl + 0.5, alpha=0.10, color='#FFD600', zorder=1)
        if 0 <= xb < tail_days:
            ax0.scatter([xb], [c[b]], s=160, marker='v', color='#D50000',
                        zorder=11, edgecolors='black', linewidths=0.8)
            ax0.annotate(f'坑底{c[b]:.2f}', (xb, c[b]), textcoords='offset points',
                        xytext=(0, -24), ha='center', fontsize=8, color='#D50000', fontweight='bold')
        if 0 <= xl < tail_days:
            ax0.scatter([xl], [c[lch]], s=200, marker='*', color='#FF6F00',
                        zorder=12, edgecolors='black', linewidths=0.8)
            ax0.annotate(f'v3出坑{c[lch]:.2f}', (xl, c[lch]), textcoords='offset points',
                        xytext=(0, 18), ha='center', fontsize=8, color='#E65100', fontweight='bold')
        midx = (xs + xl) // 2
        if 0 <= midx < tail_days:
            ax0.text(midx, c[max(0, s):lch + 1].min() * 0.97, f'坑长{lch - s}d',
                    fontsize=7, color='#F57F17', ha='center', fontweight='bold', zorder=10)

    # 阶梯分段目标价(2026-09-02 polo4111): max(reg120,250)阶梯, 偏离>20%置空
    # 必须用与V10内部一致的 double_smooth(5,5) reg 计算, 否则目标价线与图上reg250线口径不一致(目标可能显示低于reg250)
    try:
        _reg120_ds = pr.double_smooth_reg(reg120, 5, 5)
        _reg250_ds = pr.double_smooth_reg(reg250, 5, 5)
        _glv_def = (-0.09, -0.06, -0.03, 0.00, 0.03, 0.06, 0.09, 0.12)  # 完整3%间隔网格(含图上未显示的-6%/0%/+6%/+9%等档)
        _gt, _gl = pr.compute_grid_target_price(c, _reg120_ds, _reg250_ds, levels=_glv_def, max_dev=0.20, down_confirm=10)
        _gt_win = _gt[offset:offset + tail_days]
        _x_win = np.arange(tail_days)
        _m = np.isfinite(_gt_win)
        if _m.any():
            ax0.plot(_x_win[_m], _gt_win[_m], color='#D81B60', lw=3.2,
                     alpha=1.0, zorder=13, label='Grid Target (阶梯, 偏离>20%置空)')
            # 标注当前档位(按实际档位显示百分比, 支持负档)
            _cur = _gl[-1]
            if _cur >= 0:
                _tgt = _gt[-1]
                _pct = _glv_def[_cur] * 100
                ax0.annotate(f'目标{_tgt:.2f} ({_pct:+.0f}%)',
                             (tail_days-1, _tgt), textcoords='offset points',
                             xytext=(-70, 22), fontsize=9, color='#D81B60',
                             fontweight='bold', arrowprops=dict(arrowstyle='-', color='#D81B60', lw=0.8))
            else:
                ax0.annotate('目标置空(偏离reg>20%)', (tail_days-1, c[-1]),
                             textcoords='offset points', xytext=(-110, -8),
                             fontsize=8, color='#888888', fontweight='bold')
    except Exception as _e:
        print(f'[grid target] 绘制失败: {_e}')

    if out_path is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{symbol}_v10_smooth{smooth_w}_{end_date}.png')
    plt.savefig(out_path, dpi=130, bbox_inches='tight'); plt.close()
    print(f'  保存: {out_path}')
    return out_path


if __name__ == '__main__':
    symbol = sys.argv[1] if len(sys.argv) > 1 else '688099'
    end_date = sys.argv[2] if len(sys.argv) > 2 else '2025-11-01'
    tail_days = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    smooth_w = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    draw(symbol, end_date, tail_days, smooth_w)
