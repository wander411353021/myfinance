# -*- coding: utf-8 -*-
"""筹码分布(CYQ)色带 panel 绘制(fengwo版) — 供 V10 画图集成。
通达信 WINNER/COST 逐日递推, 第i天只用≤i数据 → 无未来函数(红线一致)。
渲染定版(2026-09-29 polo4111 验收):
  - 70%集中区 = COST(15%)~COST(85%), 区内按密度上色, 区外不画(悬空色带)
  - 深绿中心线 = COST50(45%/55%中点 = 中位成本)
  - 网格 1500 格, 筹码主体带定界(窗口内每天 COST5/95 中位数 ±0.2 带宽) → 最低覆盖率≥80%
  - 每天归一 × 绝对强度折扣(strength=min(峰值密度/VMULT,1), VMULT=3): 分散日假高亮被抑制
  - 配色: 浅蓝→深红渐变(gamma0.85 偏浅); 柱宽100%填满
  - Y轴=窗口价格范围(与K线面板对齐, 便于对应)
用法(作为 panel 集成): draw_chip_panel(ax, df_ohlc, tail_days, code, end_date)
df_ohlc 需含 turnover/circ_shares 列(来自 tdx_quant.get_daily_kline_from_tdx);
缺失时用 code+end_date 从 tdx_quant 补拉对齐。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CHIP_CMAP = LinearSegmentedColormap.from_list(
    'chip', ['#eef6ff', '#c9dcf7', '#94bdee', '#5f9fd8',
             '#f5b15e', '#e87a3a', '#d0451f'])
# 浅色不显示: 密度 t = (rel*strength)^0.85 < CHIP_VMIN 的格子跳过(透明),
# 颜色映射范围不变(浅蓝→深红), 只是蓝色等浅色段不画
# 0.67 = 橙色锚点 #f5b15e: 阈值提到橙色起点, 基本只留橙红
CHIP_VMIN = 0.67


def cost_series(H, L, V, turn, p):
    """fengwo COST: 获利盘占比达 p 时的价格序列(逐日递推无未来)。"""
    c = __import__('fengwo').COST(H, L, V, turn, np.array([float(p)] * len(H)))
    if c is None:
        raise RuntimeError(f'COST({p}) 返回 None')
    return np.asarray(c, dtype=float)


def chip_density_grid(H, L, V, turn, ngrid=1500, nwin=None, ext=0.2):
    """绝对筹码密度(固定网格, 筹码主体带定界):
    网格边界 = 窗口内每天 COST(5%)~COST(95%) 的中位数, 再向外扩展 ext;
    对每个网格价调用 WINNER 得累积 CDF, 差分 = 每格筹码占比(每日总和=1);
    返回 (grid, mult), mult = 每格占比/均匀占比(密度倍数, 跨股价可比)。"""
    import fengwo
    T = len(H)
    if nwin is None or nwin > T:
        nwin = T
    c_lo = np.asarray(fengwo.COST(H, L, V, turn, np.array([0.05] * T)), dtype=float)
    c_hi = np.asarray(fengwo.COST(H, L, V, turn, np.array([0.95] * T)), dtype=float)
    lo, hi = np.median(c_lo[-nwin:]), np.median(c_hi[-nwin:])
    w = hi - lo
    lo, hi = lo - ext * w, hi + ext * w
    grid = np.linspace(lo, hi, ngrid)
    cdf = np.zeros((ngrid, len(H)))
    for j in range(ngrid):
        wj = fengwo.WINNER(H, L, V, turn, np.full(len(H), grid[j]))
        cdf[j] = np.asarray(wj, dtype=float)
    cdf = np.maximum.accumulate(cdf, axis=0)          # 单调化(数值误差)
    dens = np.diff(cdf, axis=0)                       # (ngrid-1, T) 每格筹码占比
    uniform = 1.0 / (ngrid - 1)
    mult = dens / uniform
    return grid, mult


def _ensure_turnover(df_ohlc, code, end_date):
    """df_ohlc 缺 turnover 时, 用 code+end_date 从 tdx_quant 补拉并按键对齐。"""
    if 'turnover' in df_ohlc.columns:
        return df_ohlc
    if not code:
        return df_ohlc
    try:
        from tdx_quant import get_daily_kline_from_tdx
        d2 = get_daily_kline_from_tdx(code, end_date, datalen=len(df_ohlc))
        if d2 is None or len(d2) == 0:
            return df_ohlc
        m = d2[['date', 'turnover', 'circ_shares']].copy()
        m['date'] = pd.to_datetime(m['date']).dt.normalize()
        out = df_ohlc.copy()
        out['date'] = pd.to_datetime(out['date']).dt.normalize()
        out = out.merge(m, on='date', how='left')
        return out
    except Exception as e:
        print(f'[chip_panel] turnover 补拉失败: {e}')
        return df_ohlc


def draw_chip_panel(ax, df_ohlc, tail_days, code=None, end_date=None,
                    vmax=3.0, ngrid=1500, ext=0.2, show_stats=True,
                    overlay=False, alpha=0.40):
    """在指定 ax 上画筹码色带(fengwo 定版), 返回峰值密度序列。
    数据: df_ohlc 全量(逐日递推无未来), 只显示最后 tail_days 天。
    overlay=True: 叠加到已有价格坐标的 ax(如K线面板), 不设置 xlim/ylim/标题/图例,
    只画半透明色块(alpha), 不画收盘价/COST50线, 供 K线+筹码 复合面板使用。"""
    import pandas as pd
    df = _ensure_turnover(df_ohlc, code, end_date)
    if 'turnover' not in df.columns:
        ax.text(0.5, 0.5, '无 turnover 列(需 tdx_quant.get_daily_kline_from_tdx)',
                transform=ax.transAxes, ha='center', va='center', fontsize=10, color='#B71C1C')
        return np.zeros(tail_days)
    d = df.sort_values('date').reset_index(drop=True)
    H = d['high'].values.astype(float); L = d['low'].values.astype(float)
    C = d['close'].values.astype(float); V = d['volume'].values.astype(float)
    turn = d['turnover'].values.astype(float).clip(0, 1)

    ps = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    costs = {p: cost_series(H, L, V, turn, p) for p in ps}
    grid, dens = chip_density_grid(H, L, V, turn, ngrid=ngrid, nwin=tail_days, ext=ext)

    dates = [str(x)[:10] for x in d['date'].values]
    closes = list(C)
    mids = list(costs[0.45] + (costs[0.55] - costs[0.45]) / 2)
    N = min(tail_days, len(dates))
    dates = dates[-N:]; closes = closes[-N:]; mids = mids[-N:]
    for p in ps:
        costs[p] = costs[p][-N:]
    dens = dens[:, -N:]

    gmid = 0.5 * (grid[:-1] + grid[1:])
    peak_mult = np.zeros(N)
    for k in range(N):
        lo_b, hi_b = costs[0.15][k], costs[0.85][k]
        sel = (gmid >= lo_b) & (gmid <= hi_b)
        idx = np.where(sel)[0]
        if len(idx) == 0:
            continue
        mult_v = dens[idx, k]
        pk = float(mult_v.max())
        peak_mult[k] = pk
        if pk <= 0:
            continue
        strength = min(pk / vmax, 1.0)
        for jj, j in enumerate(idx):
            rel = mult_v[jj] / pk
            t = (rel * strength) ** 0.85
            if t < CHIP_VMIN:
                continue  # 蓝色等浅色段不显示(透明), 颜色映射范围不变
            ax.add_patch(plt.Rectangle((k, grid[j]), 1.0, grid[j + 1] - grid[j],
                         facecolor=CHIP_CMAP(t),
                         edgecolor='none', alpha=alpha if overlay else 1.0))

    if overlay:
        return peak_mult  # 叠加模式: 不设坐标/标题/图例, 交给宿主面板

    xs = np.arange(N) + 0.5
    ax.plot(xs, closes, color='#101018', lw=1.6, label='收盘价', zorder=6)
    ax.plot(xs, mids, color='#006400', lw=1.5, label='COST50中位成本', zorder=6)
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, facecolor='#ff6a00',
                               label='70%集中区(fengwo COST15-85)'))
    ax.set_xlim(0, N)
    ax.set_ylabel('价格')
    ylo, yhi = min(L[-N:]), max(H[-N:])
    pad = 0.02 * (yhi - ylo)
    ax.set_ylim(ylo - pad, yhi + pad)
    if show_stats:
        nz = int(np.median([(dens[:, k] > 0.05).sum() for k in range(N)]))
        cov = 100.0 * nz / (grid.size - 1)
        ax.set_title(f'筹码色带 [fengwo COST 70%集中区+COST50中线, 网格{ngrid}格(筹码主体带定界, '
                     f'非空{nz}格/{cov:.0f}%), 每天归一×绝对强度(峰值≥{vmax:.0f}x才深红), '
                     f'浅色不显示(t<{CHIP_VMIN})]',
                     fontsize=9, loc='left', pad=2)
    ax.legend(loc='upper left', fontsize=7)
    return peak_mult
