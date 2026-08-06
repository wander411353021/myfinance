"""
四阶段模式可视化工具
====================

核心思路：牛股在起涨前通常经历四个阶段

  S1 恐慌下跌  — 大幅下跌制造恐慌，清洗浮筹
  S2 拉升吸筹  — 快速拉升形成第一个明显高点（吸筹），之后自由下跌
  S3 震荡整理  — 在 S2 最高点和均值线之间来回震荡，筹码集中
  S4 突破拉升  — 突破 S2 最高点，开启主升浪

均值线：使用 mean_reversion 模块的滚动回归线
最高线：S1开始~S2结束区间的最高价（即 S2 最高点）
"""

import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from matplotlib.patches import Patch, Rectangle

# ── 中文支持（优先使用微软雅黑） ──
_CN_CANDIDATES = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'STKaiti',
                  'FangSong', 'STSong']
_CN_FONT = None
for f in fm.fontManager.ttflist:
    if f.name in _CN_CANDIDATES:
        _CN_FONT = f.name
        break
if _CN_FONT:
    plt.rcParams['font.sans-serif'] = [_CN_FONT, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdx_quant import get_daily_kline_from_tdx
from mean_reversion.signal_residual import compute_rolling_regression

# ── 阶段配色 ──
PHASE_STYLES = {
    'S1': {'face': '#FF6B6B', 'edge': '#E03131', 'alpha': 0.18, 'label': 'S1 恐慌下跌'},
    'S2': {'face': '#FFB347', 'edge': '#E67E22', 'alpha': 0.25, 'label': 'S2 拉升吸筹'},
    'S3': {'face': '#4ECDC4', 'edge': '#0D9488', 'alpha': 0.18, 'label': 'S3 震荡整理'},
    'S4': {'face': '#51CF66', 'edge': '#2F9E44', 'alpha': 0.18, 'label': 'S4 突破拉升'},
}


def visualize_four_phases(
    code: str,
    s1_start: str, s1_end: str,
    s2_start: str, s2_end: str,
    s3_start: str, s3_end: str,
    s4_start: str,
    s4_end_date: str = None,
    end_date: str = None,
    reg_window: int = 120,
    save_path: str = None,
    show: bool = True,
    xlim_left: str = None,
    xlim_right: str = None,
    title_suffix: str = '',
):
    """
    绘制四阶段模式可视化图。

    均值线：mean_reversion 滚动回归线（默认 120 日）
    最高线：S1开始~S2结束之间的最高价（即 S2 最高点）
    两条线都只在 S1~S4 范围内显示。
    """
    # ── 日期标准化 ──
    def norm_date(d):
        return d.replace('-', '')
    s1_s = norm_date(s1_start)
    s1_e = norm_date(s1_end)
    s2_s = norm_date(s2_start)
    s2_e = norm_date(s2_end)
    s3_s = norm_date(s3_start)
    s3_e = norm_date(s3_end)
    s4_s = norm_date(s4_start) if s4_start else None
    s4_e = norm_date(s4_end_date) if s4_end_date else None

    # 确定 end_date
    if end_date is None:
        if s4_s:
            s4_dt = pd.Timestamp(s4_s)
            end_date_dt = s4_dt + pd.DateOffset(days=180)
        else:
            # pending/pullback(尚未突破):从 S3 结束(数据末端)往后取
            end_date_dt = pd.Timestamp(s3_e) + pd.DateOffset(days=60)
        end_date = end_date_dt.strftime('%Y%m%d')
    else:
        end_date = norm_date(end_date)

    # ── 获取数据 ──
    print(f"获取 {code} 的历史数据，截止 {end_date}...")
    df = get_daily_kline_from_tdx(code, end_date)
    if df is None or len(df) == 0:
        print("!! 数据获取失败")
        return

    print(f"  数据量: {len(df)} 条, {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")

    # ── 日期去时区 ──
    if hasattr(df['date'].iloc[0], 'tz'):
        df['date'] = df['date'].dt.tz_localize(None)

    # ── 标记阶段 ──
    s1_ts = pd.Timestamp(s1_s); s1_te = pd.Timestamp(s1_e)
    s2_ts = pd.Timestamp(s2_s); s2_te = pd.Timestamp(s2_e)
    s3_ts = pd.Timestamp(s3_s); s3_te = pd.Timestamp(s3_e)
    s4_ts = pd.Timestamp(s4_s) if s4_s else None

    def mark_phase(d):
        if d < s1_ts:
            return 'BEFORE'
        elif d <= s1_te:
            return 'S1'
        elif d <= s2_te:
            return 'S2'
        elif s4_ts is None:
            return 'S3'     # 尚未突破:S4 开始前所有日期归 S3
        elif d < s4_ts:
            return 'S3'     # 连续覆盖 S2结束 ~ S4开始 之间所有日期
        else:
            return 'S4'     # S4 开始之后全归 S4

    df['phase'] = df['date'].map(mark_phase)
    phases_present = set(df['phase'].unique()) & {'S1', 'S2', 'S3', 'S4'}
    if not phases_present:
        print("!! 指定日期范围内没有覆盖到四个阶段")
        print(f"   数据日期范围: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
        return

    # ── 计算均值线（mean_reversion 滚动回归） ──
    closes = df['close'].values.astype(np.float64)
    reg_preds, reg_slopes = compute_rolling_regression(closes, window=reg_window, use_log=True)
    df['mean_line'] = reg_preds

    # ── 计算关键价位 ──
    s1_data = df[df['phase'] == 'S1']
    s2_data = df[df['phase'] == 'S2']
    s3_data = df[df['phase'] == 'S3']
    s4_data = df[df['phase'] == 'S4']

    s1_low = s1_data['low'].min() if len(s1_data) > 0 else None
    # 最高线 = S2 区间内的最高价(S2 拉升段天花板)。不用"S1起点~S2结束":
    # 当 S1 起点本身是历史高点时会画错(如 300437-2025 轮的 10.79 vs 10.47、
    # 600199-2020 轮的 6.71 vs 5.75)
    s2_high = s2_data['high'].max() if len(s2_data) > 0 else None

    # S2 高点日期
    s2_high_idx = s2_data['high'].idxmax() if len(s2_data) > 0 else None
    s2_high_date = df.loc[s2_high_idx, 'date'] if s2_high_idx is not None else None

    # S1 低点日期
    s1_low_idx = s1_data['low'].idxmin() if len(s1_data) > 0 else None
    s1_low_date = df.loc[s1_low_idx, 'date'] if s1_low_idx is not None else None

    s1_low_str = f"{s1_low:.2f}" if s1_low is not None else "N/A"
    s2_high_str = f"{s2_high:.2f}" if s2_high is not None else "N/A"
    print(f"  S1 最低价: {s1_low_str}  @ {s1_low_date.date() if s1_low_date is not None else 'N/A'}")
    print(f"  S2 最高价: {s2_high_str}  @ {s2_high_date.date() if s2_high_date is not None else 'N/A'}")
    print(f"  均值线: mean_reversion 滚动回归 ({reg_window}日)")

    # ── 阶段起止边界 ──
    phase_edges = {}
    for ph in ['S1', 'S2', 'S3', 'S4']:
        sub = df[df['phase'] == ph]
        if len(sub) > 0:
            phase_edges[ph] = (sub['date'].iloc[0], sub['date'].iloc[-1])

    # 整体显示范围：S1开始 ~ 数据末尾（或S4结束再多一点）
    xlim_left = pd.Timestamp(xlim_left) if xlim_left else s1_ts
    # 图右边界 = S4 终点(主升浪最高价日);未提供则延伸到数据末端
    xlim_right = (pd.Timestamp(xlim_right) if xlim_right
                  else (pd.Timestamp(s4_e) if s4_e is not None else df['date'].iloc[-1]))

    # =============================== 绘图 ===============================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)
    fig.patch.set_facecolor('#FAFAFA')

    dates = df['date'].values
    closes_arr = df['close'].values
    mean_line = df['mean_line'].values

    # ========== 主图 ==========
    ax1.set_facecolor('#FAFAFA')

    # Y轴范围
    y_axis_top = max(closes_arr) * 1.18
    y_axis_bot = min(closes_arr) * 0.85

    # 1. 阶段方框（连续延伸，无间隔）
    main_phases = ['S1', 'S2', 'S3', 'S4']
    # 对每个阶段，左边界 = 本阶段第一个交易日，右边界 = 下一阶段第一个交易日（S4到数据末尾）
    prev_left = None
    for i, ph in enumerate(main_phases):
        if ph not in phase_edges:
            continue
        style = PHASE_STYLES[ph]
        ph_left = phase_edges[ph][0]  # 本阶段第一个交易日

        if i + 1 < len(main_phases):
            next_ph = main_phases[i + 1]
            if next_ph in phase_edges:
                ph_right = phase_edges[next_ph][0]  # 延伸到下一阶段的第一天
            else:
                ph_right = xlim_right
        else:
            ph_right = xlim_right  # S4 延伸到数据末尾

        # 色块边界(保证相邻连续、不切 K 线):
        #   - S1 右界 与 S2 左界 前移一个K线宽(shared boundary = S2首日-1.5):
        #     S1 末根 K 线视觉并入 S2,S2 拉升段完整不被切;
        #   - 其余交界 = 下阶段首日-0.5(K 线间隙),逐段连续。
        offset_left = 1.5 if (ph == 'S2' and 'S1' in phase_edges) else 0.5
        offset_right = 1.5 if (ph == 'S1' and 'S2' in phase_edges) else 0.5
        x0 = mdates.date2num(ph_left) - offset_left
        if i + 1 < len(main_phases):
            next_ph = main_phases[i + 1]
            if next_ph in phase_edges:
                x1 = mdates.date2num(ph_right) - offset_right
            else:
                x1 = mdates.date2num(ph_right) + 0.5
        else:
            x1 = mdates.date2num(ph_right) + 0.5   # S4 延伸到数据末端+半个K线宽
        rect = Rectangle(
            (x0, y_axis_bot), x1 - x0, y_axis_top - y_axis_bot,
            facecolor=style['face'], edgecolor=style['edge'],
            linewidth=1.8, linestyle='-', alpha=style['alpha'],
            label='_nolegend_', zorder=0
        )
        ax1.add_patch(rect)

    # 2. K线蜡烛图(阳红阴绿,A 股配色)
    opens = df['open'].values.astype(np.float64)
    lows_arr = df['low'].values.astype(np.float64)
    highs_arr = df['high'].values.astype(np.float64)
    dates_num = mdates.date2num(dates)
    candle_w = 0.62
    price_span = max(closes_arr) - min(closes_arr)
    for i in range(len(df)):
        _up = closes_arr[i] >= opens[i]
        _col = '#E8403F' if _up else '#1FAE62'   # 红涨绿跌
        # 影线
        ax1.vlines(dates_num[i], lows_arr[i], highs_arr[i], color=_col, linewidth=0.8, alpha=0.95, zorder=2)
        # 实体(十字星给最小高度)
        _yb, _yt = min(opens[i], closes_arr[i]), max(opens[i], closes_arr[i])
        if _yt - _yb < 1e-9:
            _yt = _yb + max(price_span * 0.002, 1e-9)
        ax1.add_patch(Rectangle(
            (dates_num[i] - candle_w / 2, _yb), candle_w, _yt - _yb,
            facecolor=_col, edgecolor=_col, linewidth=0.5, zorder=3))

    # 3. 均值线（只显示 S1~S4 范围内）
    valid_mean = ~np.isnan(mean_line)
    if valid_mean.any():
        # 创建掩码：只显示 S1 开始 ~ S4 末尾的数据
        s4_end = phase_edges['S4'][1] if 'S4' in phase_edges else xlim_right
        dates_num = mdates.date2num(dates)
        s1_num = mdates.date2num(s1_ts)
        s4_num = mdates.date2num(s4_end)
        show_mask = valid_mean & (dates_num >= s1_num) & (dates_num <= s4_num)
        ax1.plot(dates[show_mask], mean_line[show_mask],
                 color='#E67E22', linewidth=1.8, linestyle='--', alpha=0.8,
                 label=f'均值线 (mean_reversion {reg_window}日)')

    # 4. 最高线（S1开始~S2结束之间的最高价），只在 S1~S4 范围内显示
    if s2_high is not None:
        s4_end = phase_edges['S4'][1] if 'S4' in phase_edges else xlim_right
        ax1.hlines(y=s2_high, xmin=s1_ts, xmax=s4_end,
                   color='#E74C3C', linewidth=1.3, linestyle='-',
                   alpha=0.7, label=f'天花板 = S2最高 {s2_high:.2f}')

        if s2_high_date is not None:
            ax1.annotate(f'S2最高 {s2_high:.2f}',
                         xy=(s2_high_date, s2_high),
                         xytext=(12, 12), textcoords='offset points',
                         fontsize=9, fontweight='bold', color='#E74C3C',
                         arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.2))

    # 5. S1 最低点标注
    if s1_low is not None and s1_low_date is not None:
        ax1.scatter([s1_low_date], [s1_low], color='#C0392B', s=90, zorder=5, marker='v')
        ax1.annotate(f'S1最低 {s1_low:.2f}',
                     xy=(s1_low_date, s1_low),
                     xytext=(10, -25), textcoords='offset points',
                     fontsize=9, fontweight='bold', color='#C0392B',
                     arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.2))

    # 6. S4 突破标注
    if s2_high is not None and len(s4_data) > 0:
        s4_above = s4_data[s4_data['close'] > s2_high]
        if len(s4_above) > 0:
            break_date = s4_above.iloc[0]['date']
            break_price = s4_above.iloc[0]['close']
            ax1.scatter([break_date], [break_price],
                        color='#2ECC40', s=140, zorder=6, marker='^',
                        edgecolors='#27AE60', linewidth=1.5)
            ax1.annotate('S4 突破 ↑',
                         xy=(break_date, break_price),
                         xytext=(18, 18), textcoords='offset points',
                         fontsize=12, fontweight='bold', color='#27AE60',
                         arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.5))

    # 7. 阶段名称标注（基于连续方框的左右边界取中点）
    for i, ph in enumerate(main_phases):
        if ph not in phase_edges:
            continue
        ph_left = phase_edges[ph][0]
        if i + 1 < len(main_phases):
            next_ph = main_phases[i + 1]
            ph_right = phase_edges[next_ph][0] if next_ph in phase_edges else xlim_right
        else:
            ph_right = xlim_right
        mid_date = ph_left + (ph_right - ph_left) / 2
        ax1.text(mid_date, y_axis_top * 0.97,
                 PHASE_STYLES[ph]['label'],
                 ha='center', va='top',
                 fontsize=11, fontweight='bold',
                 color=PHASE_STYLES[ph]['edge'],
                 bbox=dict(boxstyle='round,pad=0.25',
                           facecolor='white', edgecolor=PHASE_STYLES[ph]['edge'],
                           alpha=0.85))

    ax1.set_ylabel('价格', fontsize=11)
    ax1.set_title(f'{code} 四阶段模式 (reg_window={reg_window}){title_suffix}',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.15)
    ax1.set_ylim(y_axis_bot, y_axis_top)
    # x轴从 S1 开始显示;右界钳制到数据末端(K 线最后一天+半个K线宽),右边不留白
    _last_num = mdates.date2num(df['date'].iloc[-1])
    _xr_num = min(mdates.date2num(xlim_right), _last_num + 0.5)
    ax1.set_xlim(mdates.date2num(xlim_left) - 0.5, _xr_num)

    # ── 阶段图例 ──
    legend_elements = [
        Patch(facecolor=PHASE_STYLES['S1']['face'], edgecolor=PHASE_STYLES['S1']['edge'],
              alpha=0.5, label=PHASE_STYLES['S1']['label']),
        Patch(facecolor=PHASE_STYLES['S2']['face'], edgecolor=PHASE_STYLES['S2']['edge'],
              alpha=0.5, label=PHASE_STYLES['S2']['label']),
        Patch(facecolor=PHASE_STYLES['S3']['face'], edgecolor=PHASE_STYLES['S3']['edge'],
              alpha=0.5, label=PHASE_STYLES['S3']['label']),
        Patch(facecolor=PHASE_STYLES['S4']['face'], edgecolor=PHASE_STYLES['S4']['edge'],
              alpha=0.5, label=PHASE_STYLES['S4']['label']),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9,
               framealpha=0.85, ncol=2)

    # ========== 成交量图 ==========
    ax2.set_facecolor('#FAFAFA')
    volumes = df['volume'].values
    # 成交量按涨跌配色(阳红阴绿),与 K 线一致
    opens_v = df['open'].values.astype(np.float64)
    vol_colors = ['#E8403F' if closes_arr[i] >= opens_v[i] else '#1FAE62'
                  for i in range(len(df))]
    ax2.bar(dates, volumes, width=1, color=vol_colors, alpha=0.55)

    if len(volumes) >= 21:
        ma20v = pd.Series(volumes).rolling(20).mean().values
        ax2.plot(dates, ma20v, color='#E74C3C', linewidth=1.0, alpha=0.7,
                 label='MA20 成交量')

    ax2.set_ylabel('成交量', fontsize=11)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.15)

    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    # ── 保存 / 显示 ──
    if save_path:
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  图保存至: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 自动检测算法（Oracle版 + 无未来函数版）
# ============================================================

def _find_s1_start_idx_strict(closes, pre_low_idx, pre_low_price,
                              min_decline=0.20, min_days=10, lookback=90):
    """严格版 S1 起点(供 Oracle 版用):找 pre_low 之前**最近**一个局部高点,
    要求从该高点跌到 pre_low 低点满足"跌幅>=min_decline 且天数>=min_days"。

    - 从右往左找最近的高点(价格由升转降处),避免像旧版那样选中窗口内
      "更早更高"的远古高点(如 300437 的 2020-12 高点 vs 用户标注 2021-01);
    - 跌幅用 pre_low_price(low 价)而非 pre_low 当日收盘(S2 起爆日收盘已高,
      会低估真实跌幅);
    - min_decline 默认 0.20:跳过下跌途中的小反弹高点(如 300437 的
      2021-03-22 反弹 dec=13%),只认"显著下跌段起点"(dec>=20%);
    - 逐级扩大搜索窗口(lookback→2x→4x),全部失败返回 None。
    """
    for lb in (lookback, lookback * 2, lookback * 4):
        s = max(0, pre_low_idx - lb)
        if pre_low_idx - s < min_days:
            break
        # 从右往左扫局部极大(seg[i] > seg[i-1] 且 seg[i] >= seg[i+1]),取最近满足条件者
        for i in range(pre_low_idx - 1, s, -1):
            if not (closes[i] > closes[i - 1] and closes[i] >= closes[i + 1]):
                continue
            days = pre_low_idx - i
            if days < min_days:
                continue
            c_hi = closes[i]
            if c_hi <= 0:
                continue
            decline = (c_hi - pre_low_price) / c_hi
            if decline >= min_decline:
                return i
    return None


def _estimate_volatility(closes, window=250):
    """近一年日收益波动率(年化)。用于自适应真突破判据:
    高波动(妖股)用快严判据,低波动(慢牛)用慢宽判据。"""
    if len(closes) < 30:
        return 0.40
    rets = np.diff(closes) / closes[:-1]
    r = rets[-min(window, len(rets)):]
    return float(np.std(r, ddof=1) * np.sqrt(250))


def _find_real_breakout(closes, p_idx, ceiling, lookahead_days=30, breakout_gain=None,
                        min_close_ratio=None, quick_days=None, adaptive=True):
    """找真突破日(主升浪启动) — Oracle 专用(有未来函数)。

    从 p_idx 之后逐日扫描所有"收盘 > 天花板"的候选突破日,对每个候选看
    未来 lookahead_days 天:若 max(close) >= 突破日收盘 * breakout_gain 且
    **在 quick_days 天内达到该涨幅**(快速启动),则判为真突破,返回该日下标;
    否则视为假突破(贴顶横盘),继续找下一个候选。

    ⚠ 涨幅基准用"突破日收盘"而非"天花板"(衡量突破后动能,见 300251 案例)。

    adaptive=True(默认):按近一年波动率自动选判据——
      波动率 ≥60%(妖股):   10 天涨 30%,站上 1.05
      波动率 35~60%(活跃): 20 天涨 25%,站上 1.03
      波动率 <35%(慢牛):   30 天涨 15%,站上 1.02
    解决工商银行式慢牛(从不 10 天涨 30%、收盘很少超峰 5%)误判为无真突破。

    ⚠ 未来数据不足(突破日距数据末端太近,主升浪还没走完)时:若突破日刚发生
    (距末端 ≤ max(quick_days*2,10) 天),**降级确认为潜在真突破**——避免把
    正在进行的主升浪漏掉(如药明康德 2026-08 刚突破 129,未来无法验证)。

    无真突破返回 None。
    """
    n = len(closes)
    if adaptive:
        vol = _estimate_volatility(closes)
        if vol >= 0.60:
            breakout_gain, min_close_ratio, quick_days = 1.30, 1.05, 10
        elif vol >= 0.35:
            breakout_gain, min_close_ratio, quick_days = 1.25, 1.03, 20
        else:
            breakout_gain, min_close_ratio, quick_days = 1.15, 1.02, 30
    else:
        breakout_gain = breakout_gain if breakout_gain else 1.30
        min_close_ratio = min_close_ratio if min_close_ratio else 1.05
        quick_days = quick_days if quick_days else 10
    min_future = max(int(lookahead_days * 0.5), 5)
    for t in range(p_idx + 1, n):
        if closes[t] <= ceiling:
            continue
        if closes[t] < ceiling * min_close_ratio:
            continue  # 收盘未显著站上天花板,不算候选突破
        end = min(t + lookahead_days + 1, n)
        fut = closes[t:end]
        if len(fut) < min_future:
            # 未来数据不足:若突破日距末端近(最近刚突破),降级确认为潜在真突破
            if (n - 1 - t) <= max(quick_days * 2, 10):
                return t
            continue
        target = closes[t] * breakout_gain
        if fut.max() >= target:
            # 快速启动确认:从突破日到首次达到 target 的间隔 <= quick_days
            rel = int(np.argmax(fut >= target))
            if rel <= quick_days:
                return t
    return None


def detect_four_phases_oracle(code, end_date=None, reg_window=120,
                              lookahead_days=30, breakout_gain=1.3,
                              min_close_ratio=1.05, s3_max_days=250,
                              return_all=False, adaptive=True):
    """
    Oracle 版(有未来函数)— 用"突破后主升浪"判据找最精准的四阶段分割。

    分段语义(与人工标注一致):
      S1 恐慌下跌  : pre_low 之前的高点 → 最低点(持续下跌制造恐慌)
      S2 拉升吸筹  : 从 S1 低点快速拉升到第一个明显高点(天花板),S2 终点=天花板日
      S3 震荡整理  : 天花板日之后,在[均值线, 天花板]区间反复(含贴顶横盘的伪突破)
      S4 突破拉升  : 真突破日 = 收盘站上天花板且未来 lookahead_days 天内
                     max(close) >= 天花板 * breakout_gain,即启动主升浪

    Oracle 特有的未来函数用法:
      1. 真突破判定:看突破后未来 30 天涨幅,过滤"贴顶横盘"的伪突破
         (例:300251 20241106 首次突破但 3 个月只涨 10% → 判伪,S4 取 20250205 真突破)
      2. S1/S2 边界:严格验证下跌/拉升幅度,不把远古高点钉进 S1

    新增可调参数:
      lookahead_days : 真突破验证窗口(交易日),默认 30
      breakout_gain  : 未来窗口 max(close) 需达到 天花板*gain,默认 1.3(涨幅30%)
      min_close_ratio: 突破日收盘最低要求(天花板*ratio),默认 1.05
      s3_max_days    : S3 震荡时长上限(交易日),默认 250。过滤"跨多年整理"
                       的远古模式(如 2019 峰等 2021 主升才突破,S3 长达 535 天),
                       只保留与当前行情匹配的完整四阶段
    """
    from scipy.signal import find_peaks, savgol_filter

    if end_date is None:
        import datetime
        end_date = datetime.datetime.now().strftime('%Y%m%d')

    df = get_daily_kline_from_tdx(code, end_date)
    if df is None or len(df) < 120: return None
    if hasattr(df['date'].iloc[0], 'tz'):
        df['date'] = df['date'].dt.tz_localize(None)

    closes = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    n = len(closes)

    # 回归均线(120日对数回归)——S1 恐慌下跌必须跌破均线(基本需求)
    reg, _ = compute_rolling_regression(closes, window=reg_window, use_log=True)

    # 平滑找峰 → 候选天花板(阈值 0.02×range,小峰也检出,如 300251 的 8.67 峰)
    sw = min(11, n // 10 * 2 + 1) or 5
    if sw < 5: sw = 5
    if sw % 2 == 0: sw += 1
    smoothed = savgol_filter(closes, sw, 2)
    pr = closes.max() - closes.min()
    peaks, _ = find_peaks(smoothed, prominence=max(pr * 0.02, 0.3), distance=5, width=1)
    if len(peaks) == 0:
        peaks, _ = find_peaks(smoothed, prominence=pr * 0.01, distance=3)
    if len(peaks) == 0: return None

    patterns = []
    for p_idx in peaks:
        # 天花板 = 峰 ±2 天窗口内最高价(补偿平滑对尖峰的展平偏移,
        # 如 300437 的 11.26 尖峰被展平到 20210416 的峰);S2 终点取最高价日
        w0, w1 = max(0, p_idx - 2), min(n, p_idx + 3)
        _k = w0 + int(np.argmax(highs[w0:w1]))
        ceiling = highs[_k]
        p_eff = _k                     # 天花板日(最高价日)= S2 终点
        # S2 终点即最高价日:拉升吸筹 = 连续拉涨停形成高点的过程本身
        # (300437: 20210413~20210414 两天涨停,20210415 开始下跌归 S3;
        #  不回延到"见顶回落日",否则把下跌段混入拉升段)

        # ── 真突破(主升浪):过滤贴顶横盘的伪突破,找到即 S4 锚点 ──
        # adaptive=True: 按波动率自动选判据(妖股快严/慢牛慢宽)+ 未来不足降级
        s4_idx = _find_real_breakout(closes, p_eff, ceiling,
                                     lookahead_days, breakout_gain,
                                     min_close_ratio, None, adaptive)
        if s4_idx is None:
            continue                                    # 从未启动主升浪,不算模式
        s3_days = s4_idx - p_eff                        # 震荡天数(天花板日~突破前)

        # S2 终点 = 天花板日(p_eff);S2 起点 = 天花板前窗口内最低价(急拉起点)
        # 用 low 而非 close 找最低点:300437 的拉升起爆日(20210413 low=7.49)
        # 才是用户眼中的 S2 起点,而非更早横盘的 close 低点(20210330 7.53)
        pre_lo_idx, pre_lo_price, rise_days, rise_pct = p_eff, lows[p_eff], 0, 0
        for _look in [30, 45, 60]:
            if p_eff < _look: continue
            _seg = lows[p_eff - _look:p_eff + 1]
            _rel = int(np.argmin(_seg))
            _idx = p_eff - _look + _rel
            _rise = (ceiling - _seg[_rel]) / _seg[_rel]
            if _rise >= 0.15 and (p_eff - _idx) <= _look:
                pre_lo_idx, pre_lo_price = _idx, _seg[_rel]
                rise_days, rise_pct = p_eff - _idx, _rise
                break

        # S1 起点:严格下跌段起点;无效则放弃该候选(不钉远古高点)
        s1_st_idx = _find_s1_start_idx_strict(closes, pre_lo_idx, pre_lo_price)
        if s1_st_idx is None:
            continue
        s1_days = pre_lo_idx - s1_st_idx
        s1_decline = (closes[s1_st_idx] - pre_lo_price) / closes[s1_st_idx]

        # ── S1 必须跌破回归均线(恐慌下跌基本需求):S1 区间内低点低于同期均线 ──
        _r = reg[s1_st_idx:pre_lo_idx + 1]
        _l = lows[s1_st_idx:pre_lo_idx + 1]
        if len(_r) and not np.isnan(_r).all() and not (np.nanmin(_l) < np.nanmin(_r)):
            continue   # S1 未跌破均线,不算恐慌下跌

        # S3 回落深度(天花板 → S3 区间最低收盘,代替旧 po_lo 跌幅)
        s3_seg = closes[p_eff + 1:s4_idx]
        fall_pct = (ceiling - s3_seg.min()) / ceiling if len(s3_seg) > 0 else 0.0

        # S4 终点 = 主升浪最高价日:[s4_idx, s4_idx+250) 内最高价日
        # (窗口 250 而非 120:长主升浪也能覆盖到顶,如海光 2025-09 突破后
        #  主升浪顶 395@20260710 距突破约 200 交易日)
        top_end = min(s4_idx + 250, n)
        s4_end_idx = s4_idx + int(np.argmax(highs[s4_idx:top_end]))

        patterns.append({
            'peak_idx': p_eff, 's4_idx': s4_idx, 's4_end_idx': s4_end_idx, 'ceiling': ceiling,
            'pre_lo_idx': pre_lo_idx, 'pre_lo_price': pre_lo_price,
            's1_st_idx': s1_st_idx,
            's1_days': s1_days, 's1_decline': s1_decline,
            'rise_days': rise_days, 'rise_pct': rise_pct,
            'fall_pct': fall_pct, 's3_days': s3_days,
        })

    if len(patterns) == 0: return None

    valid = [p for p in patterns
             if p['s1_days'] >= 10 and p['s1_decline'] >= 0.10
             and p['rise_pct'] >= 0.15 and p['rise_days'] <= 35
             and p['s3_days'] >= 15 and p['s3_days'] <= s3_max_days]
    if len(valid) == 0:
        valid = [p for p in patterns
                 if p['rise_pct'] >= 0.12 and p['s3_days'] >= 10
                 and p['s3_days'] <= s3_max_days]
    if len(valid) == 0: return None

    # ── 多轮切分(主升浪视角)──
    # 1) 排除"主升浪中段浅回调"伪模式:S1 必须是恐慌下跌(跌幅>=20%),
    #    否则只是主升浪中的小回调(如 300437 2021-10 峰 dec=22%)
    deep = [p for p in valid if p['s1_decline'] >= 0.20]
    pool = deep if deep else valid

    # 2) 按 S4 聚类(15 交易日):同一波主升浪中先后突破多个峰的合并为一波
    pool.sort(key=lambda p: p['s4_idx'])
    waves = []
    for p in pool:
        if waves and p['s4_idx'] - waves[-1][-1]['s4_idx'] <= 15:
            waves[-1].append(p)
        else:
            waves.append([p])
    # 波内选优:同一波主升浪的多个候选峰,S1 起点明显更早的是跨轮候选
    # (如 300437 跨年峰 S1=20190704 vs 用户轮 S1=20210108),先取"最新 S1 簇"
    # (S1 起点最晚者 ≤30 天内的候选,即最近一轮行情),簇内选 S2 天花板最高
    def _pick_wave(w):
        s1_max = max(p['s1_st_idx'] for p in w)
        latest = [p for p in w if s1_max - p['s1_st_idx'] <= 30]
        return max(latest, key=lambda p: (p['ceiling'], p['s4_idx']))
    wave_best = [_pick_wave(w) for w in waves]

    # 3) S1 簇合并:S1 起点相近(≤30d)且 S4 相近(≤90d)的候选属同一轮行情
    #    (如 300437 的 2021-03 峰与 2021-04 峰,S1 都落在 2021-01),合并取 S4 更晚者
    wave_best.sort(key=lambda p: p['s1_st_idx'])
    merged = []
    for p in wave_best:
        if merged and abs(p['s1_st_idx'] - merged[-1]['s1_st_idx']) <= 30 \
                and abs(p['s4_idx'] - merged[-1]['s4_idx']) <= 90:
            if p['s4_idx'] > merged[-1]['s4_idx']:
                merged[-1] = p
        else:
            merged.append(p)

    # 4) 不跨轮:按 S4 升序,后轮的 S1 起点必须晚于前一轮的 S4 终点(主升浪顶),
    #    否则其 S1 落在前轮主升浪内,是伪模式(如 2021-10 中段峰 S1=20210414)
    merged.sort(key=lambda p: p['s4_idx'])
    rounds = []
    for p in merged:
        if rounds and p['s1_st_idx'] <= rounds[-1]['s4_end_idx']:
            continue
        rounds.append(p)
    if len(rounds) == 0: return None

    def ds(i): return df['date'].iloc[i].strftime('%Y%m%d')

    # 选优:主升浪幅度最大优先——识别并覆盖"真正的主升浪"(用户核心诉求)。
    # 主升浪幅度 = 主升浪最高价 / 突破日收盘(如银之杰 2024 轮 66/10.7≈6 倍,
    # 远大于 2025 轮 61/42≈1.4 倍;新易盛第一波 7.7 倍 > 第二波 1.9 倍)。
    # return_all 仍按时间升序返回全部轮次。
    def _wave_gain(p):
        c0 = closes[p['s4_idx']]
        return highs[p['s4_end_idx']] / c0 if c0 > 0 else 0.0

    def build_result(b):
        return {
            's1_start': ds(b['s1_st_idx']),
            's1_end': ds(b['pre_lo_idx'] - 1),
            's2_start': ds(b['pre_lo_idx']),
            's2_end': ds(b['peak_idx']),
            's3_start': ds(b['peak_idx'] + 1),
            's3_end': ds(b['s4_idx'] - 1),
            's4_start': ds(b['s4_idx']),
            's4_end': ds(b['s4_end_idx']),
            's2_high': round(float(b['ceiling']), 2),
            's1_low': round(float(b['pre_lo_price']), 2),
            'rise_pct': round(float(b['rise_pct'] * 100), 1),
            'fall_pct': round(float(b['fall_pct'] * 100), 1),
            's3_duration': b['s3_days'],
            'wave_gain': round(float(_wave_gain(b)), 2),
            'version': 'oracle',
        }

    if return_all:
        return [build_result(r) for r in rounds]   # 全部轮次(按时间升序)
    return build_result(max(rounds, key=_wave_gain))  # 主升浪幅度最大的一轮


def detect_four_phases(code: str, end_date: str = None, reg_window: int = 120,
                       return_all=False, min_close_ratio=1.05, pending_ratio=0.85):
    """
    无未来函数版 — 先 S1-S3,后等突破(实时流程)。

    核心逻辑:
      1. 找局部峰值(候选 S2 天花板)
      2. 对每个峰,**只用历史数据**回溯验证 S1→S2→S3:
         S1 恐慌下跌 → S2 拉升到峰(天花板) → S3 峰后震荡(进行中)
      3. S4 状态(不验证主升浪,无未来函数):
         - 'breakout' 已突破:峰后已出现收盘价站上天花板(×min_close_ratio)
         - 'pending'   待突破:S3 进行中,收盘尚未站上天花板
      4. 输出模式(默认最近一轮;return_all=True 返回全部,按峰时间降序)

    与 Oracle 版的区别:Oracle 用未来 30 天验证"主升浪"过滤假突破;
    本版只看"现在"——S3 是否成立、是否已突破,适合实盘实时判断
    (识别出 S1-S3 后"等突破")。
    """
    from scipy.signal import find_peaks, savgol_filter

    if end_date is None:
        import datetime
        end_date = datetime.datetime.now().strftime('%Y%m%d')

    df = get_daily_kline_from_tdx(code, end_date)
    if df is None or len(df) < 120:
        print(f"!! {code} 数据不足（{len(df) if df is not None else 0}条）")
        return None
    if hasattr(df['date'].iloc[0], 'tz'):
        df['date'] = df['date'].dt.tz_localize(None)

    closes = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    n = len(closes)

    # 回归均线(120日对数回归)——S1 恐慌下跌必须跌破均线(基本需求)
    reg, _ = compute_rolling_regression(closes, window=reg_window, use_log=True)

    # ── 找峰(候选 S2 天花板,与 Oracle 版一致)──
    sw = min(11, n // 10 * 2 + 1) or 5
    if sw < 5: sw = 5
    if sw % 2 == 0: sw += 1
    smoothed = savgol_filter(closes, sw, 2)
    pr = closes.max() - closes.min()
    peaks, _ = find_peaks(smoothed, prominence=max(pr * 0.02, 0.3), distance=5, width=1)
    if len(peaks) == 0:
        peaks, _ = find_peaks(smoothed, prominence=pr * 0.01, distance=3)
    if len(peaks) == 0:
        print(f"!! {code} 未检测到有效价格峰"); return None

    patterns = []
    for p_idx in peaks:
        # 天花板 = 峰 ±2 天窗口内最高价(补偿平滑偏移)
        w0, w1 = max(0, p_idx - 2), min(n, p_idx + 3)
        _k = w0 + int(np.argmax(highs[w0:w1]))
        ceiling = highs[_k]
        p_eff = _k                              # S2 终点(天花板日)

        # S2 起点 = 峰前窗口内最低价(急拉起点,用 low)
        pre_lo_idx, pre_lo_price, rise_days, rise_pct = p_eff, lows[p_eff], 0, 0
        for _look in [30, 45, 60]:
            if p_eff < _look: continue
            _seg = lows[p_eff - _look:p_eff + 1]
            _rel = int(np.argmin(_seg))
            _idx = p_eff - _look + _rel
            _rise = (ceiling - _seg[_rel]) / _seg[_rel]
            if _rise >= 0.15 and (p_eff - _idx) <= _look:
                pre_lo_idx, pre_lo_price = _idx, _seg[_rel]
                rise_days, rise_pct = p_eff - _idx, _rise
                break

        # S1 起点(严格版,只用历史数据)
        s1_st_idx = _find_s1_start_idx_strict(closes, pre_lo_idx, pre_lo_price)
        if s1_st_idx is None:
            continue
        s1_days = pre_lo_idx - s1_st_idx
        s1_decline = (closes[s1_st_idx] - pre_lo_price) / closes[s1_st_idx]

        # ── S1 必须跌破回归均线(恐慌下跌基本需求):S1 区间内低点低于同期均线 ──
        _r = reg[s1_st_idx:pre_lo_idx + 1]
        _l = lows[s1_st_idx:pre_lo_idx + 1]
        if len(_r) and not np.isnan(_r).all() and not (np.nanmin(_l) < np.nanmin(_r)):
            continue   # S1 未跌破均线,不算恐慌下跌

        # S3:峰次日 ~ 数据末端(或首次突破日-1)
        # 找峰后首次收盘站上天花板(×min_close_ratio)→ 突破日;无则 pending
        s4_idx = None
        for t in range(p_eff + 1, n):
            if closes[t] >= ceiling * min_close_ratio:
                s4_idx = t
                break
        if s4_idx is not None:
            s3_end_idx = s4_idx - 1
            s4_status = 'breakout'
        else:
            s3_end_idx = n - 1
            # pending(待突破)要求 S3 处于"贴顶震荡":当前收盘接近天花板,
            # 否则是冲高回落深跌(如长川科技当前价仅天花板 69%),不是等突破
            if closes[-1] >= ceiling * pending_ratio:
                s4_status = 'pending'
            else:
                s4_status = 'pullback'   # 冲高回落中,暂不视为待突破
        s3_days = s3_end_idx - p_eff
        s3_seg = closes[p_eff + 1:s3_end_idx + 1]
        fall_pct = (ceiling - s3_seg.min()) / ceiling if len(s3_seg) > 0 else 0.0

        patterns.append({
            'peak_idx': p_eff, 's4_idx': s4_idx, 's4_status': s4_status,
            's3_end_idx': s3_end_idx, 'ceiling': ceiling,
            'pre_lo_idx': pre_lo_idx, 'pre_lo_price': pre_lo_price,
            's1_st_idx': s1_st_idx,
            's1_days': s1_days, 's1_decline': s1_decline,
            'rise_days': rise_days, 'rise_pct': rise_pct,
            'fall_pct': fall_pct, 's3_days': s3_days,
            'age': n - 1 - p_eff,     # S2 峰距数据末端的天数(峰不能是最近的高点)
        })

    if len(patterns) == 0:
        print(f"!! {code} 未找到符合 S1-S3 结构的候选"); return None

    # 实盘过滤:避开"山顶"伪模式——S2 峰不能是最近的高点(距当前>=60天,
    # 否则是刚冲高见顶,由 age 兜底)。S3 震荡 >=30 天即可:快速整理后突破
    # 也是有效模式(如东财 2024 政策牛 S3 仅 38 天),原 60 天约束误杀这类轮。
    # S1 天数放宽到 >=10(与 Oracle 版一致):快速深跌(如 10 天跌 28%)
    # 也是恐慌下跌,不应因天数短而误杀历史大主升浪。
    valid = [p for p in patterns
             if p['s1_days'] >= 10 and p['s1_decline'] >= 0.15
             and p['rise_pct'] >= 0.15 and p['rise_days'] <= 35
             and p['s3_days'] >= 30 and p['age'] >= 60]
    if len(valid) == 0:
        valid = [p for p in patterns
                 if p['rise_pct'] >= 0.12 and p['s3_days'] >= 30 and p['age'] >= 30]
    if len(valid) == 0:
        print(f"!! {code} 未找到完整四阶段模式"); return None

    # 选优:突破后 250 天内最高价/突破日收盘最大优先——与 Oracle 版
    # "主升浪幅度(wave_gain)"完全同口径(该窗口在当前时点已是历史数据,
    # 不引入未来函数),保证两版识别同一轮主升浪(用户:无未来要"跟上")。
    n_last = len(closes) - 1
    def _proxy_gain(p):
        if p['s4_idx'] is not None:
            c0 = closes[p['s4_idx']]
            if c0 <= 0:
                return 0.0
            top_end = min(p['s4_idx'] + 250, n)
            return float(highs[p['s4_idx']:top_end].max()) / c0
        return 0.0
    valid.sort(key=lambda p: (-_proxy_gain(p), p['s4_status'] != 'pending', -p['peak_idx']))

    def ds(i): return df['date'].iloc[i].strftime('%Y%m%d')

    def build_result(b):
        cur = float(closes[-1])
        return {
            's1_start': ds(b['s1_st_idx']),
            's1_end': ds(b['pre_lo_idx'] - 1),
            's2_start': ds(b['pre_lo_idx']),
            's2_end': ds(b['peak_idx']),
            's3_start': ds(b['peak_idx'] + 1),
            's3_end': ds(b['s3_end_idx']),
            's4_start': ds(b['s4_idx']) if b['s4_idx'] is not None else None,
            's4_end': None,
            's4_status': b['s4_status'],
            'distance_to_ceiling': round((cur / b['ceiling'] - 1) * 100, 1),
            's2_high': round(float(b['ceiling']), 2),
            's1_low': round(float(b['pre_lo_price']), 2),
            'rise_pct': round(float(b['rise_pct'] * 100), 1),
            'fall_pct': round(float(b['fall_pct'] * 100), 1),
            's3_duration': b['s3_days'],
            'version': 'nofuture',
        }

    if return_all:
        return [build_result(r) for r in valid]
    return build_result(valid[0])


# ── 预设测试案例 ──
TEST_CASES = {
    "300437": {
        "code": "sz300437",
        "s1_start": "20210101", "s1_end": "20210412",
        "s2_start": "20210413", "s2_end": "20210415",
        "s3_start": "20210416", "s3_end": "20210908",
        "s4_start": "20210909",
        "note": "示例1: 2021年牛股四阶段",
    },
    "300251": {
        "code": "sz300251",
        "s1_start": "20240701", "s1_end": "20240920",
        "s2_start": "20240921", "s2_end": "20241009",
        "s3_start": "20241010", "s3_end": "20250127",
        "s4_start": "20250205",
        "note": "示例2: 近期牛股四阶段",
    },
}


def run_test_case(case_name: str, reg_window: int = 120, save: bool = False):
    """运行一个预设测试案例"""
    if case_name not in TEST_CASES:
        print(f"!! 未知案例: {case_name}，可选: {list(TEST_CASES.keys())}")
        return

    case = TEST_CASES[case_name]
    print(f"\n{'='*60}")
    print(f"  测试案例: {case_name} ({case['note']})")
    print(f"{'='*60}")

    save_path = None
    if save:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "result", "four_phase")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{case_name}_phase.png")

    visualize_four_phases(
        code=case['code'],
        s1_start=case['s1_start'], s1_end=case['s1_end'],
        s2_start=case['s2_start'], s2_end=case['s2_end'],
        s3_start=case['s3_start'], s3_end=case['s3_end'],
        s4_start=case['s4_start'],
        reg_window=reg_window,
        save_path=save_path,
        show=True,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="四阶段模式可视化工具")
    parser.add_argument("--code", type=str, default=None,
                        help="股票代码，如 sz300437")
    parser.add_argument("--s1", type=str, default=None, nargs=2,
                        metavar=('START', 'END'),
                        help="S1 起止日期 YYYYMMDD YYYYMMDD")
    parser.add_argument("--s2", type=str, default=None, nargs=2,
                        metavar=('START', 'END'),
                        help="S2 起止日期 YYYYMMDD YYYYMMDD")
    parser.add_argument("--s3", type=str, default=None, nargs=2,
                        metavar=('START', 'END'),
                        help="S3 起止日期 YYYYMMDD YYYYMMDD")
    parser.add_argument("--s4", type=str, default=None,
                        help="S4 起始日期 YYYYMMDD")
    parser.add_argument("--auto", type=str, default=None, metavar='CODE',
                        help="自动检测四阶段，如 sz300437")
    parser.add_argument("--end_date", type=str, default=None,
                        help="数据截止日期 YYYYMMDD（默认最新）")
    parser.add_argument("--reg_window", type=int, default=120,
                        help="均值回归窗口（默认 120）")
    parser.add_argument("--test", type=str, default=None,
                        choices=list(TEST_CASES.keys()),
                        help="运行预设测试案例")
    parser.add_argument("--save", action="store_true", default=False,
                        help="保存图片到 result/four_phase/")

    args = parser.parse_args()

    if args.auto:
        # ── 自动检测模式 ──
        print(f"\n{'='*60}")
        print(f"  自动检测四阶段: {args.auto}")
        print(f"{'='*60}")
        result = detect_four_phases(args.auto, end_date=args.end_date,
                                    reg_window=args.reg_window)
        if result:
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "result", "four_phase")
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f"{args.auto}_auto.png")
            visualize_four_phases(
                code=args.auto,
                s1_start=result['s1_start'], s1_end=result['s1_end'],
                s2_start=result['s2_start'], s2_end=result['s2_end'],
                s3_start=result['s3_start'], s3_end=result['s3_end'],
                s4_start=result['s4_start'],
                reg_window=args.reg_window,
                save_path=save_path,
                show=False,
            )
        return

    if args.test:
        run_test_case(args.test, reg_window=args.reg_window, save=args.save)
    elif args.code and args.s1 and args.s2 and args.s3 and args.s4:
        save_path = None
        if args.save:
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "result", "four_phase")
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f"{args.code}_phase.png")
        visualize_four_phases(
            code=args.code,
            s1_start=args.s1[0], s1_end=args.s1[1],
            s2_start=args.s2[0], s2_end=args.s2[1],
            s3_start=args.s3[0], s3_end=args.s3[1],
            s4_start=args.s4,
            reg_window=args.reg_window,
            save_path=save_path,
            show=True,
        )
    else:
        for cn in ["300437", "300251"]:
            run_test_case(cn, reg_window=args.reg_window, save=args.save)


if __name__ == "__main__":
    main()
