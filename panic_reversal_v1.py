# -*- coding: utf-8 -*-
"""
极速杀跌反转模型 (panic → reversal)

核心逻辑(全部无未来函数,滚动回归天然只用历史数据):
  1. 牛市门控(用户硬前提):事件发生时点,250 日超长对数回归斜率 > 0(斜率向上)
  2. 恐慌事件:5 日跌幅 <= -drop_pct 且 5 日均量 / 20 日均量 >= vol_ratio(放量急跌)
  3. 反转确认(可选):事件日后 0~confirm_days 天内出现「放量阳线」或「长下影阳线」
  4. 买入 = 确认日收盘价,持有 hold_days 天,统计胜率 / 平均收益 / 事件数

用法:
    python panic_reversal.py grid            # 参数网格批量测试(84只候选池)
    python panic_reversal.py one sz300437    # 单只股票检测并画图
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdx_quant import get_daily_kline_from_tdx
from mean_reversion.signal_residual import compute_rolling_regression

# 近两年大涨候选池(84 只,与四阶段批量验证一致)
CANDIDATE_POOL = [
    "sh688256", "sz300308", "sz300502", "sz300394", "sz300476", "sz002463",
    "sh600183", "sh688041", "sh603019", "sz000977", "sh688008", "sz002230",
    "sh688498", "sz002371", "sh688012", "sh688072", "sh688126", "sz300223",
    "sh688981", "sz002049", "sh600276", "sz300760", "sh688506", "sz300059",
    "sh600030", "sz000858", "sh600519", "sz000568", "sz300033", "sz300377",
    "sz300468", "sz300674", "sh600446", "sz300124", "sh688169", "sz002050",
    "sz002472", "sh603728", "sz300750", "sz002594", "sz300274", "sh688390",
    "sz300014", "sh601012", "sh601633", "sh600104", "sz000625", "sh603799",
    "sh688235", "sh600760", "sh601989", "sh600036", "sh601398", "sh603993",
    "sh601899", "sz002466", "sz002460", "sh600111", "sz000831", "sh600309",
    "sz002648", "sh600900", "sh601985", "sz300085", "sz002594", "sh600030",
    "sz300251", "sz300437", "sh600199", "sh688347", "sh688052", "sh688595",
    "sz002415", "sh601138", "sz300624", "sh688111", "sh600745", "sz002475",
    "sh603501", "sz300782", "sh688036", "sh688599", "sz002938", "sh688123",
    "sz300661", "sh603986", "sz300223", "sh688088",
]
POOL = sorted(set(CANDIDATE_POOL))


def _load_df(code, end_date=None):
    """拉取数据,带重试与容错(通达信服务器偶发连接失败)。"""
    import time
    for attempt in range(3):
        try:
            df = get_daily_kline_from_tdx(code, end_date)
            break
        except Exception as e:
            if attempt == 2:
                print(f"!! {code} 拉取失败(已重试3次): {type(e).__name__}")
                return None
            time.sleep(2.0)
    if df is None or len(df) < 260:
        return None
    if hasattr(df['date'].iloc[0], 'tz'):
        df['date'] = df['date'].dt.tz_localize(None)
    df = df.reset_index(drop=True)
    return df


def detect_panic_events(df, code='', drop_pct=0.15, vol_ratio=1.5,
                        bull_slope_min=0.0, reg_window=250,
                        require_bull=True, confirm_days=3,
                        require_confirm=True, hold_days=10,
                        min_events=1, pre_slope_annual=None):
    """在单只股票上检测「恐慌事件→反转确认」并统计持有收益。

    全部判定只用 t 时点及之前的数据(滚动回归 / 滚动均量),无未来函数。
    返回 (events: list[dict], stats: dict)
    """
    closes = df['close'].values.astype(np.float64)
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    vols = df['volume'].values.astype(np.float64)
    dates = df['date'].dt.strftime('%Y%m%d').values
    n = len(closes)

    # 超长对数回归斜率(年化):斜率>0 即趋势向上。
    # 网格批量时由调用方预计算传入 pre_slope_annual,避免每参数组重算。
    if pre_slope_annual is None:
        _, reg_slopes = compute_rolling_regression(closes, window=reg_window, use_log=True)
        slope_annual = reg_slopes * 250.0          # 年化对数斜率
    else:
        slope_annual = pre_slope_annual

    # 滚动 20 日均量
    vol_ma20 = pd.Series(vols).rolling(20).mean().values

    events = []
    last_ev = -10
    for t in range(reg_window, n - hold_days - confirm_days):
        if t < 6:
            continue
        # ── 牛市门控:事件时点 250 日斜率向上 ──
        if require_bull and not (np.isfinite(slope_annual[t]) and slope_annual[t] > bull_slope_min):
            continue
        # ── 恐慌事件:5 日跌幅 + 放量 ──
        ret5 = closes[t] / closes[t - 5] - 1.0
        if ret5 > -drop_pct:
            continue
        v5 = vols[t - 4:t + 1].mean()
        v20 = vols[t - 19:t + 1].mean() if t >= 20 else vols[:t + 1].mean()
        if v20 <= 0 or v5 / v20 < vol_ratio:
            continue
        # 同段急跌去重:事件日间隔 <5 天视为同一段,保留第一个触发日
        if t - last_ev < 5:
            continue

        # ── 反转确认:事件日后 0~confirm_days 天内找「放量阳线/长下影阳线」 ──
        confirm_idx = None
        if require_confirm:
            for i in range(t, min(t + confirm_days + 1, n)):
                if i <= 0:
                    continue
                o, c, h, l = opens[i], closes[i], highs[i], lows[i]
                body = abs(c - o)
                if c <= o:
                    continue                      # 必须阳线
                if vols[i] < 1.2 * (vol_ma20[i] if np.isfinite(vol_ma20[i]) else v20):
                    # 量能不足时,长下影阳线也算反转确认
                    lower_shadow = min(o, c) - l
                    rng = max(h - l, 1e-9)
                    if not (lower_shadow >= 2.0 * body and c >= (h + l) / 2):
                        continue
                confirm_idx = i
                break
            if confirm_idx is None:
                continue
        else:
            confirm_idx = t                       # 不要求确认,事件日直接买

        last_ev = confirm_idx                     # 去重窗口以确认日计

        # ── 持有收益(买入=确认日收盘,卖出=持有期末收盘) ──
        if confirm_idx + hold_days >= n:
            continue
        buy = closes[confirm_idx]
        sell = closes[confirm_idx + hold_days]
        events.append({
            'code': code,
            'event_date': dates[t],
            'confirm_date': dates[confirm_idx],
            'buy': round(float(buy), 3),
            'sell': round(float(sell), 3),
            'ret': round(float(sell / buy - 1.0), 4),
            'drop5': round(float(ret5), 4),
            'vol_ratio': round(float(v5 / v20), 2),
            'slope': round(float(slope_annual[t]) if np.isfinite(slope_annual[t]) else 0.0, 3),
        })

    if len(events) < min_events:
        return [], _empty_stats()

    rets = np.array([e['ret'] for e in events])
    stats = {
        'n': len(events),
        'win_rate': float((rets > 0).mean()),
        'avg_ret': float(rets.mean()),
        'med_ret': float(np.median(rets)),
        'worst': float(rets.min()),
        'best': float(rets.max()),
    }
    return events, stats


def _empty_stats():
    return {'n': 0, 'win_rate': 0.0, 'avg_ret': 0.0, 'med_ret': 0.0,
            'worst': 0.0, 'best': 0.0}


def run_grid(end_date=None, codes=None, verbose=True):
    """参数网格批量测试:遍历候选池(数据与回归只算一次,缓存复用),输出参数组合汇总表。"""
    codes = codes or POOL
    drop_list = [0.08, 0.10, 0.12, 0.15, 0.18]
    vol_list = [1.2, 1.5, 1.8]
    bull_list = [None, 0.0, 0.05]                 # None=不门控
    hold_list = [5, 10, 20]

    # 预加载:每只股票拉一次数据 + 算一次 250 日回归斜率
    dfs = {}
    for code in codes:
        df = _load_df(code, end_date)
        if df is None:
            continue
        _, slopes = compute_rolling_regression(df['close'].values.astype(np.float64),
                                               window=250, use_log=True)
        dfs[code] = (df, slopes * 250.0)          # (df, 年化斜率数组)

    rows = []
    for drop_pct in drop_list:
        for vol_ratio in vol_list:
            for bull in bull_list:
                for hold in hold_list:
                    n_evt, win_n, ret_sum = 0, 0, 0.0
                    for code, (df, slope_annual) in dfs.items():
                        _, st = detect_panic_events(
                            df, code, drop_pct=drop_pct, vol_ratio=vol_ratio,
                            bull_slope_min=(bull if bull is not None else 0.0),
                            require_bull=(bull is not None),
                            hold_days=hold, require_confirm=True,
                            pre_slope_annual=slope_annual)
                        n_evt += st['n']
                        win_n += int(st['n'] * st['win_rate'])
                        ret_sum += st['avg_ret'] * st['n']
                    if n_evt == 0:
                        continue
                    row = {
                        'drop': drop_pct, 'vol': vol_ratio,
                        'bull': ('牛市' if bull is not None else '无门控'),
                        'bull_min': bull if bull is not None else '-',
                        'hold': hold, 'n': n_evt,
                        'win_rate': win_n / n_evt,
                        'avg_ret': ret_sum / n_evt,
                    }
                    rows.append(row)
                    if verbose:
                        print(f"drop={drop_pct:.2f} vol={vol_ratio:.1f} "
                              f"bull={row['bull']}{row['bull_min']} hold={hold} "
                              f"n={n_evt:4d} 胜率={row['win_rate']:.1%} "
                              f"均值={row['avg_ret']:+.2%}")
    return pd.DataFrame(rows)


def analyze_one(code, end_date=None, save_path=None, **kw):
    """单只股票检测,画图标注恐慌事件与反转确认。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    # 中文字体(Windows)
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    df = _load_df(code, end_date)
    if df is None:
        print(f"!! {code} 数据不足")
        return
    events, stats = detect_panic_events(df, code, **kw)
    print(f"{code} 事件数={stats['n']} 胜率={stats['win_rate']:.1%} "
          f"均值={stats['avg_ret']:+.2%} 最差={stats['worst']:+.2%}")

    closes = df['close'].values.astype(np.float64)
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    vols = df['volume'].values.astype(np.float64)
    dates = df['date'].values
    n = len(df)
    _, slopes = compute_rolling_regression(closes, window=250, use_log=True)
    slope_a = slopes * 250.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#FAFAFA')
    dnum = mdates.date2num(dates)
    cw = 0.62
    up = closes >= opens
    for i in range(n):
        col = '#E8403F' if up[i] else '#1FAE62'
        ax1.vlines(dnum[i], lows[i], highs[i], color=col, lw=0.6, alpha=0.9, zorder=2)
        yb, yt = min(opens[i], closes[i]), max(opens[i], closes[i])
        if yt - yb < 1e-9:
            yt = yb + max((highs.max() - lows.min()) * 0.002, 1e-9)
        ax1.add_patch(Rectangle((dnum[i] - cw / 2, yb), cw, yt - yb,
                                facecolor=col, edgecolor=col, lw=0.4, zorder=3))
    # 斜率
    ax2.plot(dates, slope_a, color='#8E44AD', lw=1.0, label='250日斜率(年化)')
    ax2.axhline(0, color='#555555', lw=0.8, ls='--')
    ax2.legend(loc='best', fontsize=8)
    ax2.set_title('超长回归斜率(>0 = 牛市门控)', fontsize=10)
    ax2.grid(True, alpha=0.2)

    # 事件标注
    ev_dates = [pd.Timestamp(e['event_date']) for e in events]
    cf_dates = [pd.Timestamp(e['confirm_date']) for e in events]
    ax1.scatter(ev_dates, [highs[n - 1]] * len(ev_dates), marker='v',
                color='#C0392B', s=60, zorder=6, label=f'恐慌事件(n={len(ev_dates)})')
    ax1.scatter(cf_dates, [highs[n - 1] * 0.97] * len(cf_dates), marker='^',
                color='#27AE60', s=60, zorder=6, label='反转确认')
    if len(events) > 0:
        ax1.legend(loc='upper left', fontsize=9)
    ax1.set_title(f'{code} 极速杀跌反转事件 (胜率={stats["win_rate"]:.1%}, '
                  f'均值={stats["avg_ret"]:+.2%}, n={stats["n"]})', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.15)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  图保存至: {save_path}")
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'grid':
        df_res = run_grid()
        out = os.path.join('result', 'panic', 'grid_result.csv')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df_res.to_csv(out, index=False, encoding='utf-8-sig')
        print(f"\n已保存: {out}")
    elif len(sys.argv) > 2 and sys.argv[1] == 'one':
        analyze_one(sys.argv[2],
                    save_path=os.path.join('result', 'panic', f'{sys.argv[2]}_panic.png'),
                    drop_pct=0.15, vol_ratio=1.5, hold_days=10,
                    bull_slope_min=0.0, require_bull=True)
    else:
        print(__doc__)
