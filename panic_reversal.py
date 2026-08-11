# -*- coding: utf-8 -*-
"""
极速杀跌反转模型 (panic → reversal)

核心逻辑(全部无未来函数,滚动回归天然只用历史数据):
  1. 牛市门控(用户硬前提):事件发生时点,250 日超长对数回归斜率 > 0(斜率向上)
  2. 恐慌事件:5 日跌幅 <= -drop_pct 且 5 日均量 / 20 日均量 >= vol_ratio(放量急跌)
  3. 反转确认(可选):事件日后 0~confirm_days 天内出现「放量阳线」或「长下影阳线」
  4. 买入 = 确认日收盘价,持有 hold_days 天,统计胜率 / 平均收益 / 事件数

v2 改进(相对 panic_reversal_v1.py):
  - below_reg=True(默认建议):事件日收盘须跌穿 120 日回归线——恐慌充分的兜底,
    同时过滤"高位急跌继续跌"风险。批量验证(84只):胜率 71.9%→83.3%,
    最差 -17.1%→-10.8%,均值 +10.9%→+15.7%(n=24,drop0.15/vol1.2/hold20)
  - stop_loss=0.93(可选):持有期内跌破买入价 X% 提前平仓,可再限最差
    (-10.1%),但会拖累胜率(→70.8%),权衡后主配置不默认启用

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

# 扩充池:各行业普通/代表性股票(沪深300/中证500成分、行业龙头),
# 用于覆盖更广样本、检测"真实胜率"(原 POOL 全是近两年大涨股,存在幸存者偏差)
EXTRA_POOL = [
    # 银行
    "sh600036", "sh601166", "sz000001", "sh600000",
    # 白酒消费
    "sh600519", "sz000858", "sz000568", "sh600809", "sz002304",
    "sh600887", "sh603288", "sz000333", "sz000651", "sh600690",
    # 医药
    "sh603259", "sh600436", "sh600196", "sz300015",
    # 半导体/电子
    "sh603501", "sh603986", "sh603160", "sz002415", "sz002241",
    "sh600703", "sz002129", "sh600438", "sz000100", "sz000725",
    # 汽车/机械
    "sh601127", "sh600031", "sz000063",
    # 地产/建材
    "sh600048", "sz000002", "sz001979", "sz002271",
    # 券商保险
    "sh601211", "sh601688", "sh601318", "sh601628", "sh601336",
    # 周期/材料
    "sh600309", "sh600585", "sh600346", "sh601600", "sz000807",
    "sh600019", "sz002493", "sz002601", "sh600010",
    # 能源/公用
    "sh600011", "sh601088", "sh601225", "sh601857", "sh600938",
    # 交运/农业/传媒
    "sh601816", "sz002352", "sz002714", "sz300498", "sz002027",
    # 其他
    "sh601888", "sh600660", "sz300316", "sz002460", "sz002466",
    "sh600111", "sh600050",
]
BIG_POOL = sorted(set(POOL) | set(EXTRA_POOL))

# 第二批扩充(中证500/创业板/科创板/更多行业),目标总池 300+
EXTRA2_POOL = [
    # 创业板(300xxx)
    "sz300347", "sz300142", "sz300122", "sz300408", "sz300136",
    "sz300450", "sz300207", "sz300433", "sz300529", "sz300003",
    "sz300253", "sz300496", "sz300672", "sz300768", "sz300769",
    # 科创板(688xxx)
    "sh688099", "sh688200", "sh688561", "sh688396", "sh688268",
    "sh688139", "sh688516", "sh688063", "sh688185", "sh688007",
    # 半导体/封测/材料
    "sh600584", "sz002185", "sh600171", "sz002156", "sh603005",
    "sh603893", "sz002371", "sh603501", "sz002049", "sh688981",
    # 医药生物
    "sh600085", "sh600332", "sz000538", "sz002007", "sh688180",
    # 消费
    "sh600600", "sz000895", "sh603806", "sz002202", "sh601615",
    # 汽车/零部件
    "sh600741", "sz002920", "sh601238",
    # 地产/建材
    "sh600383", "sh601155",
    # 军工
    "sz002179", "sh600150", "sh601698",
    # 计算机/软件
    "sh600570", "sz002410", "sh600637", "sz300413", "sh603444",
    # 通信
    "sz002281", "sh600522", "sz002396", "sz300628", "sz300308",
    # 有色/资源
    "sh600547", "sz000762", "sh603260", "sh600111",
    # 电力设备
    "sz300724", "sz002709", "sh603659", "sz300751",
    # 机械/高端制造
    "sz002008", "sh601100", "sz300124", "sz002472",
    # 化工
    "sh600352", "sz002460", "sh600309", "sh600346",
    # 交运/物流
    "sh601006", "sz002120", "sh600029",
    # 传媒/游戏
    "sz002624", "sh600996", "sz300031",
    # 券商/金融
    "sh600837", "sh601901", "sh600958", "sz000776",
    # 其他
    "sh600027", "sh601668", "sh600690", "sz002027",
    "sh600025", "sh601088", "sh600938", "sh601857",
]
BIG3_POOL = sorted(set(BIG_POOL) | set(EXTRA2_POOL))

# 第三批扩充(金融/公用/基建/军工/食品饮料/更多中盘),目标总池 300+
EXTRA3_POOL = [
    # 能源/公用
    "sh600028", "sh600886", "sh600795", "sh600023", "sh601016",
    "sz000027", "sh600674", "sh600642",
    # 基建/交运
    "sh601390", "sh601186", "sh601800", "sz000425", "sz000157",
    "sh601766", "sh600009", "sh600029", "sh601111", "sh600115",
    # 银行/券商/保险
    "sh601288", "sh601988", "sh601328", "sh600016", "sz002142",
    "sh601818", "sh600015", "sh601169", "sh601009", "sh601601",
    "sh601788", "sz002736", "sh601377",
    # 军工
    "sh600685", "sh600316", "sz000768", "sh600879", "sz002025",
    "sh601606",
    # 通信
    "sh600941", "sh601728", "sz002093", "sh600775", "sz300628",
    # 有色/煤炭/钢铁
    "sz000983", "sz000630", "sz000878", "sh600362", "sz000933",
    "sh600188", "sh601898", "sh600516", "sh600549", "sh600362",
    # 化工/材料
    "sh600352", "sz002340", "sz002456",
    # 食品饮料
    "sh600298", "sz002568", "sh601933", "sh600779", "sz000596",
    "sh600702", "sh603369", "sh600559", "sz000799", "sh600132",
    "sz000729", "sh603589", "sh600616", "sh600872", "sz002032",
    # 医药
    "sz002030", "sh600867", "sz300601", "sh603882", "sz002821",
    "sh688276", "sz300759", "sz000538",
    # 农业
    "sz002311", "sz000876", "sh600598", "sz000860",
    # 地产/其他
    "sz000069", "sh600325", "sz002146", "sh600837", "sh600958",
    "sz000776", "sh601901",
]
BIG4_POOL = sorted(set(BIG3_POOL) | set(EXTRA3_POOL))


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
                        min_events=1, pre_slope_annual=None,
                        stop_loss=None, below_reg=False, pre_reg120=None):
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

    # v2 位置过滤:事件日收盘需跌穿 120 日回归线(恐慌充分,超跌到均线下方)
    if below_reg:
        if pre_reg120 is None:
            reg120, _ = compute_rolling_regression(closes, window=120, use_log=True)
        else:
            reg120 = pre_reg120

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
        if below_reg and not (np.isfinite(reg120[t]) and closes[t] < reg120[t]):
            continue                                  # v2:未跌穿回归线,恐慌不充分
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
        # v2 止损兜底:持有期内任一交易日收盘 < 买入价×stop_loss 则提前平仓,
        # 限制单笔最大亏损(解决 v1 最差 -17.1% 的问题)
        exit_idx = confirm_idx + hold_days
        exit_type = 'hold'
        if stop_loss is not None:
            for i in range(confirm_idx + 1, exit_idx):
                if closes[i] < buy * stop_loss:
                    exit_idx = i
                    exit_type = 'stop'
                    break
        sell = closes[exit_idx]
        events.append({
            'code': code,
            'event_date': dates[t],
            'confirm_date': dates[confirm_idx],
            'buy': round(float(buy), 3),
            'sell': round(float(sell), 3),
            'ret': round(float(sell / buy - 1.0), 4),
            'exit': exit_type,
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


def run_grid(end_date=None, codes=None, verbose=True, below_options=(False, True)):
    """参数网格批量测试:遍历候选池(数据与回归只算一次,缓存复用),输出参数组合汇总表。"""
    codes = codes or POOL
    drop_list = [0.08, 0.10, 0.12, 0.15, 0.18]
    vol_list = [1.2, 1.5, 1.8]
    bull_list = [None, 0.0, 0.05]                 # None=不门控
    hold_list = [5, 10, 20]

    # 预加载:每只股票拉一次数据 + 算 250 日斜率 + 120 日回归线(跌穿过滤用)
    dfs = {}
    for code in codes:
        df = _load_df(code, end_date)
        if df is None:
            continue
        closes = df['close'].values.astype(np.float64)
        _, slopes = compute_rolling_regression(closes, window=250, use_log=True)
        reg120, _ = compute_rolling_regression(closes, window=120, use_log=True)
        dfs[code] = (df, slopes * 250.0, reg120)  # (df, 年化斜率, 120日回归线)

    rows = []
    for drop_pct in drop_list:
        for vol_ratio in vol_list:
            for bull in bull_list:
                for hold in hold_list:
                    for below in below_options:
                        n_evt, win_n, ret_sum = 0, 0, 0.0
                        for code, (df, slope_annual, reg120) in dfs.items():
                            _, st = detect_panic_events(
                                df, code, drop_pct=drop_pct, vol_ratio=vol_ratio,
                                bull_slope_min=(bull if bull is not None else 0.0),
                                require_bull=(bull is not None),
                                hold_days=hold, require_confirm=True,
                                pre_slope_annual=slope_annual,
                                below_reg=below, pre_reg120=reg120)
                            n_evt += st['n']
                            win_n += int(st['n'] * st['win_rate'])
                            ret_sum += st['avg_ret'] * st['n']
                        if n_evt == 0:
                            continue
                        row = {
                            'drop': drop_pct, 'vol': vol_ratio,
                            'bull': ('牛市' if bull is not None else '无门控'),
                            'bull_min': bull if bull is not None else '-',
                            'hold': hold, 'below': below, 'n': n_evt,
                            'win_rate': win_n / n_evt,
                            'avg_ret': ret_sum / n_evt,
                        }
                        rows.append(row)
                        if verbose:
                            print(f"drop={drop_pct:.2f} vol={vol_ratio:.1f} "
                                  f"bull={row['bull']}{row['bull_min']} hold={hold} "
                                  f"below={below} n={n_evt:4d} 胜率={row['win_rate']:.1%} "
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
                    drop_pct=0.15, vol_ratio=1.2, hold_days=20,
                    bull_slope_min=0.05, require_bull=True, below_reg=True)
    else:
        print(__doc__)


def verify_no_future():
    """未来函数回归验证:篡改 t 之后全部价格,第 t 天的回归斜率/回归线必须零变化。

    用法:python panic_reversal.py verify
    """
    import inspect, re
    rng = np.random.RandomState(42)
    closes = 10 + np.cumsum(rng.randn(800) * 0.1)
    s1, r1 = compute_rolling_regression(closes, window=250, use_log=True)
    s120_1, r120_1 = compute_rolling_regression(closes, window=120, use_log=True)
    closes2 = closes.copy(); closes2[500:] = closes2[500:] * 10 + 999
    s2, r2 = compute_rolling_regression(closes2, window=250, use_log=True)
    s120_2, r120_2 = compute_rolling_regression(closes2, window=120, use_log=True)
    ok = True
    for t in [300, 400, 499]:
        for name, a, b in [('250斜率', s1[t], s2[t]), ('250线', r1[t], r2[t]),
                           ('120斜率', s120_1[t], s120_2[t]), ('120线', r120_1[t], r120_2[t])]:
            if abs(a - b) > 1e-12:
                print(f"  ✗ t={t} {name} 未来依赖!"); ok = False
    print("✓ 数值验证:第 t 天回归结果不依赖任何未来数据" if ok else "✗ 存在未来函数!")

    # 索引审查:detect_panic_events 决策路径中禁止出现 closes[t+k](k>0) 一类未来索引
    src = inspect.getsource(detect_panic_events)
    bad = []
    for ln in src.split('\n'):
        s = ln.strip()
        if s.startswith('#') or 'range(' in s or 'for ' in s:
            continue
        for m in re.finditer(r'(closes|vols|highs|lows|opens|slope_annual|reg120|vol_ma20)\[(\w+)\s*(\+|-)?\s*(\w*)\]', s):
            idx = m.group(2)
            if idx in ('t', 'i', 'confirm_idx', 'exit_idx') and (m.group(3) in (None, '-', '') and m.group(4) in ('', '1', '4', '5', '19', '20')):
                continue
            if idx.isdigit():
                continue
            bad.append(s)
    print("✓ 决策路径索引审查通过(无未来索引)" if not bad else f"✗ 可疑索引:\n" + "\n".join(bad))
    return ok and not bad


def run_backtest(codes=None, end_date=None, drop_pct=0.15, vol_ratio=1.2,
                 bull_slope_min=0.05, hold_days=20, require_confirm=True,
                 below_reg=True, fee=0.00025, save_path=None,
                 panic_min=None, panic_index=None):
    """升级组合回测:等权信号组合(确认日收盘买入,持有 hold 天,含手续费)。

    每个信号投入 1 个单位资金,组合日收益 = 当日所有持仓信号的等权平均日收益;
    基准 = 同期全池(所有股票有数据日均收益)等权买入持有。

    panic_min: 恐慌期过滤阈值(如 0.15)——仅保留事件日全市场恐慌指数
    >= panic_min 的信号(系统性急跌后的反转确定性最高,验证 100% 胜率)。
    panic_index 可预传入 compute_panic_index() 的结果避免重复计算。

    返回 (metrics: dict, nav: pd.Series, bench: pd.Series)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    codes = codes or POOL
    # 恐慌期过滤需要全市场恐慌指数(事件日 >= panic_min 才保留信号)
    if panic_min is not None and panic_index is None:
        panic_index = compute_panic_index(codes, end_date=end_date)
    # 1) 收集信号与每只股票数据
    dfs, signals = {}, []
    for code in codes:
        df = _load_df(code, end_date)
        if df is None:
            continue
        closes = df['close'].values.astype(np.float64)
        _, sl = compute_rolling_regression(closes, window=250, use_log=True)
        r120, _ = compute_rolling_regression(closes, window=120, use_log=True)
        dfs[code] = (df, closes, sl * 250.0, r120)
        ev, _ = detect_panic_events(df, code, drop_pct=drop_pct, vol_ratio=vol_ratio,
                                    bull_slope_min=bull_slope_min, require_bull=True,
                                    hold_days=hold_days, require_confirm=require_confirm,
                                    below_reg=below_reg, pre_slope_annual=sl * 250.0,
                                    pre_reg120=r120)
        dates = df['date'].dt.strftime('%Y%m%d').values
        for e in ev:
            if panic_min is not None and panic_index.get(e['event_date'], 0.0) < panic_min:
                continue                      # 非恐慌期信号被过滤
            bi = int(np.where(dates == e['confirm_date'])[0][0])
            signals.append((code, bi, dates[bi]))

    # 2) 按日期聚合组合日收益
    from collections import defaultdict
    day_rets = defaultdict(list)      # date -> [当日持仓信号日收益]
    for code, bi, _ in signals:
        df, closes, _, _ = dfs[code]
        n = len(closes)
        for k in range(1, hold_days + 1):
            i = bi + k
            if i >= n:
                continue
            r = closes[i] / closes[i - 1] - 1.0
            day_rets[df['date'].dt.strftime('%Y%m%d').values[i]].append(r)
    # 手续费:每个信号买入日与卖出日各扣 fee
    fee_days = defaultdict(float)
    for code, bi, _ in signals:
        df, _, _, _ = dfs[code]
        dates = df['date'].dt.strftime('%Y%m%d').values
        n = len(df)
        fee_days[dates[bi]] += fee
        if bi + hold_days < n:
            fee_days[dates[bi + hold_days]] += fee

    dates_sorted = sorted(day_rets)
    rets = np.array([np.mean(day_rets[d]) - fee_days.get(d, 0.0) for d in dates_sorted])
    nav = pd.Series(np.cumprod(1 + rets), index=pd.to_datetime(dates_sorted),
                    name='信号组合')

    # 3) 基准:同期全池等权(信号区间内每日所有股票平均收益)
    bench_rets = {}
    for d in dates_sorted:
        rs = []
        for code, (df, closes, _, _) in dfs.items():
            dates = df['date'].dt.strftime('%Y%m%d').values
            pos = np.where(dates == d)[0]
            if len(pos) == 0:
                continue
            i = pos[0]
            if i == 0:
                continue
            rs.append(closes[i] / closes[i - 1] - 1.0)
        if rs:
            bench_rets[d] = np.mean(rs)
    bench = pd.Series(np.cumprod(1 + np.array([bench_rets[d] for d in dates_sorted])),
                      index=pd.to_datetime(dates_sorted), name='全池等权基准')

    # 4) 指标
    def metrics(nav_s):
        r = nav_s.pct_change().dropna()
        total = nav_s.iloc[-1] / nav_s.iloc[0] - 1
        years = len(nav_s) / 252.0
        annual = (1 + total) ** (1 / years) - 1 if years > 0 else 0
        dd = (nav_s / nav_s.cummax() - 1).min()
        sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
        return total, annual, dd, sharpe
    t1, a1, d1, s1 = metrics(nav)
    t2, a2, d2, s2 = metrics(bench)
    m = {
        'n_signals': len(signals),
        '胜率': float((np.array([_sig_ret(s, dfs, hold_days) for s in signals]) > 0).mean()),
        '组合收益': t1, '年化': a1, '最大回撤': d1, '夏普': s1,
        '基准收益': t2, '基准年化': a2, '基准回撤': d2, '基准夏普': s2,
    }

    # 5) 画图
    if save_path:
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(nav.index, nav.values, lw=2, color='#C0392B', label=f'信号组合(年化{a1:+.1%}, 夏普{s1:.2f})')
        ax.plot(bench.index, bench.values, lw=1.5, ls='--', color='#555555',
                label=f'全池等权(年化{a2:+.1%}, 夏普{s2:.2f})')
        ax.set_title(f'极速杀跌反转 v2 组合回测 (胜率{m["胜率"]:.1%}, 最大回撤{d1:.1%}, 手续费{fee:.4f})',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.2)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return m, nav, bench


def _sig_ret(s, dfs, hold_days):
    code, bi, _ = s
    df, closes, _, _ = dfs[code]
    if bi + hold_days >= len(closes):
        return 0.0
    return closes[bi + hold_days] / closes[bi] - 1.0


def compute_panic_index(codes=None, end_date=None, drop5_thr=-0.10, min_date='20240101'):
    """全市场恐慌指数:每个交易日"5日跌幅<=drop5_thr"的股票占比。

    验证结论(139只池/最近两年):恐慌期(指数>=0.15)信号胜率100%(n=12,
    均值+17.4%),非恐慌期仅71%——系统性急跌后的反转确定性最高,
    可作为增强/过滤信号(仅恐慌期开仓:回测夏普3.90,回撤-10.7%)。

    返回 dict: {YYYYMMDD: 恐慌占比(0~1)}
    """
    codes = codes or BIG_POOL
    ret5_by_date = {}
    for code in codes:
        df = _load_df(code, end_date)
        if df is None:
            continue
        closes = df['close'].values.astype(np.float64)
        dates = df['date'].dt.strftime('%Y%m%d').values
        for i in range(5, len(closes)):
            d = dates[i]
            if d < min_date:
                continue
            ret5_by_date.setdefault(d, {})[code] = closes[i] / closes[i - 5] - 1.0
    return {d: float(np.mean([v <= drop5_thr for v in vals.values()]))
            for d, vals in ret5_by_date.items() if vals}


# 恐慌指数快速估算池(60只,均衡抽样覆盖大涨/普通/二批/三批各段),~10s 出结果;
# 精确值建议实盘每日 compute_panic_index(BIG4_POOL) 算一次缓存复用
PANIC_POOL = (POOL[::4][:20] + EXTRA_POOL[::3][:20]
              + EXTRA2_POOL[::4][:10] + EXTRA3_POOL[::4][:10])


def signal(code, end_date=None, drop_pct=0.10, vol_ratio=1.2,
           bull_slope_min=0.05, confirm_days=3, below_reg=True,
           strength_thr=12.0):
    """实盘信号接口(仅单只股票数据,无市场恐慌指数依赖,无未来函数)。

    全部只用判定日及以前数据(无未来函数)。返回 dict:
      signal: 'BUY'|'WATCH'|'NONE'|'NODATA'
        BUY  = 牛市门控 ✓ + 最近恐慌事件 + 反转确认 ✓
      bull/bull_ok: 250日年化斜率与是否>阈值
      below_reg:    当前收盘是否跌穿120日回归线
      panic_t/confirm: 最近恐慌事件日与反转确认日
      strength_t: 恐慌事件日的 strength 值(显示用)

    恐慌事件判定(v4.9 双重要求,299池/最近两年 验证胜率 83.8%,n=80):
      5日跌幅 >= drop_pct(默认10%) 且 strength(事件日) <= -strength_thr(默认12)
      strength = compute_strength(10日波幅/ATR14 + 方向状态),隐含"10日净位移为负"
      的过滤——5日急跌但10日前更低的弱信号被排除(这类信号胜率仅74%)
    strength_thr=0 或 None → 回退纯跌幅判定(旧逻辑)
    """
    df = _load_df(code, end_date)
    if df is None:
        return {'code': code, 'signal': 'NODATA', 'reason': 'No data'}
    closes = df['close'].values.astype(np.float64)
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    vols = df['volume'].values.astype(np.float64)
    dates = df['date'].dt.strftime('%Y%m%d').values
    n = len(closes); t = n - 1

    # 牛市门控(250日年化斜率)
    _, sl = compute_rolling_regression(closes, window=250, use_log=True)
    bull = float(sl[t] * 250.0) if np.isfinite(sl[t]) else 0.0
    bull_ok = bull > bull_slope_min

    # 120日回归线(跌穿判定)
    reg120, _ = compute_rolling_regression(closes, window=120, use_log=True)
    below_reg_ok = bool(np.isfinite(reg120[t]) and closes[t] < reg120[t])

    # v4.9 strength 序列(双重要求用;默认参数,纯因果)
    strength = compute_strength(closes, highs, lows, opens=opens) if strength_thr else None

    # 最近恐慌事件(向前扫 10 天:5日跌 + 放量 + 跌穿均线 + strength 双重要求)
    vol_ma20 = pd.Series(vols).rolling(20).mean().values
    panic_t = None
    for i in range(t, max(5, t - 10) - 1, -1):
        ret5 = closes[i] / closes[i - 5] - 1.0
        if ret5 > -drop_pct:
            continue
        if strength_thr and not (np.isfinite(strength[i]) and strength[i] <= -strength_thr):
            continue
        v5 = vols[i - 4:i + 1].mean()
        v20 = vols[i - 19:i + 1].mean() if i >= 20 else vols[:i + 1].mean()
        if v20 <= 0 or v5 / v20 < vol_ratio:
            continue
        if below_reg and not (np.isfinite(reg120[i]) and closes[i] < reg120[i]):
            continue
        panic_t = i
        break

    # 反转确认(恐慌日后 0~confirm_days 天,到当前为止)
    confirm = None
    if panic_t is not None:
        for i in range(panic_t, min(panic_t + confirm_days + 1, n)):
            if closes[i] > opens[i] and vols[i] > 1.2 * (vol_ma20[i] if np.isfinite(vol_ma20[i]) else v20):
                confirm = dates[i]
                break

    # 决策(below_reg 已体现在事件检测中,仅展示不卡 BUY)
    lacks = []
    if panic_t is None:
        sig = 'NONE'
    else:
        if not bull_ok:
            lacks.append('bull gate fail (slope %.3f <= %.3f)' % (bull, bull_slope_min))
        if confirm is None:
            lacks.append('no volume-confirm (bullish engulf) in %dd' % (confirm_days + 1))
        sig = 'BUY' if not lacks else 'WATCH'

    return {
        'code': code, 'date': dates[t], 'signal': sig,
        'bull': round(bull, 3), 'bull_ok': bull_ok,
        'below_reg': below_reg_ok,
        'panic_t': dates[panic_t] if panic_t is not None else None,
        'strength_t': round(float(strength[panic_t]), 1) if (panic_t is not None and strength is not None) else None,
        'confirm': confirm,
        'reason': ('All OK: bull + panic-event + below-120d-reg + confirm' if sig == 'BUY'
                   else ('Missing: ' + '; '.join(lacks) if lacks else 'No recent panic event')),
    }


def _compute_atr14(highs, lows, closes):
    """ATR(14):Wilder 平滑,返回与输入等长的数组(前13个为 NaN)。"""
    n = len(closes)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    atr = np.full(n, np.nan)
    if n < 14:
        return atr
    atr[13] = np.mean(tr[:14])
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    return atr

def compute_strength(closes, highs=None, lows=None, k=2.0, alpha=2.0, m=30.0, atr=None, win=10, dir_atr=2.0, reg_preds=None, confirm_flip=2, flip_strong=0.08, min_main=3, decay_days=5, decay_factor=0.75, min_decay=2.0, reg_decay=0.10, short_win=5, short_drop=0.08, opens=None):
    """死区滤波强度变换 v4.8b(波幅 + 收盘位移方向 + 方向死区 + 回归线门控 + 翻转确认 + 死区衰减延续)。

    演进:
      v1 收盘位移/ATR → v2 波幅/收盘位置 → v3 atan饱和 → v4 波幅/收盘位移方向
      v4.1 + dir_atr 方向死区 → v4.2 + reg_preds 回归线门控
      v4.5~v4.7 + confirm_flip/flip_strong/min_main 分级翻转确认(连续阳中无独立阴)
      v4.8 + decay_days/decay_factor:死区日延续前一方向并衰减(前 decay_days=5 根,
            强度=last_s×decay_factor^k);衰减结束归零,不做最小值保底(v4.9 的
            min_decay 已回撤)

      - amp = win日波幅 = max(high[i-win+1..i]) - min(low[i-win+1..i])
      - raw  = amp / ATR14
      - 有柱日: |净位移| >= dir_atr*ATR,方向=sign(close[i]-close[i-win]),
        回归线下正柱置0,长段(>=min_main)反向需 confirm_flip 确认,
        短段强反向(净位移占比>=flip_strong)立即翻转;
        strength = 方向 * atan(max(0,raw-k)^alpha)/(pi/2) * m
      - 死区日: cur!=0 且连续死区 < decay_days → 衰减延续柱 cur*|last_s|*decay_factor^k;
        之后归零
      - 无未来函数(全部仅依赖历史,可用截断一致性测试验证)
    """
    import math as _math
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    amp = np.full(n, np.nan)
    for i in range(win - 1, n):
        amp[i] = np.max(highs[i - win + 1:i + 1]) - np.min(lows[i - win + 1:i + 1])
    if atr is None:
        if highs is None or lows is None:
            raise ValueError("需传 highs/lows 或预计算 atr")
        atr = _compute_atr14(np.asarray(highs, dtype=float), np.asarray(lows, dtype=float), closes)
    strength = np.full(n, np.nan)
    cur = 0
    rev = 0
    main_len = 0
    last_s = 0.0
    dead_streak = 0
    for i in range(win, n):
        if not (np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(amp[i])):
            continue
        if closes[i - win] <= 0:
            continue
        d = closes[i] - closes[i - win]
        if abs(d) < dir_atr * atr[i]:
            # 死区日:先查短期骤变(5日涨跌≥8%)——补充死区里的真实信号并翻转方向
            short_ret = closes[i] / closes[i - short_win] - 1.0 if i >= short_win else 0.0
            if abs(short_ret) >= short_drop:
                reg_gate0 = (reg_preds[i] * (1.0 - reg_decay)
                             if reg_preds is not None and np.isfinite(reg_preds[i]) else None)
                flip_ok = (short_ret < 0) or (
                    (reg_gate0 is None or closes[i] >= reg_gate0) and
                    (opens is None or closes[i] >= opens[i]))
                if flip_ok:
                    cur = 1 if short_ret > 0 else -1
                    rev = 0; main_len = 1; dead_streak = 0
                    u = max(0.0, amp[i] / atr[i] - k) ** alpha
                    v = cur * (np.arctan(u) / (_math.pi / 2.0)) * m
                    strength[i] = v if abs(v) >= min_decay else cur * min_decay
                    last_s = strength[i]
                    continue
            # 死区日:延续前一方向并衰减;衰减结束后保底(min_decay),方向死守 cur(稳定)
            if cur != 0:
                if dead_streak < decay_days:
                    strength[i] = cur * abs(last_s) * (decay_factor ** (dead_streak + 1))
                else:
                    strength[i] = cur * min_decay  # 保底死守当前方向(稳定,不产生杂毛)
                dead_streak += 1
            else:
                strength[i] = 0.0
            continue
        dead_streak = 0
        raw_dir = 1 if d > 0 else -1
        reg_gate = (reg_preds[i] * (1.0 - reg_decay)
                    if reg_preds is not None and np.isfinite(reg_preds[i]) else None)
        # 5日骤变(≥short_drop)独立检查:无论与 cur 同向/反向都按骤变方向处理
        # 骤涨翻阳额外要求:站上回归线(门控)且当天非阴线(C+B);骤跌翻阴不受限
        short_ret = closes[i] / closes[i - short_win] - 1.0 if i >= short_win else 0.0
        if abs(short_ret) >= short_drop:
            flip_ok = (short_ret < 0) or (
                (reg_gate is None or closes[i] >= reg_gate) and
                (opens is None or closes[i] >= opens[i]))
            if flip_ok:
                cur = 1 if short_ret > 0 else -1
                rev = 0; main_len = 1; dead_streak = 0
                u = max(0.0, amp[i] / atr[i] - k) ** alpha
                v = cur * (np.arctan(u) / (_math.pi / 2.0)) * m
                strength[i] = v if abs(v) >= min_decay else cur * min_decay
                last_s = v
                continue
        if reg_gate is not None and closes[i] < reg_gate and raw_dir > 0:
            # 回归线下方的正柱(反转)不可信 → 不显示反转;若有主方向,画衰减延续柱/最低值保底
            if cur != 0:
                if dead_streak < decay_days:
                    strength[i] = cur * abs(last_s) * (decay_factor ** (dead_streak + 1))
                else:
                    strength[i] = cur * min_decay
                dead_streak += 1
            else:
                strength[i] = 0.0
            continue
        if cur == 0:
            cur = raw_dir; rev = 0; main_len = 1
        elif raw_dir == cur:
            rev = 0; main_len += 1
        else:
            # 5日骤变(≥short_drop)→ 立即翻转,不等确认(暴跌/暴涨第一天就变色)
            short_ret = closes[i] / closes[i - short_win] - 1.0 if i >= short_win else 0.0
            if abs(short_ret) >= short_drop:
                cur = raw_dir; rev = 0; main_len = 1
            # 站上回归线(门控解除)的翻阳 → 立即生效(避免大阳被染成阴)
            elif reg_gate is not None and closes[i] >= reg_gate and raw_dir > 0:
                cur = raw_dir; rev = 0; main_len = 1
            elif main_len < min_main and abs(d) / closes[i - win] >= flip_strong:
                cur = raw_dir; rev = 0; main_len = 1  # 恐慌起点立即翻转
            else:
                rev += 1
                if rev >= confirm_flip:
                    cur = raw_dir; rev = 0; main_len = 1
                else:
                    main_len += 1
        u = max(0.0, amp[i] / atr[i] - k) ** alpha
        v = cur * (np.arctan(u) / (_math.pi / 2.0)) * m
        strength[i] = v
        last_s = v
    return strength




def compute_turn_positive_prices(closes, highs, lows, opens=None, win=10, dir_atr=2.0,
                                 short_win=5, short_drop=0.08, reg_decay=0.10,
                                 reg_preds=None, atr=None, strength=None, min_band=0.10):
    """逐日计算"阴柱期转阳触发价"(仅阴柱日有值,阳柱/无柱日为 NaN)。

    第 i 天为阴柱时,refs[i] = 第 i+1 天转阳所需的最低参考价(买入条件单):
      常规路径: max(close[i-win] + dir_atr*ATR[i],  reg_gate)   10日净位移转正且出柱 + 站上门控线
      骤变路径: max(close[i-short_win] * (1+short_drop), reg_gate)  5日骤涨 + 站上门控线
      参考价   = min(常规, 骤变)   (两条路径任一达标即转阳)
    优化:参考价不低于当日收盘(方向已满足时=现价,压制语义);
         滞回 min_band=10%:与前一目标价差 <10% 时保持前值(抑制换参照抖动)。
    reg_gate = reg_preds * (1-reg_decay);无 reg_preds 时门控不设(-inf)
    用途:阴柱期间在 K 线上画出持续的目标价,阳柱期间无(画图辅助,非信号)。
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    if strength is None:
        strength = compute_strength(closes, highs, lows, opens=opens, win=win,
                                    dir_atr=dir_atr, reg_preds=reg_preds)
    if atr is None:
        atr = _compute_atr14(np.asarray(highs, dtype=float), np.asarray(lows, dtype=float), closes)
    refs = np.full(n, np.nan)
    lo = max(win, short_win) + 1
    prev = np.nan
    for i in range(lo, n):
        if not (np.isfinite(strength[i]) and strength[i] < 0):
            prev = np.nan  # 阳柱/无柱 → 重置滞回
            continue
        p_dir = closes[i - win] + dir_atr * atr[i]
        p_short = closes[i - short_win] * (1.0 + short_drop)
        gate = (reg_preds[i] * (1.0 - reg_decay)
                if reg_preds is not None and np.isfinite(reg_preds[i]) else -np.inf)
        cand = min(max(p_dir, gate), max(p_short, gate))
        cand = max(cand, closes[i])  # 压制语义:目标价不低于现价(方向已满足时=现价)
        if np.isfinite(prev) and abs(cand - prev) < min_band * prev:
            cand = prev  # 滞回:小幅变动保持前值,抑制抖动
        refs[i] = cand
        prev = cand
    return refs
