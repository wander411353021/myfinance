# -*- coding: utf-8 -*-
"""
首选策略路径回测 —— 敏感性与稳健性分析(增强逐信号记录)。
复用 confirm_path_backtest 的因果口径, 额外存储:
  - 两种模式各自首个成本线破位日偏移 brk_off / 卖出价 brk_px
  - T+40/60/80 收盘 px40/px60/px80 (avail<H 时为 NaN)
然后 pandas 派生: H∈{40,60,80} 下的成交/收益/持有/离场原因。
输出: result/confirm_signals_enriched.csv
用法: python3 confirm_sensitivity.py
"""
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool

CACHE = 'result/scan_cache_v3b'
SIG_CSV = 'result/backtest_v3b_5y_overcome.csv'
OUT = 'result/confirm_signals_enriched.csv'

BIG_PCT = 0.05
BIG_PCT_MAX = 0.21
TH_VOL = 1.5
COST_PCT = 0.75
EHLERS_P = 25
EXIT_K = 0.90
HS = (40, 60, 80)


def ehlers_smooth(x, period):
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    a1 = np.exp(-1.414 * np.pi / period)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / period)
    c3 = -a1 * a1
    c2 = b1
    c1 = 1.0 - c2 - c3
    out[0] = x[0]
    out[1] = c1 * (x[1] + x[0]) / 2 + (c2 + c3) * x[0]
    for k in range(2, n):
        out[k] = c1 * (x[k] + x[k - 1]) / 2 + c2 * out[k - 1] + c3 * out[k - 2]
    return out


def _bd(d):
    return str(pd.Timestamp(d))[:10].replace('-', '')


def first_break(close, cost_line, j0, j1):
    """j0..j1 内首个 close<EXIT_K×cost 的索引(绝对), 无则 None。"""
    for j in range(j0, j1 + 1):
        if not np.isfinite(cost_line[j]):
            continue
        if close[j] < EXIT_K * cost_line[j]:
            return j
    return None


def process_group(args):
    code, rows = args
    p = os.path.join(CACHE, f'{code}.csv')
    if not os.path.exists(p):
        return []
    df = pd.read_csv(p)
    n = len(df)
    if n < 260:
        return []
    dates = np.array([_bd(d) for d in df['date'].values])
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    vol = df['volume'].values.astype(float)
    turn = df['turnover'].values.astype(float).clip(0, 1)

    vol_ma60 = pd.Series(vol).shift(1).rolling(60, min_periods=60).mean().values
    try:
        cost_raw = __import__('fengwo').COST(high, low, vol, turn,
                                             np.full(n, COST_PCT))
        cost_line = ehlers_smooth(np.asarray(cost_raw, dtype=float), EHLERS_P)
    except Exception:
        cost_line = np.full(n, np.nan)

    out = []
    for r in rows:
        sd = str(int(r['sig_date']))
        i = int(np.searchsorted(dates, sd))
        if i >= n or dates[i] != sd:
            continue
        avail = n - 1 - i

        w0 = max(0, i - 250)
        win_vol = vol[w0:i]
        win_ma = vol_ma60[w0:i]
        valid = ~np.isnan(win_ma)
        bigv = np.zeros(i - w0, dtype=bool)
        bigv[valid] = win_vol[valid] > TH_VOL * win_ma[valid]
        if not bigv.any():
            continue
        last_rel = int(np.where(bigv)[0].max())
        run_start = last_rel
        while run_start - 1 >= 0 and bigv[run_start - 1]:
            run_start -= 1
        vol_high = high[w0 + run_start:w0 + last_rel + 1].max()

        ov_off = np.nan
        for k in range(1, avail + 1):
            if close[i + k] > vol_high:
                ov_off = float(k)
                break
        big_off = np.nan
        for k in range(1, min(21, avail + 1)):
            g = close[i + k] / close[i + k - 1] - 1.0
            if np.isfinite(g) and BIG_PCT <= g <= BIG_PCT_MAX:
                big_off = float(k)
                break
        if not (np.isfinite(big_off) and np.isfinite(ov_off)):
            continue

        c_off = min(big_off, ov_off)
        buy_off_A = c_off + 1.0
        rec = dict(code=code, sym=r['sym'], sig_date=sd, sig_type=r['sig_type'],
                   year=sd[:4], qfq_flag=int(r['qfq_flag']),
                   dist_high=r['dist_high'], sig_close=close[i],
                   big_off=big_off, ov_off=ov_off, avail_days=avail,
                   buy_off_A=buy_off_A)

        # T+H 到期收盘
        for H in HS:
            rec[f'px{H}'] = close[i + H] if avail >= H else np.nan

        # 模式A: 买入日后首个破位
        buy_idx_A = i + int(buy_off_A)
        if buy_idx_A < n:
            jb = first_break(close, cost_line, buy_idx_A + 1, n - 1)
            rec['brk_off_A'] = float(jb - i) if jb is not None else np.nan
            rec['brk_px_A'] = close[jb] if jb is not None else np.nan
            rec['buy_px_A'] = close[buy_idx_A]
        else:
            rec['brk_off_A'] = np.nan
            rec['brk_px_A'] = np.nan
            rec['buy_px_A'] = np.nan
        # 模式B: 信号日后首个破位
        jbB = first_break(close, cost_line, i + 1, n - 1)
        rec['brk_off_B'] = float(jbB - i) if jbB is not None else np.nan
        rec['brk_px_B'] = close[jbB] if jbB is not None else np.nan
        out.append(rec)
    return out


def main():
    df = pd.read_csv(SIG_CSV, dtype={'code': str})
    df['code'] = df.code.str.zfill(6)
    pop = ((df.big20 == 1) & (df.overcome60 == 1.0) & (df.fut60_days >= 60)
           & (df.dist_high > 30) & (df.sig_type.isin(['v3b', 'early'])))
    pop_df = df[pop].copy()
    print(f'处理总体: {len(pop_df)} 信号, codes={pop_df.code.nunique()}')
    groups = [(c, g[['sig_date', 'sym', 'sig_type', 'qfq_flag', 'dist_high']]
               .to_dict('records')) for c, g in pop_df.groupby('code')]
    rows = []
    with Pool(12) as pool:
        for res in pool.imap_unordered(process_group, groups, chunksize=16):
            rows.extend(res)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f'wrote {OUT} rows={len(out)}')


if __name__ == '__main__':
    main()
