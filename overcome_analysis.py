# -*- coding: utf-8 -*-
"""
克服前放量堆高点因子验证。
新字段: dist_vol_high / overcome60 / overcome120 / overcome_wait / overcome_volratio
口径:
  放量日 t: volume[t] > TH * mean(volume[t-60:t])  (t 之前 60 日均量, 不含 t, 无未来函数)
  前放量堆 = 信号日前 250 交易日窗口内最后一个连续放量日区间; vol_high = 区间最高价
  dist_vol_high = (sig_close - vol_high)/vol_high*100
  克服: 信号日后第 k 个交易日首次 close>vol_high; overcome60 k<=60, overcome120 k<=120
  截断: fut60_days<60 -> overcome60=NaN; fut120_days<120 -> overcome120=NaN (与 s60/s120 一致)
  overcome_volratio = 克服日 volume / 其前 20 日均量(不含克服日)
用法: python3 overcome_analysis.py
"""
import os, sys, json
import numpy as np
import pandas as pd
from multiprocessing import Pool

CACHE = 'result/scan_cache_v3b'
SIG_CSV = 'result/backtest_v3b_5y.csv'
OUT_CSV = 'result/backtest_v3b_5y_overcome.csv'
OUT_STATS = 'result/overcome_stats.json'


def _bd(d):
    t = pd.Timestamp(d)
    return str(t)[:10].replace('-', '')


def load_code(code):
    p = os.path.join(CACHE, f'{code}.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if len(df) < 260:
        return None
    dates = np.array([_bd(d) for d in df['date'].values])
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    vol = df['volume'].values.astype(float)
    return dates, close, high, vol


def compute_code(args):
    """返回该 code 所有信号的新字段 list (dict 含原始行定位)."""
    code, rows = args  # rows: list of dict with sig_date, fut60_days, fut120_days, sig_close
    ld = load_code(code)
    if ld is None:
        return [dict(r, dist_vol_high=np.nan, overcome60=np.nan, overcome120=np.nan,
                     overcome_wait=np.nan, overcome_volratio=np.nan, miss=1) for r in rows]
    dates, close, high, vol = ld
    n = len(close)
    # 60 日均量(不含当日): vol_ma60[t] = mean(vol[t-60:t])
    s = pd.Series(vol)
    vol_ma60 = s.shift(1).rolling(60, min_periods=60).mean().values
    res = []
    for r in rows:
        sd = str(int(r['sig_date']))
        i = np.searchsorted(dates, sd)
        if i >= n or dates[i] != sd:
            res.append(dict(r, dist_vol_high=np.nan, overcome60=np.nan, overcome120=np.nan,
                            overcome_wait=np.nan, overcome_volratio=np.nan, miss=2))
            continue
        sig_close = close[i]
        # 窗口 [i-250, i-1]
        w0 = max(0, i - 250)
        w1 = i  # exclusive
        if i - w0 < 60:
            res.append(dict(r, dist_vol_high=np.nan, overcome60=np.nan, overcome120=np.nan,
                            overcome_wait=np.nan, overcome_volratio=np.nan, miss=3))
            continue
        win_vol = vol[w0:w1]
        win_ma = vol_ma60[w0:w1]
        valid = ~np.isnan(win_ma)
        big = np.zeros(w1 - w0, dtype=bool)
        big[valid] = win_vol[valid] > TH * win_ma[valid]
        if not big.any():
            res.append(dict(r, dist_vol_high=np.nan, overcome60=np.nan, overcome120=np.nan,
                            overcome_wait=np.nan, overcome_volratio=np.nan, miss=0))
            continue
        # 最后一个连续放量区间
        last_rel = np.where(big)[0].max()
        run_end_rel = last_rel
        run_start_rel = last_rel
        while run_start_rel - 1 >= 0 and big[run_start_rel - 1]:
            run_start_rel -= 1
        run_abs0 = w0 + run_start_rel
        run_abs1 = w0 + run_end_rel + 1
        vol_high = high[run_abs0:run_abs1].max()
        dist = (sig_close - vol_high) / vol_high * 100.0
        # 向前克服扫描
        f60d = int(r['fut60_days']); f120d = int(r['fut120_days'])
        oc60 = np.nan; oc120 = np.nan; owait = np.nan; ovolr = np.nan
        # 首个 close>vol_high 的 k (1-based 交易日距信号日)
        overcome_k = np.nan
        for k in range(1, n - i):
            if close[i + k] > vol_high:
                overcome_k = k
                break
        if f60d >= 60:
            oc60 = 1.0 if (not np.isnan(overcome_k) and overcome_k <= 60) else 0.0
        if f120d >= 120:
            oc120 = 1.0 if (not np.isnan(overcome_k) and overcome_k <= 120) else 0.0
        if not np.isnan(overcome_k):
            owait = overcome_k
            j = i + int(overcome_k)
            if j - 20 >= 0:
                base20 = vol[j - 20:j].mean()
                if base20 > 0:
                    ovolr = vol[j] / base20
        res.append(dict(r, dist_vol_high=round(dist, 3), overcome60=oc60, overcome120=oc120,
                        overcome_wait=owait, overcome_volratio=(round(ovolr, 3) if not np.isnan(ovolr) else np.nan), miss=0))
    return res


TH = 1.5  # 放量阈值, 全局; 敏感性 2.0 主流程后单独跑


def main():
    df = pd.read_csv(SIG_CSV, dtype={'code': str})
    df['code'] = df.code.str.zfill(6)
    df['_ridx'] = np.arange(len(df))
    # 按 code 分组
    groups = [(c, g[['_ridx', 'sig_date', 'fut60_days', 'fut120_days']].to_dict('records'))
              for c, g in df.groupby('code')]
    print(f'codes={len(groups)} signals={len(df)}')
    out_rows = []
    with Pool() as pool:
        for res in pool.imap_unordered(compute_code, groups, chunksize=8):
            out_rows.extend(res)
    od = pd.DataFrame(out_rows).drop(columns=['miss']).sort_values('_ridx').reset_index(drop=True)
    newcols = od[['dist_vol_high', 'overcome60', 'overcome120', 'overcome_wait', 'overcome_volratio']].reset_index(drop=True)
    df2 = pd.concat([df.drop(columns=['_ridx']).reset_index(drop=True), newcols], axis=1)
    df2.to_csv(OUT_CSV, index=False)
    n_miss = int(df2.dist_vol_high.isna().sum())
    print(f'wrote {OUT_CSV} rows={len(df2)} dist_vol_high缺失={n_miss} ({n_miss/len(df2)*100:.1f}%)')


if __name__ == '__main__':
    main()
