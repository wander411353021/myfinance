# -*- coding: utf-8 -*-
"""early 三线制(E4) 全市场 5 年扫描
定义: 站上参考线B(COST40/E50)×1.03 ∩ 未到红压线gt_p×0.98 ∩ [T-250,T-20]曾站上红压线
      ∩ [T-70,T-5]曾跌破B×0.98 ∩ 无量能要求 ∩ 冷却20天(站B事件)
用法: python3 scan_early_5y.py <part> <nparts> <end>   (end 如 20261016)
"""
import os, sys, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from challenge_scan import compute_eff, MIN_I, V3B_REC_DIP, ehlers_smooth
from chip_panel import cost_series

WATCH = ['300171','003040','300204','300251','300437','600199','600519','600371','601579','002343','000530','000993','002460','300299','300253','600737','600721','600613','688233','688388','600865','301080','603778','600550','001696','300006','688018','600152','301487','300619','002866','603278','002923','002441','301087','300675','300389']
CACHE = 'result/scan_cache_v3b'
OUT_P = 'result/early_scan_shard{}.csv'
COOL = 20
MULT = 1.03

def scan_code(code, end):
    fp = os.path.join(CACHE, code + '.csv')
    if not os.path.exists(fp):
        return None
    try:
        df = pd.read_csv(fp)
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
        if len(df) < MIN_I + 130:
            return None
        hi = df['high'].values.astype(float); lo = df['low'].values.astype(float)
        vol = df['volume'].values.astype(float); turn = df['turnover'].values.astype(float)
        cl, darr, gt_p, gt2, gtp_e, gt2_e, c35s, base, cA = compute_eff(df)
        cB = np.asarray(ehlers_smooth(cost_series(hi, lo, vol, turn, 0.40), 50), float)
    except Exception:
        return None
    n = len(cl); last = -10**9; rows = []
    for i in range(MIN_I, n):
        pi = gtp_e[i] if np.isfinite(gtp_e[i]) else gt_p[i]
        if not (np.isfinite(pi) and pi > 0 and np.isfinite(cB[i]) and cB[i] > 0):
            continue
        if cl[i] < cB[i] * MULT:
            continue
        # E4 上限: 未站上主成本线A(红压线~A之间归early, 与v3b_B2在A处分界)
        if cl[i] >= cA[i]:
            continue
        if i - last < COOL:
            continue
        lo2, hi2 = max(MIN_I, i - 250), i - 20
        h = False
        if lo2 < hi2:
            pw = gt_p[lo2:hi2]
            h = bool(np.any(np.isfinite(pw) & (pw > 0) & (cl[lo2:hi2] >= pw)))
        if not h:
            continue
        rlo, rhi = max(0, i - 70), i - 5
        r = False
        if rlo < rhi:
            cw = cB[rlo:rhi]
            r = bool(np.any(np.isfinite(cw) & (cw > 0) & (cl[rlo:rhi] < cw * V3B_REC_DIP)))
        if not r:
            continue
        d = darr[i]
        if d < '20211001' or d > end:
            continue
        f60 = cl[i+1:min(i+61, n)]; f120 = cl[i+1:min(i+121, n)]
        f60mx = (f60.max()/cl[i]-1)*100 if len(f60) >= 20 else np.nan
        f120mx = (f120.max()/cl[i]-1)*100 if len(f120) >= 60 else np.nan
        s60 = int(np.isfinite(f60mx) and f60mx >= 10)
        s120 = int(np.isfinite(f120mx) and f120mx >= 50)
        fail = int(np.isfinite(f60mx) and f60mx < 5)
        rows.append((code, d, round(float(cl[i]), 2), round(float(pi), 2),
                     round(float(cB[i]), 2), round((cl[i]/cB[i]-1)*100, 1),
                     round((cl[i]/cA[i]-1)*100, 1) if np.isfinite(cA[i]) and cA[i] > 0 else np.nan,
                     s60, s120, fail,
                     round(float(f60mx), 1) if np.isfinite(f60mx) else np.nan))
        last = i
    if not rows:
        return None
    return pd.DataFrame(rows, columns=['code', 'sig_date', 'sig_close', 'gt_p', 'cB',
                                       'dev_b', 'dev_a', 's60', 's120', 'fail', 'fut60'])

def main():
    part, nparts, end = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    codes = sorted(f[:-4] for f in os.listdir(CACHE) if f.endswith('.csv'))
    mine = codes[part::nparts]
    t0 = time.time(); parts = []
    for k, code in enumerate(mine):
        df = scan_code(code, end)
        if df is not None:
            parts.append(df)
        if (k + 1) % 200 == 0:
            print(f'[part{part}] {k+1}/{len(mine)} {time.time()-t0:.0f}s', flush=True)
    if parts:
        out = pd.concat(parts, ignore_index=True)
        out.to_csv(OUT_P.format(part), index=False)
        print(f'[part{part}] done n={len(out)} -> {OUT_P.format(part)}', flush=True)
    else:
        print(f'[part{part}] no rows')

if __name__ == '__main__':
    main()
