# -*- coding: utf-8 -*-
"""v3 / v3b / early 三信号全市场 5 年解剖扫描 v2 (2026-10-18 polo4111)
复用 challenge_scan.compute()/signal_at()/v3b_signal_at()/v3b_early_at(), 信号判定零改动。
性能: cost_series 全数组每code预计算一次(c10/c90), 逐命中取下标 —— 与 signal_profile 公式逐字段一致
      (扫描前 --verify 抽样校验 profile_light == signal_profile)。
口径(与 backtest_v3b_5y.py 一致):
  窗口 [20211001, 20261016]; 冷却 v3b/early 各自 V3B_COOL=20, v3 独立 COOL=30, 窗口外推进冷却链
  收益 _fut 60/120; qfq_flag=信号日后60日内|qfq日涨幅|>板块涨停上限
用法: python3 anatomy_scan_v3_v3b_early.py --nshards 4 --shard K [--verify] [--resume]
产物: result/anatomy_shard{K}.csv → merge 后 result/anatomy_signals.csv
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
import challenge_scan as cs
import tdx_quant as tq
from chip_panel import cost_series

REPO = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(REPO, 'result', 'scan_cache_v3b')
WIN_START = '20211001'
WIN_END = '20261016'
MIN_I = cs.MIN_I
V3B_COOL = cs.V3B_COOL
V3_COOL = cs.COOL

COLS = ['code', 'sym', 'sig_date', 'hit_v3', 'hit_v3b', 'hit_early',
        'sig_close', 'vol_ratio', 'rec_depth', 'rec_days', 'cool_days', 'hist_touch',
        'dist_high', 'dev35', 'band', 'profit', 'gap', 'dist_purple', 'abr',
        'fut60_max', 'fut60_min', 'fut120_max', 'fut60_days', 'fut120_days',
        's60', 's120', 'fail', 'qfq_flag']


def board_limit(code):
    if code.startswith(('60', '00')):
        return 10.5
    if code.startswith(('30', '68', '69')):
        return 20.5
    if code[0] in ('8', '9', '4'):
        return 30.5
    return 10.5


def _fut(cl, i, k):
    n = len(cl)
    e = min(i + k, n - 1)
    if i + 1 > e:
        return np.nan, np.nan, 0
    seg = cl[i + 1:e + 1]
    return (round((seg.max() / cl[i] - 1) * 100, 2),
            round((seg.min() / cl[i] - 1) * 100, 2), e - i)


def profile_light(cl, hi, lo, vol, turn, gt_p, gt2, c35s, c10, c90, i):
    """与 signal_profile 完全同公式, 但 c10/c90 预计算。"""
    n = len(cl)
    ma20 = vol[max(0, i - 20):i].mean()
    vr = vol[i] / ma20 if ma20 > 0 else 0.0
    rlo, rhi = max(0, i - cs.V3B_REC_LO), i - cs.V3B_REC_HI
    if rlo < rhi:
        dip = cl[rlo:rhi] / c35s[rlo:rhi] - 1.0
        dmin = float(np.nanmin(dip)) if len(dip) else 0.0
        under = np.isfinite(dip) & (dip < cs.V3B_REC_DIP - 1.0)
        rec_days = int(under.sum())
    else:
        dmin, rec_days = 0.0, 0
    hlo, hhi = max(MIN_I, i - cs.V3B_HIST_LO), i - cs.V3B_HIST_HI
    if hlo < hhi:
        p2w = gt2[hlo:hhi]
        tj = np.where(np.isfinite(p2w) & (p2w > 0) & (cl[hlo:hhi] >= p2w * cs.V3B_TOUCH))[0]
        cool = i - (hlo + tj[-1]) if len(tj) else (i - hlo)
        hist_touch = int(len(tj))
    else:
        cool, hist_touch = 250, 0
    h250 = float(np.nanmax(hi[max(0, i - 250):i + 1]))
    ratio = h250 / cl[i] if cl[i] > 0 else 9.0
    c35i = c35s[i]
    dev35 = (cl[i] / c35i - 1) * 100 if (np.isfinite(c35i) and c35i > 0) else 0.0
    c10i, c90i = c10[i], c90[i]
    if np.isfinite(c10i) and np.isfinite(c90i) and c90i > c10i and cl[i] > 0:
        bw = (c90i - c10i) / cl[i]
        profit = float(np.clip((cl[i] - c10i) / (c90i - c10i), 0.0, 1.0))
    else:
        bw, profit = 0.5, 0.5
    pi = gt_p[i]
    gap = (cl[i] / pi - 1) * 100 if (np.isfinite(pi) and pi > 0) else 15.0
    return dict(close=round(float(cl[i]), 2), vol_ratio=round(vr, 2),
                rec_depth=round(dmin * 100, 1), rec_days=int(rec_days),
                cool_days=int(cool), hist_touch=int(hist_touch),
                dist_high=round((1 - 1 / ratio) * 100, 1) if ratio >= 1 else round((ratio - 1) * 100, 1),
                dev35=round(dev35, 1), band=round(bw * 100, 1),
                profit=round(profit, 2), gap=round(gap, 1))


def scan_code(args):
    code, sym = args
    fp = os.path.join(CACHE, f'{code}.csv')
    if not os.path.exists(fp):
        return None, code
    try:
        df = pd.read_csv(fp, parse_dates=['date'])
        if len(df) < MIN_I + 130:
            return None, code
        cl, darr, gt_p, gt2, c35s, _, _cA = cs.compute(df)
        hi = df['high'].values.astype(float)
        lo = df['low'].values.astype(float)
        vol = df['volume'].values.astype(float)
        turn = df['turnover'].values.astype(float)
        c10 = cost_series(hi, lo, vol, turn, 0.10)
        c90 = cost_series(hi, lo, vol, turn, 0.90)
    except Exception:
        return None, code
    n = len(cl)
    lim = board_limit(code)
    rets = np.full(n, np.nan)
    rets[1:] = cl[1:] / cl[:-1] - 1.0
    rows = []
    last_v3 = last_v3b = last_early = -10 ** 9
    for i in range(MIN_I, n - 1):
        d = darr[i]
        if d > WIN_END:
            break
        in_win = WIN_START <= d <= WIN_END
        hit3 = hitb = hite = False
        if i - last_v3 >= V3_COOL:
            ok3, _ = cs.signal_at(cl, gt_p, gt2, c35s, darr, i)
            if ok3:
                last_v3 = i
                hit3 = True
        if i - last_v3b >= V3B_COOL:
            okb, _ = cs.v3b_signal_at(cl, gt_p, gt2, c35s, vol, i)
            if okb:
                last_v3b = i
                hitb = True
        if i - last_early >= V3B_COOL:
            oke, _ = cs.v3b_early_at(cl, gt_p, gt2, c35s, vol, i)
            if oke:
                last_early = i
                hite = True
        if not in_win or not (hit3 or hitb or hite):
            continue
        pf = profile_light(cl, hi, lo, vol, turn, gt_p, gt2, c35s, c10, c90, i)
        f60mx, f60mn, f60d = _fut(cl, i, 60)
        f120mx, _, f120d = _fut(cl, i, 120)
        win = slice(i - 60, i)
        m = np.isfinite(gt_p[win])
        abr = round(np.sum(cl[win][m] > gt_p[win][m]) * 100 / m.sum(), 1) if m.sum() >= 42 else np.nan
        dp = round((gt2[i] / cl[i] - 1) * 100, 2) if (np.isfinite(gt2[i]) and gt2[i] > 0) else np.nan
        qf = 0
        for k in range(i + 1, min(i + 61, n)):
            if np.isfinite(rets[k]) and abs(rets[k]) * 100 > lim:
                qf = 1
                break
        rows.append(dict(code=code, sym=sym, sig_date=d,
                         hit_v3=int(hit3), hit_v3b=int(hitb), hit_early=int(hite),
                         sig_close=pf['close'], vol_ratio=pf['vol_ratio'],
                         rec_depth=pf['rec_depth'], rec_days=pf['rec_days'],
                         cool_days=pf['cool_days'], hist_touch=pf['hist_touch'],
                         dist_high=pf['dist_high'], dev35=pf['dev35'], band=pf['band'],
                         profit=pf['profit'], gap=pf['gap'], dist_purple=dp, abr=abr,
                         fut60_max=f60mx, fut60_min=f60mn, fut120_max=f120mx,
                         fut60_days=f60d, fut120_days=f120d,
                         s60=bool(np.isfinite(f60mx) and f60mx > 10),
                         s120=bool(np.isfinite(f120mx) and f120mx > 50),
                         fail=bool(np.isfinite(f60mx) and f60mx <= 0), qfq_flag=qf))
    return rows, code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nproc', type=int, default=2)
    ap.add_argument('--nshards', type=int, default=4)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--verify', action='store_true', help='抽查 profile_light 与 signal_profile 等价')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    t0 = time.time()
    all_df = tq.load_stock_list()
    all_df['code'] = all_df['code'].astype(str).str.zfill(6)
    sym_map = dict(zip(all_df['code'], all_df['name'].fillna('')))
    codes = sorted(sym_map)[args.shard::args.nshards]
    out_fp = os.path.join(REPO, 'result', f'anatomy_shard{args.shard}.csv')
    done = set()
    if args.resume and os.path.exists(out_fp):
        done = set(pd.read_csv(out_fp, dtype={'code': str}, usecols=['code'])['code'].str.zfill(6))
    print(f'[shard {args.shard}/{args.nshards}] codes={len(codes)} (skip {len(done)}) nproc={args.nproc}', flush=True)
    tasks = [(c, sym_map.get(c, '')) for c in codes if c not in done]
    rows, misses = [], []
    with Pool(args.nproc) as pool:
        for k, (res, code) in enumerate(pool.imap_unordered(scan_code, tasks, chunksize=4)):
            if res is None:
                misses.append(code)
            else:
                rows.extend(res)
            if (k + 1) % 150 == 0:
                print(f'  [{args.shard}] {k+1}/{len(tasks)} rows={len(rows)} 耗时{time.time()-t0:.0f}s', flush=True)
    out = pd.DataFrame(rows, columns=COLS)
    old = pd.read_csv(out_fp, dtype={'code': str}) if (args.resume and os.path.exists(out_fp)) else None
    out = pd.concat([old, out], ignore_index=True) if old is not None and len(old) else out
    out['code'] = out['code'].str.zfill(6)
    out = out.drop_duplicates(subset=['code', 'sig_date']).sort_values(['code', 'sig_date'])
    out.to_csv(out_fp, index=False)
    print(f'[shard {args.shard}] done n={len(out)} 缺缓存={len(misses)} 耗时{time.time()-t0:.0f}s -> {out_fp}', flush=True)


if __name__ == '__main__':
    main()
