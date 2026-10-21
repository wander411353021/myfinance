# -*- coding: utf-8 -*-
"""v3b / v3b_early 全市场最近5年回溯验证 (2026-10-17 polo4111 体系, 本脚本为批量回溯器)

信号口径全部复用 challenge_scan.py 权威实现:
  v3b_signal_at / v3b_early_at / signal_profile / compute (V10 线: 粉线gt_p/紫线gt2/COST35 Ehlers45)
  冷却去重: 同类型信号间 >= V3B_COOL(20) 天 (v3b 与 early 各自独立冷却链)

回溯窗口: 信号日 ∈ [20211001, 20261016]; 数据 end=20261016 datalen=1800(reg250预热+5年+后续120天+缓冲)
产物:     result/backtest_v3b_shard{part}.csv  每信号一行, 增量可续跑
  code/sym/sig_date/sig_type/sig_close
  + 画像(仅用<=信号日数据): vol_ratio/rec_depth/rec_days/cool_days/hist_touch/dist_high/dev35/band/profit/gap
  + 后续收益(>信号日, 事后): fut60_max/fut60_min/fut120_max/fut60_days/fut120_days
  + 成功标记: s60(后60日max>10% 主口径)/s120(后120日max>50%)/fail(后60日max<=0)
  + 事后质量字段: big20(后20日是否出大阳线>=5%)/big_wait(首根大阳等待天数, 无则-1) — bigline 已被 reasonix 推翻, 仅作事后标注
无未来函数红线: 信号判定与画像只用 <=信号日 数据; 后续收益与 bigline 为事后验证字段(与画像严格分离)。

用法:
  python3 backtest_v3b_5y.py --part N --nparts M     # 全市场分片并行(N=0..M-1)
  python3 backtest_v3b_5y.py --verify CODE DATE [datalen]   # 单只单日信号复验(含画像+后续收益)
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
try:
    import fengwo
    fengwo.showMsg(False)
except Exception:
    pass
import tdx_quant as tq
from challenge_scan import (load, compute, compute_eff, v3b_signal_at, v3b_early_at,
                            signal_profile, scan_series, COOLDOWN_MODE,
                            V3B_COOL, MIN_I, BIG_PCT, BIG_PCT_MAX)

END = '20261016'
DATALEN = 1800
WIN_START = '20211001'
WIN_END = '20261016'
CACHE = os.path.join('result', 'scan_cache_v3b')
os.makedirs(CACHE, exist_ok=True)

COLS = ['code', 'sym', 'sig_date', 'sig_type', 'sig_close',
        'vol_ratio', 'rec_depth', 'rec_days', 'cool_days', 'hist_touch',
        'dist_high', 'dev35', 'band', 'profit', 'gap',
        'fut60_max', 'fut60_min', 'fut120_max', 'fut60_days', 'fut120_days',
        's60', 's120', 'fail', 'big20', 'big_wait']


def get_df(code):
    """拉取/缓存日线(with_turnover=True 含逐日历史流通股本换手率)。"""
    fp = os.path.join(CACHE, f'{code}.csv')
    if os.path.exists(fp):
        try:
            df = pd.read_csv(fp, parse_dates=['date'])
            if len(df) and str(df['date'].iloc[-1])[:10].replace('-', '') >= END and len(df) >= 400:
                return df
        except Exception:
            pass
    df = load(code, END, DATALEN)
    if df is not None and len(df):
        df.to_csv(fp, index=False)
    return df


def _fut(cl, i, k):
    """后续k日最大/最小涨幅(相对信号日收盘), 只取可用数据。返回 (max,min,days)"""
    n = len(cl)
    e = min(i + k, n - 1)
    if i + 1 > e:
        return np.nan, np.nan, 0
    seg = cl[i + 1:e + 1]
    return (round((seg.max() / cl[i] - 1) * 100, 2),
            round((seg.min() / cl[i] - 1) * 100, 2), e - i)


def _bigline_post(cl, i):
    """事后质量字段: 后20日内首根大阳线(单日涨幅∈[5%,21%])。返回 (big20, wait)"""
    n = len(cl)
    for k in range(i + 1, min(i + 21, n)):
        g = cl[k] / cl[k - 1] - 1.0
        if np.isfinite(g) and BIG_PCT <= g <= BIG_PCT_MAX:
            return 1, k - i
    return 0, -1


def make_row(code, sym, date, stype, cl, hi, lo, vol, turn, gt_p, gt2, c35s, i,
             gt_p_eff=None, gt2_eff=None):
    pf = signal_profile(cl, hi, lo, vol, turn, gt_p, gt2, c35s, i,
                        gt_p_eff=gt_p_eff, gt2_eff=gt2_eff)
    f60mx, f60mn, f60d = _fut(cl, i, 60)
    f120mx, _, f120d = _fut(cl, i, 120)
    big20, big_wait = _bigline_post(cl, i)
    return dict(code=code, sym=sym, sig_date=date, sig_type=stype, sig_close=pf['close'],
                vol_ratio=pf['vol_ratio'], rec_depth=pf['rec_depth'], rec_days=pf['rec_days'],
                cool_days=pf['cool_days'], hist_touch=pf['hist_touch'], dist_high=pf['dist_high'],
                dev35=pf['dev35'], band=pf['band'], profit=pf['profit'], gap=pf['gap'],
                fut60_max=f60mx, fut60_min=f60mn, fut120_max=f120mx, fut60_days=f60d, fut120_days=f120d,
                s60=bool(np.isfinite(f60mx) and f60mx > 10),
                s120=bool(np.isfinite(f120mx) and f120mx > 50),
                fail=bool(np.isfinite(f60mx) and f60mx <= 0),
                big20=big20, big_wait=big_wait)


def process(code, sym, mode=None):
    """单只股票全信号扫描（P0-1: 结构冷却由 scan_series 统一）。返回 (rows, status)"""
    df = get_df(code)
    if df is None or len(df) < MIN_I + 2:
        return None, 'short'
    cl, darr, gt_p, gt2, gt_p_eff, gt2_eff, c35s, _, cA = compute_eff(df)
    vol = df['volume'].values.astype(float)
    hi = df['high'].values.astype(float)
    lo = df['low'].values.astype(float)
    turn = df['turnover'].values.astype(float)
    n = len(cl)
    rows = []
    # 冷却链从 MIN_I 全历史推进；窗口只影响输出（窗口外信号不再单独模拟）
    sig = scan_series(cl, gt_p, gt2, c35s, vol, start_i=MIN_I, end_i=n - 1,
                      types=('v3b', 'early'), mode=mode,
                      gt_p_eff=gt_p_eff, gt2_eff=gt2_eff)
    for stype, i in sig:
        d = darr[i]
        if d < WIN_START:
            continue
        if d > WIN_END:
            break
        rows.append(make_row(code, sym, d, stype, cl, hi, lo, vol, turn,
                             gt_p, gt2, c35s, i,
                             gt_p_eff=gt_p_eff, gt2_eff=gt2_eff))
    return rows, 'ok'


def log(msg, part):
    with open(os.path.join('result', f'backtest_v3b_shard{part}.log'), 'a') as f:
        f.write(time.strftime('%H:%M:%S ') + msg + '\n')
    print(msg, flush=True)


def verify(code, date, datalen):
    """复验单日信号(与批量同代码路径)。"""
    df = load(code, END, datalen)
    if df is None or len(df) < MIN_I + 5:
        print(f'{code}: 数据不足'); return
    cl, darr, gt_p, gt2, gt_p_eff, gt2_eff, c35s, _, cA = compute_eff(df)
    vol = df['volume'].values.astype(float)
    hi = df['high'].values.astype(float)
    lo = df['low'].values.astype(float)
    turn = df['turnover'].values.astype(float)
    idx = next((k for k, d in enumerate(darr) if d == date), None)
    if idx is None:
        print(f'{code} 在 {END} 前无 {date} 数据'); return
    ok_v, _ = v3b_signal_at(cl, gt_p, gt2, c35s, vol, idx,
                            gt_p_eff=gt_p_eff, gt2_eff=gt2_eff)
    ok_e, _ = v3b_early_at(cl, gt_p, gt2, c35s, vol, idx,
                           gt_p_eff=gt_p_eff, gt2_eff=gt2_eff)
    print(f'== {code} {date} idx={idx} v3b={ok_v} early={ok_e} ==')
    pf = signal_profile(cl, hi, lo, vol, turn, gt_p, gt2, c35s, idx,
                        gt_p_eff=gt_p_eff, gt2_eff=gt2_eff)
    for k, v in pf.items():
        print(f'   {k} = {v}')
    f60mx, f60mn, f60d = _fut(cl, idx, 60)
    f120mx, _, f120d = _fut(cl, idx, 120)
    big20, big_wait = _bigline_post(cl, idx)
    print(f'   fut60_max={f60mx} fut60_min={f60mn} ({f60d}d) | fut120_max={f120mx} ({f120d}d)')
    print(f'   s60={np.isfinite(f60mx) and f60mx > 10} s120={np.isfinite(f120mx) and f120mx > 50} '
          f'fail={np.isfinite(f60mx) and f60mx <= 0} | big20={big20} wait={big_wait}')


def run_shard(part, nparts, limit=None):
    t0 = time.time()
    all_df = tq.load_stock_list()
    all_df['code'] = all_df['code'].astype(str).str.zfill(6)
    codes = all_df['code'].tolist()
    sym_map = dict(zip(all_df['code'], all_df['name'].fillna('')))
    shard = codes[part::nparts]
    if limit:
        shard = shard[:limit]
    out = os.path.join('result', f'backtest_v3b_shard{part}.csv')
    done = set()
    if os.path.exists(out):
        try:
            done = set(pd.read_csv(out, usecols=['code'], dtype=str)['code'])
        except Exception:
            done = set()
    rows = []
    n_short = n_fail = 0
    for k, code in enumerate(shard):
        if code in done:
            continue
        try:
            r, st = process(code, sym_map.get(code, ''))
            if st == 'short':
                n_short += 1
            elif r:
                rows.extend(r)
        except Exception as e:
            n_fail += 1
            if n_fail <= 10:
                log(f'ERR {code}: {e}', part)
        if (k + 1) % 50 == 0:
            cur = pd.DataFrame(rows, columns=COLS) if rows else pd.DataFrame(columns=COLS)
            if os.path.exists(out) and len(done):
                old = pd.read_csv(out, dtype=str)
                cur = pd.concat([old, cur], ignore_index=True)
            cur.to_csv(out, index=False)
            done.update(shard[k - 49:k + 1])
            rows = []
            log(f'[p{part}] {k + 1}/{len(shard)} 信号累计{len(cur)} 失败{n_fail} 短数据{n_short} 耗时{time.time() - t0:.0f}s', part)
    if rows or not os.path.exists(out):
        cur = pd.DataFrame(rows, columns=COLS) if rows else pd.DataFrame(columns=COLS)
        if os.path.exists(out):
            old = pd.read_csv(out, dtype=str)
            cur = pd.concat([old, cur], ignore_index=True)
        cur.to_csv(out, index=False)
    log(f'[p{part}] done 股票{len(shard)} 短数据{n_short} 失败{n_fail} 信号{len(pd.read_csv(out))} 总耗时{time.time() - t0:.0f}s -> {out}', part)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--verify', nargs='*', default=None)
    args = ap.parse_args()
    if args.verify:
        code = args.verify[0]
        date = args.verify[1] if len(args.verify) > 1 else None
        dl = int(args.verify[2]) if len(args.verify) > 2 else 1800
        if date:
            verify(code, date, dl)
        else:
            print('--verify 需要 CODE DATE')
    else:
        run_shard(args.part, args.nparts, args.limit)
