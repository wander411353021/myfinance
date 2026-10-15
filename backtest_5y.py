# -*- coding: utf-8 -*-
"""early 模式最近五年全市场因果回测(2026-10-15 豆包)
增量滚动回归(O(n)) + 逐日因果判定, 与 scan_pressure_breakout.detect 同口径。
"""
import sys, os, time
sys.path.insert(0, '/home/user/.super_doubao/super-doubao-runtime/workspace/pressure-level-algorithm')
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import tdx_quant as tq
import panic_reversal as pr

END = '20261014'; DATALEN = 1650   # ~6年交易日(覆盖完整2021-2026) + reg250预热 + 压制期
CACHE = os.path.join('result', 'scan_cache_5y')
os.makedirs(CACHE, exist_ok=True)

LOOKBACK = 150; PRESS_VALID_MIN = 0.70; PRESS_ABOVE_MAX = 15; PRESS_GAP_MIN = 10.0
EARLY_DEV_MAX = -0.10; EARLY_GAP_MIN = 10.0; EARLY_GAP_MAX = 25.0
MIN_I = 400   # reg250(250) + 压制期(150) 预热

def _bd(d):
    t = pd.Timestamp(d)
    if t.tzinfo is None and '16:00' in str(d):
        t = t + pd.Timedelta(hours=8)
    return str(t)[:10].replace('-', '')

def rolling_reg_fast(closes, window):
    """O(n) 增量滚动对数回归(与 compute_rolling_regression 数学等价)"""
    n = len(closes); y = np.log(np.maximum(closes, 1e-8)).astype(float)
    preds = np.full(n, np.nan); slopes = np.full(n, np.nan)
    t = np.arange(window, dtype=float)
    sum_t = t.sum(); sum_tt = (t*t).sum(); denom = window*sum_tt - sum_t*sum_t
    sum_y = y[:window].sum(); sum_ty = (t*y[:window]).sum()
    for i in range(window-1, n):
        a = (window*sum_ty - sum_t*sum_y)/denom
        b = (sum_y - a*sum_t)/window
        preds[i] = np.exp(a*(window-1)+b)
        slopes[i] = a
        if i < n-1:
            yo = y[i-window+1]; yn = y[i+1]
            sum_ty_new = sum_ty - sum_y + yo + (window-1)*yn   # t权重整体左移1
            sum_y = sum_y + yn - yo
            sum_ty = sum_ty_new
    return preds, slopes

def scan_one(code):
    fp = os.path.join(CACHE, f'{code}.csv')
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        if str(df['date'].iloc[-1])[:10].replace('-','') >= END and len(df) >= 1500:
            return df
    df = tq.get_daily_kline_from_tdx(code, END, DATALEN, with_turnover=False)
    if df is not None and len(df):
        df.to_csv(fp, index=False)
    return df

def process(code):
    df = scan_one(code)
    if df is None or len(df) < MIN_I + 10:
        return []
    closes = df['close'].values.astype(float)
    dates = df['date'].values
    n = len(closes)
    reg120, _ = rolling_reg_fast(closes, 120)
    reg250, _ = rolling_reg_fast(closes, 250)
    r120s = pr.double_smooth_reg(reg120, 5, 5)
    r250s = pr.double_smooth_reg(reg250, 5, 5)
    base = np.maximum(r120s, r250s)
    gt_p, gl_p = pr.compute_grid_target_price(closes, r120s, r250s)
    dev = closes / base - 1
    ma20 = pd.Series(closes).rolling(20).mean().values
    darr = [_bd(d) for d in dates]
    out = []
    for i in range(MIN_I, n):
        win = slice(i - LOOKBACK, i + 1)          # 150天含当天
        m = np.isfinite(gt_p[win])
        if m.sum() < LOOKBACK * PRESS_VALID_MIN: continue
        c = closes[win]; gt = gt_p[win]
        above = np.sum(c[m] > gt[m]); abr = above*100/m.sum()
        if abr > PRESS_ABOVE_MAX: continue
        gap = np.median((gt[m]/c[m]-1)*100)
        if gap < PRESS_GAP_MIN or gap > EARLY_GAP_MAX: continue
        if not np.isfinite(dev[i]) or dev[i] >= EARLY_DEV_MAX: continue
        if not (np.isfinite(ma20[i]) and np.isfinite(ma20[i-1])): continue
        if not (closes[i] > ma20[i] and closes[i-1] <= ma20[i-1]): continue
        sw = int(np.sum(np.diff(gl_p[win][gl_p[win] >= 0]) != 0))
        f20 = (closes[i+1: i+21].max()/closes[i]-1)*100 if i+1 < n else np.nan
        f60 = (closes[i+1: i+61].max()/closes[i]-1)*100 if i+1 < n else np.nan
        out.append((code, darr[i], round(float(closes[i]), 3), round(gap, 1),
                    round(abr, 1), round(float(dev[i]*100), 1), sw,
                    round(float(f20), 1), round(float(f60), 1)))
    return out

if __name__ == '__main__':
    part = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nparts = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    all_df = tq.load_stock_list()
    codes = all_df['code'].astype(str).str.zfill(6).tolist()
    codes = codes[part::nparts]
    t0 = time.time(); rows = []
    for k, code in enumerate(codes):
        try:
            rows += process(code)
        except Exception:
            pass
        if (k+1) % 300 == 0:
            print(f'[part{part}] {k+1}/{len(codes)} 信号{len(rows)} 耗时{time.time()-t0:.0f}s', flush=True)
    out = os.path.join('result', f'early_5y_signals_part{part}.csv')
    pd.DataFrame(rows, columns=['code','sig_date','sig_close','gap','abr','dev','switch','fut_max20','fut_max60']).to_csv(out, index=False)
    print(f'[part{part}] done {len(rows)} -> {out}', flush=True)
