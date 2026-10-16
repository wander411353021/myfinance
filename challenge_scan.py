# -*- coding: utf-8 -*-
"""挑战信号(压制突破) v3 定版扫描器 — 2026-10-15 polo4111 定版
信号定义(到达紫线才成立, 区别于 early 启动前低位):
  1. 压制期: 前60天(不含当日) 粉线有效>=70%, 价格>粉线占比<=10%(90%低于), gap∈[3%,20%)
  2. 已突破: close > 粉线(Grid Target第一阶梯) 且 close > COST35(Ehlers45平滑)
  3. 成立:   close >= 紫线(第二阶梯)*0.98 —— 价格真正到达紫色压制线(滤掉摸不到紫线的假突破)
用法:
  python3 challenge_scan.py scan   <part> <nparts> [end]   # 全市场每日候选池(分片并行)
  python3 challenge_scan.py hist   <code> [end] [datalen] # 单只历史全信号
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import tdx_quant as tq
import panic_reversal as pr
from chip_panel import cost_series

LOOKBACK = 60; PRESS_VALID_MIN = 0.70; PRESS_ABOVE_MAX = 10
PRESS_GAP_MIN = 3.0; PRESS_GAP_MAX = 20.0
TOUCH = 0.98          # 到达紫线阈值(close >= 紫线*0.98)
MIN_I = 400           # reg250(250)+压制期(60)+缓冲 预热
COOL = 30             # 历史扫描冷却(天)

def ehlers_smooth(x, period):
    """Ehlers SuperSmoother 二阶IIR低通(因果, 第k天只用<=k数据)"""
    x = np.asarray(x, dtype=float); n = len(x); out = np.full(n, np.nan)
    if n < 2: return out
    a1 = np.exp(-1.414 * np.pi / period); b1 = 2 * a1 * np.cos(1.414 * np.pi / period)
    c3 = -a1 * a1; c2 = b1; c1 = 1.0 - c2 - c3
    out[0] = x[0]; out[1] = c1 * (x[1] + x[0]) / 2 + (c2 + c3) * x[0]
    for k in range(2, n):
        out[k] = c1 * (x[k] + x[k - 1]) / 2 + c2 * out[k - 1] + c3 * out[k - 2]
    return out

def second_stair(base, gt_p, gl_p):
    """第二阶梯(紫线): 粉线档位21天众数平滑 + 2档 + confirm=40慢速 + 封底>=粉线
    (复刻 price_segmenter_v10 682-741 行, 与 V10 图上深紫线一致)"""
    n = len(base); glr = np.asarray(gl_p, dtype=float)
    gls = np.full(n, -1.0); W = 21
    for ti in range(n):
        w = glr[max(0, ti - W + 1):ti + 1]; w = w[w >= 0]
        if len(w):
            vals, cnts = np.unique(w, return_counts=True)
            cand = vals[cnts == cnts.max()]; gls[ti] = float(cand.max())
    levs1 = (-0.09, -0.06, -0.03, 0.00, 0.03, 0.06, 0.09, 0.12)
    out = np.full(n, np.nan); cur = None; pend = None; cnt = 0
    for ti in range(n):
        if glr[ti] < 0 or not np.isfinite(base[ti]):
            out[ti] = np.nan; cur = None; pend = None; cnt = 0; continue
        tk = min(int(gls[ti]) + 2, len(levs1) - 1)
        if cur is None: cur = tk
        elif tk == cur: pend = None; cnt = 0
        else:
            if tk == pend: cnt += 1
            else: pend = tk; cnt = 1
            if cnt >= 40: cur = tk; pend = None; cnt = 0
        v2 = base[ti] * (1 + levs1[cur])
        if np.isfinite(gt_p[ti]) and v2 < gt_p[ti]: v2 = gt_p[ti]
        out[ti] = v2
    return out

def _bd(d):
    t = pd.Timestamp(d)
    if t.tzinfo is None and '16:00' in str(d): t = t + pd.Timedelta(hours=8)
    return str(t)[:10].replace('-', '')

def load(code, end, datalen):
    return tq.get_daily_kline_from_tdx(code, end, datalen, with_turnover=True)

def compute(df):
    """全量计算 V10 线: 返回 (closes, dates_str, gt_p, gt2, c35s, base)"""
    df = df.reset_index(drop=True)
    cl = df['close'].values.astype(float); hi = df['high'].values.astype(float)
    lo = df['low'].values.astype(float); vo = df['volume'].values.astype(float)
    turn = df['turnover'].values.astype(float)
    n = len(cl)
    reg120, _ = pr.compute_rolling_regression(cl, 120)
    reg250, _ = pr.compute_rolling_regression(cl, 250)
    r120s = pr.double_smooth_reg(reg120, 5, 5); r250s = pr.double_smooth_reg(reg250, 5, 5)
    base = np.maximum(r120s, r250s)
    gt_p, gl_p = pr.compute_grid_target_price(cl, r120s, r250s,
        levels=(-0.09, -0.06, -0.03, 0.00, 0.03, 0.06, 0.09, 0.12),
        max_dev=0.13, down_confirm=10)
    gt2 = second_stair(base, gt_p, gl_p)
    try:
        c35 = cost_series(hi, lo, vo, turn, 0.35); c35s = ehlers_smooth(c35, 45)
    except Exception:
        c35s = np.full(n, np.nan)
    return cl, [_bd(d) for d in df['date'].values], gt_p, gt2, c35s, base

def signal_at(cl, gt_p, gt2, c35s, darr, i):
    """判定第 i 天是否成立 v3 挑战信号(压制期不含当日)。返回 (ok, 特征dict)"""
    if i < MIN_I: return False, None
    win = slice(i - LOOKBACK, i)
    m = np.isfinite(gt_p[win])
    if m.sum() < LOOKBACK * PRESS_VALID_MIN: return False, None
    c = cl[win]; gt = gt_p[win]
    abr = np.sum(c[m] > gt[m]) * 100 / m.sum()
    if abr > PRESS_ABOVE_MAX: return False, None
    gap = np.median((gt[m] / c[m] - 1) * 100)
    if gap < PRESS_GAP_MIN or gap >= PRESS_GAP_MAX: return False, None
    if not (np.isfinite(gt_p[i]) and np.isfinite(c35s[i]) and np.isfinite(gt2[i])): return False, None
    if not (cl[i] > gt_p[i] and cl[i] > c35s[i]): return False, None
    if cl[i] < gt2[i] * TOUCH: return False, None
    feats = dict(gap=round(gap, 1), abr=round(abr, 1),
                 dev35=round((cl[i] / c35s[i] - 1) * 100, 1),
                 dist_purple=round((gt2[i] / cl[i] - 1) * 100, 1))
    return True, feats

# ─────────────────────────────────────────────────────────────
# v3b 二次买点信号(2026-10-16 polo4111 定版, 600550 模板)
#   模板: 600550 2020 特高压 — 2020-01-08 摸紫线(v3式突破) → 01-17~02-26 深回踩
#         (跌破紫线/粉线/COST35, 最深-20%) → 02-27 重新放量摸紫线 → 连板+130%
#   与 v3 区别: v3 只认"从未突破的干净压制"(abr<=10), 会漏掉"突破后深回踩再启动"的票;
#              v3b 补上该类二次买点(22池158信号, fut60>10% 59.7%, fut120>50% 22.5%)
# ─────────────────────────────────────────────────────────────
V3B_TOUCH = 0.98        # 摸紫线阈值(close >= 紫线*0.98)
V3B_HIST_LO = 250       # 历史摸线窗口 [T-250, T-20]
V3B_HIST_HI = 20
V3B_REC_LO = 70         # 回踩窗口 [T-70, T-5]
V3B_REC_HI = 5
V3B_REC_DIP = 0.98      # 回踩跌破 COST35*0.98
V3B_COOL = 20           # 与上次摸线冷却
V3B_GAP_MIN = 5.0; V3B_GAP_MAX = 20.0   # 二次突破 gap
V3B_VOL_MULT = 1.2      # 二次突破日量能 >= 1.2×前20日均量


def v3b_signal_at(cl, gt_p, gt2, c35s, vol, i):
    """判定第 i 天是否成立 v3b 二次买点(只用<=i数据, 无未来函数)。
    返回 (ok, 特征dict)"""
    if i < MIN_I: return False, None
    p2i = gt2[i]
    if not (np.isfinite(p2i) and p2i > 0): return False, None
    # 1) 二次突破: 当日摸到紫线
    if cl[i] < p2i * V3B_TOUCH: return False, None
    # 2) 历史: [T-250, T-20] 内曾摸过紫线(曾经 v3 式突破动作)
    lo, hi = max(MIN_I, i - V3B_HIST_LO), i - V3B_HIST_HI
    if lo >= hi: return False, None
    p2w = gt2[lo:hi]
    if not np.any(np.isfinite(p2w) & (p2w > 0) & (cl[lo:hi] >= p2w * V3B_TOUCH)):
        return False, None
    # 3) 回踩: [T-70, T-5] 内曾跌破 COST35(深度洗盘)
    rlo, rhi = max(0, i - V3B_REC_LO), i - V3B_REC_HI
    if rlo >= rhi: return False, None
    cw = c35s[rlo:rhi]
    if not np.any(np.isfinite(cw) & (cw > 0) & (cl[rlo:rhi] < cw * V3B_REC_DIP)):
        return False, None
    # 4) gap 过滤: 相对粉线 5%~20%
    pi = gt_p[i]
    if np.isfinite(pi) and pi > 0:
        gap = (cl[i] / pi - 1) * 100
        if not (V3B_GAP_MIN <= gap < V3B_GAP_MAX): return False, None
    else:
        gap = float('nan')
    # 5) 量能确认: 二次突破日放量 >= 1.2×前20日均量
    if vol is not None:
        ma20 = vol[max(0, i - 20):i].mean()
        if ma20 > 0 and vol[i] < ma20 * V3B_VOL_MULT: return False, None
    feats = dict(gap=round(gap, 1) if np.isfinite(gap) else None,
                 dist_purple=round((p2i / cl[i] - 1) * 100, 1))
    return True, feats


def scan_hist_v3b(code, end, datalen=1600, win_start=None, win_end=None):
    """v3b 历史全信号(冷却去重, 无未来函数)"""
    try:
        df = load(code, end, datalen)
    except Exception:
        return []
    if df is None or len(df) < MIN_I + 130: return []
    cl, darr, gt_p, gt2, c35s, _ = compute(df)
    vol = df['volume'].values.astype(float)
    n = len(cl); hits = []; last_sig = -10 ** 9
    for i in range(MIN_I, n - 125):
        if i - last_sig < V3B_COOL: continue
        if win_start and darr[i] < win_start: continue
        if win_end and darr[i] > win_end: break
        ok, feats = v3b_signal_at(cl, gt_p, gt2, c35s, vol, i)
        if not ok: continue
        fut = lambda k: round((cl[i + 1:i + k + 1].max() / cl[i] - 1) * 100, 1) if i + k < n else None
        hits.append((code, darr[i], round(float(cl[i]), 2), feats['gap'], feats['dist_purple'],
                     fut(20), fut(60), fut(120)))
        last_sig = i
    return hits


def scan_daily(code, end, datalen=1000):
    """每日候选池: 只检测最新交易日"""
    try:
        df = load(code, end, datalen)
    except Exception:
        return None
    if df is None or len(df) < MIN_I + 5: return None
    cl, darr, gt_p, gt2, c35s, _ = compute(df)
    i = len(cl) - 1
    ok, feats = signal_at(cl, gt_p, gt2, c35s, darr, i)
    if not ok: return None
    return (code, darr[i], round(float(cl[i]), 2), feats['gap'], feats['abr'],
            feats['dev35'], feats['dist_purple'])

def scan_hist(code, end, datalen=1600, win_start=None, win_end=None):
    """历史全信号(冷却去重)"""
    try:
        df = load(code, end, datalen)
    except Exception:
        return []
    if df is None or len(df) < MIN_I + 130: return []
    cl, darr, gt_p, gt2, c35s, _ = compute(df)
    n = len(cl); hits = []; last_sig = -10 ** 9
    for i in range(MIN_I, n - 125):
        if i - last_sig < COOL: continue
        if win_start and darr[i] < win_start: continue
        if win_end and darr[i] > win_end: break
        ok, feats = signal_at(cl, gt_p, gt2, c35s, darr, i)
        if not ok: continue
        fut = lambda k: round((cl[i + 1:i + k + 1].max() / cl[i] - 1) * 100, 1) if i + k < n else None
        hits.append((code, darr[i], round(float(cl[i]), 2), feats['gap'], feats['abr'],
                     feats['dev35'], feats['dist_purple'], fut(20), fut(60), fut(120)))
        last_sig = i
    return hits

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'scan'
    if mode == 'scan':
        part = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        nparts = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        END = sys.argv[4] if len(sys.argv) > 4 else '20261015'
        all_df = tq.load_stock_list()
        codes = all_df['code'].astype(str).str.zfill(6).tolist()
        codes = codes[part::nparts]
        t0 = time.time(); rows = []
        for k, code in enumerate(codes):
            try:
                r = scan_daily(code, END)
                if r: rows.append(r)
            except Exception:
                pass
            if (k + 1) % 200 == 0:
                print(f'[part{part}] {k + 1}/{len(codes)} 候选{len(rows)} 耗时{time.time() - t0:.0f}s', flush=True)
        out = os.path.join('result', f'challenge_cand_part{part}.csv')
        pd.DataFrame(rows, columns=['code', 'sig_date', 'sig_close', 'gap', 'abr', 'dev35', 'dist_purple']).to_csv(out, index=False)
        print(f'[part{part}] done {len(rows)} -> {out}', flush=True)
    elif mode == 'hist_v3b':
        code = sys.argv[2]
        end = sys.argv[3] if len(sys.argv) > 3 else '20261015'
        dl = int(sys.argv[4]) if len(sys.argv) > 4 else 1600
        hits = scan_hist_v3b(code, end, dl)
        print(f'{code} 二次买点信号(v3b): {len(hits)} 个')
        for h in hits: print('  ', h)
    else:
        code = sys.argv[2]
        end = sys.argv[3] if len(sys.argv) > 3 else '20261015'
        dl = int(sys.argv[4]) if len(sys.argv) > 4 else 1600
        hits = scan_hist(code, end, dl)
        print(f'{code} 挑战信号(v3): {len(hits)} 个')
        for h in hits: print('  ', h)
