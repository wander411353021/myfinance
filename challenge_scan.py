# -*- coding: utf-8 -*-
"""挑战信号(压制突破) v3 定版扫描器 — 2026-10-15 polo4111 定版
信号定义(到达紫线才成立, 区别于 early 启动前低位):
  1. 压制期: 前60天(不含当日) 粉线有效>=70%, 价格>粉线占比<=10%(90%低于), gap∈[3%,20%)
  2. 已突破: close > 粉线(Grid Target第一阶梯) 且 close > 信号成本线 SIGNAL_COST_PCT
     （默认 SIGNAL_COST_PCT=0.35 / SIGNAL_COST_EHLERS=60；A/B 只用于显示）
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
from algo_config import (COST_LABEL, CHIP_BAND_LO, CHIP_BAND_HI,
                         CHIP_WARMUP_DAYS, CHIP_WARMUP_TURNOVER,
                         COST_PCT, COST_EHLERS,
                         COST2_PCT, COST2_EHLERS,
                         SIGNAL_COST_PCT, SIGNAL_COST_EHLERS,
                         SIGNAL_MIN_LEVEL, SIGNAL_REARM_DAYS)

LOOKBACK = 60; PRESS_VALID_MIN = 0.70; PRESS_ABOVE_MAX = 10
PRESS_GAP_MIN = 3.0; PRESS_GAP_MAX = 20.0
TOUCH = 0.98          # 到达紫线阈值(close >= 紫线*0.98)
MIN_I = 400           # reg250(250)+压制期(60)+缓冲 预热
COOL = 30             # 历史扫描冷却(天)
# 信号成本线标准统一定义在 algo_config.py（SIGNAL_COST_*，2026-10-20 全套新标准）

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


def chip_warmup_window(df, display_days, max_days=None, min_turnover=None):
    """按“总窗口 1800 天 / 显示前累计换手 400%”确定计算起点（只提示，不阻断）。

    规则（2026-10-23 用户定版）：
      - 显示区间为最后 display_days 个交易日；
      - 从显示窗口起点往前追溯，累计 turnover（只算显示起点之前）达到 400%
        或总窗口达到 1800 天，二者谁先到就停；
      - 返回的 start 为计算初始日索引，total_days = 初始日到截至日（含显示区间）。

    不做未来函数：只使用 <= 截至日的数据，且 start 只由历史换手率决定。
    """
    _max = CHIP_WARMUP_DAYS if max_days is None else int(max_days)
    _req = CHIP_WARMUP_TURNOVER if min_turnover is None else float(min_turnover)
    out = dict(ok=False, start=0, total_days=0, display_days=0,
               warmup_days=0, warmup_turnover=0.0,
               required_days=_max, required_turnover=_req)
    if df is None or len(df) == 0:
        return out
    n = len(df)
    disp = min(max(0, int(display_days)), n)
    disp_start = max(0, n - disp)
    out.update(display_days=disp, total_days=n)
    if 'turnover' not in df.columns:
        return out
    t = pd.to_numeric(df['turnover'], errors='coerce').fillna(0).clip(0, 1).values
    i, cum = disp_start, 0.0
    while i > 0 and (n - i) < _max:
        i -= 1
        cum += t[i]
        if cum >= _req:
            break
    total = n - i
    warm = disp_start - i
    out.update(start=i, total_days=total, warmup_days=warm,
               warmup_turnover=cum, ok=(cum >= _req or total >= _max))
    if 'date' in df.columns:
        try:
            out['start_date'] = str(df['date'].iloc[i])[:10]
        except Exception:
            pass
    return out

def compute(df, cost_pct=None, ehlers_period=None):
    """全量计算 V10 原始线: 返回 (closes, dates_str, gt_p, gt2, c35s, base)。

    2026-10-20 全套新信号标准：
      c35s 与 gt_p/gt2 都用 SIGNAL_COST_PCT / SIGNAL_COST_EHLERS（COST35/E60），
      并启用动态 min_level；A/B 成本线只用于显示。
    cost_pct / ehlers_period 仅作实验覆盖（None = SIGNAL_* 默认）。
    """
    df = df.reset_index(drop=True)
    cl = df['close'].values.astype(float); hi = df['high'].values.astype(float)
    lo = df['low'].values.astype(float); vo = df['volume'].values.astype(float)
    turn = df['turnover'].values.astype(float)
    n = len(cl)
    reg120, _ = pr.compute_rolling_regression(cl, 120)
    reg250, _ = pr.compute_rolling_regression(cl, 250)
    r120s = pr.double_smooth_reg(reg120, 5, 5); r250s = pr.double_smooth_reg(reg250, 5, 5)
    base_reg = np.maximum(r120s, r250s)
    try:
        _cost_pct = SIGNAL_COST_PCT if cost_pct is None else float(cost_pct)
        _ehlers = SIGNAL_COST_EHLERS if ehlers_period is None else int(ehlers_period)
        c35 = cost_series(hi, lo, vo, turn, _cost_pct)
        c35s = ehlers_smooth(c35, max(2, _ehlers))
    except Exception:
        c35s = np.full(n, np.nan)
    # 新标准：成本线可作基准时优先用它；否则回退 REG，避免短数据/缺换手率直接失效。
    base_sig = c35s if np.isfinite(c35s).sum() >= max(20, n // 10) else base_reg
    gt_p, gl_p = pr.compute_grid_target_price(
        cl, base_sig, base_sig,
        levels=(-0.09, -0.06, -0.03, 0.00, 0.03, 0.06, 0.09, 0.12),
        max_dev=None, down_confirm=10,
        min_level=SIGNAL_MIN_LEVEL)
    gt2 = second_stair(base_sig, gt_p, gl_p)
    try:
        cA = ehlers_smooth(cost_series(hi, lo, vo, turn, COST_PCT), max(2, COST_EHLERS))
    except Exception:
        cA = np.full(n, np.nan)
    return cl, [_bd(d) for d in df['date'].values], gt_p, gt2, c35s, base_sig, np.asarray(cA, float)

def compute_eff(df, max_dev=0.13, cost_pct=None, ehlers_period=None):
    """compute + 置空备用线（2026-10-18 P0-2）。

    返回 (cl, darr, gt_p_raw, gt2_raw, gt_p_eff, gt2_eff, c35s, base, cA)。
    cost_pct / ehlers_period 透传给 compute（None = 全局 algo_config）。
    """
    cl, darr, gt_p, gt2, c35s, base, cA = compute(df, cost_pct=cost_pct,
                                                  ehlers_period=ehlers_period)
    gt_p_eff, gt2_eff = effective_lines(gt_p, gt2, base, max_dev=max_dev)
    return cl, darr, gt_p, gt2, gt_p_eff, gt2_eff, c35s, base, cA


def effective_lines(gt_p, gt2, base, max_dev=0.13):
    """gt_p/gt2 的 NaN 日用 base×(1+max_dev) 替换；返回 (p, p2)。"""
    if base is None:
        return gt_p, gt2
    base = np.asarray(base, dtype=float)
    gt_p = np.asarray(gt_p, dtype=float)
    gt2 = np.asarray(gt2, dtype=float)
    fb = np.where(np.isfinite(base) & (base > 0), base * (1.0 + max_dev), np.nan)
    p = np.where(np.isfinite(gt_p) & (gt_p > 0), gt_p, fb)
    p2 = np.where(np.isfinite(gt2) & (gt2 > 0), gt2, fb)
    return p, p2


def signal_at(cl, gt_p, gt2, c35s, darr, i, gt_p_eff=None, gt2_eff=None):
    """判定第 i 天是否成立 v3 挑战信号(压制期不含当日)。返回 (ok, 特征dict)

    P0-2: 历史压制期用原始 gt_p（置空日 NaN 不计入）；当日突破判定
    优先用 gt_p_eff/gt2_eff（置空日 = base×1.13），避免失明。
    """
    if i < MIN_I: return False, None
    win = slice(i - LOOKBACK, i)
    m = np.isfinite(gt_p[win])
    if m.sum() < LOOKBACK * PRESS_VALID_MIN: return False, None
    c = cl[win]; gt = gt_p[win]
    abr = np.sum(c[m] > gt[m]) * 100 / m.sum()
    if abr > PRESS_ABOVE_MAX: return False, None
    gap = np.median((gt[m] / c[m] - 1) * 100)
    if gap < PRESS_GAP_MIN or gap >= PRESS_GAP_MAX: return False, None
    p_i = gt_p_eff[i] if gt_p_eff is not None else gt_p[i]
    p2_i = gt2_eff[i] if gt2_eff is not None else gt2[i]
    if not (np.isfinite(p_i) and np.isfinite(c35s[i]) and np.isfinite(p2_i)): return False, None
    if not (cl[i] > p_i and cl[i] > c35s[i]): return False, None
    if cl[i] < p2_i * TOUCH: return False, None
    feats = dict(gap=round(gap, 1), abr=round(abr, 1),
                 dev35=round((cl[i] / c35s[i] - 1) * 100, 1),
                 dist_purple=round((p2_i / cl[i] - 1) * 100, 1))
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


def v3b_signal_at(cl, gt_p, gt2, c35s, vol, i, gt_p_eff=None, gt2_eff=None):
    """判定第 i 天是否成立 v3b 二次买点(只用<=i数据, 无未来函数)。
    P0-2: 当日摸紫线/ gap 用 *_eff；历史摸线窗口仍用原始线（置空日不计数）。
    返回 (ok, 特征dict)"""
    if i < MIN_I: return False, None
    p2i = gt2_eff[i] if gt2_eff is not None else gt2[i]
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
    pi = gt_p_eff[i] if gt_p_eff is not None else gt_p[i]
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


def v3b_signal_b2(cl, gt_p, cA, vol, i, gt_p_eff=None, gap_min=None, gap_max=None,
                  strong_gap=False):
    """v3b 二次买点 - 三线制 v3b_B2 (2026-10-21 polo4111 定版, 紫线 gt2 弃用):
    1) 当日站上红压线 close > gt_p, 且 close > 主成本线A(线上确认)
    2) [T-250,T-20] 曾站上红压线(曾经突破动作, touch=1.0)
    3) [T-70,T-5] 曾跌破主成本线A×0.98(深回踩洗盘)
    4) gap 相对红压线 [gap_min,gap_max) 默认 [5%,20%)
       strong_gap=True 时为 B2-early 强势分支：只保留 gap > gap_max，其他条件不变
    5) 量能 >=1.2×前20均量
    6) 冷却由 scan_series 统一负责(armed/structure 或间隔), 本函数不重复判定
       (2026-10-21 改: 原"逐日站压线事件冷却"会锁死连续站上压线的大牛,
       603778@1016 型; 冷却口径与 early_E4 对齐)
    只用<=i数据, 无未来函数。返回 (ok, 特征dict)"""
    if i < MIN_I: return False, None
    p_i = gt_p_eff[i] if gt_p_eff is not None else gt_p[i]
    if not (np.isfinite(p_i) and p_i > 0 and np.isfinite(cA[i]) and cA[i] > 0):
        return False, None
    touch = 1.0
    # 1) 当日站上红压线 + 线上确认(close>主成本线A)
    if cl[i] <= p_i: return False, None
    if cl[i] <= cA[i]: return False, None
    # 2) 历史: [T-250, T-20] 曾站上红压线
    lo, hi = max(MIN_I, i - V3B_HIST_LO), i - V3B_HIST_HI
    if lo >= hi: return False, None
    pw = gt_p[lo:hi]
    if not np.any(np.isfinite(pw) & (pw > 0) & (cl[lo:hi] >= pw * touch)):
        return False, None
    # 3) 回踩: [T-70, T-5] 曾跌破主成本线A×0.98(深回踩洗盘)
    rlo, rhi = max(0, i - V3B_REC_LO), i - V3B_REC_HI
    if rlo >= rhi: return False, None
    cw = cA[rlo:rhi]
    if not np.any(np.isfinite(cw) & (cw > 0) & (cl[rlo:rhi] < cw * V3B_REC_DIP)):
        return False, None
    # 4) gap 相对红压线
    gmin = V3B_GAP_MIN if gap_min is None else float(gap_min)
    gmax = V3B_GAP_MAX if gap_max is None else float(gap_max)
    gap = (cl[i] / p_i - 1) * 100
    if strong_gap:
        if gap <= gmax: return False, None
    elif not (gmin <= gap < gmax):
        return False, None
    # 5) 量能确认
    if vol is not None:
        ma20 = vol[max(0, i - 20):i].mean()
        if ma20 > 0 and vol[i] < ma20 * V3B_VOL_MULT: return False, None
    feats = dict(gap=round(gap, 1),
                 dev_a=round((cl[i] / cA[i] - 1) * 100, 1))
    return True, feats


def v3b_early_at(cl, gt_p, gt2, c35s, vol, i, gt_p_eff=None, gt2_eff=None):
    """v3b 提前观察信号(2026-10-16 polo4111): 与 v3b 同源, 但成立条件从
    '摸紫线'放宽为'站上 COST35×1.03 且未到紫线'——用于 COST35 与红压线
    之间长期有余量、主信号偏晚的场景, 提前标记潜在启动前兆。
    (2026-10-16 曾试连续两天站上收紧, 覆盖率60.6%→49%且提前量缩短, 用户还原)
    P0-2: 当日判定用 *_eff；历史摸线窗口仍用原始线。
    只用<=i数据, 无未来函数。返回 (ok, 特征dict)"""
    if i < MIN_I: return False, None
    p2i = gt2_eff[i] if gt2_eff is not None else gt2[i]
    c35i = c35s[i]
    if not (np.isfinite(p2i) and p2i > 0): return False, None
    if not (np.isfinite(c35i) and c35i > 0): return False, None
    # 1) 提前位: 站上 COST35 3% 之上, 但尚未摸到紫线(保证比 v3b 早)
    if cl[i] < c35i * 1.03: return False, None
    if cl[i] >= p2i * V3B_TOUCH: return False, None
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
    # 4) gap 上限过滤: 相对粉线 < 20%(防高位滞涨), **不设下限**——early 定位是
    #    站上成本线的提前观察, 价格往往还在粉线下方(gap为负), 设下限会卡掉提前点
    pi = gt_p_eff[i] if gt_p_eff is not None else gt_p[i]
    if np.isfinite(pi) and pi > 0:
        gap = (cl[i] / pi - 1) * 100
        if gap >= V3B_GAP_MAX: return False, None
    else:
        gap = float('nan')
    # 5) 量能确认: 站上成本线日放量 >= 1.2×前20日均量
    if vol is not None:
        ma20 = vol[max(0, i - 20):i].mean()
        if ma20 > 0 and vol[i] < ma20 * V3B_VOL_MULT: return False, None
    feats = dict(gap=round(gap, 1) if np.isfinite(gap) else None,
                 dist_purple=round((p2i / cl[i] - 1) * 100, 1))
    return True, feats


def early_cB(hi, lo, vol, turn):
    """参考线B(COST40/E50), try/except 防换手率异常。无未来函数(逐日)."""
    try:
        return np.asarray(ehlers_smooth(cost_series(hi, lo, vol, turn, COST2_PCT),
                                        max(2, COST2_EHLERS)), float)
    except Exception:
        return np.full(len(hi), np.nan)


def early_signal_b2(cl, gt_p, cA, cB, i, gt_p_eff=None):
    """early 三线制(E4) 提前观察信号 (2026-10-21 polo4111 定版, 紫线弃用):
    1) 当日站上参考线B×1.03(脱离底部), 且未站上主成本线A(close < cA)
       —— 覆盖"红压线之上、主成本线A之下"的中间区(300204@0415 型),
          与 v3b_B2(close>cA) 在 A 处分界互补
    2) [T-250,T-20] 曾站上红压线(曾经突破动作, touch=1.0)
    3) [T-70,T-5] 曾跌破参考线B×0.98(深洗)
    4) 无量能要求(捕捉首波缩量启动, 002460/603778 型)
    5) 冷却由 scan_series 统一负责
    只用<=i数据, 无未来函数。返回 (ok, 特征dict)"""
    if i < MIN_I: return False, None
    p_i = gt_p_eff[i] if gt_p_eff is not None else gt_p[i]
    if not (np.isfinite(p_i) and p_i > 0 and np.isfinite(cB[i]) and cB[i] > 0
            and np.isfinite(cA[i]) and cA[i] > 0):
        return False, None
    # 1) 站上参考线B×1.03, 未站上主成本线A
    if cl[i] < cB[i] * 1.03: return False, None
    if cl[i] >= cA[i]: return False, None
    # 2) 历史: [T-250, T-20] 曾站上红压线
    lo, hi = max(MIN_I, i - V3B_HIST_LO), i - V3B_HIST_HI
    if lo >= hi: return False, None
    pw = gt_p[lo:hi]
    if not np.any(np.isfinite(pw) & (pw > 0) & (cl[lo:hi] >= pw * 1.0)):
        return False, None
    # 3) 回踩: [T-70, T-5] 曾跌破参考线B×0.98
    rlo, rhi = max(0, i - V3B_REC_LO), i - V3B_REC_HI
    if rlo >= rhi: return False, None
    cw = cB[rlo:rhi]
    if not np.any(np.isfinite(cw) & (cw > 0) & (cl[rlo:rhi] < cw * V3B_REC_DIP)):
        return False, None
    gap = (cl[i] / p_i - 1) * 100
    feats = dict(gap=round(gap, 1),
                 dev_b=round((cl[i] / cB[i] - 1) * 100, 1))
    return True, feats


# ─────────────────────────────────────────────────────────────
# v3b 信号画像(2026-10-16 polo4111): 每次信号出来后, 用信号日及以前
#   数据(无未来函数)整理多维信息: 量能 / 回踩结构 / 冷却期 / 前高位置 /
#   筹码偏离 / 集中带宽 / 获利盘近似 / gap。画图时对图中每个信号日
#   做同样画像并展示在图上。用户明确: 不给分数, 只整理信息。
# ─────────────────────────────────────────────────────────────

def signal_profile(cl, hi, lo, vol, turn, gt_p, gt2, c35s, i,
                   gt_p_eff=None, gt2_eff=None):
    """信号日多维画像(纯<=i数据, 无未来函数)。返回指标dict

    P0-2: gap 优先用 gt_p_eff（置空日 = base×1.13），避免画像在失明段变 NaN。
    """
    n = len(cl)
    # ① 量能: 量比 = 当日/前20日均量
    ma20 = vol[max(0, i - 20):i].mean()
    vr = vol[i] / ma20 if ma20 > 0 else 0.0
    # ② 回踩结构: [T-70,T-5] 内相对 COST35 最低偏离 + 持续天数
    rlo, rhi = max(0, i - V3B_REC_LO), i - V3B_REC_HI
    if rlo < rhi:
        dip = cl[rlo:rhi] / c35s[rlo:rhi] - 1.0
        dmin = float(np.nanmin(dip)) if len(dip) else 0.0
        under = np.isfinite(dip) & (dip < V3B_REC_DIP - 1.0)
        rec_days = int(under.sum())
    else:
        dmin, rec_days = 0.0, 0
    # ③ 冷却期: 最近一次摸紫线距今天数([T-250,T-20])
    hlo, hhi = max(MIN_I, i - V3B_HIST_LO), i - V3B_HIST_HI
    if hlo < hhi:
        p2w = gt2[hlo:hhi]
        tj = np.where(np.isfinite(p2w) & (p2w > 0) & (cl[hlo:hhi] >= p2w * V3B_TOUCH))[0]
        cool = i - (hlo + tj[-1]) if len(tj) else (i - hlo)
        hist_touch = int(len(tj))
    else:
        cool, hist_touch = 250, 0
    # ④ 前高位置: 现价相对近250日最高价
    h250 = float(np.nanmax(hi[max(0, i - 250):i + 1]))
    ratio = h250 / cl[i] if cl[i] > 0 else 9.0
    # ⑤ 筹码: dev35(相对信号成本线 SIGNAL_COST_PCT；字段名保留兼容) + 集中带宽(COST90-COST10)/价 + 获利盘近似
    c35i = c35s[i]
    dev35 = (cl[i] / c35i - 1) * 100 if (np.isfinite(c35i) and c35i > 0) else 0.0
    c10 = cost_series(hi, lo, vol, turn, CHIP_BAND_LO)
    c90 = cost_series(hi, lo, vol, turn, CHIP_BAND_HI)
    c10i, c90i = c10[i], c90[i]
    if np.isfinite(c10i) and np.isfinite(c90i) and c90i > c10i and cl[i] > 0:
        bw = (c90i - c10i) / cl[i]
        profit = float(np.clip((cl[i] - c10i) / (c90i - c10i), 0.0, 1.0))
    else:
        bw, profit = 0.5, 0.5
    # ⑥ gap: 相对粉线
    pi = gt_p_eff[i] if gt_p_eff is not None else gt_p[i]
    gap = (cl[i] / pi - 1) * 100 if (np.isfinite(pi) and pi > 0) else 15.0
    return dict(
        close=round(float(cl[i]), 2),
        vol_ratio=round(vr, 2),
        rec_depth=round(dmin * 100, 1),
        rec_days=int(rec_days),
        cool_days=int(cool),
        hist_touch=int(hist_touch),
        dist_high=round((1 - 1 / ratio) * 100, 1) if ratio >= 1 else round((ratio - 1) * 100, 1),
        dev35=round(dev35, 1),
        band=round(bw * 100, 1),
        profit=round(profit, 2),
        gap=round(gap, 1))


def scan_hist_v3b(code, end, datalen=1800, win_start=None, win_end=None, mode=None):
    """v3b 历史全信号（结构冷却/旧天数冷却由 mode 决定，无未来函数）"""
    try:
        df = load(code, end, datalen)
    except Exception:
        return []
    if df is None or len(df) < MIN_I + 130: return []
    cl, darr, gt_p, gt2, gt_p_eff, gt2_eff, c35s, _, cA = compute_eff(df)
    vol = df['volume'].values.astype(float)
    n = len(cl)
    hits = []
    # P0-1: 冷却链从 MIN_I 全历史推进，窗口只影响输出
    sig = scan_series(cl, gt_p, gt2, c35s, vol, start_i=MIN_I,
                      end_i=n - 125, types=('v3b',), mode=mode,
                      gt_p_eff=gt_p_eff, gt2_eff=gt2_eff, cA=cA)
    for _, i in sig:
        if win_start and darr[i] < win_start: continue
        if win_end and darr[i] > win_end: break
        ok, feats = v3b_signal_b2(cl, gt_p, cA, vol, i,
                                  gt_p_eff=gt_p_eff)
        if not ok: continue
        fut = lambda k: round((cl[i + 1:i + k + 1].max() / cl[i] - 1) * 100, 1) if i + k < n else None
        hits.append((code, darr[i], round(float(cl[i]), 2), feats['gap'], feats['dev_a'],
                     fut(20), fut(60), fut(120)))
    return hits


def scan_daily(code, end, datalen=1800):
    """每日候选池: 只检测最新交易日"""
    try:
        df = load(code, end, datalen)
    except Exception:
        return None
    if df is None or len(df) < MIN_I + 5: return None
    cl, darr, gt_p, gt2, gt_p_eff, gt2_eff, c35s, _, cA = compute_eff(df)
    i = len(cl) - 1
    ok, feats = signal_at(cl, gt_p, gt2, c35s, darr, i,
                          gt_p_eff=gt_p_eff, gt2_eff=gt2_eff)
    if not ok: return None
    return (code, darr[i], round(float(cl[i]), 2), feats['gap'], feats['abr'],
            feats['dev35'], feats['dist_purple'])

def scan_hist(code, end, datalen=1800, win_start=None, win_end=None, mode=None):
    """v3 历史全信号（结构冷却/旧天数冷却由 mode 决定，无未来函数）"""
    try:
        df = load(code, end, datalen)
    except Exception:
        return []
    if df is None or len(df) < MIN_I + 130: return []
    cl, darr, gt_p, gt2, gt_p_eff, gt2_eff, c35s, _, cA = compute_eff(df)
    vol = df['volume'].values.astype(float)
    n = len(cl)
    hits = []
    sig = scan_series(cl, gt_p, gt2, c35s, vol, start_i=MIN_I,
                      end_i=n - 125, types=('v3',), mode=mode, darr=darr,
                      gt_p_eff=gt_p_eff, gt2_eff=gt2_eff)
    for _, i in sig:
        if win_start and darr[i] < win_start: continue
        if win_end and darr[i] > win_end: break
        ok, feats = signal_at(cl, gt_p, gt2, c35s, darr, i,
                              gt_p_eff=gt_p_eff, gt2_eff=gt2_eff)
        if not ok: continue
        fut = lambda k: round((cl[i + 1:i + k + 1].max() / cl[i] - 1) * 100, 1) if i + k < n else None
        hits.append((code, darr[i], round(float(cl[i]), 2), feats['gap'], feats['abr'],
                     feats['dev35'], feats['dist_purple'], fut(20), fut(60), fut(120)))
    return hits

# ─────────────────────────────────────────────────────────────
# bigline 大阳线起点确认(2026-10-16 polo4111)
#   探索结论(22池305个early): early后20天内出大阳(>=5%) 57.4%, 平均等7.7天;
#   大阳线日起点后60日中位+21.0%(81%>10%) vs early直接买+11.1%(53%>10%);
#   20天无大阳批后60日中位仅+4.4% → "等第一根大阳线=起点, 等不到=假信号"成立
# ─────────────────────────────────────────────────────────────
BIG_PCT = 0.05       # 大阳线阈值(单日涨幅>=5%)
BIG_PCT_MAX = 0.21   # 上限防异常(除权/错价, A股单日<=20%)
BIG_LOOK = 20        # early 后等待窗口(交易日)
BIG_VOL_MULT = 1.5   # 大阳线放量定义(>=1.5×前20日均量)


def bigline_hits(cl, darr, gt_p, gt2, c35s, vol, mode=None,
                 gt_p_eff=None, gt2_eff=None, cB=None):
    """对每个 early 信号（结构冷却/旧天数冷却由 mode 决定），找其后 BIG_LOOK 天内第一根大阳线。
    返回 [(big_i, early_i, wait, big_pct, vol_ratio)], 只用<=big_i数据, 无未来函数"""
    n = len(cl)
    pct = np.full(n, 0.0); pct[1:] = cl[1:] / cl[:-1] - 1.0
    early_idx = [i for _, i in scan_series(cl, gt_p, gt2, c35s, vol,
                                           start_i=MIN_I, end_i=n - 5,
                                           types=('early',), mode=mode,
                                           gt_p_eff=gt_p_eff, gt2_eff=gt2_eff,
                                           cB=cB)]
    hits = []
    for e in early_idx:
        for k in range(e + 1, min(e + BIG_LOOK + 1, n - 1)):
            if BIG_PCT_MAX >= pct[k] >= BIG_PCT:
                ma20 = vol[max(0, k - 20):k].mean()
                vr = vol[k] / ma20 if ma20 > 0 else 0.0
                hits.append((k, e, k - e, round(pct[k] * 100, 1), round(vr, 2)))
                break
    return hits


def scan_bigline(code, end, datalen=1800, fresh=10, mode=None):
    """单只: 全部 bigline 起点信号, 保留 early 日(入池点)在最近 fresh 个交易日内的新鲜信号。
    画像字段(只用<=大阳线日数据, 无未来函数):
      code/big_date/early_date/wait/big_pct/vol_ratio/close/dev35/gap/band/profit/dist_high/rec_depth/cool_days
    ⚠️ 2026-10-18 reasonix double confirm 推翻"等大阳线"结论后: 入池口径从 big_date 改为 early_date
      (大阳线只能做事后质量甄别, 不能延迟入场; 每天新增池直接吃 early 信号)"""
    try:
        df = load(code, end, datalen)
    except Exception:
        return []
    if df is None or len(df) < MIN_I + BIG_LOOK + 5: return []
    cl, darr, gt_p, gt2, gt_p_eff, gt2_eff, c35s, _, cA = compute_eff(df)
    vol = df['volume'].values.astype(float)
    hi = df['high'].values.astype(float); lo = df['low'].values.astype(float)
    turn = df['turnover'].values.astype(float)
    n = len(cl)
    cB = early_cB(hi, lo, vol, turn)
    hits = bigline_hits(cl, darr, gt_p, gt2, c35s, vol, mode=mode,
                        gt_p_eff=gt_p_eff, gt2_eff=gt2_eff, cB=cB)
    # fresh: early 日(入池点)在最近 fresh 根K线内 (2026-10-18 从 big_i 改为 early_i)
    rows = []
    for big_i, early_i, wait, bpct, vr in hits:
        if early_i < n - fresh: continue
        pf = signal_profile(cl, hi, lo, vol, turn, gt_p, gt2, c35s, big_i,
                            gt_p_eff=gt_p_eff, gt2_eff=gt2_eff)
        rows.append((code, darr[big_i], darr[early_i], wait, bpct, vr,
                     pf['close'], pf['dev35'], pf['gap'], pf['band'],
                     pf['profit'], pf['dist_high'], pf['rec_depth'], pf['cool_days']))
    return rows


# ─────────────────────────────────────────────────────────────
# 每日池扫描: 最近窗口内的 v3b / early（2026-10-18 reasonix）
#   背景: daily_pipeline 原先把 scan_bigline 当作入池来源, 但 bigline_hits 只返回
#   "已经出现大阳线"的 early —— 等大阳线已被证伪(同批子集 early 当天买 +8.9% vs
#   等大阳 +2.5%)。每日入池必须直接吃 **early 当日信号**, 大阳只做事后质量标签。
#   本函数同时返回 v3b(二次买点)与 early(提前观察), 两者各自独立冷却。
# ─────────────────────────────────────────────────────────────
RECENT_FIELDS = (
    'code', 'signal_type', 'sig_date', 'sig_close',
    'vol_ratio', 'rec_depth', 'rec_days', 'cool_days', 'hist_touch',
    'dist_high', 'dev35', 'band', 'profit', 'gap',
    'big_date', 'big_wait', 'big_pct', 'big_vol_ratio',
)


def _first_bigline(cl, i, look=BIG_LOOK):
    """i 之后 look 个交易日内第一根大阳线 -> (big_i, wait, big_pct) 或 None。
    只用 <=big_i 数据（大阳判定=当日/前收），仅作事后质量标签，不用于入场。"""
    n = len(cl)
    for k in range(i + 1, min(i + look + 1, n)):
        if cl[k - 1] <= 0:
            continue
        pct = cl[k] / cl[k - 1] - 1.0
        if BIG_PCT <= pct <= BIG_PCT_MAX:
            return k, k - i, pct
    return None


# ─────────────────────────────────────────────────────────────
# 统一信号序列扫描（2026-10-18 P0-1）
#   旧实现: 同类型信号固定间隔 V3B_COOL(20) 天。问题(688099 实测):
#     7/25 early 之后 8/13-8/19 的 early 全被 20 天冷却吞掉, 但 8/13 才是
#     本轮更靠近成本线的位置; 反过来, 若一轮结束后 20 天内又启动新的一轮,
#     新信号也会被吞掉 —— 固定天数既会漏真信号, 也会保留错时点。
#   新实现(mode='structure', 默认): 每种信号独立维护 armed;
#     收盘跌破 COST35×STRUCT_BREAK(0.98) 才重新武装;
#     同一结构区间只输出第一次触发。跌破成本线是**因果**的重新武装条件,
#     不使用未来数据; 同轮不重复提示, 避免"每次反弹都报一次"。
#   mode='days' 保留旧口径, 用于 A/B 对照与历史复现。
# ─────────────────────────────────────────────────────────────
COOLDOWN_MODE = 'structure'   # 'structure' | 'days'
STRUCT_BREAK = 0.98           # 跌破 COST35×0.98 = 本轮结构结束, 重新武装


def scan_series(cl, gt_p, gt2, c35s, vol, start_i=MIN_I, end_i=None,
                types=('v3b', 'early'), mode=None, cool=V3B_COOL, darr=None,
                gt_p_eff=None, gt2_eff=None, cA=None, cB=None,
                rearm_days=None):
    """统一扫描 v3 / v3b / early 信号序列（严格因果）。返回 [(type, i)] 升序。

    mode='structure': 跌破信号成本线 SIGNAL_COST_PCT×0.98 重新武装; 同轮只输出第一次触发。
    mode='days':      同类型信号间隔 >= cool 个交易日（旧口径）。
    v3 需要 darr（signal_at 的日期轴）；不传时 types 中的 v3 会被跳过。
    P0-2: 当日判定优先用 gt_p_eff/gt2_eff；历史窗口仍用原始 gt_p/gt2。
    rearm_days: 结构模式下，距上次同类信号 >= rearm_days 也重新武装（默认 None=纯结构）。
    v3b_B2(2026-10-21): 传入 cA(主成本线A) 时 v3b 用三线制 v3b_signal_b2(紫线弃用);
                        不传时保持旧紫线 v3b_signal_at(历史回溯兼容)。
    v3b_b2_early(2026-10-21): B2 强势分支，条件同 B2 但 gap>gap_max(默认20%)；
                              仅输出标记，不替代正式 B2。
    early_E4(2026-10-21): 传入 cB(参考线B) 时 early 用三线制 early_signal_b2(紫线弃用);
                          不传时保持旧紫线 v3b_early_at(历史回溯兼容)。
    """
    n = len(cl)
    end_i = n if end_i is None else min(end_i, n)
    mode = mode or COOLDOWN_MODE
    out = []
    last = {t: -10 ** 9 for t in types}
    armed = {t: True for t in types}
    for i in range(start_i, end_i):
        if mode == 'structure' and np.isfinite(c35s[i]) and c35s[i] > 0 \
                and cl[i] < c35s[i] * STRUCT_BREAK:
            for t in types:
                armed[t] = True
        elif mode == 'structure' and rearm_days is not None:
            for t in types:
                if i - last[t] >= int(rearm_days):
                    armed[t] = True
        for t in types:
            if mode == 'days':
                if i - last[t] < cool:
                    continue
            elif not armed[t]:
                continue
            if t == 'v3b':
                if cA is not None:
                    ok, _ = v3b_signal_b2(cl, gt_p, cA, vol, i,
                                          gt_p_eff=gt_p_eff)
                else:
                    ok, _ = v3b_signal_at(cl, gt_p, gt2, c35s, vol, i,
                                          gt_p_eff=gt_p_eff, gt2_eff=gt2_eff)
            elif t == 'v3b_b2_early':
                if cA is None:
                    continue
                ok, _ = v3b_signal_b2(cl, gt_p, cA, vol, i,
                                      gt_p_eff=gt_p_eff, strong_gap=True)
            elif t == 'early':
                if cB is not None:
                    ok, _ = early_signal_b2(cl, gt_p, cA, cB, i,
                                            gt_p_eff=gt_p_eff)
                else:
                    ok, _ = v3b_early_at(cl, gt_p, gt2, c35s, vol, i,
                                         gt_p_eff=gt_p_eff, gt2_eff=gt2_eff)
            elif t == 'v3':
                if darr is None:
                    continue
                ok, _ = signal_at(cl, gt_p, gt2, c35s, darr, i,
                                  gt_p_eff=gt_p_eff, gt2_eff=gt2_eff)
            else:
                continue
            if ok:
                out.append((t, i))
                last[t] = i
                armed[t] = False
    return out


def scan_recent(code, end, fresh=10, datalen=1800, df=None, mode=None,
                cost_pct=None, ehlers_period=None):
    """单只: 最近 fresh 个交易日内的 v3 / v3b / early 信号（含画像与事后 big 标签）。

    · 判定/画像全部只用 <=信号日 数据（无未来函数）；
    · big_* 是信号日之后的信息，仅用于入池后的质量分层/复盘，禁止参与入场判定；
    · 冷却口径由 scan_series 决定：默认结构去重（跌破 COST35×0.98 重新武装），
      mode='days' 可回退旧 20 天冷却做 A/B；
    · v3 / v3b / early 各自独立 armed/冷却；冷却链从全历史推进；
    · 2026-10-18 P1-1: 接入 v3（首次压制突破）——旧版只扫 v3b/early，
      688099 9/5、9/10 的 v3 真实可执行信号完全没有进入每日池；
    · df 可由调用方传入（daily_pipeline 的本地缓存），为 None 时才联网 load。
    返回 list[dict]，字段见 RECENT_FIELDS。
    """
    if df is None:
        try:
            df = load(code, end, datalen)
        except Exception:
            return []
    if df is None or len(df) < MIN_I + 50:
        return []
    cl, darr, gt_p, gt2, gt_p_eff, gt2_eff, c35s, _, cA = compute_eff(
        df, cost_pct=cost_pct, ehlers_period=ehlers_period)
    vol = df['volume'].values.astype(float)
    hi = df['high'].values.astype(float)
    lo = df['low'].values.astype(float)
    turn = df['turnover'].values.astype(float)
    cB = early_cB(hi, lo, vol, turn)
    n = len(cl)
    start_i = max(MIN_I, n - fresh)

    # P0-1: 统一结构冷却（跌破 COST35×0.98 重新武装；同轮只提示第一次）
    # P1-1: v3/v3b/early 三种类型独立 armed；v3 需要 darr
    sig = scan_series(cl, gt_p, gt2, c35s, vol, start_i=MIN_I,
                      types=('v3', 'v3b', 'v3b_b2_early', 'early'), mode=mode, darr=darr,
                      gt_p_eff=gt_p_eff, gt2_eff=gt2_eff, cA=cA, cB=cB,
                      rearm_days=SIGNAL_REARM_DAYS)
    hits = [(t, i) for t, i in sig if i >= start_i]

    rows = []
    for stype, i in hits:
        pf = signal_profile(cl, hi, lo, vol, turn, gt_p, gt2, c35s, i,
                            gt_p_eff=gt_p_eff, gt2_eff=gt2_eff)
        big_date = None
        big_wait = None
        big_pct = None
        big_vr = None
        if stype in ('early', 'v3'):
            bg = _first_bigline(cl, i)
            if bg is not None:
                k, wait, pct = bg
                ma20 = vol[max(0, k - 20):k].mean()
                big_date = darr[k]
                big_wait = wait
                big_pct = round(pct * 100, 1)
                big_vr = round(vol[k] / ma20, 2) if ma20 > 0 else None
        rows.append(dict(
            code=code, signal_type=stype, sig_date=darr[i], sig_close=pf['close'],
            vol_ratio=pf['vol_ratio'], rec_depth=pf['rec_depth'], rec_days=pf['rec_days'],
            cool_days=pf['cool_days'], hist_touch=pf['hist_touch'],
            dist_high=pf['dist_high'], dev35=pf['dev35'], band=pf['band'],
            profit=pf['profit'], gap=pf['gap'],
            big_date=big_date, big_wait=big_wait, big_pct=big_pct, big_vol_ratio=big_vr))
    return rows


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
    elif mode == 'scan_big':
        part = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        nparts = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        END = sys.argv[4] if len(sys.argv) > 4 else '20261015'
        all_df = tq.load_stock_list()
        codes = all_df['code'].astype(str).str.zfill(6).tolist()
        codes = codes[part::nparts]
        t0 = time.time(); rows = []
        for k, code in enumerate(codes):
            try:
                rows.extend(scan_bigline(code, END))
            except Exception:
                pass
            if (k + 1) % 200 == 0:
                print(f'[part{part}] {k + 1}/{len(codes)} 信号{len(rows)} 耗时{time.time() - t0:.0f}s', flush=True)
        out = os.path.join('result', f'challenge_bigline_part{part}.csv')
        pd.DataFrame(rows, columns=['code', 'big_date', 'early_date', 'wait_days', 'big_pct',
                                    'vol_ratio', 'close', 'dev35', 'gap', 'band',
                                    'profit', 'dist_high', 'rec_depth', 'cool_days']).to_csv(out, index=False)
        print(f'[part{part}] done {len(rows)} -> {out}', flush=True)
    elif mode == 'recent':
        # 每日池扫描: 最近 fresh 个交易日的 v3b / early（入池口径 = early 当日）
        part = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        nparts = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        END = sys.argv[4] if len(sys.argv) > 4 else '20261015'
        FRESH = int(sys.argv[5]) if len(sys.argv) > 5 else 10
        all_df = tq.load_stock_list()
        codes = all_df['code'].astype(str).str.zfill(6).tolist()
        codes = codes[part::nparts]
        t0 = time.time(); rows = []
        for k, code in enumerate(codes):
            try:
                rows.extend(scan_recent(code, END, fresh=FRESH))
            except Exception:
                pass
            if (k + 1) % 200 == 0:
                print(f'[part{part}] {k + 1}/{len(codes)} 信号{len(rows)} 耗时{time.time() - t0:.0f}s', flush=True)
        out = os.path.join('result', f'challenge_recent_part{part}.csv')
        pd.DataFrame(rows, columns=list(RECENT_FIELDS)).to_csv(out, index=False)
        print(f'[part{part}] done {len(rows)} -> {out}', flush=True)
    elif mode == 'hist_v3b':
        code = sys.argv[2]
        end = sys.argv[3] if len(sys.argv) > 3 else '20261015'
        dl = int(sys.argv[4]) if len(sys.argv) > 4 else 1600
        hits = scan_hist_v3b(code, end, dl)
        print(f'{code} 二次买点信号(v3b): {len(hits)} 个')
        for h in hits: print('  ', h)
    elif mode == 'assess':
        code = sys.argv[2]
        end = sys.argv[3] if len(sys.argv) > 3 else '20261015'
        date = sys.argv[4] if len(sys.argv) > 4 else None
        dl = int(sys.argv[5]) if len(sys.argv) > 5 else 1600
        try:
            df = load(code, end, dl)
        except Exception as e:
            print(f'取数失败: {e}'); sys.exit(1)
        if df is None or len(df) < MIN_I + 5:
            print('数据不足'); sys.exit(1)
        cl, darr, gt_p, gt2, gt_p_eff, gt2_eff, c35s, _, cA = compute_eff(df)
        hi = df['high'].values.astype(float); lo = df['low'].values.astype(float)
        vol = df['volume'].values.astype(float); turn = df['turnover'].values.astype(float)
        if date:
            idx = next((k for k, d in enumerate(darr) if d == date), None)
            if idx is None:
                print(f'{code} 在 {end} 前无 {date} 数据'); sys.exit(1)
            ok, vf = v3b_signal_b2(cl, gt_p, cA, vol, idx,
                                   gt_p_eff=gt_p_eff)
            print(f'== {code} {date} 信号判定: {ok} ==')
            if ok:
                pf = signal_profile(cl, hi, lo, vol, turn, gt_p, gt2, c35s, idx,
                                    gt_p_eff=gt_p_eff, gt2_eff=gt2_eff)
                print(f'  信号日收盘: {pf["close"]}')
                print(f'  量能       : 量比 {pf["vol_ratio"]}x (当日/前20日均量)')
                print(f'  回踩结构   : 最深 {pf["rec_depth"]}% (相对{COST_LABEL}), 跌破 {pf["rec_days"]} 天')
                print(f'  冷却期     : 距上次站上红压线 {pf["cool_days"]} 天 (近250日摸线 {pf["hist_touch"]} 次)')
                print(f'  前高位置   : 现价距近250日最高 {pf["dist_high"]}%')
                print(f'  筹码       : dev35={pf["dev35"]}% 集中带宽={pf["band"]}% 获利盘≈{pf["profit"]:.0%}')
                print(f'  gap        : 相对红压线 {pf["gap"]}% (dev_a={vf["dev_a"]}%)')
        else:
            print(f'== {code} 全部 v3b 信号画像(至 {end}) ==')
            sig = scan_series(cl, gt_p, gt2, c35s, vol, start_i=MIN_I,
                              end_i=len(cl) - 5, types=('v3b',),
                              gt_p_eff=gt_p_eff, gt2_eff=gt2_eff, cA=cA)
            for _, i in sig:
                ok, vf = v3b_signal_b2(cl, gt_p, cA, vol, i,
                                       gt_p_eff=gt_p_eff)
                if not ok: continue
                pf = signal_profile(cl, hi, lo, vol, turn, gt_p, gt2, c35s, i,
                                    gt_p_eff=gt_p_eff, gt2_eff=gt2_eff)
                print(f'  {darr[i]} 收{pf["close"]} | 量比{pf["vol_ratio"]}x | 回踩{pf["rec_depth"]}%({pf["rec_days"]}天) | '
                      f'冷却{pf["cool_days"]}天 | 距前高{pf["dist_high"]}% | dev35={pf["dev35"]}% | '
                      f'带宽{pf["band"]}% | 获利{pf["profit"]:.0%} | gap{pf["gap"]}%')
                last_sig = i
    else:
        code = sys.argv[2]
        end = sys.argv[3] if len(sys.argv) > 3 else '20261015'
        dl = int(sys.argv[4]) if len(sys.argv) > 4 else 1600
        hits = scan_hist(code, end, dl)
        print(f'{code} 挑战信号(v3): {len(hits)} 个')
        for h in hits: print('  ', h)
