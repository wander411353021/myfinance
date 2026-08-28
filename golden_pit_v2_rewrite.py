# -*- coding: utf-8 -*-
"""重写 detect_golden_pit 为纯因果单遍扫描(v2),与原版对比验证。

修正点(2026-08-28):
1. 消除 pass2 扩展段 [s0-10, e0+10](延伸到未来)与跨出坑日合并
2. 出坑即结算:出坑后再次 z<-1.5 = 新坑(坑段/坑长真实,不因 merge 失真)
3. merge_gap 仅用于"未出坑"时 z 短暂离开(≤merge_gap 天)仍延续坑
4. 坑底 b 实时维护"当日及以前最低",出坑判定参照当日已知最低(因果)
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression

def detect_golden_pit_v2(closes, reg250, z_thr=-1.5, merge_gap=15, launch_gate=0.9,
                         use_pre_std=True, require_below_gate=False):
    """纯因果单遍扫描黄金坑。返回 [(s, b, lch)] 升序。"""
    closes = np.asarray(closes, dtype=float)
    reg250 = np.asarray(reg250, dtype=float)
    n = len(closes)
    resid = closes - reg250
    # 滚动 60 日 std(含坑,因果)
    rstd = np.full(n, np.nan)
    for i in range(59, n):
        rstd[i] = np.std(resid[i - 59:i + 1])
    z_roll = np.full(n, np.nan)
    for i in range(59, n):
        z_roll[i] = resid[i] / rstd[i] if rstd[i] > 0 else 0.0
    gate_line = reg250 * launch_gate
    out = []
    i = 59
    while i < n:
        # 进坑:滚动 z < z_thr
        if not (np.isfinite(z_roll[i]) and z_roll[i] < z_thr):
            i += 1
            continue
        # 坑前 std(因果:只用 i-60 及以前)
        if use_pre_std and i >= 60:
            pre_std = np.std(resid[i - 60:i])
            if pre_std <= 0:
                pre_std = rstd[i]
        else:
            pre_std = rstd[i]
        s = i
        b = i
        bv = closes[i]
        lch = None
        leave = 0  # 连续不在坑内天数(坑内短暂离开,≤merge_gap 仍延续)
        j = i
        while j < n:
            zj = resid[j] / pre_std if pre_std > 0 else 0.0
            in_pit = np.isfinite(zj) and zj < z_thr and (
                closes[j] < gate_line[j] if require_below_gate else True)
            if in_pit:
                leave = 0
                if closes[j] < bv:
                    bv = closes[j]; b = j
            else:
                # 不在坑内: 先看出坑(收复门控线 + 高于已知最低 2%)
                if closes[j] >= gate_line[j] and closes[j] > bv * 1.02:
                    lch = j
                    break
                # 未出坑: 短暂离开(≤merge_gap)仍延续坑,最低继续维护
                leave += 1
                if closes[j] < bv:
                    bv = closes[j]; b = j
                if leave > merge_gap:
                    break  # 离开太久,坑结束(未出坑)
            j += 1
        if lch is not None:
            out.append((s, b, lch))
            i = lch + 1  # 出坑即结算,之后重新找坑
        else:
            i = n
    return out

# ── 截断自检(v2) ──
def trunc_check(stocks, n_cuts=6, seed=20260828):
    rng = np.random.default_rng(seed)
    mismatch = 0; checked = 0; checked_sig = 0
    for sym in stocks:
        f = os.path.join('.cache_kline', f'{sym}.npy')
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True).item()
        c = d['close'].astype(float)
        n = len(c)
        if n < 500:
            continue
        reg_full, _ = compute_rolling_regression(c, window=250, use_log=True)
        pits_full = detect_golden_pit_v2(c, reg_full)
        sig_full = [(s, b, lch) for s, b, lch in pits_full if lch is not None]
        cuts = np.sort(rng.choice(np.arange(int(n * 0.4), int(n * 0.95)), size=n_cuts, replace=False))
        for T in cuts:
            c_t = c[:T + 1]
            reg_t, _ = compute_rolling_regression(c_t, window=250, use_log=True)
            sig_t = set(detect_golden_pit_v2(c_t, reg_t))
            full_before = set((s, b, lch) for s, b, lch in sig_full if lch <= T)
            if full_before != sig_t:
                mismatch += 1
                print(f'  [不一致] {sym} @T={T}')
                print(f'    漏报: {sorted(full_before - sig_t)[:3]}')
                print(f'    新增: {sorted(sig_t - full_before)[:3]}')
            checked += 1
            checked_sig += len(full_before)
    return checked, checked_sig, mismatch

# ── 对比:原版 vs v2(信号量 + 胜率) ──
def compare(pool_file='stock_pool_1000.txt'):
    pool = [l.split(',')[0].strip() for l in open(pool_file, encoding='utf-8') if l.strip()]
    rows = {'old': [], 'v2': []}
    for k, sym in enumerate(pool):
        f = os.path.join('.cache_kline', f'{sym}.npy')
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True).item()
        c = d['close'].astype(float)
        n = len(c)
        reg, _ = compute_rolling_regression(c, window=250, use_log=True)
        pits_old = pr.detect_golden_pit(c, reg)
        pits_v2 = detect_golden_pit_v2(c, reg)
        for name, pits in (('old', pits_old), ('v2', pits_v2)):
            for s, b, lch in pits:
                if lch is None or lch - b > 5 or lch + 20 >= n:
                    continue  # 规则F: 快启动<=5
                rows[name].append(c[lch + 20] / c[lch] - 1)
        if (k + 1) % 300 == 0:
            print(f'  进度 {k+1}/{len(pool)}', flush=True)
    print(f'\n=== 对比(1000池, 规则F: 快启动<=5, 20日持有) ===')
    for name, R in rows.items():
        R = np.array(R)
        print(f'  {name:<4}: n={len(R):>5}  20日胜率={np.mean(R>=0):5.1%} 均值={np.mean(R):+6.1%}')

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='截断自检')
    ap.add_argument('--compare', action='store_true', help='原版vs v2对比')
    args = ap.parse_args()
    pool = [l.split(',')[0].strip() for l in open('stock_pool_1000.txt', encoding='utf-8') if l.strip()]
    if args.check:
        checked, checked_sig, mismatch = trunc_check(pool[:100], n_cuts=6)
        print(f'\n=== v2 截断自检(前100股) ===  {checked} 组 / {checked_sig} 信号 / 不一致 {mismatch}')
        print('结论:', '❌ 未来函数' if mismatch else '✅ 无未来函数')
    if args.compare:
        compare()
