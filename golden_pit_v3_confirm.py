# -*- coding: utf-8 -*-
"""v3 测试: 出坑确认机制(反弹后 N 日不跌回坑内才算真出坑, 否则大坑延续)。

因果性: 确认完成日 = 出坑日(确认需 <=lch 数据), 截断到 lch 可复现。
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression

def detect_v3(closes, reg250, z_thr=-1.5, merge_gap=15, launch_gate=0.9,
              use_pre_std=True, require_below_gate=False, confirm_days=5):
    closes = np.asarray(closes, dtype=float)
    reg250 = np.asarray(reg250, dtype=float)
    n = len(closes)
    resid = closes - reg250
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
        if not (np.isfinite(z_roll[i]) and z_roll[i] < z_thr):
            i += 1
            continue
        pre_std = np.std(resid[i - 60:i]) if use_pre_std and i >= 60 else rstd[i]
        if pre_std <= 0:
            pre_std = rstd[i]
        s = i
        b = i
        bv = closes[i]
        lch = None
        leave = 0
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
                # 出坑候选: 收复门控线 + 高于已知最低 2%
                if closes[j] >= gate_line[j] and closes[j] > bv * 1.02:
                    # 确认: 未来 confirm_days 日不跌回坑内(z<-1.5)
                    confirmed = True
                    kk = 0
                    for k in range(1, confirm_days + 1):
                        if j + k >= n:
                            confirmed = False
                            break
                        zk = resid[j + k] / pre_std if pre_std > 0 else 0.0
                        if np.isfinite(zk) and zk < z_thr:
                            confirmed = False
                            kk = k
                            break
                    if confirmed:
                        lch = j + confirm_days  # 确认完成日(需 <=lch 数据,因果)
                        break
                    else:
                        # 假出坑: 坑延续, 跳到跌回日
                        if kk > 0:
                            j += kk
                            continue
                        j += 1
                        continue
                leave += 1
                if closes[j] < bv:
                    bv = closes[j]; b = j
                if leave > merge_gap:
                    break
            j += 1
        if lch is not None:
            out.append((s, b, lch))
            i = lch + 1
        else:
            i = j + 1
    return out

def main():
    # 1) 688099 @20251001
    df = pr._load_df('sh688099', '20251001', datalen=800)
    c = df['close'].values.astype(float)
    dates = df['date'].dt.strftime('%Y-%m-%d').values
    reg, _ = compute_rolling_regression(c, window=250, use_log=True)
    print(f'\n=== 688099 @20251001 v3(confirm_days=5) ===')
    for s, b, lch in detect_v3(c, reg):
        if lch is not None:
            print(f'  s={dates[s]} b={dates[b]} lch={dates[lch]} len={b-s+1} 快启动={lch-b}')

    # 2) 全池信号量 + 胜率(规则F: 快启动<=5)
    pool = [l.split(',')[0].strip() for l in open('stock_pool_1000.txt', encoding='utf-8') if l.strip()]
    for cd in (0, 3, 5, 8):
        rows = []
        for k, sym in enumerate(pool):
            f = os.path.join('.cache_kline', f'{sym}.npy')
            if not os.path.exists(f): continue
            d = np.load(f, allow_pickle=True).item()
            cc = d['close'].astype(float); nn = len(cc)
            rg, _ = compute_rolling_regression(cc, window=250, use_log=True)
            fn = pr.detect_golden_pit_v2 if cd == 0 else lambda x, y: detect_v3(x, y, confirm_days=cd)
            for s, b, lch in fn(cc, rg):
                if lch is None or lch + 20 >= nn: continue
                rows.append(cc[lch+20]/cc[lch]-1)
            if (k+1) % 500 == 0: print(f'  confirm={cd} 进度 {k+1}', flush=True)
        R = np.array(rows)
        print(f'  confirm_days={cd}: n={len(R)}  20日胜率={np.mean(R>=0):.1%} 均值={np.mean(R):+.1%}')

if __name__ == '__main__':
    main()
