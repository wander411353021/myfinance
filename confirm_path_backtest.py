# -*- coding: utf-8 -*-
"""
首选策略严格无未来函数路径回测（dist_high>30 × 动态确认：放量克服前堆高 + 20日大阳）。

口径（全部从仓库已有脚本确认，直接复用）:
  big20/big_wait : backtest_v3b_5y._bigline_post  —— 信号日后 j∈[i+1,i+20] 首根
                   close[j]/close[j-1]-1 ∈ [0.05,0.21] 的交易日；big_wait=j-i (1-based)
  overcome/overcome_wait : overcome_analysis.py —— 前放量堆=信号日前250日窗口最后一个连续
                   放量日区间(vol>1.5×前60日均量, 不含当日)最高价 vol_high;
                   首个 j≥i+1 使 close[j]>vol_high, overcome_wait=j-i (1-based)
  主成本线: fengwo.COST(...,0.75) 逐日递推 + Ehlers SuperSmoother(period=25), 因果无未来
  离场: 买入日后逐日 close < 0.90×成本线 → 当日收盘卖出; 否则 T+H 到期收盘卖出

无未来函数保证:
  每只票一次性读全量缓存(只读≤i的数据参与第i天判定); 成本线/均量/前堆高均为因果递推;
  确认日判定在买入日之前完成; 离场逐日只看当日及之前。

输出: result/confirm_trades.csv (逐笔, 含两种模式) + 控制台统计 JSON 供报告使用。
用法: python3 confirm_path_backtest.py
"""
import os
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool

CACHE = 'result/scan_cache_v3b'
SIG_CSV = 'result/backtest_v3b_5y_overcome.csv'
OUT_TRADES = 'result/confirm_trades.csv'

BIG_PCT = 0.05
BIG_PCT_MAX = 0.21
TH_VOL = 1.5          # 放量阈值 (overcome_analysis 全局 TH)
COST_PCT = 0.75       # COST75
EHLERS_P = 25         # 主成本线 Ehlers 周期
EXIT_K = 0.90         # 跌破 0.9×成本线离场
H_MAIN = 60           # 主持有 horizon (从信号日 T 起算)


def ehlers_smooth(x, period):
    """Ehlers SuperSmoother 二阶IIR低通(因果, 第k天只用<=k数据) —— 与 challenge_scan.py 逐行一致。"""
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


def process_group(args):
    """单只股票: 读缓存→算因果成本线/均量→逐信号重算确认日并跑两种模式交易。"""
    code, rows = args
    p = os.path.join(CACHE, f'{code}.csv')
    if not os.path.exists(p):
        return [], dict(miss_file=len(rows))
    df = pd.read_csv(p)
    n = len(df)
    if n < 260:
        return [], dict(miss_short=len(rows))
    dates = np.array([_bd(d) for d in df['date'].values])
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    vol = df['volume'].values.astype(float)
    turn = df['turnover'].values.astype(float).clip(0, 1)

    # 因果因子(整段一次性递推, 第i天只用<=i数据)
    vol_ma60 = pd.Series(vol).shift(1).rolling(60, min_periods=60).mean().values
    try:
        cost_raw = __import__('fengwo').COST(high, low, vol, turn,
                                             np.full(n, COST_PCT))
        cost_line = ehlers_smooth(np.asarray(cost_raw, dtype=float), EHLERS_P)
    except Exception:
        cost_line = np.full(n, np.nan)

    trades = []
    mm = dict(big_mis=0, ov_mis=0, n_sig=0, n_big=0, n_ov=0)
    for r in rows:
        sd = str(int(r['sig_date']))
        i = int(np.searchsorted(dates, sd))
        if i >= n or dates[i] != sd:
            continue
        mm['n_sig'] += 1
        sig_close = close[i]
        avail = n - 1 - i  # 信号日后实际可用交易日数

        # ---- 重算前放量堆 vol_high (窗口 [i-250, i), 只用≤i-1数据) ----
        w0 = max(0, i - 250)
        w1 = i
        win_vol = vol[w0:w1]
        win_ma = vol_ma60[w0:w1]
        valid = ~np.isnan(win_ma)
        big = np.zeros(w1 - w0, dtype=bool)
        big[valid] = win_vol[valid] > TH_VOL * win_ma[valid]
        if not big.any():
            continue  # 无放量堆, 理论上 overcome60==1 不会出现
        last_rel = int(np.where(big)[0].max())
        run_end = last_rel
        run_start = last_rel
        while run_start - 1 >= 0 and big[run_start - 1]:
            run_start -= 1
        vol_high = high[w0 + run_start:w0 + run_end + 1].max()

        # ---- 重算克服日: 首个 j≥i+1 close[j]>vol_high (1-based 偏移) ----
        ov_off = np.nan
        for k in range(1, avail + 1):
            if close[i + k] > vol_high:
                ov_off = float(k)
                break
        # ---- 重算大阳日: j∈[i+1, i+20] 涨幅∈[5%,21%] (1-based 偏移) ----
        big_off = np.nan
        for k in range(1, min(21, avail + 1)):
            g = close[i + k] / close[i + k - 1] - 1.0
            if np.isfinite(g) and BIG_PCT <= g <= BIG_PCT_MAX:
                big_off = float(k)
                break

        # 与 CSV 原列核对(只统计, 不改写)
        csv_bw = r.get('big_wait', np.nan)
        csv_ow = r.get('overcome_wait', np.nan)
        if np.isfinite(big_off) and np.isfinite(csv_bw) and abs(big_off - csv_bw) > 1e-9:
            mm['big_mis'] += 1
        if np.isfinite(ov_off) and np.isfinite(csv_ow) and abs(ov_off - csv_ow) > 1e-9:
            mm['ov_mis'] += 1
        if np.isfinite(big_off):
            mm['n_big'] += 1
        if np.isfinite(ov_off):
            mm['n_ov'] += 1

        base = dict(code=code, sym=r['sym'], sig_date=sd, sig_type=r['sig_type'],
                    year=sd[:4], qfq_flag=int(r['qfq_flag']),
                    dist_high=r['dist_high'], sig_close=round(sig_close, 4),
                    big_off=big_off, ov_off=ov_off, avail_days=avail,
                    vol_high=round(vol_high, 4))

        # ============ 模式A: 确认后买入(最严格) ============
        # 确认日 c = min(大阳偏移, 克服偏移); 买入 = 确认日次日收盘
        if np.isfinite(big_off) and np.isfinite(ov_off):
            c_off = min(big_off, ov_off)
            buy_off_A = c_off + 1.0
            if buy_off_A <= H_MAIN and i + int(buy_off_A) < n:
                buy_idx = i + int(buy_off_A)
                buy_px = close[buy_idx]
                # 离场扫描: buy_idx+1 .. min(i+H_MAIN, n-1)
                j0 = buy_idx + 1
                j1 = min(i + H_MAIN, n - 1)
                exit_off = np.nan
                exit_px = np.nan
                reason = ''
                for j in range(j0, j1 + 1):
                    if not np.isfinite(cost_line[j]):
                        continue
                    if close[j] < EXIT_K * cost_line[j]:
                        exit_off = float(j - i)
                        exit_px = close[j]
                        reason = 'cost_break'
                        break
                if not reason:
                    if avail >= H_MAIN:
                        exit_off = float(H_MAIN)
                    else:
                        exit_off = float(avail)  # 数据末尾, 敏感性统计时剔除
                    exit_px = close[i + int(exit_off)]
                    reason = 'timeout' if avail >= H_MAIN else 'data_end'
                trades.append(dict(base, mode='A_confirm',
                                   confirm_off=c_off, buy_off=buy_off_A,
                                   buy_price=round(buy_px, 4),
                                   sell_off=exit_off,
                                   sell_price=round(exit_px, 4),
                                   holding_days=int(exit_off - buy_off_A),
                                   ret=round(exit_px / buy_px - 1.0, 6),
                                   exit_reason=reason, traded=1))
            else:
                # 买入日超出 T+60 → 未成交
                trades.append(dict(base, mode='A_confirm',
                                   confirm_off=c_off, buy_off=buy_off_A,
                                   buy_price=np.nan, sell_off=np.nan,
                                   sell_price=np.nan, holding_days=np.nan,
                                   ret=np.nan, exit_reason='no_trade_beyond_60',
                                   traded=0))

        # ============ 模式B: 信号日买入对照 ============
        buy_idx_b = i
        buy_px_b = close[i]
        j0 = buy_idx_b + 1
        j1 = min(i + H_MAIN, n - 1)
        exit_off = np.nan
        exit_px = np.nan
        reason = ''
        for j in range(j0, j1 + 1):
            if not np.isfinite(cost_line[j]):
                continue
            if close[j] < EXIT_K * cost_line[j]:
                exit_off = float(j - i)
                exit_px = close[j]
                reason = 'cost_break'
                break
        if not reason:
            if avail >= H_MAIN:
                exit_off = float(H_MAIN)
            else:
                exit_off = float(avail)
            exit_px = close[i + int(exit_off)]
            reason = 'timeout' if avail >= H_MAIN else 'data_end'
        trades.append(dict(base, mode='B_sigbuy',
                            confirm_off=np.nan, buy_off=0.0,
                            buy_price=round(buy_px_b, 4),
                            sell_off=exit_off,
                            sell_price=round(exit_px, 4),
                            holding_days=int(exit_off),
                            ret=round(exit_px / buy_px_b - 1.0, 6),
                            exit_reason=reason, traded=1))

    return trades, mm


def main():
    df = pd.read_csv(SIG_CSV, dtype={'code': str})
    df['code'] = df.code.str.zfill(6)
    # 处理总体: 两种确认都在窗口内 + 有足够后续数据 + dist_high>30 + 两类信号
    pop = ((df.big20 == 1) & (df.overcome60 == 1.0) & (df.fut60_days >= 60)
           & (df.dist_high > 30) & (df.sig_type.isin(['v3b', 'early'])))
    pop_df = df[pop].copy()
    print(f'处理总体(重算+回测): {len(pop_df)} 条信号, codes={pop_df.code.nunique()}')

    groups = [(c, g[['sig_date', 'sym', 'sig_type', 'qfq_flag', 'dist_high',
                     'big_wait', 'overcome_wait']].to_dict('records'))
              for c, g in pop_df.groupby('code')]

    all_trades = []
    agg = dict(n_sig=0, big_mis=0, ov_mis=0, n_big=0, n_ov=0,
               miss_file=0, miss_short=0)
    with Pool(12) as pool:
        for trades, mm in pool.imap_unordered(process_group, groups, chunksize=16):
            all_trades.extend(trades)
            for k in agg:
                agg[k] += mm.get(k, 0)

    out = pd.DataFrame(all_trades)
    out.to_csv(OUT_TRADES, index=False)
    print(f'wrote {OUT_TRADES} rows={len(out)}')
    print('核对统计:', json.dumps(agg, ensure_ascii=False))
    # 重算一致性核对
    ntot = agg['n_sig']
    if ntot:
        print(f"big 重算命中={agg['n_big']}/{ntot} ({agg['n_big']/ntot*100:.2f}%), "
              f"不一致={agg['big_mis']} ({agg['big_mis']/ntot*100:.3f}%)")
        print(f"ov  重算命中={agg['n_ov']}/{ntot} ({agg['n_ov']/ntot*100:.2f}%), "
              f"不一致={agg['ov_mis']} ({agg['ov_mis']/ntot*100:.3f}%)")


if __name__ == '__main__':
    main()
