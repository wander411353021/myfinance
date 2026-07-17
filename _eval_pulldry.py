"""
PullDry(缩量回调止跌) 买点专项评估 — 与卖出机制解耦。

把 V11 的买点拆成 PullDry vs 其他(BrkLvl/BrkRes/PullSup...)，分别用
入场质量分析(MAE/MFE, 前看20根, R=1.5*ATR, 目标2R/止损1R)评估胜率与期望。
并在 000066 图上标出 PullDry 触发点，验证是否落在 2024-10-17 附近的缩量下蹲反转处。
"""
import sys, os
sys.path.insert(0, r'E:\chip_analyzer_ui\new_algo')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from price_segmenter_v10 import (CausalIncrementalPriceSegmenter as Seg10,
                                  compute_buy_sell_signals as cbs10)
from price_segmenter_v11 import (CausalIncrementalPriceSegmenter as Seg11,
                                  compute_buy_sell_signals as cbs11,
                                  _compute_atr)
from juejing import get_stock_klines_from_juejing

WORK = r'E:\chip_analyzer_ui\new_algo'
RESULT = os.path.join(WORK, 'result')
CACHE = os.path.join(RESULT, '_cache')
os.makedirs(CACHE, exist_ok=True)
STOCKS = [('000066', '20241228', 250), ('300437', '20210910', 150), ('688387', '20250915', 150)]
HORIZON, R_MULT, TARGET_MULT, STOP_MULT = 20, 1.5, 2.0, 1.0


def load(code, end):
    p = os.path.join(CACHE, f'{code}_{end}.pkl')
    if os.path.exists(p):
        return pd.read_pickle(p)
    df = get_stock_klines_from_juejing(code, end, fqt=1)
    df = df[df['close'] > 0].reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df.to_pickle(p)
    return df


def eval_mask(bs_signal, mask, close, high, low, atr):
    n = len(close); buys = np.where((bs_signal > 0) & mask)[0]
    rec = []
    for t in buys:
        if t + 1 >= n:
            continue
        entry = close[t]; R = R_MULT * atr[t]
        tgt = entry + TARGET_MULT * R; stp = entry - STOP_MULT * R
        mfe = mae = 0.0; outcome = None
        end = min(t + 1 + HORIZON, n)
        for k in range(t + 1, end):
            mfe = max(mfe, (high[k] - entry) / entry)
            mae = max(mae, (entry - low[k]) / entry)
            if low[k] <= stp:
                outcome = 'LOSS'; break
            if high[k] >= tgt:
                outcome = 'WIN'; break
        if outcome is None:
            outcome = 'WIN' if (close[end - 1] - entry) / entry > 0 else 'LOSS'
        rec.append((outcome, mfe, mae))
    if not rec:
        return dict(n=0, winrate=0.0, avg_mfe=0.0, avg_mae=0.0, expect=0.0)
    wins = sum(1 for r in rec if r[0] == 'WIN'); n_ = len(rec)
    mfe = float(np.mean([r[1] for r in rec])); mae = float(np.mean([r[2] for r in rec]))
    wr = wins / n_
    return dict(n=n_, winrate=wr, avg_mfe=mfe, avg_mae=mae, expect=wr * mfe - (1 - wr) * mae)


def fmt(m):
    if m['n'] == 0:
        return f"{'0':>4}  {'-':>6}  {'-':>6}  {'-':>6}  {'-':>6}"
    return (f"{m['n']:>4}  {m['winrate']*100:>5.1f}%  {m['avg_mfe']*100:>5.1f}%  "
            f"{m['avg_mae']*100:>5.1f}%  {m['expect']*100:>5.1f}%")


def plot_000066(df, bs11, reason11, bs10):
    d = df.copy(); d['date'] = pd.to_datetime(d['date'])
    m = (d['date'] >= '2024-09-15') & (d['date'] <= '2024-11-20')
    d = d[m].reset_index(drop=True)
    x = np.arange(len(d))
    fig, ax = plt.subplots(2, 1, figsize=(20, 9), height_ratios=[4, 1], sharex=True)
    ax[0].plot(x, d['close'], color='#2C2C2A', lw=1.1, label='close')
    # map original index -> display index
    disp_idx = np.where(m.values if hasattr(m, 'values') else m)[0]
    idx_map = {int(orig): di for di, orig in enumerate(disp_idx)}
    for t in np.where(bs11 > 0)[0]:
        if t in idx_map:
            di = idx_map[t]
            if reason11[t].startswith('PullDry'):
                ax[0].plot(di, d['close'].iloc[di], '^', ms=11, color='#2E7D32', zorder=6, label='PullDry')
            else:
                ax[0].plot(di, d['close'].iloc[di], '^', ms=7, color='#1565C0', zorder=5, label='BrkLvl/other')
    for t in np.where(bs11 < 0)[0]:
        if t in idx_map:
            ax[0].plot(idx_map[t], d['close'].iloc[idx_map[t]], 'v', ms=6, color='#C62828', zorder=5)
    ax[0].set_title('000066  PullDry (green) vs other buys (blue) — 2024-09~11', fontsize=12, fontweight='bold')
    ax[0].legend(loc='upper left', fontsize=8); ax[0].grid(True, alpha=0.25)
    # volume
    ax[1].bar(x, d['volume'], color='#90A4AE', width=0.8)
    ax[1].set_ylabel('volume'); ax[1].grid(True, alpha=0.25)
    # mark 10-17
    for di, dt in enumerate(d['date']):
        if dt.strftime('%Y-%m-%d') in ('2024-10-17', '2024-10-08', '2024-10-23'):
            ax[0].axvline(di, color='gray', ls='--', lw=0.8, alpha=0.6)
            ax[0].text(di, d['close'].iloc[di], f' {dt.strftime("%m-%d")}', fontsize=7, color='gray')
    plt.tight_layout()
    save = os.path.join(RESULT, 'pulldry_000066.png')
    plt.savefig(save, dpi=130, bbox_inches='tight'); plt.close()
    return save


def main():
    print(f"{'code':<9}{'group':<16}{'n':>4}{'win%':>7}{'MFE%':>7}{'MAE%':>7}{'expect%':>8}")
    pd_dates_000066 = None
    for code, end, tail in STOCKS:
        df = load(code, end)
        close = df['close'].values; high = df['high'].values; low = df['low'].values
        atr = _compute_atr(high, low, close, period=14)
        # V10
        r10 = Seg10().segment(close, df['volume'].values, high=high, low=low, opn=df['open'].values)
        bs10, _, _, _ = cbs10(df, r10)
        # V11 (含 PullDry)
        r11 = Seg11().segment(close, df['volume'].values, high=high, low=low, opn=df['open'].values)
        bs11, reason11 = cbs11(df, r11)
        is_pd = np.array([r.startswith('PullDry') for r in reason11])
        m_v10 = eval_mask(bs10, np.ones(len(bs10), bool), close, high, low, atr)
        m_all = eval_mask(bs11, np.ones(len(bs11), bool), close, high, low, atr)
        m_pd = eval_mask(bs11, is_pd, close, high, low, atr)
        m_ot = eval_mask(bs11, ~is_pd, close, high, low, atr)
        print(f"{code:<9}{'V10 all':<16}{fmt(m_v10)}")
        print(f"{'':<9}{'V11 all':<16}{fmt(m_all)}")
        print(f"{'':<9}{'V11 PullDry':<16}{fmt(m_pd)}")
        print(f"{'':<9}{'V11 other':<16}{fmt(m_ot)}")
        if code == '000066':
            pd_dates = [df['date'].iloc[t].strftime('%Y-%m-%d') for t in np.where((bs11 > 0) & is_pd)[0]]
            print(f"  000066 PullDry 触发日: {pd_dates}")
            save = plot_000066(df, bs11, reason11, bs10)
            print(f"  chart -> {save}")
    print("\n(指标: 入场=close[t], 前看20根, R=1.5*ATR, 目标2R/止损1R; win=先触+2R)")


if __name__ == '__main__':
    main()
