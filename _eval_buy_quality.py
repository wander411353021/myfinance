"""
买点质量评估（与卖出机制完全解耦）— V10 raw vs V11 不同过滤松紧度。

关键发现：V11 默认过滤(atr_mult=0.5, vol=1.2)实际放行了全部 BrkLvl，
买点与 V10 完全相同 → 过滤是空操作。本脚本扫描多档紧度，找真正提升买点质量的设置。

指标（每个 +1 买点，入场=close[t]，前看 H=20，R=1.5*ATR，目标 2R/止损 1R）:
  win%    = 窗口内先触 +2R 的比例（买点给到盈利机会）
  MFE%/MAE% = 最大有利/不利偏移
  expect% = win%*MFE% - (1-win%)*MAE%   （买点质量分）
  n / dens = 买点数量 / 每100根bar密度（越低越"挑剔"→越稳）
"""
import sys, os
sys.path.insert(0, r'E:\chip_analyzer_ui\new_algo')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

STOCKS = [
    ('000066', '20241228', 250),
    ('300437', '20210910', 150),
    ('688387', '20250915', 150),
]
HORIZON = 20
R_MULT = 1.5
TARGET_MULT = 2.0
STOP_MULT = 1.0

STRATS = [
    ('V10 raw',          dict(vol_confirm_mult=1.2, atr_mult=0.5)),   # 实为不过滤
    ('V11 loose',        dict(vol_confirm_mult=1.2, atr_mult=0.5)),
    ('V11 med',          dict(vol_confirm_mult=1.5, atr_mult=1.0)),
    ('V11 tight',        dict(vol_confirm_mult=2.0, atr_mult=1.0)),
]


def load(code, end):
    p = os.path.join(CACHE, f'{code}_{end}.pkl')
    if os.path.exists(p):
        return pd.read_pickle(p)
    df = get_stock_klines_from_juejing(code, end, fqt=1)
    df = df[df['close'] > 0].reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df.to_pickle(p)
    print(f"  fetched {code}: {len(df)} rows")
    return df


def eval_buys(bs_signal, close, high, low, atr):
    n = len(close)
    buys = np.where(bs_signal > 0)[0]
    rec = []
    for t in buys:
        if t + 1 >= n:
            continue
        entry = close[t]; R = R_MULT * atr[t]
        tgt = entry + TARGET_MULT * R; stp = entry - STOP_MULT * R
        mfe = 0.0; mae = 0.0; outcome = None
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
        return dict(n=0, winrate=0.0, avg_mfe=0.0, avg_mae=0.0, expect=0.0, density=0.0)
    wins = sum(1 for r in rec if r[0] == 'WIN'); n_ = len(rec)
    mfe = float(np.mean([r[1] for r in rec])); mae = float(np.mean([r[2] for r in rec]))
    wr = wins / n_
    return dict(n=n_, winrate=wr, avg_mfe=mfe, avg_mae=mae,
                expect=wr * mfe - (1 - wr) * mae, density=n_ / n * 100.0)


def plot(rows):
    codes = [r[0] for r in rows]
    strats = [s[0] for s in STRATS]
    nst = len(strats); x = np.arange(len(codes)); w = 0.8 / nst
    colors = ['#BBDEFB', '#90CAF9', '#42A5F5', '#1565C0']
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for si, sname in enumerate(strats):
        wr = [next(d for c, d in r[1] if c == sname)['winrate'] * 100 for r in rows]
        dn = [next(d for c, d in r[1] if c == sname)['density'] for r in rows]
        ex = [next(d for c, d in r[1] if c == sname)['expect'] * 100 for r in rows]
        off = (si - (nst - 1) / 2) * w
        axes[0].bar(x + off, wr, w, label=sname, color=colors[si])
        axes[1].bar(x + off, dn, w, label=sname, color=colors[si])
        axes[2].bar(x + off, ex, w, label=sname, color=colors[si])
    for ax, title, ylab in [
        (axes[0], 'Buy win rate % (entry quality)', 'win %'),
        (axes[1], 'Buy density (per 100 bars, lower=more selective)', 'count/100'),
        (axes[2], 'Buy expectancy % (win%*MFE - loss%*MAE)', 'expect %')]:
        ax.set_xticks(x); ax.set_xticklabels(codes)
        ax.set_title(title); ax.set_ylabel(ylab); ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
    axes[2].axhline(0, color='gray', lw=0.7)
    plt.tight_layout()
    save = os.path.join(RESULT, 'buy_quality_sweep.png')
    plt.savefig(save, dpi=130, bbox_inches='tight'); plt.close()
    return save


def main():
    print(f"HORIZON={HORIZON} R={R_MULT}*ATR target={TARGET_MULT}R stop={STOP_MULT}R")
    print(f"{'code':<9}{'strat':<11}{'buys':>6}{'win%':>7}{'MFE%':>8}{'MAE%':>8}{'expect%':>9}{'dens/100':>10}")
    rows = []
    for code, end, tail in STOCKS:
        df = load(code, end)
        close = df['close'].values; high = df['high'].values; low = df['low'].values
        atr = _compute_atr(high, low, close, period=14)
        r10 = Seg10().segment(close, df['volume'].values, high=high, low=low, opn=df['open'].values)
        r11 = Seg11().segment(close, df['volume'].values, high=high, low=low, opn=df['open'].values)
        per = []
        for sname, p in STRATS:
            if sname == 'V10 raw':
                bs, _, _, _ = cbs10(df, r10)
            else:
                bs, _ = cbs11(df, r11, vol_confirm_mult=p['vol_confirm_mult'], atr_mult=p['atr_mult'])
            m = eval_buys(bs, close, high, low, atr)
            per.append((sname, m))
            print(f"{code:<9}{sname:<11}{m['n']:>6}{m['winrate']*100:>6.1f}%{m['avg_mfe']*100:>7.2f}%{m['avg_mae']*100:>7.2f}%{m['expect']*100:>8.2f}%{m['density']:>9.2f}")
        rows.append((code, per))
    save = plot(rows)
    print(f"\nchart -> {save}")


if __name__ == '__main__':
    main()
