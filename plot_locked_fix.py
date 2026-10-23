# -*- coding: utf-8 -*-
"""盘踞修正探索画图: 原始线 vs 修正线 + 锁定色带 + 修正后信号 (临时脚本, 不入库)"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from challenge_scan import compute_eff, early_cB, scan_series
from algo_config import SIGNAL_REARM_DAYS
from chip_panel import cost_series

def norm(d):
    s = str(d); return s.split(' ')[0].replace('-', '') if ' ' in s else str(d).zfill(8)

def main(code, ndays, end):
    df = pd.read_csv(f'result/scan_cache_v3b/{code}.csv')
    df['date'] = df['date'].apply(norm)
    endpos = np.searchsorted(df['date'].values, end, 'right') - 1
    i0 = max(0, endpos - ndays + 1)
    sub = df.iloc[i0:endpos + 1].reset_index(drop=True)
    cl = sub['close'].values.astype(float); hi = sub['high'].values.astype(float)
    lo = sub['low'].values.astype(float); V = sub['volume'].values.astype(float)
    turn = sub['turnover'].values.astype(float); d = sub['date'].values
    n = len(sub)
    # 全量计算修正(需完整序列)
    dfull = df.reset_index(drop=True)
    clf = dfull['close'].values.astype(float); hif = dfull['high'].values.astype(float)
    lof = dfull['low'].values.astype(float); Vf = dfull['volume'].values.astype(float)
    tf = dfull['turnover'].values.astype(float)
    _, _, _, _, gtp_e, gt2_e, _, _, cA = compute_eff(dfull)
    c90 = cost_series(hif, lof, Vf, tf, 0.90)
    prem = (c90 / clf - 1) * 100
    N, PREM90, RELOOK, BUF, CA_F, REF_F = 40, 15.0, 40, 0.02, 0.95, 0.90
    runhi = np.full(len(clf), np.nan)
    for i in range(N - 1, len(clf)):
        runhi[i] = np.max(clf[i - N + 1:i + 1])
    locked = np.zeros(len(clf), dtype=bool); cnt = 0
    for i in range(len(clf)):
        cnt = cnt + 1 if np.isfinite(prem[i]) and prem[i] > PREM90 else 0
        locked[i] = cnt >= RELOOK
    gtp_new = gtp_e.copy(); cA_new = cA.copy(); refb_new = gt2_e.copy()
    for i in range(len(clf)):
        if locked[i] and np.isfinite(runhi[i]):
            gtp_new[i] = min(gtp_e[i], runhi[i] * (1 + BUF))
            cA_new[i] = min(cA[i], runhi[i] * CA_F)
            refb_new[i] = min(gt2_e[i], runhi[i] * REF_F)
    # 窗口切片
    o = i0
    clw = cl; hiw = hi; loww = lo
    gp_o = gtp_e[o:o + n]; gp_m = gtp_new[o:o + n]
    ca_o = cA[o:o + n]; ca_m = cA_new[o:o + n]
    rb_m = refb_new[o:o + n]
    lock_w = locked[o:o + n]
    # 修正后信号
    ss = scan_series(clf, np.full(len(clf), np.nan), np.full(len(clf), np.nan), np.full(len(clf), np.nan),
                     Vf, types=('v3', 'v3b', 'v3b_b2_early', 'early'), mode='structure',
                     darr=dfull['date'].values, gt_p_eff=gtp_new, gt2_eff=refb_new,
                     cA=cA_new, cB=early_cB(hif, lof, Vf, tf), rearm_days=SIGNAL_REARM_DAYS)
    sigs = [(t, idx) for t, idx in ss if i0 <= idx <= endpos]
    # 画图
    fig, (ax, axv) = plt.subplots(2, 1, figsize=(16, 8.5), sharex=True,
                                  gridspec_kw={'height_ratios': [3.2, 1]})
    x = np.arange(n)
    # K线
    for i in range(n):
        c_ = '#d64545' if clw[i] >= sub['open'].values[i] else '#2e9e5b'
        ax.plot([i, i], [loww[i], hiw[i]], color=c_, lw=0.6, zorder=2)
        ax.add_patch(plt.Rectangle((i - 0.3, min(clw[i], sub['open'].values[i])), 0.6,
                                   abs(clw[i] - sub['open'].values[i]) or 0.01, color=c_, zorder=3))
    # 锁定色带
    lock_x = np.where(lock_w, x, np.nan)
    if np.any(np.isfinite(lock_x)):
        ax.fill_between(x, np.nanmin(loww), np.nanmax(hiw), where=lock_w,
                        color='#9e9e9e', alpha=0.18, zorder=1, label='盘踞锁定段(COST90溢价>15%连续40天)')
    # 线: 原始虚线 + 修正实线
    ax.plot(x, gp_o, color='#e53935', lw=1.0, alpha=0.45, ls='--', zorder=4, label='红压线·原始(COST35/E90)')
    ax.plot(x, gp_m, color='#e53935', lw=1.6, alpha=0.95, zorder=5, label='红压线·修正(下放)')
    ax.plot(x, ca_o, color='#1b5e20', lw=1.0, alpha=0.45, ls='--', zorder=4, label='主成本A·原始(COST70/E90)')
    ax.plot(x, ca_m, color='#1b5e20', lw=1.6, alpha=0.95, zorder=5, label='主成本A·修正(下放)')
    ax.plot(x, rb_m, color='#00838f', lw=1.3, alpha=0.9, ls='-.', zorder=4, label='参考线B·修正(COST40/E50)')
    # 修正后信号
    colors = {'v3': '#2e7d32', 'v3b': '#e53935', 'v3b_b2_early': '#1565c0', 'early': '#ef6c00'}
    for t, idx in sigs:
        j = idx - i0
        if 0 <= j < n:
            ax.scatter(j, clw[j], marker='^' if t != 'early' else 'v', s=90,
                       color=colors.get(t, '#000'), zorder=6)
            ax.annotate(f'{t}\n{d[idx][4:]}', (j, clw[j]), textcoords='offset points',
                        xytext=(0, 12 if t != 'early' else -18), fontsize=8, ha='center',
                        color=colors.get(t, '#000'))
    ax.set_title(f'{code} 盘踞修正探索({ndays}日,至{end})  原始线(虚线) vs 修正线(实线)+锁定色带', fontsize=12)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_ylabel('Price')
    # 量
    colors_v = ['#d64545' if V[i] >= V[max(0, i - 5):i + 1].mean() else '#4a90d9' for i in range(n)]
    axv.bar(x, V, color=colors_v, width=0.7)
    axv.set_ylabel('Volume')
    axv.grid(alpha=0.25, lw=0.4)
    tick = np.arange(0, n, max(1, n // 12))
    axv.set_xticks(tick)
    axv.set_xticklabels([d[i][:4] + '-' + d[i][4:6] for i in tick], fontsize=9)
    out = f'result/locked_fix_{code}_{end}.png'
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()
    print('已生成:', out, ' 修正后信号数:', len(sigs))

if __name__ == '__main__':
    code, nd, end = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    main(code, nd, end)
