# -*- coding: utf-8 -*-
"""600550 两口径主成本线对比: 旧V10(1023天,2016-01起) vs 新(1800天缓存,2019-05起)  (临时脚本, 不入库)"""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chip_panel import cost_series
from challenge_scan import ehlers_smooth
from tdx_quant import get_daily_kline_from_tdx

def cost_lines(hi, lo, V, turn):
    c70 = cost_series(hi, lo, V, turn, 0.70)
    c35 = cost_series(hi, lo, V, turn, 0.35)
    return ehlers_smooth(c70, 90), ehlers_smooth(c35, 90)

code, nd, end = (sys.argv[1], int(sys.argv[2]), sys.argv[3]) if len(sys.argv) > 3 else ('sh600550', 150, '20200510')
code6 = code[2:]
# 新口径: 信号缓存(1800天, 2019-05起), 截取 ≤end
dfn = pd.read_csv(f'result/scan_cache_v3b/{code6}.csv')
dfn['date'] = pd.to_datetime(dfn['date']).dt.tz_localize(None).dt.normalize()
dfn = dfn[dfn['date'] <= pd.Timestamp(end)].reset_index(drop=True)
ca_n, cp_n = cost_lines(dfn['high'].values.astype(float), dfn['low'].values.astype(float),
                         dfn['volume'].values.astype(float), dfn['turnover'].values.astype(float).clip(0, 1))
# 旧口径: tdx 1023天, 至 end
dfo = get_daily_kline_from_tdx(code, end, datalen=1023, with_turnover=True).sort_values('date').reset_index(drop=True)
ca_o, cp_o = cost_lines(dfo['high'].values.astype(float), dfo['low'].values.astype(float),
                         dfo['volume'].values.astype(float), dfo['turnover'].values.astype(float).clip(0, 1))
# 对齐到新口径窗口(缓存中 end 之前 nd 行)
i0 = max(0, len(dfn) - nd)
w = dfn.iloc[i0:].reset_index(drop=True)
cl = w['close'].values.astype(float); hi = w['high'].values.astype(float); lo = w['low'].values.astype(float)
op = w['open'].values.astype(float); V = w['volume'].values.astype(float); d = w['date'].dt.strftime('%Y%m%d').values
# 旧口径线对齐(用日期索引匹配)
dmap_o = {pd.Timestamp(x).strftime('%Y%m%d'): i for i, x in enumerate(dfo['date'])}
dmap_n = {pd.Timestamp(x).strftime('%Y%m%d'): i for i, x in enumerate(dfn['date'])}
ca_o_w = np.array([ca_o[dmap_o[x]] if x in dmap_o else np.nan for x in d])
cp_o_w = np.array([cp_o[dmap_o[x]] if x in dmap_o else np.nan for x in d])
ca_n_w = np.array([ca_n[dmap_n[x]] if x in dmap_n else np.nan for x in d])
cp_n_w = np.array([cp_n[dmap_n[x]] if x in dmap_n else np.nan for x in d])
# 画
fig, (ax, axv) = plt.subplots(2, 1, figsize=(16, 8.5), sharex=True, gridspec_kw={'height_ratios': [3.2, 1]})
x = np.arange(len(w))
for i in range(len(w)):
    c_ = '#d64545' if cl[i] >= op[i] else '#2e9e5b'
    ax.plot([i, i], [lo[i], hi[i]], color=c_, lw=0.6, zorder=2)
    ax.add_patch(plt.Rectangle((i - 0.3, min(cl[i], op[i])), 0.6, abs(cl[i] - op[i]) or 0.01, color=c_, zorder=3))
ax.plot(x, ca_o_w, color='#1b5e20', lw=1.2, alpha=0.55, ls='--', zorder=4, label='主成本A·旧(1023天/2016起)')
ax.plot(x, ca_n_w, color='#1b5e20', lw=2.0, alpha=1.0, zorder=5, label='主成本A·新(1800天/2019起)')
ax.plot(x, cp_o_w, color='#e53935', lw=1.2, alpha=0.55, ls='--', zorder=4, label='红压线·旧(1023天)')
ax.plot(x, cp_n_w, color='#e53935', lw=2.0, alpha=1.0, zorder=5, label='红压线·新(1800天)')
# 标注对比数值(窗口首/中/尾三日, 两口径按 end 对齐: 旧口径行 = 新口径行 + (len(dfo)-len(dfn)))
off = len(dfo) - len(dfn)
for tag, j in [('窗口首', 0), ('中段', len(w) // 2), ('窗口末', len(w) - 1)]:
    ax.annotate(f'{tag} {d[j][4:6]}-{d[j][6:8]}\n新CA={ca_n_w[j]:.2f} 旧CA={ca_o_w[j]:.2f}', (j, max(ca_n_w[j], ca_o_w[j])),
                textcoords='offset points', xytext=(-16, 10), fontsize=8, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='#fffbe6', ec='#8d6e63', alpha=0.9))
ax.set_title(f'{code} 主成本线两口径对比({nd}日,至{end})  旧(1023天虚线) vs 新(1800天缓存实线)', fontsize=12)
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.25, lw=0.4); ax.set_ylabel('Price')
axv.bar(x, V, color=['#d64545' if V[i] >= V[max(0, i - 5):i + 1].mean() else '#4a90d9' for i in range(len(w))], width=0.7)
axv.grid(alpha=0.25, lw=0.4)
tick = np.arange(0, len(w), max(1, len(w) // 12))
axv.set_xticks(tick); axv.set_xticklabels([d[i][:4] + '-' + d[i][4:6] for i in tick], fontsize=9)
out = f'result/compare_cost_{code6}_{end}.png'
plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()
print('已生成:', out)
