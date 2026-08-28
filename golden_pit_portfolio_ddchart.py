# -*- coding: utf-8 -*-
"""黄金坑 降回撤方案对比 资金曲线（基线 vs 止损-8%，口径B子账户复利，严格口径次日买入）

用法: python3 golden_pit_portfolio_ddchart.py
输出: golden_pit_portfolio_ddcurve.png
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden_pit_portfolio_sim as sim

for f in ['Noto Serif CJK SC', 'Noto Sans CJK SC', 'SimHei', 'AR PL UMing CN']:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [f]
        plt.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue

stocks = [l.strip().split(',')[0] for l in open(os.path.join(sim.WORKDIR, 'stock_pool_1000.txt')) if l.strip()]

r_base = sim.run_sim(stocks, buy_delay=1)
r_sl = sim.run_sim(stocks, buy_delay=1, stop_loss=0.08)

sn_base = r_base['snapshot'].set_index('date')['asset'] / 1e6
sn_sl = r_sl['snapshot'].set_index('date')['asset'] / 1e6
# 沪深300 归一化到 1（百万口径便于同图）
idx = sim.load('sh000300')
ts300 = pd.to_datetime(idx['ts'].astype('int64'), unit='s')
c300 = idx['close'].astype(float)
hs = pd.Series(np.asarray(c300), index=ts300)
hs = hs[hs.index >= '2023-01-01']
hs = hs / float(hs.iloc[0])  # 起始=1 表示100万

plt.figure(figsize=(12, 6.5))
plt.plot(sn_base.index, sn_base.values, color='#888', lw=1.4, label=f'基线(无止损)  +{r_base["total_ret"]*100:.0f}%  回撤{r_base["max_dd"]*100:.1f}%')
plt.plot(sn_sl.index, sn_sl.values, color='#c0392b', lw=1.8, label=f'止损-8%  +{r_sl["total_ret"]*100:.0f}%  回撤{r_sl["max_dd"]*100:.1f}%')
plt.plot(hs.index, hs.values, color='#888888', lw=1.2, ls='--', label='沪深300 (归一化)')

# 年度分界线
for y in [2023, 2024, 2025, 2026]:
    plt.axvline(pd.Timestamp(f'{y}-01-01'), color='#dddddd', lw=0.8, zorder=0)
plt.text(pd.Timestamp('2023-07-01'), 0.1, '2023', color='#999', fontsize=9)
plt.text(pd.Timestamp('2024-07-01'), 0.1, '2024', color='#999', fontsize=9)
plt.text(pd.Timestamp('2025-07-01'), 0.1, '2025', color='#999', fontsize=9)
plt.text(pd.Timestamp('2026-07-01'), 0.1, '2026', color='#999', fontsize=9)

plt.title('黄金坑 降回撤方案对比（100万/10份复利/严格口径次日买入）')
plt.ylabel('资产（百万元）')
plt.legend(loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('golden_pit_portfolio_ddcurve.png', dpi=140)
print('已保存 golden_pit_portfolio_ddcurve.png')
print(f'基线: +{r_base["total_ret"]*100:.1f}% 回撤{r_base["max_dd"]*100:.1f}% 胜率{r_base["win_rate"]:.1f}%')
print(f'止损-8%: +{r_sl["total_ret"]*100:.1f}% 回撤{r_sl["max_dd"]*100:.1f}% 胜率{r_sl["win_rate"]:.1f}%')
