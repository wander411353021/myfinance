# -*- coding: utf-8 -*-
"""生成v1 vs v2优化对比可视化图表"""
import sys
sys.path.insert(0, '.')
from golden_pit_v2_backtest import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 获取所有回测数据
all_v1 = []
all_v2 = []
for symbol, name in STOCK_POOL:
    df = fetch_kline_sina(symbol, datalen=1023)
    if df is None or len(df) < 100: continue
    closes = df['close'].values.astype(float)
    volumes = df['volume'].values.astype(float)
    n = len(closes)
    if n >= 310:
        reg250, _ = compute_rolling_regression(closes, window=250)
        pits_v1 = detect_golden_pit_v1(closes, reg250)
    else:
        pits_v1 = []
    pits_v2 = detect_golden_pit_v2(closes, volumes)
    r1, r2 = backtest_pits(closes, pits_v1, pits_v2, horizon=60)
    all_v1.extend(r1)
    all_v2.extend(r2)

# 定义策略
strat_slow = [r for r in all_v2 if r['launch_days'] >= 6]  # 慢启动
strat_b = [r for r in all_v2 if r['signal'] == 'B_slow_confirmed']  # B类
strat_fast = [r for r in all_v2 if r['launch_days'] <= 5]  # 快启动

strategies = [
    ('v1原始\n(快启动+长坑)', all_v1),
    ('v2全部信号', all_v2),
    ('v2慢启动≥6天\n(策略D)', strat_slow),
    ('v2-B类慢坑确认', strat_b),
    ('v2快启动≤5天', strat_fast),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('黄金坑算法 v1(原始) vs v2(优化) — 20只A股真实数据回测 (60天持有期)',
             fontsize=14, fontweight='bold')

# 图1: 胜率与信号数对比
ax = axes[0, 0]
labels = [s[0] for s in strategies]
win_rates = [stats(s[1])['win_rate']*100 for s in strategies]
counts = [stats(s[1])['n'] for s in strategies]
x = np.arange(len(labels))
bars1 = ax.bar(x - 0.2, win_rates, 0.4, label='胜率(%)', color='#1565C0', alpha=0.8)
ax2 = ax.twinx()
bars2 = ax2.bar(x + 0.2, counts, 0.4, label='信号数', color='#E53935', alpha=0.6)
ax.set_ylabel('胜率 (%)', color='#1565C0')
ax2.set_ylabel('信号数', color='#E53935')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_title('胜率 vs 信号数对比')
ax.set_ylim(0, 100)
for bar, val in zip(bars1, win_rates):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%', ha='center', fontsize=8)
for bar, val in zip(bars2, counts):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 2, str(val), ha='center', fontsize=8, color='#E53935')

# 图2: 收益分布箱线图
ax = axes[0, 1]
data = [sorted([r['ret']*100 for r in s[1]]) for s in strategies]
bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
colors = ['#1565C0', '#FF8F00', '#2E7D32', '#6A1B9A', '#E53935']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_ylabel('60天收益率 (%)')
ax.set_title('收益率分布对比 (箱线图)')
ax.tick_params(axis='x', labelsize=8)
ax.grid(True, alpha=0.2)

# 图3: 按启动天数的胜率和收益
ax = axes[1, 0]
buckets = [(0,5,'≤5天'),(6,10,'6-10天'),(11,20,'11-20天'),(21,999,'>20天')]
wr_by_days = []
ret_by_days = []
cnt_by_days = []
for lo, hi, label in buckets:
    grp = [r for r in all_v2 if lo <= r['launch_days'] <= hi]
    s = stats(grp)
    wr_by_days.append(s['win_rate']*100)
    ret_by_days.append(s['mean_ret']*100)
    cnt_by_days.append(s['n'])
x = np.arange(len(buckets))
labels_b = [b[2] for b in buckets]
ax.bar(x - 0.2, wr_by_days, 0.4, label='胜率(%)', color='#1565C0', alpha=0.8)
ax2 = ax.twinx()
ax2.bar(x + 0.2, ret_by_days, 0.4, label='均值收益(%)', color='#2E7D32', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([f'{l}\n(n={c})' for l,c in zip(labels_b, cnt_by_days)])
ax.set_ylabel('胜率 (%)', color='#1565C0')
ax2.set_ylabel('均值收益 (%)', color='#2E7D32')
ax.set_title('按坑底→出坑天数的表现 (关键发现:慢启动更优)')
ax.set_ylim(0, 100)
for i, (w, r) in enumerate(zip(wr_by_days, ret_by_days)):
    ax.text(i-0.2, w+1, f'{w:.1f}%', ha='center', fontsize=8)
    ax2.text(i+0.2, r+0.3, f'{r:.1f}%', ha='center', fontsize=8, color='#2E7D32')

# 图4: 按性质分的表现
ax = axes[1, 1]
scores = [-1, 0, 1, 2, 3, 4]
wr_by_score = []
ret_by_score = []
cnt_by_score = []
for sc in scores:
    grp = [r for r in all_v2 if r['nature_score'] == sc]
    s = stats(grp)
    wr_by_score.append(s['win_rate']*100 if s['n']>0 else 0)
    ret_by_score.append(s['mean_ret']*100 if s['n']>0 else 0)
    cnt_by_score.append(s['n'])
x = np.arange(len(scores))
ax.bar(x - 0.2, wr_by_score, 0.4, label='胜率(%)', color='#6A1B9A', alpha=0.7)
ax2 = ax.twinx()
ax2.bar(x + 0.2, ret_by_score, 0.4, label='均值收益(%)', color='#FF8F00', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([f'{s}\n(n={c})' for s,c in zip(scores, cnt_by_score)])
ax.set_ylabel('胜率 (%)', color='#6A1B9A')
ax2.set_ylabel('均值收益 (%)', color='#FF8F00')
ax.set_title('按坑性质评分的表现 (发现:分1-2最优,>=3反而下降)')
ax.set_ylim(0, 100)
for i, (w, r) in enumerate(zip(wr_by_score, ret_by_score)):
    if cnt_by_score[i] > 0:
        ax.text(i-0.2, w+1, f'{w:.1f}%', ha='center', fontsize=8)
        ax2.text(i+0.2, r+0.3, f'{r:.1f}%', ha='center', fontsize=8, color='#FF8F00')

plt.tight_layout()
out = '/home/user/.super_doubao/super-doubao-runtime/workspace/pressure-level-algorithm/v1_vs_v2_comparison.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.close()
print(f'图表已保存: {out}')
