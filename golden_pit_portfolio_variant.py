# -*- coding: utf-8 -*-
"""降回撤方案对比（严格实盘口径 次日买入）

对比 止损 / 降仓 / 大盘过滤 及组合 对收益与回撤的实际影响。
用法: python3 golden_pit_portfolio_variant.py
"""
import sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden_pit_portfolio_sim as sim

stocks = [l.strip().split(',')[0] for l in open(os.path.join(sim.WORKDIR, 'stock_pool_1000.txt')) if l.strip()]

# (标签, 参数)
VARIANTS = [
    ('基线(满仓无止损无过滤)', dict()),
    ('止损 -10%', dict(stop_loss=0.10)),
    ('止损 -8%', dict(stop_loss=0.08)),
    ('降仓 70%', dict(pos_ratio=0.70)),
    ('大盘过滤(300空头跳过)', dict(market_filter=True)),
    ('止损-10% + 降仓70%', dict(stop_loss=0.10, pos_ratio=0.70)),
    ('止损-10% + 大盘过滤', dict(stop_loss=0.10, market_filter=True)),
    ('降仓70% + 大盘过滤', dict(pos_ratio=0.70, market_filter=True)),
]

print(f'加载 {len(stocks)} 只, 跑 {len(VARIANTS)} 组变体(严格口径/次日买入)...\n')
t0 = time.time()
results = []
for label, kw in VARIANTS:
    r = sim.run_sim(stocks, buy_delay=1, **kw)
    results.append((label, r))
    print(f'[{label}] 成交{r["n_trades"]:>3} 胜率{r["win_rate"]:5.1f}% 收益{r["total_ret"]*100:7.1f}% 回撤{r["max_dd"]*100:6.1f}% ({time.time()-t0:.0f}s)', flush=True)

print('\n' + '=' * 78)
print('降回撤方案对比（100万 / 10份复利 / 严格口径次日买入 / 2023-2026）')
print('=' * 78)
print(f'{"方案":<22}{"成交":>5}{"胜率":>7}{"总收益":>10}{"最大回撤":>9}{"收益/回撤":>10}')
for label, r in results:
    print(f'{label:<22}{r["n_trades"]:>5}{r["win_rate"]:>6.1f}%{r["total_ret"]*100:>9.1f}%{r["max_dd"]*100:>8.1f}%'
          f'{r["total_ret"]/abs(r["max_dd"]):>9.1f}x')
