# -*- coding: utf-8 -*-
"""黄金坑组合模拟 压力测试（随机子集稳健性验证）

验证收益是否依赖"特定精选股票"：从 989 只缓存池中按固定随机种子抽多组随机子集，
每组独立跑完整模拟（100万/10份/子账户复利），比较总收益率、胜率、交易数的分布。
若随机子集收益与全池接近 → 策略稳健，非"选了好股票"。

用法: python3 golden_pit_portfolio_stress.py
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden_pit_portfolio_sim as sim

POOL = 'stock_pool_1000.txt'
rng = np.random.default_rng(42)  # 固定种子，结果可复现


def load_stocks(pool_file=POOL):
    return [l.strip().split(',')[0] for l in open(os.path.join(sim.WORKDIR, pool_file)) if l.strip()]


def main():
    all_stocks = load_stocks()
    print(f'缓存池共 {len(all_stocks)} 只（stock_pool_1000.txt, 全市场分层抽样）\n')

    # 各组: (标签, 股票数)
    groups = []
    for i in range(3):
        s = rng.choice(all_stocks, size=300, replace=False)
        groups.append((f'随机300-#{i+1}', list(s)))
    for i in range(2):
        s = rng.choice(all_stocks, size=500, replace=False)
        groups.append((f'随机500-#{i+1}', list(s)))
    s = rng.choice(all_stocks, size=700, replace=False)
    groups.append(('随机700-#1', list(s)))
    # 对照: 池子头部300只(非随机, 顺序)
    groups.append(('顺序前300(对照)', all_stocks[:300]))

    results = []
    t0 = time.time()
    for label, stocks in groups:
        r = sim.run_sim(stocks)
        results.append((label, r))
        print(f'[{label}] 信号{r["n_signals"]:>4} 成交{r["n_trades"]:>3} 胜率{r["win_rate"]:5.1f}% '
              f'收益{r["total_ret"]*100:7.1f}% 回撤{r["max_dd"]*100:6.1f}% '
              f'({time.time()-t0:.0f}s)', flush=True)

    print('\n' + '=' * 78)
    print('压力测试汇总 (100万 / 10份子账户复利 / 2023-01-01 ~ 2026-08-28)')
    print('=' * 78)
    print(f'{"组合":<18}{"股票数":>6}{"信号":>6}{"成交":>5}{"胜率":>7}{"总收益":>10}{"最大回撤":>9}')
    for label, r in results:
        print(f'{label:<18}{r["n_stocks"]:>6}{r["n_signals"]:>6}{r["n_trades"]:>5}'
              f'{r["win_rate"]:>6.1f}%{r["total_ret"]*100:>9.1f}%{r["max_dd"]*100:>8.1f}%')

    rets = np.array([r['total_ret'] * 100 for _, r in results])
    print('\n随机子集收益率: min {:.1f}% / mean {:.1f}% / max {:.1f}% / std {:.1f}pp'.format(
        rets.min(), rets.mean(), rets.max(), rets.std()))
    print('→ 若各组收益接近（波动小），说明策略不依赖特定股票组合，稳健。')


if __name__ == '__main__':
    main()
