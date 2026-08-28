# -*- coding: utf-8 -*-
"""黄金坑 深度未来函数截断自检（可审计, 2026-08-28 polo4111）

方法: 对每只股票取若干截断日T, 只用 closes[:T+1] 重算信号,
      与全量计算中"出坑日 lch<=T"的信号逐一比对, 必须完全一致(s,b,lch)。
      任一不一致 = 该信号依赖了 T 之后的数据 = 未来函数。

- 回归一致性: reg250 在 T 之前必须逐点一致(验证滚动回归因果)
- 信号一致性: 全量 lch<=T 的信号集合 == 截断重算信号集合

用法: python3 golden_pit_lookahead_deep.py [股票数=40]
输出: 0 不一致 = ✅ 无未来函数
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression
import golden_pit_portfolio_sim as sim

N_STOCKS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
_pool = os.path.join(sim.WORKDIR, 'stock_pool_1000.txt')
stocks = [l.strip().split(',')[0] for l in open(_pool) if l.strip()][:N_STOCKS]
rng = np.random.default_rng(20260828)

mismatch = 0; checked = 0; checked_sig = 0; checked_reg = 0
for sym in stocks:
    d = sim.load(sym)
    if d is None: continue
    c = d['close'].astype(float); n = len(c)
    if n < 500: continue
    # 全量
    reg_full, _ = compute_rolling_regression(c, window=250, use_log=True)
    pits_full = pr.detect_golden_pit(c, reg_full)
    sig_full = [(s, b, lch) for s, b, lch in pits_full if lch is not None]
    # 随机6个截断点, 分布在后60%区间
    cuts = np.sort(rng.choice(np.arange(int(n * 0.4), int(n * 0.95)), size=6, replace=False))
    for T in cuts:
        c_t = c[:T + 1]
        reg_t, _ = compute_rolling_regression(c_t, window=250, use_log=True)
        pits_t = pr.detect_golden_pit(c_t, reg_t)
        sig_t = set((s, b, lch) for s, b, lch in pits_t if lch is not None)
        full_before = set((s, b, lch) for s, b, lch in sig_full if lch <= T)
        # 回归一致性: T 之前逐点
        rbad = np.isfinite(reg_full[:T + 1]) & np.isfinite(reg_t) & (np.abs(reg_full[:T + 1] - reg_t) > 1e-9)
        if rbad.any():
            print(f'[回归不一致] {sym} @T={T}  {rbad.sum()} 点')
            mismatch += 1
        # 信号一致性
        if full_before != sig_t:
            print(f'[信号不一致] {sym} @T={T}')
            only_full = full_before - sig_t
            only_t = sig_t - full_before
            if only_full:
                print(f'   全量有截断无(漏报): {sorted(only_full)[:5]}')
            if only_t:
                print(f'   截断有全量无(新增): {sorted(only_t)[:5]}')
            mismatch += 1
        checked += 1
        checked_sig += len(full_before)
        checked_reg += 1

print(f'\n=== 截断自检结果 ===')
print(f'股票×截断点 {checked} 组, 回归一致性检查 {checked_reg} 组')
print(f'逐信号核对 {checked_sig} 个(全量 lch<=T 信号), 不一致 {mismatch} 处')
print('结论:', '❌ 存在未来函数' if mismatch else '✅ 无未来函数(全部截断可复现)')
sys.exit(1 if mismatch else 0)
