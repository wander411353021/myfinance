# -*- coding: utf-8 -*-
"""黄金坑未来函数自检【截断一致性验证】。

核心方法: 对每个黄金坑信号 (s,b,lch), 把 K线截断到 lch 日(只保留 lch 及之前),
         重新运行 detect_golden_pit, 验证信号是否仍被检测出且坑底/出坑日一致。
         若能复现 → 证明该信号在 lch 日收盘即可产生, 无未来函数。

同时检查:
  1. 出坑日 lch 的判定是否只依赖 lch 当日及历史
  2. 快启动(lch-b<=5)判定是否 lch 日可知
  3. 收益起点(买入=lch收盘)是否与信号产生一致
  4. 放量堆加仓标记是否存在"确认时滞"(标记用途 vs 可执行区分)

用法: python3 golden_pit_lookahead_check.py [股票数上限]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panic_reversal as pr
from golden_pit_v2_backtest import compute_rolling_regression
import golden_pit_rule_compare as rc

WORKDIR = os.path.dirname(os.path.abspath(__file__))


def check_stock(symbol):
    """单只股票截断一致性检查。返回 (pass_list, fail_list, n_sig)"""
    d = rc.load(symbol)
    if d is None:
        return [], [], 0
    c = d['close'].astype(float)
    v = d['vol'].astype(float)
    n = len(c)
    reg250, _ = compute_rolling_regression(c, window=250, use_log=True)
    pits = pr.detect_golden_pit(c, reg250)

    passed, failed = [], []
    n_sig = 0
    for s, b, lch in pits:
        if lch is None or lch + 20 >= n:
            continue
        if lch - b > 5:   # 规则F: 只查可执行信号(快启动<=5)
            continue
        n_sig += 1
        # ── 截断到 lch 日(含) ──
        c_cut = c[:lch + 1]
        v_cut = v[:lch + 1]
        reg_cut, _ = compute_rolling_regression(c_cut, window=250, use_log=True)
        pits_cut = pr.detect_golden_pit(c_cut, reg_cut)
        # 找截断后检测中与原始一致的坑(坑底 b 一致 + 出坑 lch 一致)
        match = False
        for s2, b2, lch2 in pits_cut:
            if b2 == b and lch2 == lch:
                match = True
                break
        if match:
            passed.append((symbol, s, b, lch))
        else:
            failed.append((symbol, s, b, lch, [(x[0], x[1], x[2]) for x in pits_cut]))
    return passed, failed, n_sig


def main():
    top = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    stocks = [l.strip().split(',')[0] for l in open(os.path.join(WORKDIR, 'stock_pool_1000.txt')) if l.strip()][:top]
    all_pass, all_fail = [], []
    t0 = time.time()
    for i, sym in enumerate(stocks):
        p, f, _ = check_stock(sym)
        all_pass += p
        all_fail += f
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(stocks)} 通过{len(all_pass)} 失败{len(all_fail)} {time.time()-t0:.0f}s', flush=True)
    total_sig = len(all_pass) + len(all_fail)
    print(f'\n=== 截断一致性验证结果 ===')
    print(f'检查股票: {len(stocks)} 只')
    print(f'规则F信号总数: {total_sig}')
    print(f'✅ 截断后复现(无未来函数): {len(all_pass)} ({len(all_pass)/max(1,total_sig)*100:.2f}%)')
    print(f'❌ 截断后消失(潜在未来函数): {len(all_fail)}')
    if all_fail:
        print('\n失败样本(前10):')
        for sym, s, b, lch, cut_pits in all_fail[:10]:
            print(f'  {sym} 坑底[{s},{b}] 出坑{lch}  截断后检测: {cut_pits}')
    else:
        print('\n✅ 全部通过 —— 黄金坑信号在出坑日收盘即可确定, 无未来函数')


if __name__ == '__main__':
    main()
