# -*- coding: utf-8 -*-
"""验证用户洞察: 旧版高胜率 = 合并大坑后"最后的出坑信号"。

事后标注(允许未来, 只做研究不做交易):
- 对每个坑: 未来30天内是否还会再进坑(同大坑内的后续坑)
- "最后出坑" = 该坑之后不再进坑(大坑彻底反转)
- "中间出坑" = 该坑之后还进坑(假出坑, 反弹后跌回)
对比 20/60 日胜率。若"最后出坑"胜率显著高 → 识别最后出坑是赚胜率的关键;
若不高 → 旧版高胜率纯靠未来函数买点优化, 无因果可复制。
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression

def main():
    pool = [l.split(',')[0].strip() for l in open('stock_pool_1000.txt', encoding='utf-8') if l.strip()]
    rows = []  # [是否最后出坑, r20, r60, 与下一坑间隔]
    for k, sym in enumerate(pool):
        f = os.path.join('.cache_kline', f'{sym}.npy')
        if not os.path.exists(f): continue
        d = np.load(f, allow_pickle=True).item()
        c = d['close'].astype(float); nn = len(c)
        reg, _ = compute_rolling_regression(c, window=250, use_log=True)
        pits = [p for p in pr.detect_golden_pit_v2(c, reg) if p[2] is not None]
        pits.sort(key=lambda p: p[2])
        for idx, (s, b, lch) in enumerate(pits):
            if lch + 60 >= nn: continue
            nxt = pits[idx + 1][2] if idx + 1 < len(pits) else None
            gap = (nxt - lch) if nxt is not None else 999
            last = (nxt is None) or (nxt - lch) > 30  # 30天内不再进坑 = 最后出坑
            r20 = c[lch + 20] / c[lch] - 1
            r60 = c[lch + 60] / c[lch] - 1
            rows.append([last, r20, r60, gap])
        if (k + 1) % 500 == 0: print(f'  进度 {k+1}', flush=True)
    R = np.array(rows)
    def show(label, m, col):
        M = R[m]
        if len(M) < 15:
            print(f'  {label:<34} n={len(M):>4} 样本少'); return
        print(f'  {label:<34} n={len(M):>4}  20日胜率={np.mean(M[:,col]>=0):5.1%} 均值={np.mean(M[:,col]):+6.1%} | '
              f'60日胜率={np.mean(M[:,3] if col==1 else M[:,2]>=0):.1%}')
    print(f'\n=== 事后标注: 最后出坑 vs 中间出坑 (1000池 {len(R)} 坑) ===')
    print(f'  {"最后出坑(30天内不再进坑)":<34} n={np.sum(R[:,0]==1):>4}  20日胜率={np.mean(R[R[:,0]==1][:,1]>=0):5.1%} 均值={np.mean(R[R[:,0]==1][:,1]):+6.1%} | 60日胜率={np.mean(R[R[:,0]==1][:,2]>=0):.1%}')
    print(f'  {"中间出坑(30天内又进坑)":<34} n={np.sum(R[:,0]==0):>4}  20日胜率={np.mean(R[R[:,0]==0][:,1]>=0):5.1%} 均值={np.mean(R[R[:,0]==0][:,1]):+6.1%} | 60日胜率={np.mean(R[R[:,0]==0][:,2]>=0):.1%}')
    # 按间隔分层
    print('\n--- 间隔分层(事后) ---')
    for lo, hi, tag in [(1, 10, '间隔1-10天(紧密嵌套)'), (11, 30, '间隔11-30天'), (31, 999, '间隔>30天(=最后)')]:
        m = (R[:, 3] >= lo) & (R[:, 3] <= hi)
        print(f'  {tag:<22} n={np.sum(m):>4}  20日胜率={np.mean(R[m][:,1]>=0):5.1%} 均值={np.mean(R[m][:,1]):+6.1%}')

if __name__ == '__main__':
    main()
