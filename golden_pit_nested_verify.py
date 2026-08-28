# -*- coding: utf-8 -*-
"""验证用户假设: 大坑被拆成小坑 → 小坑胜率低; 合并大坑胜率更高。

全池 v2 检测, 按"嵌套关系"分组:
- 嵌套坑: 出坑后 15 天内又有新坑(属于大坑内部的小坑)
- 独立坑: 出坑后 15 天内无新坑(真正独立的坑)
对比 20日胜率; 再用 v3(confirm=5)合并后的大坑胜率对比。
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression
from golden_pit_v3_confirm import detect_v3

def main():
    pool = [l.split(',')[0].strip() for l in open('stock_pool_1000.txt', encoding='utf-8') if l.strip()]
    rows = []  # [嵌套标记, r20(出坑日买)]
    for k, sym in enumerate(pool):
        f = os.path.join('.cache_kline', f'{sym}.npy')
        if not os.path.exists(f): continue
        d = np.load(f, allow_pickle=True).item()
        c = d['close'].astype(float); nn = len(c)
        reg, _ = compute_rolling_regression(c, window=250, use_log=True)
        pits = [p for p in pr.detect_golden_pit_v2(c, reg) if p[2] is not None]
        pits.sort(key=lambda p: p[2])
        for idx, (s, b, lch) in enumerate(pits):
            if lch + 20 >= nn: continue
            # 嵌套: 下一个坑在 15 天内出现
            nxt = pits[idx + 1][2] if idx + 1 < len(pits) else None
            nested = nxt is not None and (nxt - lch) <= 15
            r20 = c[lch + 20] / c[lch] - 1
            rows.append([nested, r20])
        if (k + 1) % 500 == 0: print(f'  进度 {k+1}', flush=True)
    R = np.array(rows, dtype=float)
    def show(label, m):
        M = R[m]
        if len(M) < 15:
            print(f'  {label:<30} n={len(M):>4} 样本少'); return
        print(f'  {label:<30} n={len(M):>4}  20日胜率={np.mean(M[:,1]>=0):5.1%} 均值={np.mean(M[:,1]):+6.1%}')
    print(f'\n=== v2 全池 {len(R)} 个坑, 嵌套 vs 独立 ===')
    show('全部坑', np.ones(len(R), bool))
    show('嵌套坑(出坑15天内又进坑)', R[:, 0] == 1)
    show('独立坑(出坑15天内无新坑)', R[:, 0] == 0)

    # v3 合并大坑
    rows3 = []
    for k, sym in enumerate(pool):
        f = os.path.join('.cache_kline', f'{sym}.npy')
        if not os.path.exists(f): continue
        d = np.load(f, allow_pickle=True).item()
        c = d['close'].astype(float); nn = len(c)
        reg, _ = compute_rolling_regression(c, window=250, use_log=True)
        for s, b, lch in detect_v3(c, reg, confirm_days=5):
            if lch is None or lch + 20 >= nn: continue
            rows3.append(c[lch + 20] / c[lch] - 1)
        if (k + 1) % 500 == 0: print(f'  v3 进度 {k+1}', flush=True)
    R3 = np.array(rows3)
    print(f'\n=== v3(confirm=5) 合并后大坑 {len(R3)} 个 ===')
    if len(R3) >= 15:
        print(f'  {"合并大坑":<30} n={len(R3):>4}  20日胜率={np.mean(R3>=0):5.1%} 均值={np.mean(R3):+6.1%}')

if __name__ == '__main__':
    main()
