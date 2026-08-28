# -*- coding: utf-8 -*-
"""确认增强测试(因果, 当日/次日可知): 出坑日放量 / 连续确认 / 按坑型定制确认。

全部条件只用 <= 买入日的数据, 无未来函数。对比胜率与信号量。
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression
from golden_pit_v2_rewrite import detect_golden_pit_v2

def variants(c, v, s, b, lch):
    """返回该信号在各确认变体下是否通过 + 买入日收益。"""
    n = len(c)
    if lch + 20 >= n:
        return None
    r20 = c[lch + 20] / c[lch] - 1
    # A. 原样
    # B. 出坑日放量(量 > 前5日均量)
    vp = np.mean(v[max(0, lch - 5):lch])
    vol_ok = v[lch] > vp if vp > 0 else False
    vol12 = v[lch] > 1.2 * vp if vp > 0 else False
    # D. 连续2日站上门控线(第2日买入)
    reg = None  # 需要 reg, 由外部传入
    return r20, vol_ok, vol12

def main():
    pool = [l.split(',')[0].strip() for l in open('stock_pool_1000.txt', encoding='utf-8') if l.strip()]
    A, B, C = [], [], []
    D, E = [], []
    for k, sym in enumerate(pool):
        f = os.path.join('.cache_kline', f'{sym}.npy')
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True).item()
        c = d['close'].astype(float)
        v = d['vol'].astype(float)
        n = len(c)
        reg, _ = compute_rolling_regression(c, window=250, use_log=True)
        pits = detect_golden_pit_v2(c, reg)
        for s, b, lch in pits:
            if lch is None or lch - b > 5 or lch + 20 >= n:
                continue
            r20 = c[lch + 20] / c[lch] - 1
            vp = np.mean(v[max(0, lch - 5):lch])
            vol_ok = v[lch] > vp if vp > 0 else False
            vol12 = v[lch] > 1.2 * vp if vp > 0 else False
            plen = b - s + 1
            A.append([r20, plen, vol_ok, vol12, c[lch] / reg[lch]])
            # D. 连续2日站上门控线(买入=第2日)
            if lch + 1 < n:
                gate2 = reg[lch + 1] * 0.9
                if c[lch + 1] >= gate2:
                    r20d = c[lch + 21] / c[lch + 1] - 1 if lch + 21 < n else np.nan
                    if np.isfinite(r20d):
                        D.append(r20d)
        if (k + 1) % 300 == 0:
            print(f'  进度 {k+1}/{len(pool)}', flush=True)
    A = np.array(A)
    def show(label, R):
        if len(R) < 15:
            print(f'  {label:<34} n={len(R):>4} 样本少'); return
        print(f'  {label:<34} n={len(R):>4}  20日胜率={np.mean(R>=0):5.1%} 均值={np.mean(R):+6.1%}')
    print(f'\n=== 确认增强对比(1000池, v2算法, 快启动<=5, 20日) ===')
    show('A. 基线(v2 原样)', A[:, 0])
    show('B. 出坑日放量(>前5均量)', A[A[:, 2] == 1][:, 0])
    show('   B 被过滤(缩量出坑)', A[A[:, 2] == 0][:, 0])
    show('C. 出坑日放量>1.2x', A[A[:, 3] == 1][:, 0])
    show('D. 连续2日站上(第2日买)', np.array(D))
    # E. 按坑型定制: 短坑(<6)需放量, 长坑(>=6)直接
    E_r = A[(A[:, 1] >= 6) | ((A[:, 1] < 6) & (A[:, 2] == 1))][:, 0]
    show('E. 定制:短坑放量/长坑直接', E_r)
    show('   长坑(>=6)子集', A[A[:, 1] >= 6][:, 0])
    show('   短坑+放量子集', A[(A[:, 1] < 6) & (A[:, 2] == 1)][:, 0])
    show('   短坑无放量(放弃)', A[(A[:, 1] < 6) & (A[:, 2] == 0)][:, 0])

if __name__ == '__main__':
    main()
