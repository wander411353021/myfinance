# -*- coding: utf-8 -*-
"""格栅线压制突破算法(2026-09-02 用户定义):
以 reg120/reg250 为基准, 上下平移若干档形成格栅线;
某条格栅线"长期压制"股价(close 在其下方连续>=min_suppress 天), 向上突破后=信号。
全部因果(只用<=突破日数据)。返回 [(压制起点, 突破日, 基准reg类型, 档位, 压制天数)]。
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression

def detect_grid_break(closes, reg120, reg250,
                      levels=(-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20),
                      min_suppress=20):
    """格栅线压制突破。levels 为相对 reg 的偏移(0.05=reg上方5%档)。"""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    regs = {'r120': np.asarray(reg120, dtype=float), 'r250': np.asarray(reg250, dtype=float)}
    out = []
    for rname, reg in regs.items():
        for lev in levels:
            line = reg * (1 + lev)
            # 扫描压制段
            sup_start = None
            for i in range(len(closes)):
                below = np.isfinite(line[i]) and closes[i] < line[i]
                if below:
                    if sup_start is None:
                        sup_start = i
                else:
                    if sup_start is not None:
                        sup_days = i - sup_start
                        if sup_days >= min_suppress:
                            # 突破日 = i(收盘站上该线)
                            out.append((sup_start, i, rname, lev, sup_days))
                    sup_start = None
            if sup_start is not None:
                sup_days = n - sup_start
                if sup_days >= min_suppress:
                    out.append((sup_start, n - 1, rname, lev, sup_days))  # 未突破(到数据末)不算
    # 只保留已突破的信号(有明确突破日)
    out = [o for o in out if o[1] < len(closes)]
    out.sort(key=lambda o: o[1])
    return out

if __name__ == '__main__':
    # 300171 示例
    df = pr._load_df('sz300171', '20260902', datalen=800)
    c = df['close'].values.astype(float)
    dates = df['date'].dt.strftime('%Y-%m-%d').values
    r120, _ = compute_rolling_regression(c, window=120, use_log=True)
    r250, _ = compute_rolling_regression(c, window=250, use_log=True)
    print('=== 300171 格栅突破信号(最近240天) ===')
    i240 = len(c) - 240
    for s, br, rn, lev, days in detect_grid_break(c, r120, r250):
        if br >= i240:
            print(f'  {rn} {lev:+.0%}档: 压制 {dates[s]}~{dates[br]}({days}天) → 突破 {dates[br]}')
