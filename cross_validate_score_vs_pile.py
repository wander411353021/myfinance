# -*- coding: utf-8 -*-
"""交叉验证:6维评分层 vs 放量堆确认层(黄金坑 v2.0 vs 本地超高信号)。

统一样本:本地 panic_reversal.detect_golden_pit(与 v2.0 信号口径一致)。
对比:
  A. 6维评分(v2.0 体系):甜点区42-51 / 放弃≥52 / 正常30-41 / 观望<30
  B. 放量堆确认(本地体系):出坑后7天巨量放量堆(超高信号 93%)
  C. 组合:两者叠加
收益:20日持有(v2.0 口径)与 60日(本地口径)都算。
"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import panic_reversal as pr
from golden_pit_final_strategy import score_signal, add_rs_score
try:
    from tdx_quant_index_extension import get_index_kline_from_tdx
    _idx_df = get_index_kline_from_tdx('hs300', count=1200)
    _idx_dates = pd.to_datetime(_idx_df['date']).dt.strftime('%Y-%m-%d').values
    _idx_close = _idx_df['close'].values.astype(float)
    print(f'HS300 数据: {len(_idx_close)} 天')
except Exception as e:
    print(f'HS300 不可用: {e}')
    _idx_dates, _idx_close = None, None

def hs300_ret(date_str, days=20):
    if _idx_close is None:
        return None
    m = {d: i for i, d in enumerate(_idx_dates)}
    i = m.get(str(date_str)[:10])
    if i is None or i < days:
        return None
    return _idx_close[i] / _idx_close[i - days] - 1

def main():
    pool = []
    with open('stock_pool_300.txt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                code = line.split(',')[0].strip()
                if code:
                    pool.append(code)
    print(f'股票池: {len(pool)} 只')

    rows = []  # [评分, 是否放量堆确认, 20日收益, 60日收益]
    years = []
    for k, code in enumerate(pool):
        fcode = code if code[:2] in ('sh', 'sz') else ('sh' if code[0] in '69' else 'sz') + code
        df = pr._load_df(fcode, '20260827')
        if df is None or len(df) < 400:
            continue
        c = df['close'].values.astype(float)
        v = df['volume'].values.astype(float)
        dates = df['date'].dt.strftime('%Y-%m-%d').values
        n = len(c)
        from mean_reversion.signal_residual import compute_rolling_regression
        reg250, _ = compute_rolling_regression(c, window=250, use_log=True)
        pits = pr.detect_golden_pit(c, reg250)
        vcl = pr.detect_volume_clusters(c, v)
        for s, b, lch in pits:
            if lch is None or lch - b > 5 or b - s + 1 < 8:
                continue  # 快启动 + 坑长(与 v2.0 一致)
            if lch + 60 >= n:
                continue
            # 6维评分
            score, shrink, fill, depth = score_signal(c, v, reg250, s, b, lch)
            rs20 = None
            if b >= 20:
                sret = c[b] / c[b - 20] - 1
                mret = hs300_ret(dates[lch], 20)
                if mret is not None:
                    rs20 = sret - mret
            score = add_rs_score(score, rs20)
            # 放量堆确认:出坑后7天内 HIGH 堆且峰值量比>=5
            pile = False
            for ss, ee, kd, dr, pk, vr in vcl:
                if kd == 'HIGH' and lch < ss <= lch + 7 and pk >= 5.0:
                    pile = True
                    break
            r20 = c[lch + 20] / c[lch] - 1
            r60 = c[lch + 60] / c[lch] - 1
            rows.append([score, pile, r20, r60])
            years.append(dates[lch][:4])
        if (k + 1) % 100 == 0:
            print(f'  进度 {k+1}/{len(pool)}', flush=True)
    R = np.array(rows, dtype=float)
    years = np.array(years)
    print(f'\n信号总数: {len(R)}')

    def show(label, mask):
        m = R[mask]
        if len(m) < 15:
            print(f'  {label:<34} n={len(m):>4} (样本少)')
            return
        w20 = np.mean(m[:, 2] >= 0); med20 = np.median(m[:, 2])
        w60 = np.mean(m[:, 3] >= 0); med60 = np.median(m[:, 3])
        print(f'  {label:<34} n={len(m):>4}  20日胜率={w20:5.1%} 均值={np.mean(m[:,2]):+6.1%} | 60日胜率={w60:5.1%} 均值={np.mean(m[:,3]):+6.1%}')

    sc = R[:, 0]
    print('\n--- A. 6维评分分组 ---')
    show('基线(全部信号)', np.ones(len(R), bool))
    show('甜点区 42-51', (sc >= 42) & (sc < 52))
    show('放弃区 >=52', sc >= 52)
    show('正常 30-41', (sc >= 30) & (sc < 42))
    show('观望 <30', sc < 30)
    print('\n--- B. 放量堆确认 ---')
    show('有放量堆确认(超高)', R[:, 1] == 1)
    show('无放量堆确认', R[:, 1] == 0)
    print('\n--- C. 组合 ---')
    show('甜点区 + 放量堆', (sc >= 42) & (sc < 52) & (R[:, 1] == 1))
    show('甜点区 无放量堆', (sc >= 42) & (sc < 52) & (R[:, 1] == 0))
    show('非甜点 + 放量堆', ((sc < 42) | (sc >= 52)) & (R[:, 1] == 1))
    print('\n--- 按年份(20日胜率) ---')
    for y in sorted(set(years)):
        m = R[years == y]
        if len(m) < 10:
            continue
        print(f'  {y}: n={len(m):>3} 20日胜率={np.mean(m[:,2]>=0):.1%}')

if __name__ == '__main__':
    main()
