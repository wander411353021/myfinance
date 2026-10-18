# -*- coding: utf-8 -*-
"""阈值敏感性网格 + 可行组合搜索 (承接 fake_filter_v3b_5y.py 的因子表)
输出: result/fake_filter_sensitivity.csv / result/fake_filter_feasible_combo.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings, itertools
warnings.filterwarnings('ignore')
from fake_filter_v3b_5y import evaluate_flag, BASE_N, FAKE_N, TRUE_N

REPO = os.path.dirname(os.path.abspath(__file__))
FACTOR_CSV = os.path.join(REPO, 'result', 'fake_filter_factors.csv')


def main():
    df = pd.read_csv(FACTOR_CSV, dtype={'code': str})
    df['code'] = df.code.str.zfill(6)
    assert len(df) == BASE_N
    rows = []
    # F1 二元(无阈值), 仅记录
    rows.append(evaluate_flag(df.f1_ok != 1.0, df, 'F1', 'close>MA20 且 MA20>MA60'))
    # F2 阈值: 距紫线 > x%
    for x in (10, 15, 20, 25, 30, 35, 40, 45):
        rows.append(evaluate_flag(df.f2_dist_purple > x, df, 'F2', f'距紫线>{x}%'))
    # F3 阈值: 量/前250日最大量 < t
    for t in (0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40):
        rows.append(evaluate_flag(df.f3_vr250 < t, df, 'F3', f'量比250<{t:.2f}'))
    # F4 区间: dist_vol_high ∈ [0, hi]
    for hi in (2, 3, 4, 5, 7, 10, 15):
        rows.append(evaluate_flag(df.dist_vol_high.between(0, hi), df, 'F4', f'前堆压∈[0,{hi}]'))
    sens = pd.DataFrame(rows)
    sens.to_csv(os.path.join(REPO, 'result', 'fake_filter_sensitivity.csv'), index=False)
    print('== 阈值敏感性(全) ==')
    print(sens[['label', 'thr', 'n_rem', 'rem_fake', 'rem_true', 'rem_mz', 'fake_rm_rate', 'true_keep_rate',
                'keep_n', 'keep_s60', 'keep_s120', 'keep_fake_rate', 'keep_fmed']].to_string(index=False))

    # ── 可行操作点: 真保留率>=90% ──
    feas = sens[sens.true_keep_rate >= 90.0].copy()
    print('\n== 真保留率>=90% 的因子操作点 ==')
    print(feas[['label', 'thr', 'n_rem', 'rem_fake', 'rem_true', 'rem_mz', 'fake_rm_rate', 'true_keep_rate',
                'keep_n', 'keep_s120']].to_string(index=False))
    feas.to_csv(os.path.join(REPO, 'result', 'fake_filter_feasible_points.csv'), index=False)

    # ── 可行组合搜索: 用保留率>=90%的操作点 OR 组合 ──
    fea_variants = {}
    for _, r in feas.iterrows():
        if r['label'] == 'F1':
            fea_variants['F1'] = df.f1_ok != 1.0
        elif r['label'] == 'F2':
            x = float(r['thr'].replace('距紫线>', '').replace('%', ''))
            fea_variants[f"F2>{x:.0f}%"] = df.f2_dist_purple > x
        elif r['label'] == 'F3':
            t = float(r['thr'].replace('量比250<', ''))
            fea_variants[f"F3<{t:.2f}"] = df.f3_vr250 < t
        elif r['label'] == 'F4':
            hi = int(r['thr'].split('∈[0,')[1].rstrip(']'))
            fea_variants[f"F4[0,{hi}]"] = df.dist_vol_high.between(0, hi)
    names = list(fea_variants)
    crows = []
    for k in (1, 2, 3):
        for combo in itertools.combinations(names, k):
            flag = pd.Series(False, index=df.index)
            for nm in combo:
                flag = flag | fea_variants[nm]
            crows.append(evaluate_flag(flag, df, '+'.join(combo), '+'.join(combo)))
    cdf = pd.DataFrame(crows)
    cdf = cdf[cdf.true_keep_rate >= 90.0].sort_values('fake_rm_rate', ascending=False)
    cdf.to_csv(os.path.join(REPO, 'result', 'fake_filter_feasible_combo.csv'), index=False)
    print('\n== 可行组合(真保留率>=90%, 按剔假率降序, 全) ==')
    print(cdf[['label', 'n_rem', 'rem_fake', 'rem_true', 'rem_mz', 'fake_rm_rate', 'true_keep_rate',
               'keep_n', 'keep_s120', 'keep_s60']].head(30).to_string(index=False))
    # 组合内部冗余清理: 去掉'组合与单点等效'的 (先保留全部, 报告中再讨论)


if __name__ == '__main__':
    main()
