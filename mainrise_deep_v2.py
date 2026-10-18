# -*- coding: utf-8 -*-
"""
v3b/v3b_early 五年回溯：用户点名案例 + 主升浪归因 深度分析。
所有数字 pandas 实算自 result/backtest_v3b_5y.csv，不修改既有文件。
用法: python3 mainrise_deep_v2.py
产出:
  result/cases_37.csv   —— 37 票逐信号明细
  result/deep_stats.json —— 三组归因/假设/门槛枚举统计
"""
import os, json
import pandas as pd
import numpy as np

CSV = os.path.join('result', 'backtest_v3b_5y.csv')
CASES = os.path.join('result', 'cases_37.csv')
STATS = os.path.join('result', 'deep_stats.json')

CODES37 = ['300171','003040','300204','300251','300437','600199','600519','600371',
           '601579','002343','000530','000993','002460','300299','300253','600737',
           '600721','600613','688233','688388','600865','301080','603778','600550',
           '001696','300006','688018','600152','301487','300619','002866','603278',
           '002923','002441','301087','300675','300389']

WIN_LO, WIN_HI = 20211001, 20261016

# 用户图上重点关注日期
KEY_DATES = {
    '300204': 20250730, '300251': 20250205, '001696': 20240318,
    '603778': [20251120, 20251120],  # 两信号
    '600865': 20261016,
}

DIMS = ['vol_ratio','rec_depth','rec_days','cool_days','dist_high','dev35','band','profit','gap']


def load():
    df = pd.read_csv(CSV, dtype={'code': str})
    df['code'] = df['code'].str.zfill(6)
    return df


# ============ 第一批：37 票 ============
def part1_cases(df):
    sub = df[(df.code.isin(CODES37)) & (df.sig_date >= WIN_LO) & (df.sig_date <= WIN_HI)].copy()
    # 主升浪命中判定（数据末端截断单独标注）
    def hit(r):
        if r.fut120_days < 120:
            return 'truncated'
        return 'mainrise' if r.fut120_max >= 50 else 'no'
    sub['mainrise'] = sub.apply(hit, axis=1)
    sub['is_key'] = sub.apply(
        lambda r: (KEY_DATES.get(r.code) is not None and
                   (r.sig_date in (KEY_DATES[r.code] if isinstance(KEY_DATES[r.code], list) else [KEY_DATES[r.code]]))),
        axis=1)
    cols = ['code','sym','sig_date','sig_type','sig_close','vol_ratio','rec_depth','rec_days',
            'cool_days','dist_high','dev35','band','profit','gap',
            'fut60_max','fut60_min','fut120_max','fut60_days','fut120_days',
            's60','s120','qfq_flag','big20','big_wait','mainrise','is_key']
    sub[cols].sort_values(['code','sig_date']).to_csv(CASES, index=False)

    # 汇总：每票 主升浪命中 X/N（N=可评估信号数，即 fut120_days>=120）
    summ = []
    for c in CODES37:
        d = sub[sub.code == c]
        ev = d[d.fut120_days >= 120]
        mr = int((ev.fut120_max >= 50).sum())
        summ.append({
            'code': c, 'sym': d.sym.iloc[0] if len(d) else '',
            'n_sig': len(d), 'n_eval': len(ev), 'n_mainrise': mr,
            'n_trunc': int((d.fut120_days < 120).sum()),
            'dates': d.sig_date.tolist(),
        })
    return sub, summ


# ============ 第二批：三组归因 ============
def part2_groups(df):
    w120 = df[df.fut120_days >= 120].copy()
    w60 = df[df.fut60_days >= 60].copy()
    # 主升组
    g_main = w120[w120.fut120_max >= 50].copy()
    # 平庸组：w60 & w120 & s60 & fut120_max<50
    g_mid = df[(df.fut60_days >= 60) & (df.fut120_days >= 120) &
               (df.s60 == True) & (df.fut120_max < 50)].copy()
    # 失败组：w60 & fut60_max<=0
    g_fail = w60[w60.fut60_max <= 0].copy()

    def grp_stats(g, name):
        out = {'name': name, 'n': len(g),
               'n_qfq1': int((g.qfq_flag == 1).sum()),
               'qfq1_ratio': round(float((g.qfq_flag == 1).mean()), 4),
               'big20_rate': round(float(g.big20.mean()), 4),
               'big_wait_median': float(g.big_wait.median()),
               'big_wait_median_flag0': float(g[g.qfq_flag==0].big_wait.median())}
        for d in DIMS:
            s = g[d]
            out[d] = {'mean': round(float(s.mean()), 3),
                      'median': round(float(s.median()), 3),
                      'p25': round(float(s.quantile(.25)), 3),
                      'p75': round(float(s.quantile(.75)), 3)}
        # 60 日先行确认（仅主升组有意义）
        if name == 'main':
            g6 = g[g.fut60_days >= 60]
            out['p_fut60max_ge20'] = round(float((g6.fut60_max >= 20).mean()), 4)
            out['p_fut60max_ge30'] = round(float((g6.fut60_max >= 30).mean()), 4)
            out['n_main_w60'] = len(g6)
        return out

    res = {'main': grp_stats(g_main, 'main'),
           'mid': grp_stats(g_mid, 'mid'),
           'fail': grp_stats(g_fail, 'fail')}

    # 关键区分因子：中位(主升) - 中位(平庸)
    diffs = []
    for d in DIMS:
        diffs.append((d, round(res['main'][d]['median'] - res['mid'][d]['median'], 3)))
    diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    res['distinguish_median_diff_main_minus_mid'] = diffs
    return res, g_main, g_mid, g_fail


def layer_s120(base, col, bins, labels):
    """在 base(v3b∩flag0∩w120) 上分层算 S120 率。"""
    b = base.copy()
    b['_bin'] = pd.cut(b[col], bins=bins, labels=labels, include_lowest=True)
    rows = []
    for lab in labels:
        s = b[b._bin == lab]
        rows.append({'layer': lab, 'n': len(s),
                     's120_rate': round(float(s.s120.mean()), 4) if len(s) else None,
                     'fail_rate': round(float(s.fail.mean()), 4) if len(s) else None})
    return rows


def part3_hypotheses(df):
    # 门槛枚举与假设共用底座：v3b ∩ flag0 ∩ fut120_days>=120
    base = df[(df.sig_type == 'v3b') & (df.qfq_flag == 0) & (df.fut120_days >= 120)].copy()
    out = {'base_n': len(base),
           'base_s120': round(float(base.s120.mean()), 4),
           'base_s60': round(float(base.s60.mean()), 4),
           'base_bigniu': round(float((base.fut120_max >= 100).mean()), 4)}

    # ① big20 / big_wait 已在三组里给；这里在底座补 v3b 口径
    out['h1_big20_by_s120'] = {
        's120=1_big20_rate': round(float(base[base.s120].big20.mean()), 4),
        's120=0_big20_rate': round(float(base[~base.s120].big20.mean()), 4),
    }
    # ② cool_days 分层
    out['h2_cool'] = layer_s120(base, 'cool_days',
        [-1, 20, 40, 80, 150, 10**9],
        ['0-20','20-40','40-80','80-150','>150'])
    # ③ band 分层（筹码带宽%）
    out['h3_band'] = layer_s120(base, 'band',
        [-1, 20, 40, 60, 10**9],
        ['<20','20-40','40-60','>60'])
    # ④ profit 分层
    out['h4_profit'] = layer_s120(base, 'profit',
        [-0.01, 0.3, 0.5, 0.7, 0.85, 1.01],
        ['<0.3','0.3-0.5','0.5-0.7','0.7-0.85','>0.85'])
    # ⑤ dist_high 分层（距前高%，越大越低位）
    out['h5_dist_high'] = layer_s120(base, 'dist_high',
        [-1, 5, 15, 30, 10**9],
        ['贴前高<5','5-15','15-30','>30低位'])
    return out, base


def part4_threshold(base):
    """枚举 2-3 维组合门槛，n>=300 才保留。"""
    B = base.copy()
    candidates = []

    def eval_cond(name, mask):
        s = B[mask]
        n = len(s)
        if n < 300:
            return None
        return {
            'cond': name, 'n': n,
            's120': round(float(s.s120.mean()), 4),
            'bigniu_ge100': round(float((s.fut120_max >= 100).mean()), 4),
            's60': round(float(s.s60.mean()), 4),
            'fut120_max_median': round(float(s.fut120_max.median()), 2),
            'per_month': round(n / 60.0, 1),
        }

    # 单维先看甜区
    # band 甜区 >40；gap 大；dist_high；cool_days；profit
    combos = [
        # 核心甜区：低位吸筹(profit<0.3) + 低位起涨(dist_high>30) + 大阳线确认(big20=1)
        ('profit<0.3 & dist_high>30 & big20=1 & cool>=40',
         (B.profit < 0.3) & (B.dist_high > 30) & (B.big20 == 1) & (B.cool_days >= 40)),
        ('profit<0.3 & dist_high>30 & big20=1',
         (B.profit < 0.3) & (B.dist_high > 30) & (B.big20 == 1)),
        ('profit<0.3 & big20=1',
         (B.profit < 0.3) & (B.big20 == 1)),
        ('profit<0.3 & dist_high>30 & band>40 & big20=1',
         (B.profit < 0.3) & (B.dist_high > 30) & (B.band > 40) & (B.big20 == 1)),
        ('dist_high>30 & band>60 & big20=1',
         (B.dist_high > 30) & (B.band > 60) & (B.big20 == 1)),
        # 弱对照（不带 big20）
        ('profit<0.3 & dist_high>30', (B.profit < 0.3) & (B.dist_high > 30)),
        ('profit<0.3', B.profit < 0.3),
    ]
    for name, mask in combos:
        r = eval_cond(name, mask)
        if r:
            candidates.append(r)
    candidates.sort(key=lambda x: x['s120'], reverse=True)
    return candidates


def main():
    df = load()
    sub, summ = part1_cases(df)
    print(f'[part1] cases_37.csv rows={len(sub)}; 命中主升浪票数={sum(1 for s in summ if s["n_mainrise"]>0)}/37')

    groups, g_main, g_mid, g_fail = part2_groups(df)
    print(f'[part2] main n={groups["main"]["n"]} mid n={groups["mid"]["n"]} fail n={groups["fail"]["n"]}')

    hyp, base = part3_hypotheses(df)
    thresh = part4_threshold(base)
    print(f'[part4] threshold candidates={len(thresh)}')

    out = {'groups': groups, 'hypotheses': hyp, 'thresholds': thresh,
           'cases_summary': [{k: v for k, v in s.items() if k != 'dates'} for s in summ]}
    with open(STATS, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f'[done] -> {CASES}, {STATS}')


if __name__ == '__main__':
    main()
