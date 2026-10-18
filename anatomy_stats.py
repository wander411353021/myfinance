# -*- coding: utf-8 -*-
"""三信号解剖统计 A-F (2026-10-18) — 输入 result/anatomy_signals.csv
A 样本量按年 / B 重叠矩阵 / C 特征分布与后验(w60+qfq) / D early→v3b转化 / E v3→v3b转化 / F v3定义实验
产物: result/anatomy_stats_summary.csv + 控制台输出(报告生成用)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

REPO = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(REPO, 'result')

df = pd.read_csv(os.path.join(R, 'anatomy_signals.csv'), dtype={'code': str})
df['code'] = df.code.str.zfill(6)
df['year'] = df.sig_date.astype(str).str[:4]
n = len(df)

V3 = df[df.hit_v3 == 1].copy()
V3B = df[df.hit_v3b == 1].copy()
EARLY = df[df.hit_early == 1].copy()
V3ONLY = df[(df.hit_v3 == 1) & (df.hit_v3b == 0)].copy()
OV = df[(df.hit_v3 == 1) & (df.hit_v3b == 1)].copy()


def perf(sub, tag):
    w60 = sub[sub.fut60_days >= 60]
    w120 = sub[sub.fut120_days >= 120]
    d = dict(tag=tag, n=len(sub), w60=len(w60),
             s60=round(w60.s60.mean() * 100, 2) if len(w60) else np.nan,
             fail=round(w60.fail.mean() * 100, 2) if len(w60) else np.nan,
             f60med=round(float(w60.fut60_max.median()), 2) if len(w60) else np.nan,
             f60min_med=round(float(w60.fut60_min.median()), 2) if len(w60) else np.nan,
             s120=round(w120.s120.mean() * 100, 2) if len(w120) else np.nan,
             qfq=round(sub.qfq_flag.mean() * 100, 2))
    return d


def feats(sub):
    cols = ['gap', 'abr', 'dev35', 'dist_purple', 'vol_ratio', 'band', 'profit', 'dist_high']
    return {c: round(float(sub[c].median()), 2) for c in cols}


print('========== A. 样本量(按年) ==========')
by = pd.DataFrame({'v3': V3.groupby('year').size(), 'v3b': V3B.groupby('year').size(),
                   'early': EARLY.groupby('year').size()}).fillna(0).astype(int)
by.loc['合计'] = by.sum()
print(by.to_string())

print('\n========== B. 重叠矩阵 ==========')
print('总行(唯一code,date):', n)
print('v3:', len(V3), 'v3b:', len(V3B), 'early:', len(EARLY))
print('v3&v3b:', len(OV), '| v3&early:', int((V3.hit_early == 1).sum()),
      '| v3b&early:', int((V3B.hit_early == 1).sum()), '| 三者:', int((df.hit_v3 & df.hit_v3b & df.hit_early).sum()))
print('v3b∩v3 / v3b = %.2f%%' % (len(OV) / len(V3B) * 100))
print('v3∩v3b / v3  = %.2f%%' % (len(OV) / len(V3) * 100))
print('仅v3:', len(V3ONLY), '仅v3b:', len(df[(df.hit_v3b == 1) & (df.hit_v3 == 0)]), '仅early:', len(df[df.hit_early == 1]))

print('\n========== C. 特征中位数 ==========')
for tag, sub in [('v3', V3), ('v3b', V3B), ('early', EARLY), ('仅v3', V3ONLY), ('v3∩v3b', OV)]:
    print(tag, feats(sub))

print('\n========== C2. 后验表现(w60完整窗口 + qfq) ==========')
perf_rows = [perf(V3, 'v3'), perf(V3B, 'v3b'), perf(EARLY, 'early'),
             perf(V3ONLY, '仅v3'), perf(OV, 'v3∩v3b')]
print(pd.DataFrame(perf_rows).to_string(index=False))

# qfq 分层表现
print('\n--- qfq 分层 (w60) ---')
for tag, sub in [('v3', V3), ('v3b', V3B), ('early', EARLY)]:
    for q in (0, 1):
        s = sub[(sub.qfq_flag == q) & (sub.fut60_days >= 60)]
        if len(s):
            print(f'{tag} qfq={q}: n={len(s)} s60={s.s60.mean()*100:.2f}% fail={s.fail.mean()*100:.2f}% f60med={s.fut60_max.median():.2f}')

print('\n========== D. early→v3b 转化率(60/120日内出现v3b) ==========')
v3b_dates = {c: np.sort(g.sig_date.values) for c, g in V3B.groupby('code')}
early_codes = EARLY['code'].values
early_dates = EARLY['sig_date'].values
conv60 = np.zeros(len(EARLY), dtype=bool)
conv120 = np.zeros(len(EARLY), dtype=bool)
next_v3b_idx = np.full(len(EARLY), -1)
for i in range(len(EARLY)):
    c = early_codes[i]
    arr = v3b_dates.get(c)
    if arr is None or len(arr) == 0:
        continue
    d = early_dates[i]
    j = np.searchsorted(arr, d, side='right')
    if j < len(arr):
        if arr[j] <= d + 60:
            conv60[i] = True
        if arr[j] <= d + 120:
            conv120[i] = True
            next_v3b_idx[i] = j
EARLY = EARLY.copy()
EARLY['conv60'] = conv60
EARLY['conv120'] = conv120
print('early n=%d, 60日内转化 %.2f%% (%d), 120日内转化 %.2f%% (%d)' % (
    len(EARLY), conv60.mean() * 100, conv60.sum(), conv120.mean() * 100, conv120.sum()))
# 转化组 vs 未转化组: early 自身表现 + 后续 v3b 表现
e120 = EARLY[EARLY.fut60_days >= 60].copy()
conv_g = e120[e120.conv120]
noc_g = e120[~e120.conv120]
print('--- early自身(w60): 120日内转化 vs 未转化 ---')
print(pd.DataFrame([perf(conv_g, 'early转化(120d内)'), perf(noc_g, 'early未转化')]).to_string(index=False))
# 后续 v3b 的表现: 取转化组对应的 v3b 行
v3b_lookup = V3B.set_index(['code', 'sig_date'])
nxt = []
for i in np.where(conv120)[0]:
    c = early_codes[i]
    arr = v3b_dates[c]
    d = arr[next_v3b_idx[i]]
    r = v3b_lookup.loc[(c, d)]
    nxt.append(r)
nxt_df = pd.DataFrame(nxt)
print('--- 转化后的首个v3b(120d内)表现(w60) ---')
print(pd.DataFrame([perf(nxt_df, 'early后首个v3b')]).to_string(index=False))
print('对比全量v3b:', pd.DataFrame([perf(V3B, '全量v3b')]).to_string(index=False))

print('\n========== E. v3→v3b 转化率(120日内出现v3b) ==========')
v3_codes = V3['code'].values
v3_dates = V3['sig_date'].values
conv120_v3 = np.zeros(len(V3), dtype=bool)
for i in range(len(V3)):
    c = v3_codes[i]
    arr = v3b_dates.get(c)
    if arr is None or len(arr) == 0:
        continue
    d = v3_dates[i]
    j = np.searchsorted(arr, d, side='right')
    if j < len(arr) and arr[j] <= d + 120:
        conv120_v3[i] = True
print('v3 n=%d, 120日内出现v3b: %.2f%% (%d)' % (len(V3), conv120_v3.mean() * 100, conv120_v3.sum()))
V3 = V3.copy(); V3['conv120'] = conv120_v3
v3c = V3[V3.fut60_days >= 60]
print('--- v3自身(w60): 120日内转v3b vs 未转 ---')
print(pd.DataFrame([perf(v3c[v3c.conv120], 'v3转v3b'), perf(v3c[~v3c.conv120], 'v3未转')]).to_string(index=False))

print('\n========== F. v3 定义实验 ==========')
V3w = V3[V3.fut60_days >= 60].copy()
# F1: 加量能 vol_ratio>=1.2
v3_vol = V3w[V3w.vol_ratio >= 1.2]
v3_novol = V3w[V3w.vol_ratio < 1.2]
print('--- F1 v3+量能(量≥1.2×前20均量) ---')
print(pd.DataFrame([perf(v3_vol, 'v3+量能≥1.2'), perf(v3_novol, 'v3+量能<1.2')]).to_string(index=False))
# F2: 排除历史摸线 hist_touch==0 (无 [T-250,T-20] 摸紫线)
v3_nt = V3w[V3w.hist_touch == 0]
v3_t = V3w[V3w.hist_touch > 0]
print('--- F2 v3排除历史摸线(hist_touch==0) ---')
print(pd.DataFrame([perf(v3_nt, 'v3无历史摸线'), perf(v3_t, 'v3有历史摸线')]).to_string(index=False))
# 完整对照: 全量v3 / v3+量能 / v3无历史摸线 / 仅v3(剔除v3b重叠)
print('--- F3 定义改进候选全对照(w60) ---')
print(pd.DataFrame([perf(V3w, '全量v3'), perf(v3_vol, 'v3+量能≥1.2'), perf(v3_nt, 'v3无历史摸线'),
                    perf(V3ONLY[V3ONLY.fut60_days >= 60], '仅v3(剔除v3b重叠)')]).to_string(index=False))
# 组合: 量能+无历史摸线
v3_both = V3w[(V3w.vol_ratio >= 1.2) & (V3w.hist_touch == 0)]
print(pd.DataFrame([perf(v3_both, 'v3+量能&无历史摸线')]).to_string(index=False))
# 量能分布
print('v3 vol_ratio 分位:', np.percentile(V3.vol_ratio, [10, 25, 50, 75, 90]).round(2))
print('v3 hist_touch>0 占比: %.2f%%' % (V3.hist_touch.gt(0).mean() * 100))

# 保存汇总
summary = {'n_v3': len(V3), 'n_v3b': len(V3B), 'n_early': len(EARLY),
           'ov_v3b_n_v3': len(OV), 'ov_frac_v3b': round(len(OV) / len(V3B) * 100, 2),
           'ov_frac_v3': round(len(OV) / len(V3) * 100, 2),
           'early_conv60': round(conv60.mean() * 100, 2), 'early_conv120': round(conv120.mean() * 100, 2),
           'v3_conv120': round(conv120_v3.mean() * 100, 2)}
pd.DataFrame([summary]).to_csv(os.path.join(R, 'anatomy_stats_summary.csv'), index=False)
print('\nsummary saved:', summary)
