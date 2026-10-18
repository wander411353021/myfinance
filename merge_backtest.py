# -*- coding: utf-8 -*-
"""合并 4 个分片 CSV → result/backtest_v3b_5y.csv, 并做完整性校验与已知案例抽查。
用法: python3 merge_backtest.py
校验项:
  1. 各分片行数之和 == 合并后行数(无遗漏)
  2. 无 (code, sig_date, sig_type) 完全重复行
  3. sig_date ∈ [20211001, 20261016], sig_type ∈ {v3b, early}
  4. 已知案例抽查(来自 SKILL.md / 用户验收口径):
     300204 20250418 / 001696 20240318 应存在且 s60=True(成功组)
     001696 20240802 应存在(追高型, dev35 高/距前高近, 深回撤, 属于'失败组附近'的讨论对象)
     600550 20200227 在 5 年窗口之外(2020), 仅作流水线复验(已在脚本 verify 模式单独验证过), 不在合并表内
  5. 汇总统计: 信号总数 / 按类型 / 按年份 / s60 / s120 / fail 计数
"""
import pandas as pd, numpy as np, os, sys

REPO = '/home/user/.super_doubao/super-doubao-runtime/workspace/pressure-level-algorithm'
COLS = ['code', 'sym', 'sig_date', 'sig_type', 'sig_close',
        'vol_ratio', 'rec_depth', 'rec_days', 'cool_days', 'hist_touch',
        'dist_high', 'dev35', 'band', 'profit', 'gap',
        'fut60_max', 'fut60_min', 'fut120_max', 'fut60_days', 'fut120_days',
        's60', 's120', 'fail', 'big20', 'big_wait']

shard_files = [os.path.join(REPO, 'result', f'backtest_v3b_shard{i}.csv') for i in range(4)]
frames = []
for i, fp in enumerate(shard_files):
    if not os.path.exists(fp):
        print(f'!! 缺分片 {i}: {fp}')
        sys.exit(1)
    df = pd.read_csv(fp, dtype={'code': str})
    frames.append(df)
    print(f'shard{i}: {len(df)} 行, {df["code"].nunique()} 只股票')

all_df = pd.concat(frames, ignore_index=True)
all_df['code'] = all_df['code'].astype(str).str.zfill(6)
all_df['sig_date'] = all_df['sig_date'].astype(str).str.zfill(8)
n_before = sum(len(f) for f in frames)
print(f'\n合并: 各片之和 {n_before} == 合并行数 {len(all_df)} : {"OK" if n_before == len(all_df) else "FAIL"}')

# 重复检查
dup = all_df.duplicated(subset=['code', 'sig_date', 'sig_type']).sum()
print(f'重复 (code,sig_date,sig_type) 行: {dup}')

# 字段合法性
bad_date = all_df[(all_df.sig_date < '20211001') | (all_df.sig_date > '20261016')]
bad_type = ~all_df.sig_type.isin(['v3b', 'early'])
print(f'sig_date 越界: {len(bad_date)}, sig_type 非法: {int(bad_type.sum())}')
missing = [c for c in COLS if c not in all_df.columns]
print(f'缺失列: {missing if missing else "无"}')

# 已知案例抽查
def probe(code, date):
    r = all_df[(all_df.code == code) & (all_df.sig_date == date)]
    if len(r) == 0:
        return f'{code} {date}: 未找到!'
    row = r.iloc[0]
    return (f'{code} {date} [{row.sig_type}] 收{row.sig_close} dev35={row.dev35} 距前高={row.dist_high} '
            f'fut60_max={row.fut60_max} fut60_min={row.fut60_min} fut120_max={row.fut120_max} '
            f's60={row.s60} s120={row.s120} fail={row.fail}')

print('\n== 已知案例抽查 ==')
print(probe('300204', '20250418'))
print(probe('001696', '20240318'))
print(probe('001696', '20240802'))

# 汇总统计
print('\n== 汇总 ==')
print('按类型:', all_df.sig_type.value_counts().to_dict())
all_df['year'] = all_df.sig_date.str[:4]
print('\n按年份×类型:')
print(all_df.pivot_table(index='year', columns='sig_type', values='code', aggfunc='count', fill_value=0))
full60 = all_df[all_df.fut60_days >= 60]
print(f'\n完整60日窗口信号: {len(full60)} (占 {len(full60)/len(all_df):.1%})')
print('  s60 率:', (full60.s60.mean() * 100).round(1), '% | s120 率(60日窗内统计):', (full60.s120.mean() * 100).round(1), '% | fail 率:', (full60.fail.mean() * 100).round(1), '%')
by_type = full60.groupby('sig_type')
for t, g in by_type:
    print(f'  {t}: n={len(g)} s60={g.s60.mean()*100:.1f}% s120={g.s120.mean()*100:.1f}% fail={g.fail.mean()*100:.1f}% '
          f'fut60中位={g.fut60_max.median():.1f}% fut60_min中位={g.fut60_min.median():.1f}%')

out = os.path.join(REPO, 'result', 'backtest_v3b_5y.csv')
all_df[COLS].to_csv(out, index=False)
print(f'\n已写出: {out} ({len(all_df)} 行)')
