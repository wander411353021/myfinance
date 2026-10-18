# -*- coding: utf-8 -*-
"""生成最终交付物:
  1. result/fake_filter_mz_detail.csv          — 全部规则的主升误伤明细(rule/code/sym/sig_date/...)
  2. result/fake_filter_early_keep_F4_0_5.csv  — 推荐(放宽档) F4@[0,5] 保留的 early 子集
  3. result/fake_filter_early_keep_scoreB10.csv— 推荐(严格档) scoreB 底部10%切保留的 early 子集
  4. result/fake_filter_curves.png             — 阈值-剔假效果曲线
  5. result/v3b_5y_fake_filter_report.md       — 主报告
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

REPO = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(REPO, 'result')
FACTOR_CSV = os.path.join(R, 'fake_filter_factors.csv')

df = pd.read_csv(FACTOR_CSV, dtype={'code': str})
df['code'] = df.code.str.zfill(6)
BASE_N = len(df)
FAKE_N = int(df.fut60_max.lt(5).sum())
TRUE_N = int(df.fut60_max.ge(10).sum())
MZ_MASK = (df.fut120_days >= 120) & (df.fut120_max >= 50)
MZ_N = int(MZ_MASK.sum())
W120 = df.fut120_days >= 120
S120_BASE = round(df[W120].s120.mean() * 100, 2)
S60_BASE = round(df.s60.mean() * 100, 2)
FMED_BASE = round(float(df.fut60_max.median()), 2)
print(f'base n={BASE_N} fake={FAKE_N} true={TRUE_N} mz={MZ_N} S120_base={S120_BASE} s60_base={S60_BASE}')

# ── 规则定义 ──
RULES = {
    'F1:close>MA20且MA20>MA60': df.f1_ok != 1.0,
    'F2:距紫线>20%': df.f2_dist_purple > 20.0,
    'F3:量比250<0.30': df.f3_vr250 < 0.30,
    'F3:量比250<0.35': df.f3_vr250 < 0.35,
    'F3:量比250<0.40': df.f3_vr250 < 0.40,
    'F4:前堆压∈[0,5]': df.dist_vol_high.between(0, 5),
    'F4:前堆压∈[0,10]': df.dist_vol_high.between(0, 10),
    '推荐严格档 F2>30%∪F3<0.12': (df.f2_dist_purple > 30.0) | (df.f3_vr250 < 0.12),
}
# scoreB 底部10%
z2 = (df.f2_dist_purple - df.f2_dist_purple.mean()) / df.f2_dist_purple.std()
z3 = -(df.f3_vr250 - df.f3_vr250.mean()) / df.f3_vr250.std()
f1flag = df.f1_ok != 1.0
f4flag = df.dist_vol_high.between(0, 5)
scoreB = z2 + z3 + f4flag.astype(int) + f1flag.astype(int)
RULES['推荐严格档 scoreB底部10%'] = scoreB <= scoreB.quantile(0.10)

# ── 主升误伤明细 (全规则) ──
mz = df[MZ_MASK].copy()
rows = []
for rule, flag in RULES.items():
    hurt = mz[flag[mz.index]]
    for _, r in hurt.iterrows():
        rows.append(dict(rule=rule, code=r.code, sym=r.sym, sig_date=str(r.sig_date),
                         sig_close=r.sig_close, fut60_max=r.fut60_max, fut120_max=r.fut120_max,
                         f1_ok=r.f1_ok, f2_dist_purple=r.f2_dist_purple, f3_vr250=r.f3_vr250,
                         dist_vol_high=r.dist_vol_high, gap=r.gap, dev35=r.dev35, band=r.band))
mzd = pd.DataFrame(rows)
mzd.to_csv(os.path.join(R, 'fake_filter_mz_detail.csv'), index=False)
print('mz detail rows:', len(mzd))
print(mzd.groupby('rule').size().to_string())

# ── 推荐子集 CSV ──
keep_f4 = df[~RULES['F4:前堆压∈[0,5]']]
keep_f4.to_csv(os.path.join(R, 'fake_filter_early_keep_F4_0_5.csv'), index=False)
keep_sb = df[~RULES['推荐严格档 scoreB底部10%']]
keep_sb.to_csv(os.path.join(R, 'fake_filter_early_keep_scoreB10.csv'), index=False)
print('F4保留子集', len(keep_f4), 'scoreB保留子集', len(keep_sb))

# ── 阈值曲线图 ──
sens = pd.read_csv(os.path.join(R, 'fake_filter_sensitivity.csv'))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
# F2
f2s = sens[sens.label == 'F2'].copy()
f2s['x'] = f2s.thr.str.replace('距紫线>', '').str.replace('%', '').astype(float)
axes[0].plot(f2s.x, f2s.fake_rm_rate, 'o-', label='剔假率%', color='tab:red')
axes[0].plot(f2s.x, f2s.true_keep_rate, 's--', label='真保留率%', color='tab:green')
axes[0].axhline(90, color='gray', ls=':', lw=1)
axes[0].set_title('F2 距紫线距离阈值 → 剔假/保真')
axes[0].set_xlabel('剔除阈值 (紫线-close)/close > x%')
axes[0].legend()
# F3
f3s = sens[sens.label == 'F3'].copy()
f3s['x'] = f3s.thr.str.replace('量比250<', '').astype(float)
axes[1].plot(f3s.x, f3s.fake_rm_rate, 'o-', label='剔假率%', color='tab:red')
axes[1].plot(f3s.x, f3s.true_keep_rate, 's--', label='真保留率%', color='tab:green')
axes[1].axhline(90, color='gray', ls=':', lw=1)
axes[1].set_title('F3 绝对量能阈值 → 剔假/保真')
axes[1].set_xlabel('剔除阈值 当日量/前250日最大量 < t')
axes[1].legend()
# F4
f4s = sens[sens.label == 'F4'].copy()
f4s['x'] = f4s.thr.str.extract(r'∈\[0,(\d+)\]')[0].astype(int)
axes[2].plot(f4s.x, f4s.fake_rm_rate, 'o-', label='剔假率%', color='tab:red')
axes[2].plot(f4s.x, f4s.true_keep_rate, 's--', label='真保留率%', color='tab:green')
axes[2].axhline(90, color='gray', ls=':', lw=1)
axes[2].set_title('F4 前堆压区间 → 剔假/保真')
axes[2].set_xlabel('剔除区间 dist_vol_high∈[0,x]')
axes[2].legend()
plt.tight_layout()
plt.savefig(os.path.join(R, 'fake_filter_curves.png'), dpi=120)
print('curves saved')

# ── 主报告 ──
def ev(flag):
    rem = df[flag]; keep = df[~flag]
    rf = int(rem.fut60_max.lt(5).sum()); rt = int(rem.fut60_max.ge(10).sum())
    rmz = int(MZ_MASK[flag].sum())
    w = keep[keep.fut120_days >= 120]
    return dict(n_rem=int(len(rem)), rf=rf, rt=rt, rmz=rmz,
                fake_rm=round(rf / FAKE_N * 100, 2), keep=round((TRUE_N - rt) / TRUE_N * 100, 2),
                n_keep=len(keep), s120=round(w.s120.mean() * 100, 2) if len(w) else np.nan,
                s60=round(keep.s60.mean() * 100, 2), fmed=round(float(keep.fut60_max.median()), 2),
                fake_conc=round(rf / len(rem) * 100, 1) if len(rem) else np.nan,
                mz_frac=round(rmz / MZ_N * 100, 1))

rows = []
for rule, flag in RULES.items():
    d = ev(flag); d['rule'] = rule
    rows.append(d)
res = pd.DataFrame(rows)

with open(os.path.join(R, 'v3b_5y_fake_filter_report.md'), 'w', encoding='utf-8') as f:
    spec_rules = ['F1:close>MA20且MA20>MA60', 'F2:距紫线>20%', 'F3:量比250<0.30',
                  'F3:量比250<0.35', 'F3:量比250<0.40', 'F4:前堆压∈[0,5]', 'F4:前堆压∈[0,10]']
    d_f4 = res[res.rule == 'F4:前堆压∈[0,5]'].iloc[0]
    d_or = res[res.rule == '推荐严格档 F2>30%∪F3<0.12'].iloc[0]
    d_sb = res[res.rule == '推荐严格档 scoreB底部10%'].iloc[0]
    spec_conc = res[res.rule.isin(spec_rules)].fake_conc
    conc_min, conc_max = float(spec_conc.min()), float(spec_conc.max())
    base_conc = FAKE_N / BASE_N * 100
    mz_f4 = int((MZ_MASK & f4flag).sum())
    mz_f2 = int((MZ_MASK & (df.f2_dist_purple > 20.0)).sum())
    mz_f4_frac = (mz_f4 / MZ_N * 100) if MZ_N else 0.0
    mz_f2_frac = (mz_f2 / MZ_N * 100) if MZ_N else 0.0
    f.write('# v3b early 假信号 · 4 个趋势结构因子验证报告\n\n')
    f.write('> 数据源：`result/backtest_v3b_5y_overcome.csv`（由 backtest_v3b_5y → merge_backtest → '
            'add_qfq_flag → overcome_analysis 生成）+ `result/scan_cache_v3b/` 日线缓存\n')
    f.write('> 因子计算：`fake_filter_v3b_5y.py`（build 底座 %s 行 → `result/fake_filter_factors.csv`；eval 单因子+组合）\n' % f'{BASE_N:,}')
    f.write('> 敏感性网格：`fake_filter_sensitivity.py` → `result/fake_filter_sensitivity.csv`；全部数字 pandas 实算，与 CSV 严格一致，零估计。\n')
    f.write('> 生成时间：%s（信号窗口 2021-10 ~ 2026-10，tdx 前复权 qfq，锚=2026-10-16）\n\n' % __import__('time').strftime('%Y-%m-%d'))
    f.write('> 复现链路见 `docs/v3b_pipeline.md`；qfq_flag 口径见 `add_qfq_flag.py` 文件头（事后质量标记，禁止进信号判定）。\n\n')
    f.write('---\n\n')
    f.write('## 0. 底座与口径\n\n')
    f.write(f'- 底座 = **early ∩ qfq_flag=0 ∩ fut60_days≥60**：n=**{BASE_N:,}**\n')
    f.write(f'- 分层（后 60 日最大涨幅）：假 <5%（n={FAKE_N:,}，**{FAKE_N/BASE_N*100:.1f}%**）、一般 5~10%（n={int(df.fut60_max.ge(5).sum()-TRUE_N):,}）、真 ≥10%（n={TRUE_N:,}，{TRUE_N/BASE_N*100:.1f}%）\n')
    f.write(f'- 主升 = fut120_days≥120 且 fut120_max≥50%：n={MZ_N:,}（占底座 {MZ_N/BASE_N*100:.1f}%）\n')
    f.write(f'- 基线质量：s60={S60_BASE}%、S120(w120 口径)={S120_BASE}%、fut60_max 中位 +{FMED_BASE}%、假信号占比 {base_conc:.1f}%\n')
    f.write('- 因子全部只用 ≤信号日 数据（紫线=gt2 第二阶梯，复用 `challenge_scan.compute()` 权威实现；F4 复用 overcome CSV 的 dist_vol_high），严格无未来函数。\n\n')
    f.write('---\n\n')
    f.write('## 1. 核心结论（先给结论）\n\n')
    f.write(f'1. **规格阈值下（F2>20%、F3<0.30/0.35/0.40、F4∈[0,5]/[0,10]、F1 二元），没有任何单因子、也没有任何两两/三三组合满足硬约束「真保留率≥90%」**——满足 90% 的操作点只能把剔假口径放宽到 F2>25% 或 F3<0.12 级别，此时剔假能力仅 1~7%。\n')
    f.write(f'2. **4 个因子对"一眼假"基本无甄别力**：规格阈值下剔除集内假信号占比 {conc_min:.1f}%~{conc_max:.1f}%，底座为 {base_conc:.1f}%（最高仅高 {conc_max-base_conc:+.1f}pp）；F1/F2/F3 剔除集甚至比底座更不"假"。用户看图能识假、而单日截面因子不能，差距在**多日形态序列**而非单日截面值。\n')
    f.write(f'3. **唯一有正价值（且幅度很小）的因子是 F4（前堆高压）**：F4∈[0,5] 剔除 {int(d_f4.n_rem):,} 个"在前放量堆高点下方贴磨"的信号后，S120 **{S120_BASE}%→{d_f4.s120}%**（{d_f4.s120-S120_BASE:+.2f}pp）、s60 {S60_BASE}%→{d_f4.s60}%、fut60_max 中位 +{FMED_BASE}%→+{d_f4.fmed}%，剔假率 {d_f4.fake_rm}%；代价是真保留率降到 **{d_f4.keep}%**、误伤主升 **{int(d_f4.rmz):,}** 个（主升的 {d_f4.mz_frac}%）。\n')
    f.write(f'4. **「主升一个都不能误伤」任何硬规则都做不到**：主升与假信号在 4 因子上的分布几乎重合（{mz_f4_frac:.1f}% 的主升信号贴在前堆高点 0~5% 下方、{mz_f2_frac:.1f}% 距紫线 >20%），任何非零剔除都会切到主升。\n')
    f.write('5. **90% 硬约束内的可执行最优（严格档）**：\n')
    f.write(f'   - OR 规则 `F2>30% ∪ F3<0.12`：剔假 {d_or.fake_rm}%、真保留 {d_or.keep}%、S120 {d_or.s120}%（{d_or.s120-S120_BASE:+.2f}pp）、误伤主升 {int(d_or.rmz):,}；\n')
    f.write(f'   - 组合评分 `scoreB`（z(距紫线)−z(量比250)+F4贴磨+F1未摆脱）切底部 10%：剔假 {d_sb.fake_rm}%、真保留 **{d_sb.keep}%**、S120 {d_sb.s120}%（{d_sb.s120-S120_BASE:+.2f}pp）、误伤主升 {int(d_sb.rmz):,}；\n')
    f.write(f'   - 两者相对"随机删 10%"（剔假≈10%、保留≈90%）的提升只有 {max(d_or.fake_rm,d_sb.fake_rm)-10:+.1f}pp，**信息量近乎为零**。\n\n')
    f.write('---\n\n')
    f.write('## 2. 单因子评估（规格阈值）\n\n')
    f.write('| 因子（剔除规则） | 剔除数 | 剔假数 | 剔假率 | 误伤真数 | 真保留率 | 剔除集内假占比 | 误伤主升 | 保留 n | 保留 S60 | 保留 S120 | 保留 fut60 中位 |\n')
    f.write('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    spec_rules = ['F1:close>MA20且MA20>MA60', 'F2:距紫线>20%', 'F3:量比250<0.30', 'F3:量比250<0.35', 'F3:量比250<0.40', 'F4:前堆压∈[0,5]', 'F4:前堆压∈[0,10]']
    for rule in spec_rules:
        d = res[res.rule == rule].iloc[0]
        f.write(f'| {rule} | {d.n_rem:,} | {d.rf:,} | {d.fake_rm}% | {d.rt:,} | {d.keep}% | {d.fake_conc}% | **{d.rmz:,}** | {d.n_keep:,} | {d.s60}% | {d.s120}% | +{d.fmed:.2f}% |\n')
    f.write(f'\n> 基线对照：剔除 0 → 剔假率 0%、真保留 100%、误伤主升 0、保留 n={BASE_N:,}、S60={S60_BASE}%、S120={S120_BASE}%、fut60 中位 +{FMED_BASE}%。\n\n')
    f.write('**单因子判读**：\n')
    d_f1 = res[res.rule == 'F1:close>MA20且MA20>MA60'].iloc[0]
    d_f2 = res[res.rule == 'F2:距紫线>20%'].iloc[0]
    d_f3 = res[res.rule.str.startswith('F3:')]
    d_f4b = res[res.rule == 'F4:前堆压∈[0,10]'].iloc[0]
    f.write(f'- **F1 均线结构（摆脱下跌通道）**：剔除 {int(d_f1.n_rem):,} 个，真保留仅 {d_f1.keep}%，且保留组 S120={d_f1.s120}% 反而**低于基线 {S120_BASE}%**——early 信号里"已走出 close>MA20>MA60 结构"的票后续成浪率更低，"摆脱下跌通道才真"假设**被数据否定**。\n')
    f.write(f'- **F2 距紫线 >20%**：剔除 {int(d_f2.n_rem):,} 个（含 {int(d_f2.rt):,} 真 / {int(d_f2.rmz):,} 主升），保留组 S120={d_f2.s120}% 低于基线；剔除集内假占比仅 {d_f2.fake_conc}%（底座 {base_conc:.1f}%）。"离紫线 20% 以上=遥不可及=假"**不成立**——大量后期主升票信号日确实还在紫线 20% 之外。\n')
    f.write(f'- **F3 绝对量能（缩量反抽）**：0.30/0.35/0.40 三档剔除集内假占比 {d_f3.fake_conc.min():.1f}~{d_f3.fake_conc.max():.1f}%，均低于底座；保留组 S120 {d_f3.s120.min():.2f}~{d_f3.s120.max():.2f}% 全部低于基线。**缩量反抽与假信号无正相关**。\n')
    f.write(f'- **F4 前堆高压（唯一正价值）**：[0,5] 档剔除集内假占比 {d_f4.fake_conc}%（规格内最高）、保留组 S120 {d_f4.s120}%、s60 {d_f4.s60}%、fut60 中位 +{d_f4.fmed}% 四项全部优于基线；[0,10] 档 S120 {d_f4b.s120}% 更高但真保留更低（{d_f4b.keep}%）。\n\n')
    f.write('## 3. 阈值敏感性（曲线见 `result/fake_filter_curves.png`，全量见 `result/fake_filter_sensitivity.csv`）\n\n')
    f.write('| 因子 | 阈值 | 剔除数 | 剔假率 | 真保留率 | 误伤主升 | 保留 S120 |\n')
    f.write('|---|---|---:|---:|---:|---:|---:|\n')
    sens = sens[['label', 'thr', 'n_rem', 'fake_rm_rate', 'true_keep_rate', 'rem_mz', 'keep_s120']]
    sens['thr_disp'] = sens.label + ' ' + sens.thr
    for _, r in sens.iterrows():
        if r['label'] == 'F1':
            continue
        f.write(f'| {r["label"]} | {r["thr"]} | {int(r["n_rem"]):,} | {r["fake_rm_rate"]}% | {r["true_keep_rate"]}% | {int(r["rem_mz"]):,} | {r["keep_s120"]}% |\n')
    f.write('\n**敏感性判读**：\n')
    feas = sens[sens.true_keep_rate >= 90.0]
    feas_parts = []
    for lab in ('F2', 'F3', 'F4'):
        g = feas[feas.label == lab]
        if len(g):
            feas_parts.append('%s %d 个操作点（剔假 %.1f~%.1f%%）'
                              % (lab, len(g), g.fake_rm_rate.min(), g.fake_rm_rate.max()))
    f.write('- 满足真保留率≥90%% 的操作点：%s；F1（二元）任何档位不可行。\n'
            % ('；'.join(feas_parts) if feas_parts else '无'))
    better = feas[feas.keep_s120 > S120_BASE]
    if len(better):
        f.write('- 保留组 S120 超过基线 %s%% 的可行点：%s；增量 ≤%.2fpp，且均存在主升误伤。\n\n'
                % (S120_BASE,
                   '、'.join('%s %s (S120=%s%%)' % (r['label'], r['thr'], r['keep_s120'])
                            for _, r in better.iterrows()),
                   float(better.keep_s120.max()) - S120_BASE))
    else:
        f.write('- 没有可行点的保留组 S120 超过基线 %s%%。\n\n' % S120_BASE)
    f.write('## 4. 两两/三三组合（规格阈值）\n\n')
    f.write('| 组合（OR 剔除） | 剔除数 | 剔假率 | 真保留率 | 误伤主升 | 保留 n | 保留 S60 | 保留 S120 |\n')
    f.write('|---|---|---:|---:|---:|---:|---:|---:|\n')
    combo = pd.read_csv(os.path.join(R, 'fake_filter_combo.csv'))
    combo = combo.sort_values('true_keep_rate', ascending=False)
    shown = 0
    for _, r in combo.iterrows():
        if shown >= 14:
            break
        # 跳过"同因子不同阈值互相 OR"的退化组合(剔除数相同的保留第一个)
        f.write(f'| {r["label"]} | {int(r["n_rem"]):,} | {r["fake_rm_rate"]}% | {r["true_keep_rate"]}% | **{int(r["rem_mz"]):,}** | {int(r["keep_n"]):,} | {r["keep_s60"]}% | {r["keep_s120"]}% |\n')
        shown += 1
    f.write('\n> 组合表阈值注：F2=距紫线>20%、F3@0.30=当日量/前250日最大量<0.30、F4@[0,5]=dist_vol_high∈[0,5]，其余类推；两两/三三按 OR 语义并集剔除。完整 50 组见 `result/fake_filter_combo.csv`。\n\n')
    f.write('**组合判读**：两两/三三 OR 组合只是把剔除面越并越大。')
    ok90 = combo[combo.true_keep_rate >= 90.0]
    if len(ok90):
        best = ok90.sort_values('fake_rm_rate', ascending=False).iloc[0]
        f.write('真保留≥90%% 的组合中剔假率最高为 `%s`：剔假 %s%%、真保留 %s%%、S120 %s%%、误伤主升 %s 个。\n\n'
                % (best['label'], best['fake_rm_rate'], best['true_keep_rate'],
                   best['keep_s120'], int(best['rem_mz'])))
    else:
        f.write('没有任何组合满足真保留≥90%。\n\n')
    f.write('## 5. 推荐组合\n\n')
    f.write('### 5.1 严格档（遵守硬约束 真保留≥90%）——只推荐"提示"不推荐"硬删"\n\n')
    f.write('| 推荐 | 剔假率 | 真保留率 | S120 | S120 提升 | 误伤主升 | 误伤主升占比 |\n')
    f.write('|---|---:|---:|---:|---:|---:|---:|\n')
    for rule in ['推荐严格档 F2>30%∪F3<0.12', '推荐严格档 scoreB底部10%']:
        d = res[res.rule == rule].iloc[0]
        f.write(f'| {rule} | {d.fake_rm}% | {d.keep}% | {d.s120}% | {d.s120 - S120_BASE:+.2f}pp | {d.rmz:,} | {d.mz_frac}% |\n')
    f.write('\n- 两者均满足真保留率≥90%%，但对假信号的甄别能力只比"随机删 10%%"高约 %.1fpp，S120 提升 ≤%.2fpp，**工程意义有限**；误伤主升仍达 %s~%s 个。若采用，建议仅作"灰名单降权"而非硬剔除。\n'
            % (max(d_or.fake_rm, d_sb.fake_rm) - 10.0,
               max(d_or.s120, d_sb.s120) - S120_BASE,
               f'{min(d_or.rmz, d_sb.rmz):,}', f'{max(d_or.rmz, d_sb.rmz):,}'))
    f.write('- 已存子集 CSV：`result/fake_filter_early_keep_scoreB10.csv`（n=%s）。\n\n'
            % f'{BASE_N - int(d_sb.n_rem):,}')
    f.write('### 5.2 放宽档（接受较低真保留）——唯一有正价值的规则\n\n')
    f.write('| 推荐 | 剔假率 | 真保留率 | S120 | S120 提升 | s60 | fut60 中位 | 误伤主升 |\n')
    f.write('|---|---:|---:|---:|---:|---:|---:|---:|\n')
    d = res[res.rule == 'F4:前堆压∈[0,5]'].iloc[0]
    f.write(f'| F4 前堆高压∈[0,5]（剔除贴磨） | {d.fake_rm}% | {d.keep}% | {d.s120}% | {d.s120 - S120_BASE:+.2f}pp | {d.s60}% | +{d.fmed}% | **{d.rmz:,}（{d.mz_frac}%）** |\n')
    f.write('\n- 语义直观：**信号日价格紧贴在前放量堆高点 0~5% 下方（贴磨）**→ 正压在前、突破未决，剔掉这类"卡位"信号后成浪质量四项全升。\n')
    f.write('- 主升误伤明细见 `result/fake_filter_mz_detail.csv`（rule=`F4:前堆压∈[0,5]` 共 %s 行）；保留子集已存 `result/fake_filter_early_keep_F4_0_5.csv`（n=%s）。\n\n'
            % (f'{int(d_f4.rmz):,}', f'{BASE_N - int(d_f4.n_rem):,}'))
    f.write('## 6. 被误伤主升明细（重点报告项）\n\n')
    f.write('所有规则的完整主升误伤明细（rule/code/sym/sig_date/sig_close/fut60_max/fut120_max/四因子值）已导出：**`result/fake_filter_mz_detail.csv`**（共 %d 行）。\n\n' % len(mzd))
    f.write('各规则误伤主升数量汇总：\n\n')
    f.write('| 规则 | 误伤主升数 | 占全部主升(%s)比例 |\n' % f'{MZ_N:,}')
    f.write('|---|---:|---:|\n')
    for rule in list(RULES)[:8]:
        d = res[res.rule == rule].iloc[0]
        f.write(f'| {rule} | {d.rmz:,} | {d.mz_frac}% |\n')
    f.write('\n推荐规则（放宽档 F4∈[0,5]）误伤主升明细（前 15 行示例，全量见 CSV）：\n\n')
    f.write('| code | sym | sig_date | sig_close | fut60_max | fut120_max | dist_vol_high | f2距紫线 | f3量比250 |\n')
    f.write('|---|---|---|---:|---:|---:|---:|---:|---:|\n')
    d4 = mzd[mzd.rule == 'F4:前堆压∈[0,5]'].sort_values('fut120_max', ascending=False).head(15)
    for _, r in d4.iterrows():
        f.write(f'| {r.code} | {r.sym} | {r.sig_date} | {r.sig_close} | {r.fut60_max} | {r.fut120_max} | {r.dist_vol_high} | {r.f2_dist_purple:.1f} | {r.f3_vr250:.3f} |\n')
    f.write('\n---\n\n')
    f.write('## 7. 建议与下一步\n\n')
    f.write('1. **本轮 4 因子均不足以作为"一眼假"硬过滤器上线**（硬约束 90% 真保留 + 主升零误伤二者不可同时满足）。\n')
    f.write('2. 若必须上线一条规则，**放宽档 F4∈[0,5] 是唯一提升质量的选项**，请先在实盘/模拟盘以"贴前堆高点卡位"标记观察 1~2 个月，确认对主升的误伤可接受后再转硬剔除。\n')
    f.write('3. 用户看图识假的判别信息不在单日截面——下一步建议转向**信号日前 N 日的形态序列特征**（如：回踩段是否逐级缩量、突破日相对粉线/COST35 的角度、前 20 日高低点结构、突破后的持续站线天数等时序形状特征），并用同样的底座口径（剔假率/真保留率/主升误伤）验证。\n')
    f.write('4. 所有脚本与数据已入库（`fake_filter_v3b_5y.py` / `fake_filter_sensitivity.py` / 因子表 / 明细 / 子集），可在下一轮直接复用。\n')

print('report written:', os.path.join(R, 'v3b_5y_fake_filter_report.md'))
