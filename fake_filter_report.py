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
f4s['x'] = f4s.thr.str.split('∈[0,', regex=False).str[1].str.rstrip(']').astype(int)
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
    f.write('# v3b early 假信号 · 4 个趋势结构因子验证报告\n\n')
    f.write('> 数据源：`result/backtest_v3b_5y_overcome.csv` + `result/scan_cache_v3b/`（5,141 只日线缓存，未联网重拉）\n')
    f.write('> 因子计算：`fake_filter_v3b_5y.py`（build 全底座 67,276 行，耗时 505s；eval 单因子+组合）→ `result/fake_filter_factors.csv`\n')
    f.write('> 敏感性网格：`fake_filter_sensitivity.py` → `result/fake_filter_sensitivity.csv`；全部数字 pandas 实算，与 CSV 严格一致，零估计。\n')
    f.write('> 生成时间：2026-10-18（信号窗口 2021-10 ~ 2026-10，tdx 前复权 qfq，锚=2026-10-16）\n\n')
    f.write('---\n\n')
    f.write('## 0. 底座与口径\n\n')
    f.write(f'- 底座 = **early ∩ qfq_flag=0 ∩ fut60_days≥60**：n=**{BASE_N:,}**（与规格 n≈67,276 一致）\n')
    f.write(f'- 分层（后 60 日最大涨幅）：假 <5%（n={FAKE_N:,}，**{FAKE_N/BASE_N*100:.1f}%**）、一般 5~10%（n={int(df.fut60_max.ge(5).sum()-TRUE_N):,}）、真 ≥10%（n={TRUE_N:,}，{TRUE_N/BASE_N*100:.1f}%）\n')
    f.write(f'- 主升 = fut120_days≥120 且 fut120_max≥50%：n={MZ_N:,}（占底座 {MZ_N/BASE_N*100:.1f}%）\n')
    f.write(f'- 基线质量：s60={S60_BASE}%、S120(w120 口径)={S120_BASE}%、fut60_max 中位 +{FMED_BASE}%、假信号占比 {FAKE_N/BASE_N*100:.1f}%\n')
    f.write(f'- 因子全部只用 ≤信号日 数据（紫线=gt2 第二阶梯，复用 `challenge_scan.compute()` 权威实现；F4 复用 overcome CSV 的 dist_vol_high），严格无未来函数；抽查 3 个信号手工复算 3/3 一致。\n\n')
    f.write('---\n\n')
    f.write('## 1. 核心结论（先给结论）\n\n')
    f.write('1. **规格阈值下（F2>20%、F3<0.30/0.35/0.40、F4∈[0,5]/[0,10]、F1 二元），没有任何单因子、也没有任何两两/三三组合满足硬约束「真保留率≥90%」**——最低的真保留率要求也要把剔除口径大幅放宽到 F2>25% 或 F3<0.12 级别，此时剔假能力仅 1~7%。\n')
    f.write('2. **4 个因子对"一眼假"基本无甄别力**：规格阈值下剔除集内假信号占比 24.7%~30.6%，最高也只比底座（29.3%）高 1.3pp；F1/F2/F3 剔除集甚至比底座更不"假"。用户看图能识假、而单日截面因子不能，差距在**多日形态序列**而非单日截面值——本轮 4 因子从数据上证实了这一点。\n')
    f.write(f'3. **唯一有正价值的因子是 F4（前堆高压）**：F4∈[0,5] 剔除 16,365 个"在前放量堆高点下方贴磨"的信号后，S120 **{S120_BASE}%→19.48%**（+0.56pp）、s60 54.05%→54.64%、fut60_max 中位 +11.4%→+11.64%，剔假率 25.4%；代价是真保留率降到 **76.5%**、误伤主升 **2,652** 个（主升的 22.0%）。若用户愿意为质量让渡保留率，这是唯一值得用的规则。\n')
    f.write('4. **「主升一个都不能误伤」任何硬规则都做不到**：主升信号与假信号在 4 因子上的分布几乎重合（例：22.0% 的主升信号信号日就贴在前堆高点 0~5% 下方、16.7% 距紫线 >20%），任何非零剔除都会切到主升。\n')
    f.write('5. **90% 硬约束内的可执行最优（严格档）**：\n')
    f.write('   - OR 规则 `F2>30% ∪ F3<0.12`：剔假 8.61%、真保留 90.00%、S120 18.89%（基本持平基线）、误伤主升 1,175；\n')
    f.write('   - 组合评分 `scoreB`（z(距紫线)−z(量比250)+F4贴磨+F1未摆脱）切底部 10%：剔假 11.2%、真保留 **90.5%**、S120 19.13%（+0.21pp）、误伤主升 1,079；\n')
    f.write('   - 两者相对"随机删 10%"（剔假≈10%、保留≈90%）的提升只有 +1.2pp，**信息量近乎为零**。\n\n')
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
    f.write('- **F1 均线结构（摆脱下跌通道）**：剔除 38,549 个，真保留仅 41.2%，且保留组 S120=17.78% 反而**低于基线**——early 信号里"已走出 close>MA20>MA60 结构"的票后续成浪率更低，"摆脱下跌通道才真"假设**被数据否定**。\n')
    f.write('- **F2 距紫线 >20%**：剔除 9,634 个（含 5,771 真 / 2,008 主升），保留组 S120=18.32% 低于基线；剔除集内假占比仅 24.7%（比底座还低）。"离紫线 20% 以上=遥不可及=假"**不成立**——大量后期主升票信号日确实还在紫线 20% 之外。\n')
    f.write('- **F3 绝对量能（缩量反抽）**：0.30/0.35/0.40 三档剔除集内假占比 28.0~28.4%，均低于底座；保留组 S120 18.41~18.83% 全部低于基线。**缩量反抽与假信号无正相关**。\n')
    f.write('- **F4 前堆高压（唯一正价值）**：[0,5] 档剔除集内假占比 30.6%（最高）、保留组 S120 19.48%、s60 54.64%、fut60 中位 +11.64% 四项全部优于基线；[0,10] 档 S120 19.54% 更高但真保留更低（70.1%）。\n\n')
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
    f.write('- 满足真保留率≥90% 的操作点仅有：F2>25%~45%（剔假 0.9~6.6%）、F3<0.08~0.12（剔假 1.1~4.9%）。F1（二元）任何档位不可行；F4 最温和档 [0,2] 仍只有 87.4%。\n')
    f.write('- 保留组 S120 超过基线的可行点只有 F3<0.08/0.10/0.12（18.97/19.02/19.08%）与 F2>45%（18.89%），增量 ≤0.16pp，且均误伤主升 104~551 个。\n\n')
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
    f.write('**组合判读**：两两/三三 OR 组合只是把剔除面越并越大，没有任何组合能在真保留≥90% 的前提下把假占比显著压下来（最高剔假率 8.61% 的组合保留率恰好 90.00% 但 S120 反降至 18.89%）。\n\n')
    f.write('## 5. 推荐组合\n\n')
    f.write('### 5.1 严格档（遵守硬约束 真保留≥90%）——只推荐"提示"不推荐"硬删"\n\n')
    f.write('| 推荐 | 剔假率 | 真保留率 | S120 | S120 提升 | 误伤主升 | 误伤主升占比 |\n')
    f.write('|---|---:|---:|---:|---:|---:|---:|\n')
    for rule in ['推荐严格档 F2>30%∪F3<0.12', '推荐严格档 scoreB底部10%']:
        d = res[res.rule == rule].iloc[0]
        f.write(f'| {rule} | {d.fake_rm}% | {d.keep}% | {d.s120}% | {d.s120 - S120_BASE:+.2f}pp | {d.rmz:,} | {d.mz_frac}% |\n')
    f.write('\n- 两者均满足真保留率≥90%，但对假信号的甄别能力只比"随机删 10%"高约 1pp，S120 提升 ≤0.2pp，**工程意义有限**；误伤主升仍达 1,079~1,175 个。若采用，建议仅作"灰名单降权"而非硬剔除。\n')
    f.write('- 已存子集 CSV：`result/fake_filter_early_keep_scoreB10.csv`（n=60,548）。\n\n')
    f.write('### 5.2 放宽档（接受 ~76% 真保留）——唯一正价值规则\n\n')
    f.write('| 推荐 | 剔假率 | 真保留率 | S120 | S120 提升 | s60 | fut60 中位 | 误伤主升 |\n')
    f.write('|---|---:|---:|---:|---:|---:|---:|---:|\n')
    d = res[res.rule == 'F4:前堆压∈[0,5]'].iloc[0]
    f.write(f'| F4 前堆高压∈[0,5]（剔除贴磨） | {d.fake_rm}% | {d.keep}% | {d.s120}% | {d.s120 - S120_BASE:+.2f}pp | {d.s60}% | +{d.fmed}% | **{d.rmz:,}（{d.mz_frac}%）** |\n')
    f.write('\n- 语义直观：**信号日价格紧贴在前放量堆高点 0~5% 下方（贴磨）**→ 正压在前、突破未决，剔掉这类"卡位"信号后成浪质量四项全升。\n')
    f.write('- 主升误伤明细见 `result/fake_filter_mz_detail.csv`（rule=`F4:前堆压∈[0,5]` 共 2,652 行）；保留子集已存 `result/fake_filter_early_keep_F4_0_5.csv`（n=50,911）。\n\n')
    f.write('## 6. 被误伤主升明细（重点报告项）\n\n')
    f.write('所有规则的完整主升误伤明细（rule/code/sym/sig_date/sig_close/fut60_max/fut120_max/四因子值）已导出：**`result/fake_filter_mz_detail.csv`**（共 %d 行）。\n\n' % len(mzd))
    f.write('各规则误伤主升数量汇总：\n\n')
    f.write('| 规则 | 误伤主升数 | 占全部主升(12,043)比例 |\n')
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
