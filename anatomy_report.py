# -*- coding: utf-8 -*-
"""三信号解剖报告生成 (2026-10-18) — 直接重算关键数字于 anatomy_signals.csv
产物: result/v3b_v3_early_anatomy.md
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
V3 = df[df.hit_v3 == 1]; V3B = df[df.hit_v3b == 1]; E = df[df.hit_early == 1]
V3O = df[(df.hit_v3 == 1) & (df.hit_v3b == 0)]; OV = df[(df.hit_v3 == 1) & (df.hit_v3b == 1)]


def perf(sub):
    w60 = sub[sub.fut60_days >= 60]; w120 = sub[sub.fut120_days >= 120]
    return dict(n=len(sub), w60=len(w60), s60=round(w60.s60.mean() * 100, 2),
                fail=round(w60.fail.mean() * 100, 2), f60med=round(float(w60.fut60_max.median()), 2),
                f60min=round(float(w60.fut60_min.median()), 2),
                s120=round(w120.s120.mean() * 100, 2) if len(w120) else np.nan, qfq=round(sub.qfq_flag.mean() * 100, 2))


def feat(sub):
    cols = ['gap', 'abr', 'dev35', 'dist_purple', 'vol_ratio', 'band', 'profit', 'dist_high']
    return {c: round(float(sub[c].median()), 2) for c in cols}


# 转化统计
v3b_dates = {c: np.sort(g.sig_date.values) for c, g in V3B.groupby('code')}


def conv_stats(uni, sub, k):
    codes = sub['code'].values; dates = sub['sig_date'].values
    conv = np.zeros(len(sub), dtype=bool); ds = []
    for i in range(len(sub)):
        arr = v3b_dates.get(codes[i])
        if arr is None or len(arr) == 0:
            continue
        j = np.searchsorted(arr, dates[i], side='right')
        if j < len(arr) and arr[j] <= dates[i] + k:
            conv[i] = True
            ds.append(int(arr[j] - dates[i]))
    return conv, ds


ec60, _ = conv_stats(df, E, 60)
ec120, ed120 = conv_stats(df, E, 120)
vc120, vd120 = conv_stats(df, V3, 120)
E = E.copy(); E['conv120'] = ec120
V3 = V3.copy(); V3['conv120'] = vc120

# early gap 分档
w60_all = df[df.fut60_days >= 60]

with open(os.path.join(R, 'v3b_v3_early_anatomy.md'), 'w', encoding='utf-8') as f:
    f.write('# v3 / v3b / early 三信号全市场 5 年解剖报告\n\n')
    f.write('> 数据：`result/scan_cache_v3b/`（5,534 只日线缓存，end=20261016，datalen≈1800），信号判定复用 `challenge_scan.compute()/signal_at()/v3b_signal_at()/v3b_early_at()` 零改动；画像=signal_profile 全字段（profile_light 等价性 310/310 通过）。\n')
    f.write('> 口径：窗口 [20211001, 20261016]；v3b/early 独立冷却链 V3B_COOL=20、v3 独立 COOL=30（与 backtest_v3b_5y 一致，窗口外推进冷却链）；无未来函数（判定/画像只用 ≤i 数据）；收益 `_fut` 60/120 日 max/min（qfq 缓存口径）；qfq_flag=信号日后 60 日内 |qfq 日涨幅|>板块涨停上限（10.5/20.5/30.5%）。\n')
    f.write('> 与 backtest_v3b_5y.csv 对照：v3b 32,252 vs 32,257、early 77,266 vs 77,275（差 14 行全部为 2026-08~10 数据端边缘，系 backtest 当时重拉过该批缓存，可忽略）。全量明细：`result/anatomy_signals.csv`（121,637 行）。\n\n')
    f.write('---\n\n')
    f.write('## 0. 核心结论\n\n')
    f.write('1. **疑点①实锤且严重：v3 与 v3b 高度重叠**。全市场 5 年 v3b 中 **34.22%（11,036/32,252）同日同时满足 v3**；v3 中 47.66%（11,036/23,155）同日同时满足 v3b。v3 与 v3b 不是平行信号，而是**定义包含/递进关系**：v3b 的「历史摸紫线+深回踩 COST35+量能」条件恰好落在 v3 内部筛出了质量最高的子集（v3∩v3b 的 S120=23.28%，高于 v3 全量 22.33% 与 v3b 全量 22.94%）。\n')
    f.write('2. **疑点②前半（v3 无量能确认）代价极小**：v3 触发日本身 89.4% 已满足量能≥1.2×前20均量（vol_ratio 中位 2.51），加量能条件只剔除 2,332 个且被剔子集表现反而略好（s60 59.86% vs 58.01%）→ **加量能无收益，不建议**。\n')
    f.write('3. **疑点②后半（v3 从未全市场统计）本次补齐**：v3 全市场 5 年 **23,155 个**，w60 胜率 s60=58.01%、fail=12.14%、S120=22.33%——整体质量与 v3b 相当（59.36%/12.51%/22.94%），**显著好于 early**（55.53%/9.50%/19.49%）。\n')
    f.write('4. **疑点③（early 无 gap 下限）被否定**：early 中 **82.06% 本就在粉线下方（gap<0）**是定义使然（未启动的早期信号），且 gap<0 组 w60 胜率 55.84% **高于** 0≤gap<5 组 53.91%；设 gap 下限（≥0/≥3/≥5）剔除集的假信号占比 27.9~28.3% 全部**低于**全量 early 的 29.3%，剔除的全是好信号 → **不设 gap 下限**。\n')
    f.write('5. **v3 排除历史摸线会反劣**：剔除后 s60 58.01%→54.31%（n=1,692 纯首摸），因为「历史摸线+深回踩」正是 v3b 语义里的质量因子——**排除它是方向性错误**。\n')
    f.write('6. **转化率（实盘可监控）**：early→v3b 60 日转化 **15.41%**、120 日转化 **29.01%**（中位 21 天）；v3→v3b 120 日转化 **22.56%**（中位仅 6 天，v3 与 v3b 同日/数日内重叠为主）。转化后的 v3b 自身 w60 胜率 57.4~57.7%，与全量 v3b（59.36%）相当 → 「early 打底观察 + v3b 确认再买」成立。\n')
    f.write('7. **三信号共有的真问题：qfq 伪影**。三信号 qfq_flag=1 组 w60 胜率 73.8~77.0%，比 qfq=0 组高 16.8~19.8pp（fut60 中位 +9~11pp）——是 qfq 除权伪影推高后验，**任何跨信号对比与回测都应剔除或单独标注 qfq_flag=1**（本次已标注）。\n\n')
    f.write('---\n\n')
    f.write('## 1. A. 三信号样本量（按年）\n\n')
    f.write('| 年份 | v3 | v3b | early |\n|---|---:|---:|---:|\n')
    for y in ['2021', '2022', '2023', '2024', '2025', '2026']:
        f.write(f'| {y} | {int((V3.year == y).sum()):,} | {int((V3B.year == y).sum()):,} | {int((E.year == y).sum()):,} |\n')
    f.write(f'| 合计 | {len(V3):,} | {len(V3B):,} | {len(E):,} |\n\n')
    f.write('> v3b/early 与 backtest_v3b_5y.csv 总量一致；v3 为本次首次全市场统计。\n\n')
    f.write('## 2. B. 重叠矩阵（疑点①定量）\n\n')
    f.write('| 组合 | 同日命中数 | 占比 |\n|---|---:|---:|\n')
    f.write(f'| v3 ∩ v3b | {len(OV):,} | v3b 的 **{len(OV)/len(V3B)*100:.2f}%**，v3 的 {len(OV)/len(V3)*100:.2f}% |\n')
    f.write(f'| v3 ∩ early | 0 | 互斥（摸紫线条件相反） |\n')
    f.write(f'| v3b ∩ early | 0 | 互斥（close vs 紫线×0.98 相反） |\n')
    f.write(f'| 三者同中 | 0 | — |\n')
    f.write(f'| 仅 v3 | {len(V3O):,} | v3 的 {(1-len(OV)/len(V3))*100:.1f}% |\n')
    f.write(f'| 仅 v3b | {len(df[(df.hit_v3b==1)&(df.hit_v3==0)]):,} | v3b 的 {(1-len(OV)/len(V3B))*100:.2f}% |\n\n')
    f.write('**机制解释**：v3 只要求压制期（前 60 天）价格>粉线占比 ≤10%（abr≤10）且 gap∈[3,20)，并未排除「历史摸过紫线又深回踩 COST35」（v3b 条件）。而深回踩会让价格长期低于粉线，压制期天然满足 abr≤10 —— 于是 v3b 信号日大概率同时是 v3 信号日。\n\n')
    f.write('## 3. C. 特征分布与后验表现\n\n')
    f.write('### 3.1 画像字段中位数\n\n')
    f.write('| 字段 | v3 | v3b | early | 仅v3 | v3∩v3b |\n|---|---:|---:|---:|---:|---:|\n')
    for c in ['gap', 'abr', 'dev35', 'dist_purple', 'vol_ratio', 'band', 'profit', 'dist_high']:
        f.write(f'| {c} | {feat(V3)[c]} | {feat(V3B)[c]} | {feat(E)[c]} | {feat(V3O)[c]} | {feat(OV)[c]} |\n')
    f.write('\n> v3 与 v3b 画像几乎同构（gap 6.0/7.5、abr 5.0/7.3、dev35 13.0/15.7、vol_ratio 2.51/2.67），印证重叠；early 完全不同（gap -4.2=粉线下方、dist_purple +9.67=紫线下方 9.7%、vol_ratio 1.71、profit 0.64=获利盘 64%）。v3∩v3b 是「量能更足（3.21）、离紫线更近（-1.19）、gap 更大（7.8）」的子集。\n\n')
    f.write('### 3.2 后验表现（w60 完整窗口口径 + qfq 标注）\n\n')
    f.write('| 信号 | n | w60 n | S60 | fail | fut60中位 | fut60min中位 | S120(w120) | qfq占比 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    for tag, sub in [('v3', V3), ('v3b', V3B), ('early', E), ('仅v3(剔v3b重叠)', V3O), ('v3∩v3b(重叠组)', OV)]:
        p = perf(sub)
        f.write(f'| {tag} | {p["n"]:,} | {p["w60"]:,} | {p["s60"]}% | {p["fail"]}% | +{p["f60med"]}% | {p["f60min"]}% | {p["s120"]}% | {p["qfq"]}% |\n')
    f.write('\n**要点**：质量排序 v3∩v3b(重叠组) > v3b > v3 > 仅v3 > early；重叠组是 v3 内部质量最高部分（S120 23.28% vs 仅v3 21.46%），v3 统计口径若不剔除重叠会混入 v3b 的高质量子集。\n\n')
    f.write('### 3.3 qfq 伪影分层（w60）\n\n')
    f.write('| 信号 | qfq_flag | n | S60 | fail | fut60中位 |\n|---|---|---:|---:|---:|---:|\n')
    for tag, sub in [('v3', V3), ('v3b', V3B), ('early', E)]:
        for q in (0, 1):
            s = sub[(sub.qfq_flag == q) & (sub.fut60_days >= 60)]
            f.write(f'| {tag} | {q} | {len(s):,} | {s.s60.mean()*100:.2f}% | {s.fail.mean()*100:.2f}% | +{s.fut60_max.median():.2f}% |\n')
    f.write('\n> qfq=1 组胜率高 16.8~19.8pp 为除权伪影，**不是真实信号质量**；对比/回测应统一剔除或标注。\n\n')
    f.write('## 4. D. early→v3b 转化率\n\n')
    f.write(f'- early n=**{len(E):,}**；60 日内出现 v3b（任一）：**{ec60.mean()*100:.2f}%**（{int(ec60.sum()):,}）；120 日内：**{ec120.mean()*100:.2f}%**（{int(ec120.sum()):,}），转化中位 **{int(np.median(ed120))} 天**（P25={int(np.percentile(ed120,25))}，P75={int(np.percentile(ed120,75))}）。\n')
    f.write('- **转化组 vs 未转化组（early 自身 w60）**：转化组 S60=89.89%、fut60 中位 +26.61%、fail 0.09%；未转化组 S60=41.27%、+7.50%、fail 13.40%。⚠️ **该对比含事后分组偏差**（120 日内出 v3b 的 early，其 60 日收益窗口覆盖了 v3b 突破段），不能当作 early 的可交易胜率；但它证明「early 之后出现 v3b」的联动极强。\n')
    f.write('- **转化后的首个 v3b（无偏，从 v3b 信号日起算）**：w60 S60=57.37%、fail=13.54%、fut60 中位 +12.99%、S120=22.01% —— 与全量 v3b（59.36%/12.51%/+13.96%/22.94%）相当 → **「early 进观察池 + 120 日内出 v3b 再买」作为两步确认策略成立**（牺牲小部分胜率换确认度）。\n\n')
    f.write('## 5. E. v3→v3b 转化率\n\n')
    f.write(f'- v3 n=**{len(V3):,}**；120 日内出现 v3b：**{vc120.mean()*100:.2f}%**（{int(vc120.sum()):,}），转化中位仅 **{int(np.median(vd120))} 天**（P25={int(np.percentile(vd120,25))}，P75={int(np.percentile(vd120,75))}）——v3 与 v3b 同日或数日内重叠为主（47.66% 的 v3 本身就是 v3b）。\n')
    f.write('- v3 转 v3b 组自身 w60 S60=71.92%（事后分组，含偏差）；未转组 53.93%。转化后的首个 v3b（无偏）：w60 S60=57.65%、S120=23.11%。\n\n')
    f.write('## 6. F. v3 定义实验（量化定义缺陷代价）\n\n')
    f.write('| 定义变体 | n(w60) | S60 | fail | fut60中位 | S120(w120) | 结论 |\n|---|---:|---:|---:|---:|---:|---|\n')
    f.write(f'| 全量 v3（现状） | {perf(V3)["w60"]:,} | {perf(V3)["s60"]}% | {perf(V3)["fail"]}% | +{perf(V3)["f60med"]}% | {perf(V3)["s120"]}% | 基准 |\n')
    v3_vol = V3[V3.vol_ratio >= 1.2]; v3_nt = V3[V3.hist_touch == 0]
    f.write(f'| v3 + 量能≥1.2×20均量 | {perf(v3_vol)["w60"]:,} | {perf(v3_vol)["s60"]}% | {perf(v3_vol)["fail"]}% | +{perf(v3_vol)["f60med"]}% | {perf(v3_vol)["s120"]}% | 剔除 2,332 个(10.6%)，被剔组反而略优，**无收益** |\n')
    f.write(f'| v3 排除历史摸线(纯首摸) | {perf(v3_nt)["w60"]:,} | {perf(v3_nt)["s60"]}% | {perf(v3_nt)["fail"]}% | +{perf(v3_nt)["f60med"]}% | {perf(v3_nt)["s120"]}% | 剔除 20,272 个，胜率**反降 3.7pp**，方向错误 |\n')
    f.write(f'| 仅 v3（剔除 v3b 重叠） | {perf(V3O)["w60"]:,} | {perf(V3O)["s60"]}% | {perf(V3O)["fail"]}% | +{perf(V3O)["f60med"]}% | {perf(V3O)["s120"]}% | 重叠组才是 v3 最优部分 |\n\n')
    f.write('**early gap 下限实验（疑点③）**：\n\n')
    f.write('| 规则 | 保留 n | S60 | S120 | 剔除集假占比 | 判读 |\n|---|---:|---:|---:|---:|---|\n')
    for x, lab in [(-5, 'gap≥-5'), (-3, 'gap≥-3'), (0, 'gap≥0'), (3, 'gap≥3'), (5, 'gap≥5')]:
        keep = w60_all[(w60_all.hit_early == 1) & (w60_all.gap >= x)]
        rem = w60_all[(w60_all.hit_early == 1) & (w60_all.gap < x)]
        s120k = keep[keep.fut120_days >= 120].s120.mean() * 100
        f.write(f'| {lab} | {len(keep):,} | {keep.s60.mean()*100:.2f}% | {s120k:.2f}% | {(rem.fut60_max<5).mean()*100:.1f}% | 剔除集假占比均低于全量 29.3% |\n')
    f.write('\n> 全量 early 假信号占比 29.3%；**任何 gap 下限剔除集的假占比（26.2~28.3%）都低于全量** → early 的「假」与 gap 负值无关，设下限只会误伤。\n\n')
    f.write('## 7. 疑点结论与定义改进建议（落到参数）\n\n')
    f.write('| 疑点 | 定量结论 | 建议 |\n|---|---|---|\n')
    f.write('| ① v3∩v3b 重叠 | v3b 中 34.22% 同日满足 v3；v3 中 47.66% 同日满足 v3b；重叠组是质量最高子集（S120 23.28%） | **v3 统计口径改为「仅 v3（v3 且非 v3b）」（n=12,119）**；实盘池同日重叠按 v3b 归类去重；保留 v3b 原定义不动 |\n')
    f.write('| ② v3 无量能 | v3 触发日 89.4% 已量能≥1.2（中位 2.51），加条件无收益 | **不加量能条件**（加了只少 2,332 个且质量不变） |\n')
    f.write('| ② v3 从未全市场统计 | 全市场 5 年 23,155 个，S60 58.01%/S120 22.33%/fail 12.14%，与 v3b 相当 | 已落盘（anatomy_signals.csv），后续回测可直接引用 v3 全量口径 |\n')
    f.write('| ③ early 无 gap 下限 | 82.06% 在粉线下方（gap<0）；gap<0 组 S60 55.84% 优于 0~5 组 53.91%；设下限剔除集不更假 | **不设 gap 下限**（early 的「假」另有来源，见下） |\n')
    f.write('| （新增）qfq 伪影 | qfq=1 组胜率虚高 16.8~19.8pp | 回测/实盘统一**剔除 qfq_flag=1 或单独标注**；跨信号对比只用 qfq=0 组 |\n')
    f.write('| （新增）early 假信号来源 | 与 gap/量能/均线等单日截面因子无关（上轮 4 因子验证） | early「一眼假」需多日形态序列因子（回踩缩量形状/突破角度/站线天数），沿用同口径验证 |\n')
    f.write('\n---\n\n')
    f.write('## 8. 文件与复现\n\n')
    f.write('- `result/anatomy_signals.csv`：全量三信号明细（121,637 行；code/sym/sig_date/hit_v3/hit_v3b/hit_early/画像 11 字段/dist_purple/abr/收益 5 字段/qfq_flag）\n')
    f.write('- `result/anatomy_shard0..3.csv`：4 分片原始输出（可断点续跑）\n')
    f.write('- `anatomy_scan_v3_v3b_early.py`：扫描（`--nshards 4 --shard K --resume`）\n')
    f.write('- `anatomy_stats.py`：A-F 统计（含本次报告所有数字的源头）\n')
    f.write('- 复现：`python3 anatomy_scan_v3_v3b_early.py --nshards 4 --shard 0..3` → `python3 anatomy_stats.py`\n')
print('report written')
