# v3b / early 五年回溯与假信号研究：完整复现链路

> 适用仓库：`pressure-level-algorithm`（工作区 `E:\chip_analyzer_ui\new_algo`）
> 最后更新：2026-10-18
> 运行环境：`conda activate chip_analyzer`（`eltdx==3.1.7`、`fengwo==0.0.7`、pandas、numpy）
> 红线：所有信号判定 / 画像只用 **≤ 信号日** 数据；`qfq_flag` 是事后质量标记，**禁止进入 signal 层**。

---

## 1. 这套流水线解决什么问题

`challenge_scan.py` 的 `v3b` / `early` 是全市场信号定义，`backtest_v3b_5y.py` 做五年回溯。
但回溯产物、`qfq_flag` 生成脚本、假信号报告都放在被 `.gitignore` 忽略的 `result/` 下，
2026-10-18 环境重置后**只剩脚本和结论，无法端到端重建**。

本文件把链路固定下来：任何人在干净仓库里按顺序执行即可重建全部中间表与结论
（首次全市场跑批约数小时，主要耗时是联网取数与 `compute()`）。

---

## 2. 端到端命令

```bash
# 0) 环境
conda activate chip_analyzer

# 1) 五年回溯（8 片并行；可断点续跑，已完成的 code 会跳过）
for i in 0 1 2 3 4 5 6 7; do
  python backtest_v3b_5y.py --part $i --nparts 8 &
done
wait

# 2) 合并分片 + 完整性校验（各片之和、重复、日期越界、已知案例）
python merge_backtest.py
#   产物 result/backtest_v3b_5y.csv

# 3) 附加 qfq_flag（除权/股本事件标记；带缓存，可断点续跑）
python add_qfq_flag.py --horizon 60 --workers 4
#   产物 result/backtest_v3b_5y_qfq.csv + result/xdxr_cache/{code}.json

# 4) 前放量堆高点 / 克服字段（纯本地缓存计算，不联网）
python overcome_analysis.py
#   产物 result/backtest_v3b_5y_overcome.csv

# 5) 假信号 4 因子：建因子表 → 单因子+组合评估
python fake_filter_v3b_5y.py build     # 产物 result/fake_filter_factors.csv
python fake_filter_v3b_5y.py eval      # 产物 fake_filter_single/combo.csv

# 6) 阈值敏感性网格 + 可行操作点
python fake_filter_sensitivity.py

# 7) 生成最终报告（报告 md 会入库，CSV/PNG 仍忽略）
python fake_filter_report.py
#   产物 result/v3b_5y_fake_filter_report.md
```

单只复验 / 画像：

```bash
python backtest_v3b_5y.py --verify 300204 20250418 1800
python challenge_scan.py hist_v3b 600550 20261016 1600
python challenge_scan.py assess 600550 20261016 20260904
```

---

## 3. 字段口径

### 3.1 信号与画像（`backtest_v3b_5y.py`）

| 字段 | 含义 | 是否因果 |
|---|---|---|
| `sig_type` | `v3b`（回踩后二次摸紫线）/ `early`（站上 COST35 未到紫线） | — |
| `sig_date` / `sig_close` | 信号日 / 当日收盘 | 当日可见 |
| `vol_ratio` | 当日量 / 前 20 日均量 | ✅ |
| `rec_depth` / `rec_days` | [T-70,T-5] 相对 COST35 最深偏离 / 跌破天数 | ✅ |
| `cool_days` / `hist_touch` | 距上次摸紫线天数 / 近 250 日摸线次数 | ✅ |
| `dist_high` | 现价距近 250 日最高价 | ✅ |
| `dev35` | 现价相对 COST35 偏离 | ✅ |
| `band` / `profit` | 筹码集中带宽 / 获利盘近似 | ✅ |
| `gap` | 现价相对粉线（Grid Target）偏离 | ✅ |
| `fut60_max` / `fut60_min` / `fut120_max` | 信号日**之后** 60/120 日最大/最小涨幅 | ❌ 事后验证 |
| `s60` / `s120` / `fail` | 后 60 日 max>10% / 后 120 日 max>50% / 后 60 日 max≤0 | ❌ 事后 |
| `big20` / `big_wait` | 后 20 日是否出 ≥5% 大阳 / 等待天数 | ❌ 事后质量标签 |

### 3.2 `qfq_flag`（`add_qfq_flag.py`）

| 值 | 定义 |
|---|---|
| `1` | 信号日之后 60 个交易日内存在除权/股本变动记录（`category_raw ∈ {1,2,3,5,9,14}`） |
| `0` | 窗口内无此类记录 |
| `NaN` | 信号日后不足 60 个交易日，或接口失败 |

数据源：主路径 `client.corporate.capital_changes(code)`；回退 `client.get_xdxr(code)`。
**用途**：回溯样本口径剔除（避免复权因子重锚污染 fut60 收益），不得用于买入判定。

### 3.3 假信号分层（`fake_filter_v3b_5y.py`）

底座 = `sig_type == 'early' AND qfq_flag == 0 AND fut60_days >= 60`

- 假：`fut60_max < 5%`
- 一般：`5% ≤ fut60_max < 10%`
- 真：`fut60_max ≥ 10%`
- 主升：`fut120_days ≥ 120 AND fut120_max ≥ 50%`

4 个因子（全部 ≤ 信号日）：

| 因子 | 定义 |
|---|---|
| F1 | `close > MA20 且 MA20 > MA60`（剔除不满足者） |
| F2 | 距紫线 `(gt2-close)/close > 20%` |
| F3 | 当日量 / 前 250 日最大单日量 `< 阈值` |
| F4 | 前放量堆高点压力 `dist_vol_high ∈ [0,5]`（来自 `overcome_analysis.py`） |

### 3.4 P0/P1 算法修正（2026-10-18，688099 案例驱动）

| 项 | 口径 | 状态 |
|---|---|---|
| **P0-1 结构冷却** | `scan_series` 默认 `mode='structure'`：跌破 COST35×0.98 重新武装，同一结构区间只输出第一次触发；`mode='days'` 复现旧 20 天冷却 | 已定版 |
| **P0-2 置空备用线** | `compute_eff` 返回 `gt_p_raw/gt2_raw` + `gt_p_eff/gt2_eff`；当日判定用 eff（置空日 = base×1.13），历史压制期/摸线窗口用 raw（NaN 不计数） | 已定版 |
| **P1-1 v3 接入每日池** | `scan_recent` types = `('v3','v3b','early')`，三类独立结构 armed | 已定版 |
| **P1-2 early 趋势确认** | MA20/MA60、MA20 斜率、close>MA20、3 日不破：50 只/438 信号均未超过基线（42.0%/-0.66%），**不采用** | 负结果 |

- 验证要点：688099 的 8/13–8/19 early 被旧 20 天冷却吞掉；8/25–9/4 置空失明 9 天；9/5、9/10 的 v3 旧管线看不到。
- 截断一致性必须在 `compute_eff` 口径下重算（raw+eff），不能只截断价格不重算线。
- **全局参数入口**：`algo_config.py`（`COST_PCT=0.70` / `COST_EHLERS=90` 主成本线 A，仅显示；`COST_PCT_LOW/MID/HIGH=0.35/0.50/0.75`
  板块三档；`CHIP_BAND_LO/HI` 带宽；`COST2_PCT=0.40` / `COST2_EHLERS=50` 参考线 B，仅显示）。
  **信号标准**：`SIGNAL_COST_PCT=0.35` / `SIGNAL_COST_EHLERS=90`，`SIGNAL_MIN_LEVEL=4`（目标价不能低于基准）；
  `SIGNAL_REARM_DAYS=30`（结构跌破或距上次同类信号 30 日重新武装）；
  筹码预热：`CHIP_WARMUP_DAYS=1800`（总窗口，含显示区间）/ `CHIP_WARMUP_TURNOVER=1.0`（初始日→显示起点累计换手 100%），
  从显示起点往前追溯，100% 或 1800 天先到为准；不足只提示不阻断；
  数据默认 1800 天。`challenge_scan.compute` 的 c35s 与 gt_p/gt2 统一使用该标准。环境变量可覆盖。已接入
  `challenge_scan` / `price_segmenter_v10` / `concept_chip_scan` / `pages/1_Concept_Chip` / `_compare_k`。
- **B2-early 强势分支（2026-10-21）**：正式 `v3b_B2` 仍要求 `gap∈[5,20)`；`v3b_b2_early` 复用 B2
  全部结构/放量条件，仅允许 `gap>20%`，仅作标记、不替代正式 B2。`scan_recent` 会同时返回；
  V10 信号面板蓝色 ▲（y=0.20）。600865：early 0828 → B2-early 0904（gap 28.5%）→ 正式 B2 0915（gap 13.3%）。
- **streamlit 滑轨（A/B 仅显示）**：`主成本线 A · 分位` + `主成本线 A · Ehlers 周期`（只改 A 线和填充）；
  `参考成本线 B · 分位` + `参考成本线 B · Ehlers 周期`（当前 B 线与填色已隐藏，滑轨保留）。v3/v3b/early 固定使用 `SIGNAL_*`。
  拖动后自动重绘；不修改全局默认。
- **主图颜色**：A 深绿虚线 + 绿色填充 alpha 0.15；B 参考线/红色填充当前隐藏（`_SHOW_COST2=False`）。
- **自动重算**默认开启：拖动滑轨/tail_days 后用上次行情重绘（不重新取数）；改代码/日期仍需点按钮。
- ⚠️ `dev35` 字段名保留兼容，值已是主成本线偏离；历史回测 CSV 仍是 COST35 口径。

---

## 4. 口径陷阱（引用结论前必须核对）

1. **`fut60_max` 口径 ≠ 持有 60 日末收益**。
   前者是区间最高价，后者是第 60 日收盘；五年回溯中前者 77.7%、后者仅 51.3%。
2. **`s60`（>10%）≠ `fut60_max` 中位**，也不同 `max>5%` 口径。
3. **比较两个入场时点必须锁定同一样本集合**。bigline 的“+21% vs +11.1%”就是
   不同批次对比（选择偏差），同批子集为 +2.5% vs +8.9%。
4. **`qfq_flag=1` 的信号不是“差信号”**，只是收益口径可能受除权影响；不要把它当因子。
5. 报告中的比例分母随数据重跑变化：`fake_filter_*` 已改为运行时计算，不再断言 67,276。

---

## 5. 产物与入库策略

| 产物 | 路径 | 是否入库 |
|---|---|---|
| 回溯分片 / 合并表 | `result/backtest_v3b_5y*.csv` | ❌ 体积大、可重建 |
| qfq 除权缓存 | `result/xdxr_cache/*.json` | ❌ 可重建 |
| 日线缓存 | `result/scan_cache_v3b/*.csv` | ❌ 可重建（首次联网） |
| 因子表 / 明细 / 子集 | `result/fake_filter_*.csv` | ❌ 可重建 |
| **假信号报告** | `result/v3b_5y_fake_filter_report.md` | ✅ `.gitignore` 已放行 `result/*_report.md` |
| **本复现文档** | `docs/v3b_pipeline.md` | ✅ |

> 入库原则：**脚本 + 结论文档入库，原始数据/中间大表可重建不入库**。
> 若某轮结论只存在于提交信息或 skill 中，必须同步把关键数字写进报告 md。

---

## 6. 已知限制

- `backtest_v3b_5y.py` 的 `END` / `WIN_START` / `WIN_END` / `DATALEN` 是模块常量；
  换锚点需改代码（或后续加 CLI 参数）。
- `add_qfq_flag.py` 的除权记录依赖 `eltdx` 接口结构；若接口升级，`_extract_dates`
  回退解析可能失效，表现为 `qfq_flag=NaN` 增多（日志会打印 `esrc=error`）。
- 首次全市场跑批需要通达信连接；无网络时可以复用 `result/scan_cache_v3b/` 已有缓存，
  但缓存缺失的股票会被跳过（日志计数）。
- `early` 的定位是观察层：22 池验证中仅 28% 在 20 日内兑现为 v3b，不能替代 v3b 确认。
