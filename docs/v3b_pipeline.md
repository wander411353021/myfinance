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
