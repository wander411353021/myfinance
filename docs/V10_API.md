# V10 接口文档（price_segmenter_v10）

> 版本基线：`2026-10-08`（Gitee `247fa32`）
> 模块：`price_segmenter_v10.py`
> 依赖环境：`conda activate chip_analyzer`（`eltdx==3.1.7`、`fengwo==0.0.7`）

---

## 1. 概述

V10 是「因果分段 + 买卖信号 + 多层面板绘图」的一体化模块。它**不自行拉取数据**，输入是标准化后的日线 DataFrame，输出分段结果、信号序列与 PNG 图表。

### 分层

```
调用方（streamlit_demo.py / pages/*.py / 扫描脚本 / CLI）
        │  df_ohlc（date/open/high/low/close/volume[/turnover]）
        ▼
run_segmentation()               ← 计算入口：分段 + 买卖信号（+ 可选恐慌信号）
        │  (c_result, bs_signal, bs_reason, bs_strength, all_levels)
        ▼
plot_price_segmentation_v10()    ← 绘图入口：K线/量/强度/筹码/黄金坑多面板
        │
        └── 内部调用外部模块：panic_reversal / chip_panel / aben_patterns
                            / mean_reversion.signal_residual
```

### 数据来源（模块外）

| 接口 | 位置 | 说明 |
|---|---|---|
| `get_daily_kline_from_tdx(code, end_date, datalen=800, with_turnover=True)` | `tdx_quant.py` | 通达信直连个股日线（前复权）；`with_turnover=True` 附加 `turnover` / `circ_shares`（逐日历史股本，因果） |
| `get_bars(cli, code, period='day', count=800, anchor_date=None, page_size=800)` | `eltdx_compat.py` | 板块/指数 K 线兼容层（新版 `all_pages` / 旧版 `start` 分页降级） |
| `Client(timeout=...)` | `eltdx`（3.1.7） | 通达信客户端；**本地必须 3.1.7**（1.0.2 会报 `invalid kline date`） |

---

## 2. 主入口

### 2.1 `run_segmentation(...)`

```python
run_segmentation(df_ohlc, tail_days=200, name="",
                 lookback=15, min_reversal_pct=0.02, confirm_bars=3,
                 save_path=None, fast_mode=False, same_type_merge_gap=20,
                 dur_horizon=120, touch_norm=3,
                 reg_window=120, reg_window_long=250,
                 hide_ma=True,
                 code=None, end_date=None, panic_index=None,
                 enable_panic=False, show_chip=False,
                 hide_overlay_lines=False)
```

**参数**

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `df_ohlc` | DataFrame | — | 必需列见 §4 数据契约 |
| `tail_days` | int | 200 | 绘图显示窗口（最近 N 根）；计算仍用全量 |
| `name` | str | `""` | 图标题名称 |
| `lookback` | int | 15 | 分段器候选回看窗口 |
| `min_reversal_pct` | float | 0.02 | 最小反转幅度（分段判据） |
| `confirm_bars` | int | 3 | 拐点确认所需根数（因果） |
| `save_path` | str \| None | None | PNG 输出路径；None 则 `plt.show()` |
| `fast_mode` | bool | False | True = 跳过绘图，返回 bool（最后一天是否有买入信号） |
| `same_type_merge_gap` | int | 20 | 同类型分段合并间隔 |
| `dur_horizon` | int | 120 | 信号强度评估的回看/持有视窗 |
| `touch_norm` | int | 3 | 触碰标准化系数 |
| `reg_window` | int | 120 | 中期回归窗口（reg120）；`0` 不计算 |
| `reg_window_long` | int | 250 | 长期回归窗口（reg250）；`0` 不计算 |
| `hide_ma` | bool | True | 是否隐藏 MA120 / EMA |
| `code` | str \| None | None | 股票/板块代码（`show_chip` 补拉 turnover、恐慌信号用） |
| `end_date` | str \| None | None | 结束日期 `YYYYMMDD` |
| `panic_index` | — | None | 市场恐慌指数（旧接口，保留） |
| `enable_panic` | bool | **False** | 是否计算 `panic_reversal.signal()`；**默认关**（计算耗时 2 分钟+） |
| `show_chip` | bool | **False** | True = 筹码模式（2 面板 + 筹码带，见 §6） |
| `hide_overlay_lines` | bool | False | True = 隐藏 MA120/EMA/REG 基准/Grid Target 主目标/第二阶梯 5 条线 |

**返回**

```python
(c_result, bs_signal, bs_reason, bs_strength, all_levels)
# fast_mode=True 时返回 bool
```

| 返回值 | 说明 |
|---|---|
| `c_result` | `CausalIncrementalPriceSegmenter.segment()` 的结果 dict（含 phase/pivot/signal 等） |
| `bs_signal` | `np.ndarray[int]`，逐日买卖信号（`>0` 买 / `<0` 卖） |
| `bs_reason` | 信号原因（突破位/破位等） |
| `bs_strength` | 突破分量 0~1 评分 |
| `all_levels` | 阻力/支撑位生命周期列表 |

> ⚠️ 当 `code` 非空且 `show_chip=True` 时，模块内部会经 `chip_panel._ensure_turnover` 按需补拉 `turnover`。

---

### 2.2 `plot_price_segmentation_v10(...)`

```python
plot_price_segmentation_v10(df_ohlc, result, bs_signal, bs_reason,
                            tail_days=200, name="", save_path=None,
                            bs_strength=None, all_levels=None,
                            reg_preds=None, reg_preds_long=None,
                            hide_ma=True,
                            reg_win=120, reg_win_long=250,
                            panic_info=None,
                            strength_win=10,
                            dir_atr=2.0,
                            despeckle=False,
                            hide_mid_panels=True,
                            show_chip=False,
                            code=None, end_date=None,
                            hide_overlay_lines=False)
```

**参数**

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `df_ohlc` / `result` / `bs_signal` / `bs_reason` | — | — | 来自 `run_segmentation` |
| `tail_days` | int | 200 | 显示窗口 |
| `save_path` | str \| None | None | PNG 路径 |
| `bs_strength` | ndarray \| None | None | 信号柱高（分量） |
| `all_levels` | list \| None | None | 支撑/阻力位生命周期 |
| `reg_preds` | ndarray \| None | None | reg120 序列（**全量**）；建议传 `run_segmentation` 内同源值 |
| `reg_preds_long` | ndarray \| None | None | reg250 序列（**全量**） |
| `hide_ma` | bool | True | 隐藏 MA120/EMA |
| `reg_win` / `reg_win_long` | int | 120 / 250 | 缺失时自算回归的窗口 |
| `panic_info` | dict \| None | None | `panic_reversal.signal()` 结果；None 时面板显示提示 |
| `strength_win` | int | 10 | `compute_strength` 强度窗口 |
| `dir_atr` | float | 2.0 | 方向死区（ATR 倍数） |
| `despeckle` | bool | **False** | 强度去斑；**含未来函数，仅供事后可视化**（见 §8） |
| `hide_mid_panels` | bool | True | True = 4 面板；False = 6 面板（见 §6） |
| `show_chip` | bool | False | True = 筹码模式（2 面板 + COST 带） |
| `code` / `end_date` | str \| None | None | 筹码补拉与坑标注用 |
| `hide_overlay_lines` | bool | False | 隐藏 5 条叠加线 |

**返回**：无（写文件或 `plt.show()`）。

**模块内统一处理**：进入函数即对 `reg_preds` / `reg_preds_long` 做 `panic_reversal.double_smooth_reg(..., 5, 5)`，所有下游（显示 / 坑检测 / 包络线）共用平滑后序列。

---

## 3. 次级公开接口

### 3.1 `compute_buy_sell_signals(...)`

```python
compute_buy_sell_signals(df_ohlc, result, dur_horizon=120, touch_norm=3,
                         W_DUR=0.7, W_TOUCH=0.3, tt=0.005)
```

基于分段结果计算买卖信号、原因、突破分量、支撑/阻力位生命周期。
返回 `(bs_signal, bs_reason, bs_strength, all_levels)`。

### 3.2 `class CausalIncrementalPriceSegmenter`

因果分段器（主用），逐日递推。

```python
CausalIncrementalPriceSegmenter(lookback=15, min_reversal_pct=0.02, confirm_bars=3,
                                ema_span=15, ground_pct=20, sky_pct=85,
                                rolling_window=120, same_type_merge_gap=20)
```

| 方法 | 说明 |
|---|---|
| `segment(close, volume=None, high=None, low=None, opn=None)` | 主入口，返回结果 dict |
| `_ema_close(close)` | 内部：EMA |
| `_detect_candidates(close)` | 内部：候选拐点 |
| `_confirm_pivots(close, candidates)` | 内部：拐点确认（`confirm_bars` 延迟 → 因果） |
| `_assign_phases(n, pivots, close)` | 内部：阶段划分 |
| `_annotate_volume(volume)` | 内部：量能标注 |
| `_compute_touch_signal(close, high, low, opn, volume, n, confirmed_pivots)` | 内部：触碰信号 |

### 3.3 `class FutureLookingPriceSegmenter`（**仅回测参考**）

```python
FutureLookingPriceSegmenter(sg_window=11, sg_poly=3, peak_distance=3, min_reversal_pct=0.02)
    .segment(close)
```

基于 `scipy.signal.savgol_filter` 的**全局平滑**分段，**含未来函数**，只用于对照实验，禁止用于实盘信号。

### 3.4 模块内部辅助（非公开）

| 函数 | 说明 |
|---|---|
| `_compute_rolling_percentile(log_vol, ground_pct, sky_pct, rolling_window)` | 滚动分位数（地量/天量） |
| `_build_price_result(close, smooth, phase_id, phase_name, pivots, is_pending, pending_confidence, vol_annotation)` | 组装结果 dict |

---

## 4. 输入数据契约（`df_ohlc`）

| 列 | 必需 | 说明 |
|---|---|---|
| `date` | ✅ | 日期（datetime 或可排序值） |
| `open` / `high` / `low` / `close` | ✅ | 前复权价 |
| `volume` | ✅ | 成交量（单位与 `tdx_quant` 一致：手） |
| `turnover` | 条件 | `show_chip=True` 时经 `_ensure_turnover` 使用；缺失时用 `code`+`end_date` 补拉 |
| `circ_shares` | 条件 | 伴随 `turnover` 返回（当日流通股本，股） |

要求：**按日期升序**、无空收盘。`streamlit_demo.fetch_tdx_kline` 已做标准化。

---

## 5. V10 调用的外部接口

### 5.1 `panic_reversal`

| 函数 | 签名 | V10 用途 |
|---|---|---|
| `double_smooth_reg` | `(reg, w1=5, w2=5)` | 进入 plot 即对 reg120/250 两轮平滑 |
| `detect_volume_clusters` | `(closes, volumes, win=60, hi_ratio=1.5, lo_ratio=0.6, hi_pct=0.75, lo_pct=0.15, min_len=3, merge_gap=0, exit_confirm=2, dir_pct=0.02)` | 成交量面板放量堆底色（**必须用全量序列算**，窗口仅影响显示） |
| `detect_golden_pit_v6` | `(closes, reg250, z_thr=-1.5, launch_gate=0.9, confirm_days=3, min_len=1, min_depth=0.08, mom_main=0.05, mom_win=10, mom_sup=0.07, sup_near=0.0, near_z_thr=-0.5)` | 主路径坑检测（z + 动量过滤 + 急跌补充） |
| `detect_golden_pit_v3` | `(closes, reg250, reg120=None, z_thr=-1.5, merge_gap=15, launch_gate=0.9, use_pre_std=True, require_below_gate=False, confirm_days=3, min_len=1, min_depth=0.08, use_dual=False, smooth_w=5)` | reg120 基准坑（V10 传 `min_len=3`） |
| `compute_pit_quality` | `(pits, closes, volumes, pre_win=20, fill_win=20, fill_lead=2)` | 坑质量标签（strong/normal/weak） |
| `mark_high_pos` | `(pits, closes, thr=1.5)` | 高位坑软标注 |
| `mark_super_pits` | `(pits, closes, volumes, min_len=8, peak_ratio=5.0, fast_days=5, window_days=7)` | 加仓确认坑（★） |
| `compute_strength` | `(closes, highs=None, lows=None, k=2.0, alpha=2.0, m=30.0, atr=None, win=10, dir_atr=2.0, reg_preds=None, confirm_flip=2, flip_strong=0.08, min_main=3, decay_days=5, decay_factor=0.75, min_decay=2.0, reg_decay=0.1, short_win=5, short_drop=0.08, opens=None)` | Strength 面板 |
| `despeckle_strength` | ⚠️ **该函数当前不存在于 `panic_reversal`** | 仅 `despeckle=True` 时调用（默认 False） |
| `compute_grid_target_price` | `(closes, reg120, reg250, levels=(-0.09,-0.06,-0.03,0,0.03,0.06,0.09,0.12), max_dev=0.13, down_confirm=10, up_confirm=5)` | 粉阶梯 Grid Target（基准 `max(reg120,250)`） |
| `signal` | `(code, end_date=None, drop_pct=0.10, vol_ratio=1.2, bull_slope_min=0.05, confirm_days=3, below_reg=True, strength_thr=12.0)` | 仅 `enable_panic=True` 时经 `run_segmentation` 调用 |

### 5.2 `chip_panel`

| 函数 | 签名 | V10 用途 |
|---|---|---|
| `cost_series` | `(H, L, V, turn, p)` | `fengwo.COST` 逐日递推（**因果**），p=0.35/0.50/0.75 |
| `_ensure_turnover` | `(df_ohlc, code, end_date)` | 缺 `turnover` 时补拉对齐 |
| `draw_chip_panel` | `(ax, df_ohlc, tail_days, code=None, end_date=None, vmax=3.0, ngrid=1500, ext=0.2, show_stats=True, overlay=False, alpha=0.4)` | 独立筹码色带面板（V10 主图未用） |
| `chip_density_grid` | `(H, L, V, turn, ngrid=1500, nwin=None, ext=0.2)` | 绝对密度网格（⚠️ `nwin` 取序列末尾 → 渲染层含未来信息，见 §8） |

### 5.3 `mean_reversion.signal_residual`

| 函数 | 签名 | V10 用途 |
|---|---|---|
| `compute_rolling_regression` | `(closes: np.ndarray, window: int = 120, use_log: bool = True)` | 滚动回归预测线（reg120/reg250）；`reg_preds` 缺失时自算 |

> `run_segmentation` 用 `reg_window` / `reg_window_long` 调它；plot 内部兜底各自 `window=250` / `window=120`。

### 5.4 `aben_patterns`

| 函数 | 签名 | V10 用途 |
|---|---|---|
| `detect_guyan` | `(closes, volumes, pre_win=60, post_win=60, min_trend=0.08, stop_ok=-0.03, mega=8.0)` | 股眼标注（volume 面板色块 + 类型文字） |

---

## 6. 面板结构（由开关组合决定）

| 开关组合 | 面板数 | `height_ratios` | 内容 |
|---|---|---|---|
| `show_chip=True` | **2** | `[4, 1.3]` | K线（+筹码 COST 带）+ Volume（+股眼/量堆） |
| `hide_mid_panels=True`（默认，非筹码） | **4** | `[4, 1.3, 0.55, 0.6]` | K线 + Volume + Strength + GOLD PIT |
| `hide_mid_panels=False` | **6** | `[4, 1.3, 0.45, 0.9, 0.55, 0.6]` | 追加 买卖信号面板 与 阻力/支撑位生命周期面板 |

**筹码模式（`show_chip=True`）的 K 线叠加层**：

| 元素 | 颜色/线型 | 平滑 |
|---|---|---|
| COST35（筹码峰下沿） | 深绿虚线 `#1B5E20` | Ehlers 45 |
| COST75（筹码峰上沿） | 橙红虚线 `#E65100` | Ehlers 20，**仅放量段显示**（换手 `EMA10/MA60 > 1.5`，相对阈值） |
| 跌破 COST35 区段 | 深绿填充 `#2E7D32` alpha 0.30 | 用 `low < COST35(平滑)` 判定 |
| 黄金坑标记 | 金色背景 / ★ / reg120 紫带 | — |
| ~~COST50 中线、COST10-90 集中区带、COST90 线~~ | 已移除（2026-10-01 用户决定） | — |

**Ehlers SuperSmoother**（`_ehlers_smooth(_x, _period)`，模块内定义）：
`out[k] = c1*(x[k]+x[k-1])/2 + c2*out[k-1] + c3*out[k-2]`，系数 `a1=exp(-1.414π/period)`、`b1=2*a1*cos(1.414π/period)`、`c3=-a1²`、`c2=b1`、`c1=1-c2-c3` —— **只用当根/前一根与历史输出，因果**。

---

## 7. 未来函数红线

| 项 | 状态 |
|---|---|
| `CausalIncrementalPriceSegmenter` / `compute_buy_sell_signals` | ✅ 因果（`confirm_bars` 延迟确认） |
| `panic_reversal` 全部坑检测 / `compute_strength` / `compute_grid_target_price` | ✅ 因果（函数注释均有「无未来函数」约定） |
| `fengwo.COST` / `WINNER` | ✅ 因果（截断自检 0 不一致） |
| `chip_panel.cost_series`（V10 使用的路径） | ✅ 因果 |
| `FutureLookingPriceSegmenter`、`despeckle=True` | ❌ **含未来函数，禁止用于实盘信号** |
| `chip_density_grid`（`nwin` 取序列末尾） | ⚠️ 渲染层含未来信息；V10 主图未调用 |

---

## 8. 已知问题 / 待办

1. **`panic_reversal.despeckle_strength` 缺失** —— V10 在 `despeckle=True` 时会 `AttributeError`（默认 False 不触发）。需补实现或移除该分支。
2. `chip_density_grid` 的 `nwin` 末窗网格：启用密度色带前应先改为纯历史窗口（如 `c_lo[-2*nwin:-nwin]`）。
3. `add_turnover` 在无历史股本记录时 fallback **当前快照** → 历史换手率高估（次新/无记录股）。用户暂缓修复。
4. GitHub 镜像推送依赖网络，常 443 超时；本机无 `github_ed25519` 时需改用 HTTPS remote 强制推送。

---

## 9. 调用示例

### 9.1 最小可用（非筹码模式）

```python
import numpy as np, matplotlib
matplotlib.use('Agg')
import price_segmenter_v10 as v10
from tdx_quant import get_daily_kline_from_tdx
from mean_reversion.signal_residual import compute_rolling_regression

df = get_daily_kline_from_tdx('sz003040', '20261008', datalen=800)
c = df['close'].values.astype(float)
reg250, _ = compute_rolling_regression(c, window=250, use_log=True)
reg120, _ = compute_rolling_regression(c, window=120, use_log=True)

res, bs, br, bst, lv = v10.run_segmentation(df, tail_days=200, name='003040')
v10.plot_price_segmentation_v10(df, res, bs, br, tail_days=180, name='003040',
                                save_path='out.png', bs_strength=bst, all_levels=lv,
                                reg_preds=reg120, reg_preds_long=reg250)
```

### 9.2 筹码模式（streamlit 默认路径）

```python
res, bs, br, bst, lv = v10.run_segmentation(df, tail_days=200, name='003040',
                                            show_chip=True, code='sz003040',
                                            end_date='20261008')
v10.plot_price_segmentation_v10(df, res, bs, br, tail_days=180, name='003040',
                                save_path='out_chip.png', bs_strength=bst, all_levels=lv,
                                reg_preds=reg120, reg_preds_long=reg250,
                                show_chip=True, code='sz003040', end_date='20261008')
```

> `df` 需含 `turnover`（`with_turnover=True`）或提供 `code`/`end_date` 由 `_ensure_turnover` 补拉。

### 9.3 快速信号扫描（不绘图）

```python
has_buy = v10.run_segmentation(df, fast_mode=True)   # → bool
```
