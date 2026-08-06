---
name: price-segmenter-arch
description: 查询 price_segmenter_v10 架构速查——类/函数清单、参数默认值、信号体系、与 mean_reversion 耦合、设计演进(只读参考)
---

# price_segmenter_v10 架构参考

> 适用项目:`E:\chip_analyzer_ui\new_algo`。只读速查,不修改代码时使用;
> 改动代码前请配合 `price-segmenter-optimize` 阅读。行号基于 HEAD `0557a4d`。

## 1. 模块定位

`price_segmenter_v10.py`(895 行)实现"价格分段 + 买卖信号 + 可视化"一体的量化分析管线。
核心设计约束:**`CausalIncrementalPriceSegmenter` 无未来函数**——任何 bar t 的判定只能使用
`confirm_idx <= t` 的信息;`FutureLookingPriceSegmenter` 有未来函数,仅作 Oracle 基准对照。

## 2. 类/函数清单与数据流

```
df_ohlc(open/high/low/close/volume/date)
  │
  ▼
run_segmentation(df_ohlc, ...)                         # 便捷入口 (L852)
  ├─► CausalIncrementalPriceSegmenter.segment()        # 因果分段 (L109)
  │     ├─ _detect_candidates(close)                   # 局部极值候选 (L137)
  │     ├─ _confirm_pivots(close, candidates)          # 确认 + 同类型合并 (L149)
  │     ├─ _assign_phases(n, pivots, close)            # 阶段分配 + pending (L188)
  │     ├─ _annotate_volume(volume)                    # 量能标注 (L234)
  │     └─ _compute_touch_signal(close,high,low,opn,volume,n,pivots)  # 触摸信号 (L245)
  │              └─ 内嵌 _fkc(s,e,ib)                  # key candle 定位 (L261)
  │     ⇒ 产出 result DataFrame(result.attrs['pivots'])
  │
  ├─► compute_buy_sell_signals(df_ohlc, result, ...)   # 逐层突破买卖信号 (L359)
  │     ⇒ (bs_signal, bs_reason, bs_strength, all_levels)
  │
  ├─► (仅非 fast_mode) mean_reversion.signal_residual.compute_rolling_regression
  │     ⇒ reg_preds(120d) / reg_preds_long(250d)       # 对数滚动回归 (signal_residual.py L131)
  │
  └─► plot_price_segmentation_v10(df_ohlc, ...)        # 4 面板图 (L549)
        ├─ ax0: K线 + phase 背景 + 回归线 + gap + key candle + 阻力位
        ├─ ax1: 成交量(红=VOL_EXPANDING 绿=VOL_SHRINKING)
        ├─ ax2: 买卖信号柱(柱高/圆点大小 = strength)
        └─ ax3: 阻力/支撑位生命周期(★形成 · ▽/▲测试 · ●突破)
```

### 共享工具函数
| 函数 | 行号 | 作用 |
|---|---|---|
| `_compute_rolling_percentile` | L32 | 滚动分位数(成交量 ground/sky 阈值),前段 NaN 用首个有效值回填 |
| `_build_price_result` | L41 | 构造 result DataFrame 骨架(供 Oracle 版使用) |

### 两个分段器
| 类 | 行号 | 角色 |
|---|---|---|
| `FutureLookingPriceSegmenter` | L57 | savgol_filter + find_peaks 找 pivot;有未来函数;仅基准对照 |
| `CausalIncrementalPriceSegmenter` | L100 | **生产用核心**:EMA 平滑 + 候选/确认/合并/phase/触摸信号,全程因果 |

### 信号与绘图
| 函数 | 行号 | 作用 |
|---|---|---|
| `compute_buy_sell_signals` | L359 | V10 逐层突破买卖信号 + 阻力/支撑生命周期追踪 |
| `plot_price_segmentation_v10` | L549 | 4 面板可视化(内部另有一份 `_fkc`/`_zhug` 副本,见优化指南) |
| `run_segmentation` | L852 | 便捷入口,串起 分段→信号→回归线→画图 |

### 关键数据流细节
- `segment()` 的 pivot 三元组为 `(idx, type, confirm_idx)`;`compute_buy_sell_signals` 按
  `confirm_idx <= t` 过滤可见 pivot(L414-415),保证因果性。
- `result` DataFrame 列:`close, smooth, phase, phase_id, is_pivot, pivot_type,
  is_pending, pending_confidence, vol_annotation, touch_signal, touch_source`;
  `result.attrs['pivots']` 存确认后的 pivot 列表。
- `compute_buy_sell_signals` 额外用 `zones_phase`(L387-396)把 phase 连续区间并入
  zone 追踪,覆盖"无 TROUGH 回踩的短 UP run"(如 300437 的 10.34 尖峰)。
- 绘图轴范围聚焦逻辑(L716-742):纳入所有叠加曲线(MA/回归线)的 y 值,仅对
  离群极值(>4× 或 <0.12× 最近收盘)封顶,上下留 5% 余量;ax3 复用该区间。

## 3. 关键参数默认值与含义

### `FutureLookingPriceSegmenter.__init__`(L59)
| 参数 | 默认 | 含义 |
|---|---|---|
| `sg_window` | 11 | savgol 平滑窗口 |
| `sg_poly` | 3 | savgol 多项式阶数 |
| `peak_distance` | 3 | find_peaks 峰最小间距 |
| `min_reversal_pct` | 0.02 | 最小反转幅度阈值(过滤噪声 pivot) |

### `CausalIncrementalPriceSegmenter.__init__`(L102)
| 参数 | 默认 | 含义 |
|---|---|---|
| `lookback` | 15 | 局部极值回看窗口(候选检测) |
| `min_reversal_pct` | 0.02 | 反转确认阈值,兼作 phase 死区缓冲(见 L216-217) |
| `confirm_bars` | 3 | pivot 确认延迟 bar 数(因果性来源) |
| `ema_span` | 15 | EMA 平滑跨度(收盘价 + 成交量共用) |
| `ground_pct` | 20 | 成交量 ground 分位数 |
| `sky_pct` | 85 | 成交量 sky 分位数 |
| `rolling_window` | 120 | 成交量分位数滚动窗口 |
| `same_type_merge_gap` | 20 | 同类型 pivot 合并最大间隔(穿透合并) |

### `compute_buy_sell_signals`(L359)
| 参数 | 默认 | 含义 |
|---|---|---|
| `dur_horizon` | 120 | 压制时长归一化上限(超过即 dscore=1) |
| `touch_norm` | 3 | 影线测试次数归一化上限 |
| `W_DUR` | 0.7 | 时长分量权重(strength 计算) |
| `W_TOUCH` | 0.3 | 测试次数分量权重 |
| `tt` | 0.005 | 突破/触摸容差(0.5%) |

### `_compute_touch_signal` 内部魔法数(L246)
`tt=0.005`(触摸)、`at=0.05`(接近)、`gma=10`(gap 最小年龄,bar)。

### `run_segmentation`(L852)
| 参数 | 默认 | 含义 |
|---|---|---|
| `tail_days` | 200 | 绘图尾段长度 |
| `reg_window` | 120 | 中期回归窗口;=0 不显示(红虚线) |
| `reg_window_long` | 250 | 长期回归窗口;=0 不显示(蓝实线) |
| `hide_ma` | True | 隐藏 MA120/EMA 叠加线 |
| `fast_mode` | False | True=跳过画图,返回 `bs_signal[-1] > 0`(bool) |

### `mean_reversion.signal_residual.compute_rolling_regression`(L131)
`window=120`, `use_log=True`(对数回归,消除复利偏差)。`compute_residual_signal`
默认 `z_strong_buy=-2.0 / z_weak_buy=-1.5 / z_weak_sell=1.5 / z_strong_sell=2.0`,
`robust_std=True`(剔除当日点算 std)。

## 4. 信号体系与输出结构

### 6 种买卖信号(每 bar 至多一个,优先级从高到低)
| 方向 | 名称 | 触发条件 | 说明 |
|---|---|---|---|
| +1 | `BrkLvl` | 收盘突破一个未突破的前 UP 区高点 | V10 核心,逐层突破,带 0~1 分量评分 |
| +1 | `BrkRes` | 收盘突破之前触碰过的压力位 | 基于 `pending_resistance_level`(touch_signal=2 设置) |
| +1 | `PullSup` | UP 待定区中回踩支撑 | 条件:`is_pending & phase==UP & touch_signal<=-1` |
| -1 | `BrkLow` | 收盘跌破一个未跌破的前 DOWN 区低点 | 对称于 BrkLvl,带分量评分 |
| -1 | `BrkSup` | 收盘跌破之前触碰过的支撑位 | 基于 `pending_support_level`(touch_signal=-2 设置) |
| -1 | `BncRes` | DOWN 待定区中反弹到阻力 | 条件:`is_pending & phase==DOWN & touch_signal>=1` |

- 优先级实现:`compute_buy_sell_signals` 主循环 L412-505 先处理 BrkLvl/BrkLow
  (命中即 `continue`),L507 之后按 BrkRes → PullSup → BrkSup → BncRes 顺序。
- `BrkLvl/BrkLow` 突破目标选择:买侧取**价格最低**的未突破前高
  (`min(broken)`,L471),卖侧取**价格最高**的未跌破前低(`max(broken)`,L492)。
- strength 计算(L473-476):`strength = W_DUR·min(1, dur/dur_horizon) + W_TOUCH·min(1, touch_count/touch_norm)`,
  其中 `dur = t - form_idx`(压制时长),`touch_count` 为未破期间影线测试次数(去连续)。

### 输出结构
| 输出 | 类型 | 说明 |
|---|---|---|
| `result` | DataFrame | 10 列 + `attrs['pivots']`(确认后 pivot 三元组列表) |
| `bs_signal` | np.ndarray[int] | 取值 {1, 0, -1},长度 n |
| `bs_reason` | np.ndarray[str] | 如 `'BrkLvl(10.34,s=0.56)'` / `'PullSup(DN_LOW)'` |
| `bs_strength` | np.ndarray[float] | 0~1,仅 BrkLvl/BrkLow 非零(绘图柱高/圆点大小依据) |
| `all_levels` | list[dict] | 阻力/支撑生命周期:未突破(unbroken)+ 已突破(completed)合并 |

`result` DataFrame 列:`close, smooth, phase, phase_id, is_pivot, pivot_type,
is_pending, pending_confidence, vol_annotation, touch_signal, touch_source`。

`all_levels` 每项 dict 字段:`price`(位价)、`form_idx`(形成位置)、`touch_count`、
`tests`(测试 bar 下标列表)、`kind`('RES'/'SUP')、`break_idx`(突破位置或 None)、
`break_strength`(突破分量,未破为 0.0)。

### 绘图语义(plot_price_segmentation_v10)
- ax2 柱高/圆点:`height = signal · (0.4 + 0.6·strength)`(strength>0 时),否则 `0.5`。
- ax3 生命周期标记:★ 形成(form_idx)、▽(RES 测试)/▲(SUP 测试)、● 突破(break_idx,
  附 `s=strength` 标注)。
- 只有价格落在 ax0 聚焦区间(`_focus_lo/_focus_hi`)内的 level 才绘制(L803-807)。

## 5. 与 mean_reversion 模块的耦合

### 耦合面(单向、仅绘图)
- **唯一调用点**:`run_segmentation` L880-887 —— `import mean_reversion.signal_residual
  .compute_rolling_regression`,生成 `reg_preds`(120d)/`reg_preds_long`(250d)传入绘图。
- **只影响可视化,不影响分段与信号**:回归线不参与 `CausalIncrementalPriceSegmenter`
  或 `compute_buy_sell_signals` 的任何计算;`hide_ma=True` 只隐藏 MA120/EMA,回归线独立显示。
- `reg_window=0` / `reg_window_long=0` 时跳过对应回归计算与绘制。

### mean_reversion 包内部结构(供优化时对照)
| 文件 | 关键符号 | 角色 |
|---|---|---|
| `signal_residual.py` | `compute_residual_signal`(L27)、`compute_rolling_regression`(L131)、`compute_reversion_debt`(L183) | 对数回归残差信号 / 滚动回归线 / 回归负债 |
| `signal_energy.py` | `compute_energy_signal`(L16) | 能量信号 |
| `fuser.py` | `SignalResult`(L24)、`fuse_signal`(L43)、`compute_signal`(L73) | 多信号融合 |
| `backtest.py` | `run_backtest`(L27)、`run_backtest_many`(L172)、`summarize`(L204) | 事件回测框架(独立于 V10 分段) |
| `smoke_test.py` / `test_backtest.py` | `main()` | 冒烟测试 / 回测单测(独立入口) |

### 依赖方向
```
price_segmenter_v10.run_segmentation ──► mean_reversion.signal_residual(compute_rolling_regression)
```
`fuser` / `signal_energy` / `backtest` 当前不被 `price_segmenter_v10` 引用;若后续要把
残差/能量信号与 V10 买卖点融合,入口在 `fuser.compute_signal`。

### 回归算法要点(signal_residual.py)
- 对数线性回归 `log(price) = a·t + b`(use_log=True),消除复利偏差;
- 残差 = log(实际价) − log(预测价),z_residual = 残差/稳健 std(robust_std=True 剔除当日点);
- `compute_reversion_debt` 计算"回归负债"(偏离累积量),配合 fuser 使用。

## 6. 设计意图与演进史

### 演进时间线(git log)
| 提交 | 内容 | 意义 |
|---|---|---|
| `ed21115` | snapshot current finance code | V8/V9 时代基线 |
| `b7269ac` | 更新 volume 显示超出区域的问题 | 修复量轴显示 |
| `cd61cda` | 集成滚动回归线到 V10 面板0(ax0) | 引入回归线 |
| `39392fd` | reg_window 默认值改为 120,调用时自动显示回归线 | 参数默认化 |
| `7a6a397` | mean_reversion 信号改造 + V10 回归线修复 + 事件回测框架 | 建立回测体系 |
| `9382410` | refactor: 对数回归 + overhang 连续衰减 | 回归算法改造 |
| `0557a4d` | panel0 双回归线(120d+250d),隐藏 MA120/EMA | 双周期回归线(当前 HEAD) |

### V9 → V10 核心变化(模块头 L5-17)
- **买点改"逐层突破"**:追踪所有未突破的前 UP 区高点,每个被突破都发信号;
- **卖点对称**:追踪所有未跌破的前 DOWN 区低点;
- **废弃 V9 的 `max(last_3)` 逻辑**——它在下降趋势中会被远古高点卡住信号,逐层突破
  更及时、更密集(compute_buy_sell_signals docstring L365-372)。

### 关键设计决策与动机(带出处)
1. **无未来函数铁律**:`CausalIncrementalPriceSegmenter` 全程因果(confirm_bars 延迟确认,
   pivot 按 `confirm_idx <= t` 可见);`FutureLookingPriceSegmenter` 明确标注"有未来函数,
   仅基准对照"(L58),用于评估因果版精度损失。
2. **phase 区间并入 zone 追踪**(L383-386):pivot 配对(TROUGH→PEAK)组不成 zone 的
   短 UP run(如 300437 的 10.34 尖峰)会被漏登记;并入 phase 连续区间后进入
   unbroken_highs/completed_highs,参与 BrkLvl/BrkLow 并在 ax3 显示;用
   `round(price,2)` 与 pivot 区间去重。
3. **死区缓冲**(L215-217):phase 翻转需越过 `min_reversal_pct` 死区,避免 pending 区域
   产生 1-2 天的碎片 zone;pending_confidence = 穿越幅度/阈值 归一化(0~1)。
4. **逐层突破目标选择**:买侧突破"价格最低的未突破前高"(最近),卖侧跌破"价格最高的
   未跌破前低"——信号密集且按层次消费,不会重复打同一个位。
5. **轴聚焦**(L716-719):轴范围纳入 MA/EMA/回归线全部可见 y 值,防止 pre-window 历史价
   被裁切到轴外;仅离群极值(>4× / <0.12× 最近收盘)封顶,健康区间不收缩;ax3 复用同轴。
6. **对数回归**(signal_residual.py 头注释):`log(price) = a·t + b` 消除复利偏差,
   大牛股不会被误判为"持续超买";robust_std 剔除当日点防自相关噪声。
