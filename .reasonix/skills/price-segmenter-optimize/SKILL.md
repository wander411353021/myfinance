---
name: price-segmenter-optimize
description: 改动 price_segmenter_v10/mean_reversion 前的优化指南——可维护性痛点、性能瓶颈、参数调优地图、验证命令(铁律:无未来函数、行为对齐)
---

# price_segmenter_v10 优化指南

> 适用项目:`E:\chip_analyzer_ui\new_algo`。改动 `price_segmenter_v10.py` /
> `mean_reversion/` 前必读。行号基于当前 HEAD(`0557a4d`),改动前先核对。

## 0. 铁律(改动前必读)

1. **无未来函数**:任何 bar t 的判定只能使用 `confirm_idx <= t` 的信息;禁止用全量数据
   回填。破坏此约束 = 回测失真(参考 `FutureLookingPriceSegmenter` 的"仅基准"定位 L58)。
2. **行为对齐**:优化后必须与优化前在相同输入上输出一致(或已论证的改善),用
   `mean_reversion/test_backtest.py` + `smoke_test.py` 回归验证。
3. **信号优先级不变**:BrkLvl/BrkLow → BrkRes → PullSup → BrkSup → BncRes(每 bar 至多一个)。
4. **不要顺手改参数默认值**:默认值是被调过的,除非本次任务明确调参。

## 1. 可维护性痛点(按优先级)

### P1 重复实现(结构性,优先处理)
| 痛点 | 位置 | 说明 |
|---|---|---|
| `_fkc` 重复两份 | L261-270(`_compute_touch_signal` 内嵌)与 L648-657(plot 内嵌) | key candle 定位逻辑几乎相同(仅变量名/数组来源不同),应提取模块级共享函数 |
| gap 检测/填充重复 | `_compute_touch_signal` L251-260、plot L625-646、plot `_zhug` L658-666 | 三处各自实现"检测 gap + 找填充 bar"逻辑;应提取 `_detect_gaps(high, low) -> (gaps, gap_fills)` 共用 |
| `_zhug` 逻辑重叠 | plot L658-666 vs `_compute_touch_signal` L296-302 | gap 未填充判断重复,可与上项合并 |
| `result` 构建重复 | `_build_price_result` L41-55 vs `CausalIncrementalPriceSegmenter.segment` L125-128 | 两处几乎相同列定义;`segment` 应复用 `_build_price_result` |
| zone 构建重复 | `_assign_phases` L202-212、`_compute_touch_signal` L274-289、`compute_buy_sell_signals` L417-421 | 三处从可见 pivot 配对 TROUGH→PEAK/PEAK→TROUGH 构建 zones;应提取 `_build_zones(visible)` |
| 可见 pivot 过滤重复 | `_assign_phases` L193-194、`_compute_touch_signal` L272、`compute_buy_sell_signals` L414-415 | `[p for p in pivots if confirm <= t] + sort` 三处重复;应提取 helper |

### P2 魔法数字散落(可读性)
- `tt=0.005 / at=0.05 / gma=10`(L246):触摸阈值,写死在 `_compute_touch_signal` 局部;
  `tt` 又在 `compute_buy_sell_signals` 签名里(默认 0.005),两处来源不统一。
- key candle 放大系数 `lp=20.0/10.0`、`sc*=10`(L267-268 与 L654-655):同逻辑两份。
- 轴聚焦离群阈值 `4×`/`0.12×`(L733-738):无命名常量。
- zone 去重 `round(price, 2)`(L434):精度假设散落,若处理 <0.01 价位需全局调整。
- 信号强度兜底 `0.5`(L782):"无 strength 时柱高"的语义未命名。

### P3 函数过大
- `plot_price_segmentation_v10` L549-846(~300 行):ax0 内 phase 背景 / K线 / 回归线 /
  gap / key candle / 阻力线逐段堆叠;应拆 `_draw_ax0/_draw_ax1/_draw_ax2/_draw_ax3`。
- `compute_buy_sell_signals` L359-543(~185 行)单函数承载 zone 追踪 + 5 个信号分支 +
  生命周期汇总;可拆"追踪状态更新 / 各信号判定 / 汇总"。

### P4 其他
- 除 `mean_reversion/signal_*.py` 外几乎无类型标注;`run_segmentation` 每次新建
  `CausalIncrementalPriceSegmenter`,同一参数多股票批量调用可复用实例。
- 文档注释中英混杂;模块头(L1-18)是唯一系统性说明,函数级 docstring 缺失。

## 2. 性能瓶颈(Python 纯循环)

### 复杂度总览(热点在分段器 4 个方法)
| 函数 | 位置 | 现状复杂度 | 瓶颈原因 |
|---|---|---|---|
| `_ema_close` | L132-135 | O(n) 纯循环 | 手写 EMA 递推;可用 `pd.Series.ewm(span, adjust=False)` 或 `scipy.signal.lfilter` 向量化 |
| `_annotate_volume` | L234-243 | O(n) 纯循环 | 同样手写 EMA(L236-237);`_compute_rolling_percentile` 已用 pandas rolling,可接受 |
| `_detect_candidates` | L137-147 | O(n·lookback) | 每 bar 切片 `close[t-lookback:t+1]` 求 max/min |
| `_confirm_pivots` | L149-186 | 最坏 O(n²) | 每候选内层扫描至反转;合并段 `filtered=filtered[:j]+[...]` 反复重建列表 |
| `_assign_phases` | L188-232 | O(n² log n) | 每 bar 重建 vis 并 `sort`(L193-194) |
| `_compute_touch_signal` | L245-352 | 最坏 O(n²) | 每 bar 重建 zones、重复求 zone 极值、可能调 `_fkc` |
| `compute_buy_sell_signals` | L359-543 | O(n·k)(k=可见 pivot) | 每 bar 重建 zones + `_all_zones` 拼接;unbroken dict 遍历 |
| plot `_fkc`/`_zhug` | L648-666 | O(n·interval) | tail_days=200,非热点,可不优化 |

### 向量化建议(按性价比排序)
1. **`_detect_candidates`**:用 `pd.Series(close).rolling(lookback+1).max()/min()` 一次算出
   滑动极值,再按 `close[t]==wmax[t] & close[t]>close[t-1]` 取候选;O(n·lookback) → O(n)。
2. **`_ema_close`/`_annotate_volume`**:`pd.Series(x).ewm(span=ema_span, adjust=False).mean()`,
   结果与当前递推一致(adjust=False 即 a=2/(span+1) 递推)。
3. **`_assign_phases`**:按 `confirm_idx` 增量插入 pivot 到有序 vis,替代每 bar 全量
   `[filter+sort]`;死区判定命中即 continue。
4. **`_compute_touch_signal`**:对每个 zone 预计算 `high.max()/low.min()` 极值表
   (区间 max/min 可用 prefix/suffix 或稀疏表),消除循环内重复求极值;
   `gap_fills` 已一次性预计算(L256-260),保持。
5. **`_confirm_pivots` 合并段**:改为指针式就地合并,避免 `filtered[:j]+[...]` 整表重建。

### 注意
- **向量化必须保持逐位等价**:`_ema_close` 的递推 `s[i]=a*close[i]+(1-a)*s[i-1]`
  与 `ewm(adjust=False)` 完全等价,可放心替换;但 `_assign_phases` 的增量重构是
  行为敏感的(死区缓冲/pending_confidence),改后必须跑 `test_backtest.py` 逐位对比。
- 性能收益只有当 `n` 数千 bar、多股票批量跑 `fast_mode=True` 时才有实际意义;
  单次画图(200 bar)无需为此冒险。

## 3. 参数调优地图

### 分段器参数(级联影响信号层,调后必须回测验证)
| 参数 | 默认 | ↑ 调大 | ↓ 调小 | 推荐区间 |
|---|---|---|---|---|
| `lookback` | 15 | 候选更少更稳,漏小拐点 | 候选更多,噪声增加 | 10-20(短线 10,中线 20) |
| `min_reversal_pct` | 0.02 | 信号更少更可靠,pending 死区变大 | 信号更多,pending 碎片增多 | 0.015-0.03 |
| `confirm_bars` | 3 | 确认更晚更稳,信号滞后 | 更快,假 pivot 增多 | 2-5 |
| `same_type_merge_gap` | 20 | 合并更远同类型 pivot,zone 更少 | 保留细节,zone 更碎 | 10-40 |
| `ema_span` | 15 | smooth 更平滑(不参与 phase 判定) | 更敏感 | 10-30 |

> ⚠ `min_reversal_pct` 是**全局最敏感参数**:它同时充当反转确认阈值(L154-155)与
> phase 死区缓冲(L217),调大会同时减少 pivot 数量、加宽 pending 死区。

### 信号层参数
| 参数 | 默认 | ↑ 调大 | ↓ 调小 | 推荐区间 |
|---|---|---|---|---|
| `dur_horizon` | 120 | 压制时长更难满 dscore,强度普遍偏低 | 更快满分量 | 60-180 |
| `touch_norm` | 3 | 需更多影线测试才满 tscore | 更易满 | 2-5 |
| `W_DUR / W_TOUCH` | 0.7/0.3 | 时长分量更主导 | 测试更主导 | 0.5/0.5 ~ 0.8/0.2 |
| `tt`(触摸容差) | 0.005 | 更易触发触摸/突破判定 | 更严格 | 0.003-0.01 |

### 量能标注参数(仅影响 vol_annotation 与 ax1 颜色)
| 参数 | 默认 | ↑ 调大 | ↓ 调小 |
|---|---|---|---|
| `ground_pct` | 20 | VOL_EXPANDING 更少 | 更多 |
| `sky_pct` | 85 | VOL_EXPANDING 更少 | 更多 |
| `rolling_window` | 120 | 分位数更平滑 | 更敏感 |

### 回归线参数(仅绘图,不影响信号)
| 参数 | 默认 | 说明 |
|---|---|---|
| `reg_window` | 120 | 中期趋势(红虚线);60=短线,180-250=大级别 |
| `reg_window_long` | 250 | 长期趋势(蓝实线);=0 隐藏 |
| `use_log` | True | 对数回归,消除复利偏差 |

### 验证纪律
- 调**分段器/信号层**参数:`mean_reversion/backtest.run_backtest`(事件回测)或
  Oracle 对比(参考 `_test_compare.py` 思路,比对信号差异),不能只看单图。
- 调**绘图/量能/回归**参数:无需回测,`run_segmentation(save_path=...)` 出图核对即可。
- 一次只调一个参数;记录基准与对照组,避免耦合效应。

## 4. 验证手段

### 可用命令(已在 Windows 实测)
| 命令 | 用途 | 现状(实测) |
|---|---|---|
| `python mean_reversion/smoke_test.py` | 冒烟测试(构造 buy/downtrend/debt 序列) | ⚠ **失败**:`AttributeError: 'SignalResult' object has no attribute 'in_debt'`(smoke_test.py L125 引用了 fuser.SignalResult 不存在的字段)——已知不一致,优化时可顺手修 |
| `python mean_reversion/test_backtest.py` | 回测单测(构造 episode 序列) | ⚠ **跑通但结尾崩溃**:回测结果正常打印(good_zone 命中率/超额收益),但 `UnicodeEncodeError`(GBK 终端无法编码 ✅)——用 `PYTHONIOENCODING=utf-8` 可规避 |
| `python _test_compare.py` | Oracle(four_phase_visualizer)vs 无未来函数对比 | 属于 four_phase 模块,与本文件解耦;参考其"逐项 diff 关键字段"的验证思路 |
| `PYTHONIOENCODING=utf-8 python ...` | 上述命令的编码安全变体 | Windows GBK 终端下推荐统一加前缀 |

### fast_mode 冒烟(最轻量)
```python
from price_segmenter_v10 import run_segmentation
# 取任意 df_ohlc(需 open/high/low/close/volume/date 列)
buy = run_segmentation(df_ohlc, fast_mode=True)   # 返回 bool:最后一天是否有买入信号
```

### 信号级回归对比(改代码后的铁律验证)
1. 改前:对固定股票集跑 `run_segmentation`(非 fast_mode),保存
   `bs_signal / bs_reason / result['phase']` 到 npz/pkl 作为基准。
2. 改后:同输入重跑,`np.array_equal(old_signal, new_signal)` 必须 True
   (若改动是有意的行为变更,则 diff 每个差异点的 `bs_reason`,逐条论证)。
3. 批量:`run_segmentation` 的 `fast_mode=True` 已支持多股票扫描(参考
   `daily_scan.py` 的批量用法)。

### 事件回测(验证信号质量)
```python
from mean_reversion.backtest import run_backtest, summarize
# 构造 episode 序列或接入真实 df,评估 good_zone 命中率/超额收益
```
- 阈值参考:`test_backtest.py` 的合成序列基准——good_zone 命中率 100%、基线 7.7%、超额 +92.3%、
  回撤 49.4%;真实股票数据以"命中率 > 基线"为最低门槛。

### 出图核对
`run_segmentation(df_ohlc, tail_days=200, name="xxx", save_path="result/xxx.png")`
——检查:ax0 phase 背景与回归线、ax2 信号柱/圆点、ax3 生命周期标记完整性;
改绘图相关代码(轴聚焦/颜色/标签)后必须出图目检。
