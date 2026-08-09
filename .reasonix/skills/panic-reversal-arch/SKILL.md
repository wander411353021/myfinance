---
name: panic-reversal-arch
description: 极速杀跌反转模型与 V10 第5面板 strength 柱架构速查——函数清单/参数默认值/演进史/未来函数边界,改动前必读
---

# 极速杀跌反转模型架构速查(panic_reversal.py + V10 第5面板)

## 核心文件与函数(当前版)

- `panic_reversal.py`:
  - `signal(code, end_date=None, drop_pct=0.18, vol_ratio=1.2, bull_slope_min=0.05, confirm_days=3, below_reg=True)`
    - 实盘信号接口(单只,无未来函数);判定用**固定 18% 跌度**(83% 胜率档),未用 strength
  - `detect_panic_events(df, code, drop_pct=0.15, ...)` — 事件研究主函数(默认 15% 档)
  - `compute_strength(closes, highs, lows, k=2.0, alpha=2.0, m=30.0, atr=None, win=10, dir_atr=2.0, reg_preds=None, confirm_flip=2, flip_strong=0.08, min_main=3, decay_days=5, decay_factor=0.75, min_decay=2.0)`
    - **第5面板 strength 柱最终版 v4.8**(无未来函数)
  - `despeckle_strength(strength, min_seg=3)` — ⚠️ 用到右侧(未来)柱段,**存在未来函数**,
    仅事后可视化,面板默认 despeckle=False,绝不用于 signal()
  - `_compute_atr14(highs, lows, closes)` — Wilder ATR(14),前 13 个 NaN
- `price_segmenter_v10.py`:
  - `plot_price_segmentation_v10(..., panic_info=None, strength_win=10, dir_atr=2.0, despeckle=False)`
  - 第5面板 ax4:strength 柱(绿正红负,±30),panic_t ▼/confirm ●/BUY-WATCH-NONE 标注

## compute_strength 最终公式(默认参数 v4.8)

```
amp     = win日波幅 = max(high[i-win+1..i]) - min(low[i-win+1..i])   win=10
raw     = amp / ATR14                                  # 波幅是几个单日波动
有柱日   = |10日净位移| >= dir_atr*ATR14 (dir_atr=2.0),方向 = sign(净位移)
回归线门控: V10 面板传 reg_preds_long(250日回归线,缺省回退120日);close < reg250 时,正柱(反转)不显示——
           若有主方向且衰减期内 → 画衰减延续柱(非硬0)
翻转确认: 与当前方向相反时——短段(<min_main=3)且净位移占比>=flip_strong(8%)
          立即翻转(恐慌起点);否则需连续 confirm_flip(2) 根反向才翻转
u       = max(0, raw - k)^alpha                          k=2.0, alpha=2.0
有柱日   = 方向 * atan(u)/(π/2) * m                     m=30(峰值 ±30)
死区/门控日 = 延续前方向衰减: cur*|last_s|*decay_factor^k (decay_days=5, factor=0.75);
          衰减结束若方向未变 → 最低值保底 cur*min_decay(min_decay=2.0,方向永续可见)
```

## 演进史(重要,理解为什么是现在这样)

| 版本 | 变更 | 动机 |
|---|---|---|
| v1 | 收盘位移/ATR | close 不动但盘中波动大时看不见 |
| v2 | 波幅/ATR + 收盘位置方向 | 大阳/大阴留窗口内来回翻转(锚点效应) |
| v3 | + atan 饱和压缩 | 平方放大爆表到几百 |
| v4 | 波幅/ATR + 收盘位移方向 | 消除锚点翻转(300437 8/31~9/06) |
| v4.1 | + dir_atr 方向死区 | 横盘净位移不足 → 柱置0,交替 72→25 |
| v4.2 | + reg_preds 回归线门控 | 回归线下正柱(反转)不可信 → 置0 |
| v4.5 | + flip_strong(ATR 版)/confirm_flip | 连续阳不轻易转阴;ATR 联动稀释 |
| v4.6 | flip_strong 改净位移占比(8%) | 恐慌日 ATR 升高导致阈值失效 |
| v4.7 | + min_main 主段长度 | 连续同色段中无独立反向柱(反向孤立=0) |
| v4.8 | + decay_days/decay_factor 死区衰减延续 | 死区日延续前方向衰减,视觉连贯 |
| v4.8b | 门控日也画衰减延续柱 | 大柱后反弹被门控硬置0,视觉突兀(603986 案例) |
| v4.9 | + min_decay 最低值保底(衰减结束后方向未变→±2.0) | 长段空白判别不了方向;用户最终确认要保底 |

## 无未来函数边界(铁律)

- ✅ `compute_strength` 全部逻辑从左到右扫描,只看已发生历史:dir_atr 死区、回归线门控、
  confirm_flip/flip_strong/min_main 翻转确认、decay 衰减(cur/last_s/dead_streak 纯历史累计)
  ——**截断一致性测试通过**(任意时点只用历史前缀计算=全序列计算)
- ✅ `signal()` / `detect_panic_events` 无未来函数(verify_no_future() 返回 True)
- ⚠️ `despeckle_strength` 用右邻段判断 → 未来函数,仅画图参考,默认关闭

## 关键验证数据(299 池/最近两年)

- 固定 18% 跌度 + 放量1.2 + 牛市门控 + 破线 + 确认:胜率 83.3%(n=30)
- strength 阈值单独做信号仅 60~65%(波幅大但净跌小);需叠加固定跌度才回 80%
- 死区滤波视觉价值 > 独立信号价值(柱覆盖率 ~27-37%,反向孤立柱=0)

## 常用命令

```
python panic_reversal.py grid          # 参数网格(84只)
python panic_reversal.py one sz300437  # 单只检测+画图
# streamlit: streamlit run streamlit_demo.py(走 run_segmentation,默认 strength_win=10 dir_atr=2.0 decay_days=5;门控=250日回归线)
# 注意:修改 panic_reversal.py/price_segmenter_v10.py 后必须重启 streamlit 进程才生效
```
