---
name: panic-reversal-arch
description: 极速杀跌反转模型与 V10 第5面板 strength 柱架构速查——函数清单/参数默认值/演进史/未来函数边界,改动前必读
---

# 极速杀跌反转模型架构速查(panic_reversal.py + V10 第5面板)

## 核心文件与函数(当前版)

- `panic_reversal.py`:
  - `signal(code, end_date=None, drop_pct=0.10, vol_ratio=1.2, bull_slope_min=0.05, confirm_days=3, below_reg=True, strength_thr=12.0)`
    - 实盘信号接口(单只,无未来函数);恐慌判定=**5日跌≥10% AND strength≤-12 双重要求**
      (299池复测胜率 83.8%,n=80);strength_thr=0/None 回退纯跌幅旧逻辑
  - `detect_panic_events(df, code, drop_pct=0.15, ...)` — 事件研究主函数(默认 15% 档)
  - `compute_strength(closes, highs, lows, k=2.0, alpha=2.0, m=30.0, atr=None, win=10, dir_atr=2.0, reg_preds=None, confirm_flip=2, flip_strong=0.08, min_main=3, decay_days=5, decay_factor=0.75, min_decay=2.0, reg_decay=0.10, short_win=5, short_drop=0.08, opens=None)`
    - **第5面板 strength 柱最终版 v4.8**(无未来函数)
  - `compute_turn_positive_prices(closes, highs, lows, opens=None, win=10, dir_atr=2.0, short_win=5, short_drop=0.08, reg_decay=0.10, reg_preds=None, atr=None, strength=None, min_band=0.10)`
    - 阴柱期"转阳触发价"序列(阳柱/无柱日=NaN);V10 panel0 画蓝粗虚线目标价
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
回归线门控: reg_gate = reg_preds × (1-reg_decay=0.9);close < reg_gate 时正柱(反转)不显示,
           画衰减延续柱/保底(V10 面板传 reg_preds_long=250日,缺省回退120日)
5日骤变(short_win=5, short_drop=8%): 独立检查(同向/反向都触发)——
           骤跌立即翻阴;骤涨翻阳需 站上回归线(C) 且 当天非阴线(B,需传 opens)
翻转确认: ①骤变 ②站上回归线翻阳 ③短段(<min_main=3)强反向(净位移占比>=8%)
          立即翻转;否则需 confirm_flip(2) 根确认(1天延迟)
u       = max(0, raw - k)^alpha                          k=2.0, alpha=2.0
有柱日   = 方向 * atan(u)/(π/2) * m                     m=30(峰值 ±30)
死区/门控日 = 延续前方向衰减: cur*|last_s|*decay_factor^k (decay_days=5, factor=0.75);
          衰减结束 → 保底 cur*min_decay(±2.0,死守 cur,不跟K线)
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
| v4.9+ | + reg_decay=0.10(门控线=reg×0.9) | reg250 太严,接近回归线的反弹不压 |
| v4.9+ | + 站上回归线翻阳立即(不等2根) | 300476 9/30 大阳被染阴,延迟不合理 |
| v4.9+ | + 5日骤变(±8%)独立检查 | 5日暴跌但10日方向正被染阳(000066 10/15) |
| v4.9+ | 骤涨翻阳需站上回归线+当天非阴线(C+B) | 688099 4/14 阴K阳柱(骤涨+8%但reg下) |
| v4.9+ | opens 参数(骤涨阳线条件) | signal/V10 已传 opens |

## 转阳触发价(compute_turn_positive_prices)— V10 panel0 蓝粗虚线

- 用途:阴柱期间在 K 线上画一条"目标/压制价",突破该价 = 次日大概率转阳(买入条件单参考);**阳柱期延续显示,直到真突破才消失**
- 第 i 天(阴柱)触发价 = min(常规路径, 骤变路径) 再优化:
  - 常规路径 = max(close[i-win] + dir_atr×ATR[i], 门控线 reg×0.9)
  - 骤变路径 = max(close[i-short_win]×(1+short_drop), 门控线)
- 优化①:触发价不低于当日收盘(方向已满足时=现价,压制语义)——688552 5/19 根因:暴跌后 10 日前参照价过低,min(常规,骤变)选到低价路径出现"压制价<现价"
- 优化②:滞回 min_band=10%:新候选与当前显示值差 <10% 时保持前值(抑制 close[i-10]/close[i-5] 换参照抖动;688552 变动率 17%→9%,段数 39→21)
- **优化③延续(核心语义)**:阴柱日计算/更新目标价并激活;阳柱/无柱日延续显示最后值,直到出现**阳K(close>open,需传 opens)盘中触及(high≥目标价)**才结束;结束前一直显示(601138 10/14~10/24 持续 70.85,10/27 high=72.60 突破才消失)
- 只依赖第 i 天及之前数据,无未来函数(active/prev 为历史累计状态)
- V10 绘制:ax0 `plot(x, tp_win, drawstyle='steps-post', color='#1565C0', linewidth=2.6, ls='--', label='Turn-Up Target')`;完整序列算完再切片(offset:offset+n)
- ⚠️ 历史教训:2026-08 曾因替换脚本 docstring 断言失败导致延续逻辑未写入文件(assert 在 write 前抛错),用户反馈"结束太突然"实为旧版 bug——改 panic_reversal.py 后必须实际验证行为,不能只看 py_compile

## 无未来函数边界(铁律)

- ✅ `compute_strength` 全部逻辑从左到右扫描,只看已发生历史:dir_atr 死区、回归线门控、
  confirm_flip/flip_strong/min_main 翻转确认、decay 衰减(cur/last_s/dead_streak 纯历史累计)
  ——**截断一致性测试通过**(任意时点只用历史前缀计算=全序列计算)
- ✅ `signal()` / `detect_panic_events` 无未来函数(verify_no_future() 返回 True)
- ⚠️ `despeckle_strength` 用右邻段判断 → 未来函数,仅画图参考,默认关闭

## 关键验证数据(299 池/最近两年)

- signal 双重要求(跌≥10% + strength≤-12 + 放量 + 破线120 + 确认):胜率 83.8%(n=80)
- 旧固定 18% 档:胜率 88.0%(n=25,事件少 1/3)
- strength 单独做信号仅 59~64%(波幅大但净跌小,假信号)
- 关键:区分度来自 strength 的方向状态(10日净位移为负),不是幅度(跌≥10%的事件波幅都≥3ATR);
  300251 1/06(5日跌11.3%但10日前更低)属弱信号(74%区间),被 strength 过滤是设计使然
- 死区滤波视觉价值 > 独立信号价值(柱覆盖率 ~27-37%,反向孤立柱=0)

## 已知遗留(2026-08 用户选择暂不处理)

- 非骤变"波幅大但10日方向转负"的有柱日(如 000066 6/05,5日仅+2.1%),方向仍跟随旧 cur
  (confirm 延迟染色)——用户"先接受,后面再说"
- 死区保底死守 cur:窄幅阴跌期保底阳(000400 8/27~9/04),与"保底跟K线"权衡后选稳定

## 常用命令

```
python panic_reversal.py grid          # 参数网格(84只)
python panic_reversal.py one sz300437  # 单只检测+画图
# streamlit: streamlit run streamlit_demo.py(走 run_segmentation,默认 strength_win=10 dir_atr=2.0 decay_days=5;门控=250日回归线;目标价线自动在 panel0)
# 触发价图:plot_price_segmentation_v10(..., reg_preds_long=250日回归线) 即自动画蓝粗虚线(阴柱期)
# 注意:修改 panic_reversal.py/price_segmenter_v10.py 后必须重启 streamlit 进程才生效
```
