# -*- coding: utf-8 -*-
"""
黄金坑算法独立测试与分析脚本
从 panic_reversal.py 提取核心算法，用模拟数据运行并可视化。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# 1. 从 signal_residual.py 提取的滚动回归
# ============================================================
def compute_rolling_regression(closes, window=250, use_log=True):
    """滚动对数线性回归: log(price) = a*t + b, 返回 (preds, slopes)"""
    n = len(closes)
    preds = np.full(n, np.nan)
    slopes = np.full(n, np.nan)
    t = np.arange(window, dtype=np.float64)
    sum_t = np.sum(t)
    sum_tt = np.sum(t * t)
    denom = window * sum_tt - sum_t * sum_t
    for i in range(window - 1, n):
        y = closes[i - window + 1:i + 1].astype(np.float64)
        if use_log:
            y = np.log(np.maximum(y, 1e-8))
        sum_y = np.sum(y)
        sum_ty = np.sum(t * y)
        a = (window * sum_ty - sum_t * sum_y) / denom
        b = (sum_y - a * sum_t) / window
        pred = a * (window - 1) + b
        preds[i] = np.exp(pred) if use_log else pred
        slopes[i] = a
    return preds, slopes

# ============================================================
# 2. 从 panic_reversal.py 提取的黄金坑检测
# ============================================================
def detect_golden_pit(closes, reg250, z_thr=-1.5, merge_gap=15,
                       launch_gate=0.9, use_pre_std=True,
                       max_pre_gain=None, require_below_gate=False):
    """黄金坑检测 — 无未来函数。
    坑 = close 相对 250日回归线残差 z-score 持续 < z_thr(-1.5) 的深跌段;
    坑底 = 段内最低 close; 启动 = 坑底后首次收复门控线(reg250×0.9)且 >坑底×1.02。
    use_pre_std: 两遍扫描,第二遍用坑前60日std重算段内z,精确定边界。
    返回 [(段起点s, 坑底b, 启动日lch或None)] 升序。
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    resid = closes - np.asarray(reg250, dtype=float)
    rstd = np.full(n, np.nan)
    for i in range(59, n):
        rstd[i] = np.std(resid[i - 59:i + 1])
    z = np.full(n, np.nan)
    for i in range(59, n):
        z[i] = resid[i] / rstd[i] if rstd[i] > 0 else 0.0
    gate_line = np.asarray(reg250, dtype=float) * launch_gate

    # pass1: 滚动std粗定坑段
    pits = []
    st = None
    for i in range(59, n):
        in_pit = np.isfinite(z[i]) and z[i] < z_thr and (
            closes[i] < gate_line[i] if require_below_gate else True)
        if in_pit and st is None:
            st = i
        elif (not in_pit) and st is not None:
            pits.append([st, i - 1]); st = None
    if st is not None:
        pits.append([st, n - 1])
    merged = []
    for p in pits:
        if merged and p[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = p[1]
        else:
            merged.append(list(p))

    # pass2: 坑前std重算,精确定边界
    refined = []
    for s0, e0 in merged:
        if use_pre_std and s0 >= 60:
            pre_std = np.std(resid[s0 - 60:s0])
            if pre_std <= 0:
                pre_std = rstd[s0]
        else:
            pre_std = rstd[s0]
        s_a = max(0, s0 - 10); e_a = min(n - 1, e0 + 10)
        seg = []
        st2 = None
        for i in range(s_a, e_a + 1):
            zz = resid[i] / pre_std if pre_std > 0 else 0.0
            in_pit2 = zz < z_thr and (
                closes[i] < gate_line[i] if require_below_gate else True)
            if in_pit2 and st2 is None:
                st2 = i
            elif (not in_pit2 or not np.isfinite(zz)) and st2 is not None:
                seg.append([st2, i - 1]); st2 = None
        if st2 is not None:
            seg.append([st2, e_a])
        if not seg:
            continue
        m2 = []
        for p in seg:
            if m2 and p[0] - m2[-1][1] <= merge_gap:
                m2[-1][1] = p[1]
            else:
                m2.append(list(p))
        s, e = m2[0][0], m2[-1][1]
        b = int(s + np.argmin(closes[s:e + 1]))
        lch = None
        for i in range(b + 1, n):
            if closes[i] >= reg250[i] * launch_gate and closes[i] > closes[b] * 1.02:
                lch = i
                break
        refined.append((s, b, lch))
    return refined

# ============================================================
# 3. 坑量能质量标签
# ============================================================
def compute_pit_quality(pits, closes, volumes, pre_win=20, fill_win=20, fill_lead=2):
    out = []
    for s, b, lch in pits:
        v_pre = np.mean(volumes[max(0, s - pre_win):s]) if s >= pre_win else np.nan
        v_pit = np.mean(volumes[s:b + 1])
        shrink = v_pit / v_pre if v_pre and np.isfinite(v_pre) and v_pre > 0 else np.nan
        if lch is not None:
            v_base = np.mean(volumes[max(0, lch - fill_win):lch]) if lch >= fill_win else np.nan
            v_lch = np.mean(volumes[max(0, lch - fill_lead):lch + 1])
            fill = v_lch / v_base if v_base and np.isfinite(v_base) and v_base > 0 else np.nan
        else:
            fill = np.nan
        sh_ok = np.isfinite(shrink) and shrink < 1.0
        fl_ok = np.isfinite(fill) and fill > 1.2
        q = 'strong' if (sh_ok and fl_ok) else ('normal' if (sh_ok or fl_ok) else 'weak')
        out.append((shrink, fill, q))
    return out

# ============================================================
# 4. 构造模拟数据: 典型黄金坑形态
# ============================================================
np.random.seed(42)
n_days = 600
dates = pd.date_range('2022-06-01', periods=n_days, freq='B')

# 构造价格: 慢牛上涨 → 急跌挖坑 → 坑底横盘缩量 → 放量出坑 → 主升
t = np.arange(n_days)
# 基础慢牛趋势(年化约30%)
base = 10.0 * np.exp(0.0012 * t)
# 挖坑段(第380-440天): 前10天急跌45%, 后50天坑底横盘
pit_drop = np.ones(n_days)
for i in range(380, 390):
    pit_drop[i] = 1.0 - 0.45 * (i - 380) / 10  # 急跌到0.55
for i in range(390, 440):
    pit_drop[i] = 0.55 + 0.03 * np.sin((i - 390) * 0.3)  # 坑底震荡
# 出坑后主升(第445天后)
rally = np.ones(n_days)
for i in range(445, n_days):
    rally[i] = np.exp(0.006 * (i - 445))
# 加噪声
noise = np.random.normal(0, 0.012, n_days)
closes = base * pit_drop * rally * np.exp(noise)

# 构造成交量: 正常量 + 挖坑初期放量(恐慌) + 坑底缩量 + 出坑放量
vol_base = 1_000_000 * (1 + 0.2 * np.sin(t * 0.08))
vol_pit_panic = np.where((t >= 380) & (t <= 392), 2.8, 1.0)
vol_pit_shrink = np.where((t >= 393) & (t <= 440), 0.35, 1.0)
vol_launch = np.where((t >= 442) & (t <= 460), 3.2, 1.0)
volumes = vol_base * vol_pit_panic * vol_pit_shrink * vol_launch * np.random.uniform(0.85, 1.15, n_days)

# OHLC
opens = closes * np.random.uniform(0.99, 1.01, n_days)
highs = np.maximum(opens, closes) * np.random.uniform(1.0, 1.02, n_days)
lows = np.minimum(opens, closes) * np.random.uniform(0.98, 1.0, n_days)

df = pd.DataFrame({'date': dates, 'open': opens, 'high': highs, 'low': lows,
                   'close': closes, 'volume': volumes})

# ============================================================
# 5. 运行算法
# ============================================================
print("=" * 70)
print("黄金坑算法运行测试")
print("=" * 70)

reg250, slopes = compute_rolling_regression(closes, window=250, use_log=True)
pits = detect_golden_pit(closes, reg250, z_thr=-1.5, merge_gap=15,
                          launch_gate=0.9, use_pre_std=True)
qualities = compute_pit_quality(pits, closes, volumes)

print(f"\n数据长度: {n_days} 天")
print(f"检测到黄金坑数量: {len(pits)}")

for idx, ((s, b, lch), (shrink, fill, q)) in enumerate(zip(pits, qualities)):
    print(f"\n--- 坑 #{idx+1} ---")
    print(f"  坑段起点: 第{s}天 ({dates[s].strftime('%Y-%m-%d')}), 价格={closes[s]:.2f}")
    print(f"  坑底:     第{b}天 ({dates[b].strftime('%Y-%m-%d')}), 价格={closes[b]:.2f}")
    print(f"  坑长:     {b - s + 1} 天")
    print(f"  坑跌幅:   {(closes[b]/closes[s]-1)*100:.1f}%")
    if lch is not None:
        print(f"  出坑日:   第{lch}天 ({dates[lch].strftime('%Y-%m-%d')}), 价格={closes[lch]:.2f}")
        print(f"  快启动:   {'是(≤5天)' if lch - b <= 5 else '否(>5天)'} ({lch - b}天)")
        # 出坑后收益
        if lch + 60 < n_days:
            r20 = closes[min(lch+20, n_days-1)] / closes[lch] - 1
            r60 = closes[min(lch+60, n_days-1)] / closes[lch] - 1
            print(f"  出坑后r20: {r20*100:+.1f}%, r60: {r60*100:+.1f}%")
    else:
        print(f"  出坑日:   未出坑(数据末端)")
    print(f"  缩量挖比: {shrink:.2f}" if np.isfinite(shrink) else "  缩量挖比: N/A")
    print(f"  放量填比: {fill:.2f}" if np.isfinite(fill) else "  放量填比: N/A")
    print(f"  量能质量: {q}")

# ============================================================
# 6. 可视化
# ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                                      gridspec_kw={'height_ratios': [3, 1, 1.2]})
fig.suptitle('黄金坑算法检测结果 — 模拟数据', fontsize=14, fontweight='bold')

# K线
x = np.arange(n_days)
for i in range(n_days):
    color = '#E15759' if closes[i] >= opens[i] else '#2E7D32'
    ax1.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.5)
    ax1.add_patch(plt.Rectangle((i - 0.3, min(opens[i], closes[i])), 0.6,
                                  abs(closes[i] - opens[i]) or 0.01, color=color))

# 250日回归线
ax1.plot(x, reg250, color='#1565C0', linewidth=1.2, label='250日回归线', alpha=0.8)
# 门控线
ax1.plot(x, reg250 * 0.9, color='#FF8F00', linewidth=1.0, linestyle='--',
         label='门控线(reg×0.9)', alpha=0.7)

# 标注黄金坑
for idx, (s, b, lch) in enumerate(pits):
    # 坑段背景
    ax1.axvspan(s, b if lch is None else lch, alpha=0.12, color='#E53935',
                 label=f'黄金坑#{idx+1}' if idx == 0 else None)
    # 坑底
    ax1.plot(b, closes[b], 'v', color='#D32F2F', markersize=12, zorder=5)
    ax1.annotate(f'坑底\n{closes[b]:.2f}', (b, closes[b]),
                 textcoords='offset points', xytext=(0, -25), ha='center',
                 fontsize=8, color='#D32F2F', fontweight='bold')
    # 出坑日
    if lch is not None:
        ax1.plot(lch, closes[lch], '^', color='#1B5E20', markersize=12, zorder=5)
        ax1.annotate(f'出坑\n{closes[lch]:.2f}', (lch, closes[lch]),
                     textcoords='offset points', xytext=(0, 12), ha='center',
                     fontsize=8, color='#1B5E20', fontweight='bold')

ax1.set_ylabel('价格')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.2)
ax1.set_title('K线 + 回归线 + 门控线 + 黄金坑标注')

# 成交量
colors_vol = ['#E15759' if closes[i] >= opens[i] else '#2E7D32' for i in range(n_days)]
ax2.bar(x, volumes, color=colors_vol, width=0.8, alpha=0.7)
ax2.set_ylabel('成交量')
ax2.grid(True, alpha=0.2)
ax2.set_title('成交量 (挖坑期缩量 / 出坑期放量)')

# z-score
resid = closes - reg250
rstd = np.full(n_days, np.nan)
for i in range(59, n_days):
    rstd[i] = np.std(resid[i - 59:i + 1])
z = np.full(n_days, np.nan)
for i in range(59, n_days):
    z[i] = resid[i] / rstd[i] if rstd[i] > 0 else 0.0
ax3.plot(x, z, color='#6A1B9A', linewidth=1.0)
ax3.axhline(-1.5, color='#E53935', linestyle='--', linewidth=1.0, label='z_thr=-1.5')
ax3.axhline(0, color='#999', linewidth=0.5)
ax3.fill_between(x, z, -1.5, where=(z < -1.5), alpha=0.3, color='#E53935')
ax3.set_ylabel('z-score')
ax3.set_xlabel('交易日')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.2)
ax3.set_title('残差 z-score (黄金坑 = z持续<-1.5)')

# x轴日期
tick_idx = np.linspace(0, n_days - 1, 10, dtype=int)
ax3.set_xticks(tick_idx)
ax3.set_xticklabels([dates[i].strftime('%Y-%m') for i in tick_idx], rotation=30)

plt.tight_layout()
out_path = '/home/user/.super_doubao/super-doubao-runtime/workspace/pressure-level-algorithm/golden_pit_analysis.png'
plt.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"\n可视化图表已保存: {out_path}")

# ============================================================
# 7. 算法压力测试: 边界情况
# ============================================================
print("\n" + "=" * 70)
print("算法边界情况测试")
print("=" * 70)

# 测试1: 无坑(平稳上涨)
print("\n[测试1] 平稳上涨无坑:")
t1 = np.arange(300)
c1 = 10.0 * np.exp(0.002 * t1) + np.random.normal(0, 0.01, 300)
reg1, _ = compute_rolling_regression(c1, window=250)
pits1 = detect_golden_pit(c1, reg1)
print(f"  检测到坑数: {len(pits1)} (预期: 0)")

# 测试2: V型急跌急涨(短坑)
print("\n[测试2] V型急跌急涨(短坑<8天):")
c2 = np.ones(300) * 10.0
c2[150:155] = 10.0 * np.linspace(1, 0.7, 5)  # 5天跌30%
c2[155:165] = 7.0 * np.linspace(1, 1.4, 10)   # 10天涨回
c2 = c2 * np.exp(np.random.normal(0, 0.01, 300))
reg2, _ = compute_rolling_regression(c2, window=250)
pits2 = detect_golden_pit(c2, reg2)
print(f"  检测到坑数: {len(pits2)}")
for s, b, lch in pits2:
    print(f"    坑长={b-s+1}天, 快启动={'是' if lch and lch-b<=5 else '否'}")
print(f"  注: 回测策略要求坑长≥8天,短坑会被过滤")

# 测试3: 持续阴跌(下跌中继)
print("\n[测试3] 持续阴跌(下跌中继,无明显坑底):")
c3 = 10.0 * np.exp(-0.003 * np.arange(300)) + np.random.normal(0, 0.005, 300)
reg3, _ = compute_rolling_regression(c3, window=250)
pits3 = detect_golden_pit(c3, reg3)
print(f"  检测到坑数: {len(pits3)}")
print(f"  注: 持续阴跌中z可能持续<-1.5,算法会标记为一个长坑,但出坑日可能不存在")

# 测试4: 次新股(数据不足250天)
print("\n[测试4] 次新股(数据<250天,reg250不稳定):")
c4 = 10.0 * np.exp(0.005 * np.arange(200)) + np.random.normal(0, 0.02, 200)
c4[100:120] *= np.linspace(1, 0.6, 20)  # 挖坑
reg4, _ = compute_rolling_regression(c4, window=250)
valid_reg = np.sum(np.isfinite(reg4))
pits4 = detect_golden_pit(c4, reg4)
print(f"  有效回归点数: {valid_reg}/200")
print(f"  检测到坑数: {len(pits4)}")
print(f"  注: 次新股reg250数据不足,坑检测可能漏检(代码注释中已标注此盲区)")

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
