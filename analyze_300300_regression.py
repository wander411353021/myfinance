"""复盘 300300：tdx_quant 取数，复现 V10 回归线(120日滚动回归)，
量化两种"波段/上涨起点"结构：
  (A) 回归斜率局部谷底(slope 减速后重新加速)
  (B) 价格从回归线下方重新站回线上方(回踩趋势基线后 reclaim)
对比随机基准。"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tdx_quant import get_daily_kline_from_tdx
from mean_reversion.signal_residual import compute_rolling_regression

CODE = "300300"
END = "2025-11-30"
TAIL = 200
REG_W = 120

df = get_daily_kline_from_tdx(CODE, END).reset_index(drop=True)
close = df["close"].values.astype(float)
high = df["high"].values.astype(float)
low = df["low"].values.astype(float)
n = len(df)

preds, slopes = compute_rolling_regression(close, window=REG_W)
start = n - TAIL
xs = np.arange(TAIL)
vis_close = close[start:start + TAIL]
vis_pred = preds[start:start + TAIL]
vis_slope = slopes[start:start + TAIL]

HORIZONS = [5, 10, 20]

def fwd_ret(t, h):
    j = t + h
    return np.nan if j >= n else close[j] / close[t] - 1.0

# ── 事件 (A): 斜率局部谷底（先平滑再找极小值）──
ss = pd.Series(vis_slope).ewm(span=5, min_periods=1).mean().values
evA = [i for i in range(2, TAIL - 2)
       if ss[i] <= ss[i - 1] and ss[i] <= ss[i + 1] and vis_slope[i] > 0]
# ── 事件 (B): 价格从线下 reclaim 到线上 ──
below = vis_close <= vis_pred
evB = [i for i in range(1, TAIL) if below[i - 1] and not below[i]]
print(f"[data] {CODE} rows={n}, window={TAIL}")
print(f"[evA] 斜率局部谷底(波段重启): {len(evA)} 个")
print(f"[evB] 价格reclaim回归线(回踩结束): {len(evB)} 个")

def stats(indices_abs, label):
    rows = {h: [] for h in HORIZONS}
    maxfwd = []
    for t in indices_abs:
        mf = -1e9
        for h in HORIZONS:
            r = fwd_ret(t, h)
            if not np.isnan(r):
                rows[h].append(r); mf = max(mf, r)
        maxfwd.append(mf)
    print(f"\n=== {label} (n={len(indices_abs)}) ===")
    for h in HORIZONS:
        a = np.array(rows[h])
        if a.size:
            print(f"  fwd{h}d: mean={a.mean()*100:+.2f}%  med={np.median(a)*100:+.2f}%  "
                  f"win%={(a>0).mean()*100:.1f}  n={a.size}")
    m = np.array(maxfwd)
    print(f"  max-up-in-20d>=8% (good_zone): {(m>=0.08).mean()*100:.1f}%  n={m.size}")

stats([start + i for i in evA], "A: slope trough (band restart)")
stats([start + i for i in evB], "B: price reclaims regression line")
rng = np.random.default_rng(7)
rand = [start + int(x) for x in rng.integers(1, TAIL, size=60)]
rand = [t for t in rand if t + max(HORIZONS) < n]
stats(rand, "RANDOM baseline")

# ── 图 ──
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
for i in range(TAIL):
    c = "#ef5350" if vis_close[i] >= vis_close[i] else "#26a69a"
    ax0.plot([i, i], [low[start+i], high[start+i]], color="#888", linewidth=0.4)
    ax0.add_patch(plt.Rectangle((i-0.3, vis_close[i]), 0.6, 1e-6,
                 facecolor=c, edgecolor=c))
ax0.plot(xs, vis_pred, color="#d62728", linewidth=1.8, linestyle="--",
         label="RegLine(120d)")
for i in evB:
    ax0.scatter(i, vis_close[i]*0.985, marker="^", s=70, color="green", zorder=5)
for i in evA:
    ax0.scatter(i, vis_pred[i]*1.01, marker="o", s=40, color="orange", zorder=5)
ax0.set_title(f"{CODE} RegLine + band-start markers (^=reclaim, o=slope-trough)")
ax0.legend(loc="upper left"); ax0.grid(alpha=0.3)
ax1.plot(xs, vis_slope, color="#1565C0", linewidth=1.2, label="slope(120d)")
ax1.plot(xs, ss, color="#ff7f0e", linewidth=1.0, alpha=0.7, label="slope EMA5")
ax1.axhline(0, color="k", linewidth=0.8)
for i in evA:
    ax1.scatter(i, vis_slope[i], color="orange", zorder=5)
ax1.set_title("Regression slope (local troughs = band restart)")
ax1.legend(loc="upper left"); ax1.grid(alpha=0.3)
plt.tight_layout()
out = "result/300300_regression_bandstart.png"
plt.savefig(out, dpi=110)
print(f"\n[plot] {out}")
