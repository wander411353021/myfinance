"""mean_reversion 合成数据冒烟测试（不依赖网络）

验证：
  T1  融合矩阵 → confidence 1-5 各级可达（fuse_signal 直接单测）
  T2  kwargs (debt_z_buy / overhang_min / max_below_bars) 真正生效
  T3  锤子线维度 (hammer_like) 与旧 near_low 行为相反
  T4  残差 std 稳健化 / use_log
  T5  下跌趋势防护 consec_below
  T6  止跌确认 (stalled / not_new_low)
  T8  交易纪律字段 stop_loss_pct / position_pct

运行:
  python mean_reversion/smoke_test.py
"""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mean_reversion.fuser import compute_signal, fuse_signal
from mean_reversion.signal_energy import compute_energy_signal
from mean_reversion.signal_residual import compute_residual_signal


def make_df(closes, vols, hammer_last=False):
    n = len(closes)
    opens = np.empty(n)
    highs = np.empty(n)
    lows = np.empty(n)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]
    for i in range(n):
        o, c = opens[i], closes[i]
        hi = max(o, c) * 1.01
        lo = min(o, c) * 0.99
        highs[i], lows[i] = hi, lo
    if hammer_last:
        # 最后一根做锤子线：开盘近低、收盘靠高、长下影
        lo = closes[-1] * 0.95
        hi = closes[-1] * 1.03
        o = lo * 1.001
        c = hi * 0.99
        opens[-1], closes[-1] = o, c
        highs[-1], lows[-1] = hi, lo
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates, "open": opens, "high": highs,
        "low": lows, "close": closes, "volume": vols,
    })


def build_buy_setup(seed=0, n=200):
    """强回归买点：沿升势运行，末段急跌破趋势，末端锤子线+缩量+减速。"""
    np.random.seed(seed)
    closes = 10.0 + 0.01 * np.arange(n)
    closes[:170] += np.random.normal(0, 0.07, n)[:170]
    base = closes[169]
    closes[170:190] = np.linspace(base, base - 0.5, 20) + np.random.normal(0, 0.03, 20)
    closes[190:195] = np.linspace(closes[189], base - 2.4, 5) + np.random.normal(0, 0.02, 5)
    closes[195:] = np.linspace(closes[194], closes[194] - 0.1, 5) + np.random.normal(0, 0.02, 5)
    vols = np.random.uniform(1e5, 2e5, n)
    vols[-12:] *= 0.35
    return make_df(closes, vols, hammer_last=True)


def build_downtrend(seed=1, n=200):
    """先升后持续回落到趋势线下方（consec_below 大，无反转确认）。"""
    np.random.seed(seed)
    closes = np.full(n, 20.0)
    closes[:160] = 20.0 + 0.025 * np.arange(160) + np.random.normal(0, 0.08, 160)
    closes[160:] = np.linspace(closes[159], 21.5, n - 160) + np.random.normal(0, 0.05, n - 160)
    vols = np.random.uniform(1e5, 2e5, n)
    return make_df(closes, vols)


def build_debt(seed=2, n=200):
    """先横盘大涨脱离趋势线（累积 overhang），再跌回趋势线下方触发负债期。"""
    np.random.seed(seed)
    closes = np.full(n, 10.0)
    closes[:150] = 10.0 + np.random.normal(0, 0.05, 150)
    closes[150:172] = np.linspace(10.0, 16.0, 22) + np.random.normal(0, 0.05, 22)
    closes[172:] = np.linspace(16.0, 10.0, n - 172) + np.random.normal(0, 0.05, n - 172)
    vols = np.random.uniform(1e5, 2e5, n)
    return make_df(closes, vols)


def main():
    fails = []

    # ---- T1: fuse_signal 直接单测，覆盖 confidence 1-5 ----
    cases = [
        ((3, 5, -3), ("buy", 5)),
        ((3, 3, -3), ("buy", 4)),
        ((3, 1, -3), ("neutral", 2)),
        ((3, 0, -3), ("neutral", 1)),
        ((1, 3, -1), ("buy", 3)),
        ((1, 2, -1), ("neutral", 2)),
        ((1, 0, -1), ("neutral", 1)),
        ((-3, 0, -3), ("sell", 3)),
        ((0, 0, 0), ("neutral", 0)),
    ]
    confs_seen = set()
    for args, expect in cases:
        got = fuse_signal(*args)
        confs_seen.add(got[1])
        if got != expect:
            fails.append(f"fuse_signal{args}={got} 期望 {expect}")
    print(f"[FUSE] conf 各级可达: {sorted(confs_seen)}")
    if not {1, 2, 3, 4, 5}.issubset(confs_seen):
        fails.append("fuse_signal: confidence 1-5 未全部可达")

    # ---- T1 + T3 + T6: 强回归买点 → buy conf 5 ----
    r = compute_signal(build_buy_setup(), code="T_BUY")
    print(f"[BUY ] signal={r.signal} conf={r.confidence} "
          f"z={r.z_residual:.2f} energy={r.energy_score} "
          f"hammer={r.details.get('hammer_like')} not_new_low={r.details.get('not_new_low')}")
    if r.signal != "buy" or r.confidence != 5:
        fails.append(f"BUY: 期望 buy/5, 实际 {r.signal}/{r.confidence}")
    if not r.details.get("hammer_like"):
        fails.append("BUY: 期望 hammer_like=True")

    # ---- T5: 下跌趋势防护（无反转确认）→ 压制，不买 ----
    r = compute_signal(build_downtrend(), code="T_DT")
    print(f"[DOWN] signal={r.signal} conf={r.confidence} "
          f"z={r.z_residual:.2f} in_debt={r.in_debt} "
          f"downtrend_guard={r.downtrend_guard} consec={r.debt_remaining}")
    if not r.downtrend_guard:
        fails.append("DOWN: 期望触发 downtrend_guard")
    # 无反转确认时不应买
    if r.signal == "buy":
        if not (r.details.get("hammer_like") or r.details.get("not_new_low")):
            fails.append("DOWN: 无反转确认却买入")

    # ---- T2: 关闭防护 (max_below_bars=0) → 防护标志为 False ----
    r2 = compute_signal(build_downtrend(), code="T_DT2", max_below_bars=0)
    print(f"[DOWN-off] signal={r2.signal} conf={r2.confidence} downtrend_guard={r2.downtrend_guard}")
    if r2.downtrend_guard:
        fails.append("DOWN-off: max_below_bars=0 应关闭防护")

    # ---- T2: 负债压制 + debt_z_buy 生效 ----
    r = compute_signal(build_debt(), code="T_DEBT")
    r_loose = compute_signal(build_debt(), code="T_DEBT2", debt_z_buy=10.0)
    print(f"[DEBT] signal={r.signal} conf={r.confidence} "
          f"z={r.z_residual:.2f} in_debt={r.in_debt} overhang={r.overhang}")
    print(f"[DEBT-loose] signal={r_loose.signal} conf={r_loose.confidence} "
          f"z={r_loose.z_residual:.2f} in_debt={r_loose.in_debt}")
    if not r.in_debt:
        fails.append("DEBT: 期望 in_debt=True")
    if r_loose.confidence == r.confidence and r_loose.signal == r.signal:
        fails.append("DEBT: debt_z_buy 参数未改变结果（应已生效）")

    # ---- T3: hammer_like 与 旧 near_low（收盘近低）行为相反 ----
    n = 25
    closes = np.full(n, 10.0)
    vols = np.full(n, 1e5)
    highs = np.full(n, 10.1)
    lows = np.full(n, 9.9)
    # 锤子线：收盘靠高、长下影
    h_h, h_l, h_c = 10.8, 9.5, 10.7
    # 收盘近低：收盘贴最低、上影长
    l_h, l_l, l_c = 10.6, 10.2, 10.25
    ch, cl = closes.copy(), closes.copy()
    hh, hl = highs.copy(), lows.copy()
    hh[-1], hl[-1], ch[-1] = h_h, h_l, h_c
    lh, ll = highs.copy(), lows.copy()
    lh[-1], ll[-1], cl[-1] = l_h, l_l, l_c
    e_hammer = compute_energy_signal(ch, vols, hh, hl)
    e_low = compute_energy_signal(cl, vols, lh, ll)
    print(f"[HAMMER] hammer_like={e_hammer['hammer_like']}  "
          f"[LOWCLOSE] hammer_like={e_low['hammer_like']}")
    if not e_hammer["hammer_like"]:
        fails.append("HAMMER: 锤子线应 hammer_like=True")
    if e_low["hammer_like"]:
        fails.append("LOWCLOSE: 收盘近低不应 hammer_like=True")

    # ---- T4: robust_std 与 use_log 不报错且 z 合理 ----
    df = build_buy_setup()
    closes = df["close"].values.astype(float)
    ra = compute_residual_signal(closes, reg_window=120, robust_std=True)
    rb = compute_residual_signal(closes, reg_window=120, robust_std=False)
    rc = compute_residual_signal(closes, reg_window=120, use_log=True)
    print(f"[ROBUST] z(robust)={ra['z_residual']:.2f} "
          f"z(plain)={rb['z_residual']:.2f} z(log)={rc['z_residual']:.2f}")
    if not (-10 < ra["z_residual"] < 0):
        fails.append("ROBUST: z 应在合理负区间")

    # ---- T8: 交易纪律字段 ----
    r = compute_signal(build_buy_setup(), code="T_DISC", stop_loss_pct=-0.1, position_pct=0.25)
    print(f"[DISC] stop_loss={r.stop_loss_pct} position={r.position_pct}")
    if r.stop_loss_pct != -0.1 or r.position_pct != 0.25:
        fails.append("DISC: 交易纪律字段未生效")

    if fails:
        print("\n❌ 失败项:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("\n✅ 全部冒烟测试通过")


if __name__ == "__main__":
    main()
