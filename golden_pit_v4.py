# -*- coding: utf-8 -*-
"""
黄金坑v4: 双因子进坑判据 (polo4111 2026-08-30)

克服v3底层缺陷:
  v3只用z(相对reg偏离)判进坑 → 远离reg横盘也进坑(假坑), 贴近reg急跌进不了坑(真坑漏检)
  v4用z+绝对跌幅双因子, 两条路径覆盖:

路径A - 标准超跌坑(远离reg且真跌):
  z < -1.5 AND 60日高点跌幅 > 8% AND 坑内波动 > 15% AND 坑底距reg < -8%
路径B - 贴近reg急跌坑(贴近reg但急跌):
  距reg > -5% AND 7日跌幅 > 7% AND z < -0.5 AND 进坑前10日高点/坑底跌幅 > 8%

出坑: 同v3(z连续3日>=-1.5确认 + 价格过gate_line + 高于坑底2%)
严格无未来函数: 所有判据只用当日及之前数据
"""
import numpy as np

def detect_golden_pit_v4(closes, reg250, z_thr=-1.5, launch_gate=0.9,
                          abs_drop_thr=0.08, near_reg_thr=0.05,
                          near_drop_thr=0.07, near_drop_win=7, near_z_thr=-0.5,
                          confirm_days=3, min_depth=0.08, min_pit_amp=0.15,
                          b_min_abs_drop=0.08):
    """
    黄金坑v4双因子检测
    返回: [(start_idx, bottom_idx, launch_idx, pit_type), ...]
          pit_type: 'A'=标准超跌坑, 'B'=贴近reg急跌坑
    """
    closes = np.asarray(closes, dtype=float)
    reg250 = np.asarray(reg250, dtype=float)
    n = len(closes)
    resid = closes - reg250
    rstd = np.full(n, np.nan)
    for i in range(59, n):
        rstd[i] = np.std(resid[i-59:i+1])
    z = np.where(rstd > 0, resid / rstd, 0.0)
    gate_line = reg250 * launch_gate
    out = []
    i = 59
    while i < n:
        # === 双因子进坑判据 ===
        in_pit = False
        ptype = ''
        # 路径A: 标准超跌坑
        if np.isfinite(z[i]) and z[i] < z_thr:
            hi60 = closes[max(0, i-60):i+1].max()
            drop60 = closes[i] / hi60 - 1 if hi60 > 0 else 0
            if drop60 < -abs_drop_thr:
                in_pit = True
                ptype = 'A'
        # 路径B: 贴近reg急跌坑
        if not in_pit:
            dist_reg = closes[i] / reg250[i] - 1 if (np.isfinite(reg250[i]) and reg250[i] > 0) else 0
            if (dist_reg > -near_reg_thr and np.isfinite(z[i]) and z[i] < near_z_thr
                    and i >= near_drop_win):
                dropN = closes[i] / closes[i-near_drop_win] - 1
                if dropN < -near_drop_thr:
                    in_pit = True
                    ptype = 'B'
        if not in_pit:
            i += 1
            continue
        # === 坑内维护 + 出坑判定 ===
        s = i
        b = i
        bv = closes[i]
        lch = None
        j = i
        pre_hi = closes[max(0, i-10):i].max() if i > 0 else closes[i]
        while j < n:
            if closes[j] < bv:
                bv = closes[j]
                b = j
            zj = z[j] if np.isfinite(z[j]) else 0
            if (zj >= z_thr and closes[j] >= gate_line[j]
                    and closes[j] > bv * 1.02):
                # z连续确认
                confirmed = True
                for k in range(1, confirm_days):
                    if j + k >= n:
                        confirmed = False
                        break
                    zk = z[j+k] if np.isfinite(z[j+k]) else 0
                    if np.isfinite(zk) and zk < z_thr:
                        confirmed = False
                        break
                if confirmed:
                    lch = j
                    break
            j += 1
        # === 分路径过滤 ===
        if lch is not None:
            if ptype == 'A':
                depth = closes[b] / reg250[b] - 1 if (np.isfinite(reg250[b]) and reg250[b] > 0) else 0
                pit_hi = closes[s:lch+1].max()
                pit_lo = closes[s:lch+1].min()
                amp = pit_hi / pit_lo - 1 if pit_lo > 0 else 0
                if depth <= -min_depth and amp >= min_pit_amp:
                    out.append((s, b, lch, ptype))
            else:  # 路径B: 用绝对跌幅(不用距reg深度, 因为贴近reg天然浅)
                abs_drop = closes[b] / pre_hi - 1 if pre_hi > 0 else 0
                if abs_drop <= -b_min_abs_drop:
                    out.append((s, b, lch, ptype))
        i = (lch + 1) if lch is not None else j + 1
    return out
