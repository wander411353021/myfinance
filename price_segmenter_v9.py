"""
Price Segmentation V9 — 买卖点信号版
=====================================

基于 V8，新增：因果性买卖点检测 + 第4个 Trade 面板。

V9 相比 V8 的变化:
  - 新函数: compute_buy_sell_signals() — 因果性买卖点检测
  - 4面板图表: K线 + 成交量 + 触碰信号 + 买卖信号
  - 所有买卖点逻辑都是因果性的（无未来函数）

买卖信号:
  +2 (强买): BrkHigh — 突破最近3个可见前高中最高的那个（真正的压力线）
  +1 (买入): BrkRes — 突破之前触碰过的压力位 / PullSup — UP趋势中回踩支撑
   0 (中性)
  -1 (卖出): BrkSup — 跌破支撑位 / BncRes — DOWN趋势中反弹到阻力位
  -2 (强卖): BrkLow — 跌破最近3个可见前低中最低的那个
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
import finshare
warnings.filterwarnings("ignore")

# ============================================================
# 共享工具函数（从 V8 沿用，未修改）
# ============================================================
def _compute_rolling_percentile(log_vol, ground_pct, sky_pct, rolling_window):
    """
    滚动分位数阈值 — 纯向后看，无未来函数。
    参数:
        log_vol: 对数成交量序列
        ground_pct: 地量分位（默认20，即成交量低于此分位视为地量）
        sky_pct: 天量分位（默认85，即成交量高于此分位视为天量）
        rolling_window: 滚动窗口大小
    返回:
        ground_thresh: 地量阈值数组
        sky_thresh: 天量阈值数组
    """
    log_s = pd.Series(log_vol)
    ground_thresh = (log_s.rolling(rolling_window, min_periods=20)
                     .quantile(ground_pct / 100).values.copy())
    sky_thresh = (log_s.rolling(rolling_window, min_periods=20)
                  .quantile(sky_pct / 100).values.copy())
    fv_g = np.where(~np.isnan(ground_thresh))[0]
    fv_s = np.where(~np.isnan(sky_thresh))[0]
    if len(fv_g) > 0:
        ground_thresh[:fv_g[0]] = ground_thresh[fv_g[0]]
    if len(fv_s) > 0:
        sky_thresh[:fv_s[0]] = sky_thresh[fv_s[0]]
    return ground_thresh, sky_thresh


def _build_price_result(close, smooth, phase_id, phase_name, pivots,
                        is_pending=None, pending_confidence=None,
                        vol_annotation=None):
    """
    构建价格分段的标准输出 DataFrame。
    """
    n = len(close)
    is_pivot = np.zeros(n, dtype=bool)
    pivot_type = np.array([""] * n, dtype='U8')
    for p in pivots:
        idx = p[0]; ptype = p[1]
        if idx < n:
            is_pivot[idx] = True; pivot_type[idx] = ptype

    if is_pending is None: is_pending = np.zeros(n, dtype=bool)
    if pending_confidence is None: pending_confidence = np.zeros(n, dtype=float)
    if vol_annotation is None: vol_annotation = np.array(["NEUTRAL"] * n, dtype='U14')

    result = pd.DataFrame({
        "close": close, "smooth": smooth,
        "phase": phase_name, "phase_id": phase_id,
        "is_pivot": is_pivot, "pivot_type": pivot_type,
        "is_pending": is_pending, "pending_confidence": pending_confidence,
        "vol_annotation": vol_annotation,
        "touch_signal": np.zeros(n, dtype=int),
        "touch_source": np.array([""] * n, dtype='U20'),
    })
    result.attrs["pivots"] = list(pivots)
    return result


# ============================================================
# 一、上帝视角分段器（有未来函数，仅用作基准对照）
# ============================================================
class FutureLookingPriceSegmenter:
    """使用 Savgol 平滑 + 全局峰值检测。有未来函数，仅作为基准对照。实盘不能用！"""

    def __init__(self, sg_window=11, sg_poly=3, peak_distance=3, min_reversal_pct=0.02):
        self.sg_window = sg_window; self.sg_poly = sg_poly
        self.peak_distance = peak_distance; self.min_reversal_pct = min_reversal_pct

    def segment(self, close):
        close = np.asarray(close, dtype=float); n = len(close)
        smooth = close.copy() if n < self.sg_window else savgol_filter(close, self.sg_window, self.sg_poly)
        peaks, _ = find_peaks(smooth, distance=self.peak_distance)
        troughs, _ = find_peaks(-smooth, distance=self.peak_distance)
        pivots = []
        for idx in peaks: pivots.append((idx, "PEAK"))
        for idx in troughs: pivots.append((idx, "TROUGH"))
        pivots.sort(key=lambda x: x[0])

        filtered = []
        for p in pivots:
            if len(filtered) == 0: filtered.append(p); continue
            last_idx, last_type = filtered[-1]; curr_idx, curr_type = p
            if curr_type == last_type:
                if curr_type == "PEAK" and smooth[curr_idx] > smooth[last_idx]: filtered[-1] = p
                elif curr_type == "TROUGH" and smooth[curr_idx] < smooth[last_idx]: filtered[-1] = p
                continue
            reversal = abs(smooth[curr_idx] - smooth[last_idx]) / smooth[last_idx]
            if reversal >= self.min_reversal_pct: filtered.append(p)
            elif curr_type == "PEAK" and smooth[curr_idx] > smooth[last_idx]: filtered[-1] = p
            elif curr_type == "TROUGH" and smooth[curr_idx] < smooth[last_idx]: filtered[-1] = p
        pivots = filtered

        phase_id = np.zeros(n, dtype=int)
        phase_name = np.array(["NEUTRAL"] * n, dtype='U8')
        if len(pivots) == 0:
            return _build_price_result(close, smooth, phase_id, phase_name, pivots)

        first_type = pivots[0][1]
        if first_type == "PEAK":
            phase_id[:pivots[0][0]] = 1; phase_name[:pivots[0][0]] = "UP"
        else:
            phase_id[:pivots[0][0]] = -1; phase_name[:pivots[0][0]] = "DOWN"

        for i in range(len(pivots) - 1):
            s_idx = pivots[i][0]; e_idx = pivots[i + 1][0]
            if pivots[i][1] == "TROUGH" and pivots[i + 1][1] == "PEAK":
                phase_id[s_idx:e_idx + 1] = 1; phase_name[s_idx:e_idx + 1] = "UP"
            elif pivots[i][1] == "PEAK" and pivots[i + 1][1] == "TROUGH":
                phase_id[s_idx:e_idx + 1] = -1; phase_name[s_idx:e_idx + 1] = "DOWN"
            else:
                mid = (s_idx + e_idx) // 2
                if smooth[e_idx] > smooth[s_idx]:
                    phase_id[s_idx:mid + 1] = 1; phase_name[s_idx:mid + 1] = "UP"
                    phase_id[mid + 1:e_idx + 1] = -1; phase_name[mid + 1:e_idx + 1] = "DOWN"
                else:
                    phase_id[s_idx:mid + 1] = -1; phase_name[s_idx:mid + 1] = "DOWN"
                    phase_id[mid + 1:e_idx + 1] = 1; phase_name[mid + 1:e_idx + 1] = "UP"

        last_type = pivots[-1][1]
        if last_type == "TROUGH":
            phase_id[pivots[-1][0]:] = 1; phase_name[pivots[-1][0]:] = "UP"
        else:
            phase_id[pivots[-1][0]:] = -1; phase_name[pivots[-1][0]:] = "DOWN"

        return _build_price_result(close, smooth, phase_id, phase_name, pivots)


# ============================================================
# 二、因果增量式分段器（无未来函数，实盘可用）
# ============================================================
class CausalIncrementalPriceSegmenter:
    """
    因果增量式价格分段器。
    核心思路：
      1. 只向前看检测候选 pivot + 向后看确认 pivot（回撤确认）
      2. 确认后的 pivot 锁定区间方向（LOCKED），不再改变
      3. 最后一个确认 pivot 之后的区间是 PENDING 状态，可以翻转
    """

    def __init__(self, lookback=15, min_reversal_pct=0.02, confirm_bars=3,
                 ema_span=15, ground_pct=20, sky_pct=85, rolling_window=120):
        self.lookback = lookback; self.min_reversal_pct = min_reversal_pct
        self.confirm_bars = confirm_bars; self.ema_span = ema_span
        self.ground_pct = ground_pct; self.sky_pct = sky_pct
        self.rolling_window = rolling_window

    def segment(self, close, volume=None, high=None, low=None, opn=None):
        """主分段入口。返回包含 phase, is_pending, touch_signal 等列的 DataFrame。"""
        close = np.asarray(close, dtype=float); n = len(close)
        volume = np.asarray(volume, dtype=float) if volume is not None else None
        high = np.asarray(high, dtype=float) if high is not None else close
        low = np.asarray(low, dtype=float) if low is not None else close
        opn = np.asarray(opn, dtype=float) if opn is not None else close

        candidates = self._detect_candidates(close)
        confirmed_pivots = self._confirm_pivots(close, candidates)
        phase_id, phase_name, is_pending, pending_confidence = \
            self._assign_phases(n, confirmed_pivots, close)

        vol_annotation = np.array(["NEUTRAL"] * n, dtype='U14')
        if volume is not None: vol_annotation = self._annotate_volume(volume)
        smooth = self._ema_close(close)

        touch_signal, touch_source = self._compute_touch_signal(
            close, high, low, opn, volume, n, confirmed_pivots)

        is_pivot = np.zeros(n, dtype=bool); pivot_type = np.array([""] * n, dtype='U8')
        for p in confirmed_pivots:
            if p[0] < n: is_pivot[p[0]] = True; pivot_type[p[0]] = p[1]

        result = pd.DataFrame({
            "close": close, "smooth": smooth,
            "phase": phase_name, "phase_id": phase_id,
            "is_pivot": is_pivot, "pivot_type": pivot_type,
            "is_pending": is_pending, "pending_confidence": pending_confidence,
            "vol_annotation": vol_annotation,
            "touch_signal": touch_signal, "touch_source": touch_source,
        })
        result.attrs["pivots"] = list(confirmed_pivots)
        return result

    def _ema_close(self, close):
        n = len(close); s = np.zeros(n); a = 2.0 / (self.ema_span + 1); s[0] = close[0]
        for i in range(1, n): s[i] = a * close[i] + (1 - a) * s[i - 1]
        return s

    def _detect_candidates(self, close):
        """
        检测候选 PEAK 和 TROUGH — 纯左看，无未来函数。
        候选 PEAK:  当前收盘是过去 lookback 根K线中最高，且比前一根高
        候选 TROUGH: 当前收盘是过去 lookback 根K线中最低，且比前一根低
        """
        n = len(close); candidates = []
        for t in range(self.lookback, n):
            window = close[t - self.lookback:t + 1]
            if close[t] == window.max() and close[t] > close[t - 1]:
                if len(candidates) == 0 or candidates[-1][1] != "PEAK" or candidates[-1][0] < t - 1:
                    candidates.append((t, "PEAK"))
                elif close[t] >= close[candidates[-1][0]]: candidates[-1] = (t, "PEAK")
            if close[t] == window.min() and close[t] < close[t - 1]:
                if len(candidates) == 0 or candidates[-1][1] != "TROUGH" or candidates[-1][0] < t - 1:
                    candidates.append((t, "TROUGH"))
                elif close[t] <= close[candidates[-1][0]]: candidates[-1] = (t, "TROUGH")
        return candidates

    def _confirm_pivots(self, close, candidates):
        """
        回撤确认候选 pivot — 从候选向后扫描，当价格回撤 ≥ min_reversal_pct 时确认。
        PEAK 确认: close[t] ≤ peak_price × (1 - min_reversal_pct)  → 真的顶
        TROUGH 确认: close[t] ≥ trough_price × (1 + min_reversal_pct) → 真的底
        返回: [(idx, type, confirm_time), ...]
        """
        n = len(close); confirmed = []
        for cand_idx, cand_type in candidates:
            confirmed_time = None
            for t in range(cand_idx + self.confirm_bars, n):
                if cand_type == "PEAK" and close[t] <= close[cand_idx] * (1 - self.min_reversal_pct):
                    confirmed_time = t; break
                if cand_type == "TROUGH" and close[t] >= close[cand_idx] * (1 + self.min_reversal_pct):
                    confirmed_time = t; break
            if confirmed_time is not None: confirmed.append((cand_idx, cand_type, confirmed_time))

        # 过滤相邻同类型 pivot：保留更极端的
        filtered = []
        for p in confirmed:
            if not filtered: filtered.append(p); continue
            last = filtered[-1]
            if p[1] == last[1]:
                # 同类型合并：间隔>20天说明趋势结构已变，不再合并
                if abs(p[0] - last[0]) <= 20:
                    if p[1] == "PEAK" and close[p[0]] > close[last[0]]: filtered[-1] = p
                    elif p[1] == "TROUGH" and close[p[0]] < close[last[0]]: filtered[-1] = p
                else:
                    filtered.append(p)
            else: filtered.append(p)
        return filtered

    def _assign_phases(self, n, pivots, close):
        """
        根据已确认 pivot 分配 UP/DOWN 阶段。
        两个 pivot 之间 = LOCKED（永不变）
        最后确认 pivot 之后 = PENDING（可翻转）
        """
        phase_id = np.zeros(n, dtype=int); phase_name = np.array(["NEUTRAL"] * n, dtype='U8')
        is_pending = np.zeros(n, dtype=bool); pending_confidence = np.zeros(n, dtype=float)
        if len(pivots) == 0: return phase_id, phase_name, is_pending, pending_confidence

        sorted_by_idx = sorted(pivots, key=lambda x: x[0])
        for t in range(n):
            visible = [(p_idx, p_type) for p_idx, p_type, p_confirm in sorted_by_idx if p_confirm <= t]
            visible.sort(key=lambda x: x[0])
            if len(visible) == 0: continue

            if len(visible) == 1:
                p_idx, p_type = visible[0]
                if p_type == "TROUGH": phase_id[t] = 1; phase_name[t] = "UP"
                else: phase_id[t] = -1; phase_name[t] = "DOWN"
                if t < visible[0][0]: is_pending[t] = True
                continue

            assigned = False
            for seg_i in range(len(visible) - 1):
                s_idx = visible[seg_i][0]; e_idx = visible[seg_i + 1][0]
                if s_idx <= t <= e_idx:
                    s_type = visible[seg_i][1]; e_type = visible[seg_i + 1][1]
                    if s_type == "TROUGH" and e_type == "PEAK":
                        phase_id[t] = 1; phase_name[t] = "UP"
                    elif s_type == "PEAK" and e_type == "TROUGH":
                        phase_id[t] = -1; phase_name[t] = "DOWN"
                    else:
                        mid = (s_idx + e_idx) // 2
                        if close[e_idx] > close[s_idx]:
                            phase_id[t] = 1 if t <= mid else -1
                            phase_name[t] = "UP" if t <= mid else "DOWN"
                        else:
                            phase_id[t] = -1 if t <= mid else 1
                            phase_name[t] = "DOWN" if t <= mid else "UP"
                    assigned = True; break

            if not assigned:
                # 在最后一个可见 pivot 之后 → PENDING 区
                last_p = visible[-1]; p_idx, p_type = last_p
                last_pivot_price = close[p_idx]
                if p_type == "PEAK":
                    if close[t] > last_pivot_price: phase_id[t] = 1; phase_name[t] = "UP"
                    else: phase_id[t] = -1; phase_name[t] = "DOWN"
                else:
                    if close[t] < last_pivot_price: phase_id[t] = -1; phase_name[t] = "DOWN"
                    else: phase_id[t] = 1; phase_name[t] = "UP"
                is_pending[t] = True

                move = (close[t] - last_pivot_price) / last_pivot_price if phase_name[t] == "UP" \
                    else (last_pivot_price - close[t]) / last_pivot_price
                pending_confidence[t] = min(1.0, max(0.0, move / self.min_reversal_pct))

        return phase_id, phase_name, is_pending, pending_confidence

    def _annotate_volume(self, volume):
        """量能标注：VOL_EXPANDING / VOL_SHRINKING / NEUTRAL。不影响方向判断。"""
        n = len(volume); log_vol = np.log1p(volume.astype(float))
        alpha = 2.0 / (self.ema_span + 1)
        smooth = np.zeros(n); smooth[0] = log_vol[0]
        for i in range(1, n): smooth[i] = alpha * log_vol[i] + (1 - alpha) * smooth[i - 1]
        ground_thresh, _ = _compute_rolling_percentile(log_vol, self.ground_pct, self.sky_pct, self.rolling_window)
        annotation = np.array(["NEUTRAL"] * n, dtype='U14')
        for i in range(1, n):
            if smooth[i] > smooth[i - 1] and log_vol[i] > ground_thresh[i]: annotation[i] = "VOL_EXPANDING"
            elif smooth[i] < smooth[i - 1] or log_vol[i] <= ground_thresh[i]: annotation[i] = "VOL_SHRINKING"
        return annotation

    def _compute_touch_signal(self, close, high, low, opn, volume, n, confirmed_pivots):
        """
        触碰信号检测 — 因果性，无未来函数。
        
        阻力来源（按优先级）:
          1. 前一个 UP 区的最高价
          2. 前一个 DOWN 区的未填补向上缺口
          3. 前一个 DOWN 区的关键空头K线
          4. 更早的 UP 区最高价（上升阶梯）
        
        支撑来源（对称）:
          1. 前一个 DOWN 区的最低价
          2. 前一个 UP 区的未填补向下缺口
          3. 前一个 UP 区的关键多头K线
          4. 更早的 DOWN 区最低价（下降阶梯）
        
        信号: +2=精确触碰阻力, +1=逼近阻力, -1=逼近支撑, -2=精确触碰支撑
        """
        touch_threshold = 0.005; approach_threshold = 0.05; gap_min_age = 10
        touch_signal = np.zeros(n, dtype=int); touch_source = np.array([""] * n, dtype='U20')
        if len(confirmed_pivots) == 0: return touch_signal, touch_source

        sorted_pivots = sorted(confirmed_pivots, key=lambda x: x[0])
        all_gaps = []
        for k in range(1, n):
            if low[k] > high[k - 1]: all_gaps.append((k, low[k], high[k - 1], True))
            if high[k] < low[k - 1]: all_gaps.append((k, low[k - 1], high[k], False))

        gap_fills = {}
        for gap_idx, gap_top, gap_bottom, is_up in all_gaps:
            fill_bar = n
            for k in range(gap_idx + 1, n):
                if is_up and low[k] <= gap_bottom: fill_bar = k; break
                if not is_up and high[k] >= gap_top: fill_bar = k; break
            gap_fills[gap_idx] = fill_bar

        def _find_key_candle(s, e, is_bullish):
            """在区间内找关键K线。is_bullish=True 找最大阳线, False 找最大阴线。"""
            best_idx, best_score = None, -1
            for k in range(s, e + 1):
                body_pct = (close[k] - opn[k]) / opn[k] * 100 if opn[k] > 0 else 0
                if (is_bullish and body_pct <= 0) or (not is_bullish and body_pct >= 0): continue
                vol_w = volume[k] / max(volume[s:e+1].mean(), 1) if volume is not None else 1.0
                score = abs(body_pct) * vol_w
                limit_pct = 20.0 if opn[k] >= 50 else 10.0
                if (is_bullish and body_pct >= limit_pct * 0.9) or (not is_bullish and body_pct <= -limit_pct * 0.9):
                    score *= 10
                if score > best_score: best_score = score; best_idx = k
            return best_idx

        for t in range(n):
            visible = [(p_idx, p_type) for p_idx, p_type, p_confirm in sorted_pivots if p_confirm <= t]
            visible.sort(key=lambda x: x[0])
            if len(visible) < 2: continue

            zones = []
            for vi in range(len(visible) - 1):
                s_idx, s_type = visible[vi]; e_idx, e_type = visible[vi + 1]
                if s_type == "TROUGH" and e_type == "PEAK": zones.append((s_idx, e_idx, "UP"))
                elif s_type == "PEAK" and e_type == "TROUGH": zones.append((s_idx, e_idx, "DOWN"))
            if not zones: continue

            cur_zi = None
            for zi, (zs, ze, zp) in enumerate(zones):
                if zs <= t <= ze: cur_zi = zi; break
            if cur_zi is None:
                last_p_idx, last_p_type = visible[-1]
                if t >= last_p_idx:
                    last_zs, last_ze, last_zp = zones[-1]
                    if last_ze == last_p_idx:
                        cur_zi = len(zones)
                        if last_zp == "UP": zones.append((last_ze, t, "DOWN"))
                        else: zones.append((last_ze, t, "UP"))
            if cur_zi is None or cur_zi == 0: continue

            cur_zs, cur_ze, cur_zp = zones[cur_zi]; prev_zs, prev_ze, prev_zp = zones[cur_zi - 1]

            # ── 构建阻力列表 ──
            resistances = []
            if prev_zp == "UP":
                zone_high = high[prev_zs:prev_ze + 1].max()
                cur_max = high[cur_zs:min(t + 1, cur_ze + 1)].max()
                if zone_high > close[t] and cur_max < zone_high: resistances.append((zone_high, "UP_HIGH"))
            elif prev_zp == "DOWN":
                has_gap = False
                for gap_idx, gap_top, gap_bottom, is_up in all_gaps:
                    if gap_idx < prev_zs or gap_idx > prev_ze: continue
                    if not is_up or t - gap_idx < gap_min_age: continue
                    if gap_fills.get(gap_idx, n) <= t: continue
                    if gap_top > close[t]:
                        cur_max = high[cur_zs:min(t + 1, cur_ze + 1)].max()
                        if cur_max < gap_top: resistances.append((gap_top, "GAP")); has_gap = True; break
                if not has_gap:
                    key_idx = _find_key_candle(prev_zs, prev_ze, is_bullish=False)
                    if key_idx is not None:
                        key_high = high[key_idx]
                        if key_high > close[t]:
                            cur_max = high[cur_zs:min(t + 1, cur_ze + 1)].max()
                            if cur_max < key_high: resistances.append((key_high, "KEY"))

            # 上升阶梯
            cur_max = high[cur_zs:min(t + 1, cur_ze + 1)].max()
            existing_max = max((lv for lv, _ in resistances), default=cur_max)
            for zi in range(cur_zi - 2, -1, -1):
                zs, ze, zp = zones[zi]
                if zp != "UP": continue
                zh = high[zs:ze + 1].max()
                if zh <= existing_max or cur_max >= zh or zh <= close[t]: continue
                resistances.append((zh, f"UP_HIGH+{cur_zi - zi}")); existing_max = zh
            resistances.sort(key=lambda x: x[0])

            # ── 构建支撑列表 ──
            supports = []
            if prev_zp == "DOWN":
                zone_low = low[prev_zs:prev_ze + 1].min()
                cur_min = low[cur_zs:min(t + 1, cur_ze + 1)].min()
                if zone_low < close[t] and cur_min > zone_low: supports.append((zone_low, "DN_LOW"))
            elif prev_zp == "UP":
                has_gap = False
                for gap_idx, gap_top, gap_bottom, is_up in all_gaps:
                    if gap_idx < prev_zs or gap_idx > prev_ze: continue
                    if is_up or t - gap_idx < gap_min_age: continue
                    if gap_fills.get(gap_idx, n) <= t: continue
                    if gap_bottom < close[t]:
                        cur_min = low[cur_zs:min(t + 1, cur_ze + 1)].min()
                        if cur_min > gap_bottom: supports.append((gap_bottom, "GAP")); has_gap = True; break
                if not has_gap:
                    key_idx = _find_key_candle(prev_zs, prev_ze, is_bullish=True)
                    if key_idx is not None:
                        key_low = low[key_idx]
                        if key_low < close[t]:
                            cur_min = low[cur_zs:min(t + 1, cur_ze + 1)].min()
                            if cur_min > key_low: supports.append((key_low, "KEY"))

            # 下降阶梯
            cur_min = low[cur_zs:min(t + 1, cur_ze + 1)].min()
            existing_min = min((lv for lv, _ in supports), default=cur_min)
            for zi in range(cur_zi - 2, -1, -1):
                zs, ze, zp = zones[zi]
                if zp != "DOWN": continue
                zl = low[zs:ze + 1].min()
                if zl >= existing_min or cur_min <= zl or zl >= close[t]: continue
                supports.append((zl, f"DN_LOW+{cur_zi - zi}")); existing_min = zl
            supports.sort(key=lambda x: -x[0])

            # ── 检测触碰 ──
            for level, source in resistances:
                if high[t] >= level * (1 - touch_threshold): touch_signal[t] = 2; touch_source[t] = source; break
                elif close[t] >= level * (1 - approach_threshold): touch_signal[t] = 1; touch_source[t] = source; break
            if touch_signal[t] == 0:
                for level, source in supports:
                    if low[t] <= level * (1 + touch_threshold): touch_signal[t] = -2; touch_source[t] = source; break
                    elif close[t] <= level * (1 + approach_threshold): touch_signal[t] = -1; touch_source[t] = source; break

        return touch_signal, touch_source


# ============================================================
# 三、买卖点信号计算（V9 新增 — 完全因果性）
# ============================================================
def compute_buy_sell_signals(df_ohlc, result):
    """
    因果性买卖点检测 — 无未来函数，只用 bar t 之前的数据。

    信号体系:
        +2 强买 BrkHigh: 收盘价突破最近3个可见前高中最高的那个（真正的压力线）
        +1 买入 BrkRes:   收盘价突破之前触碰过的压力位
        +1 买入 PullSup:  UP 待定区中回踩支撑
         0 中性
        -1 卖出 BrkSup:   收盘价跌破之前触碰过的支撑位
        -1 卖出 BncRes:   DOWN 待定区中反弹到阻力
        -2 强卖 BrkLow:   收盘价跌破最近3个可见前低中最低的那个

    返回: (bs_signal, bs_reason)
    """
    n = len(df_ohlc); close = df_ohlc['close'].values
    high = df_ohlc['high'].values; low = df_ohlc['low'].values

    bs_signal = np.zeros(n, dtype=int); bs_reason = np.array([''] * n, dtype='U50')
    pivots = result.attrs.get('pivots', [])

    # 状态追踪
    last_up_high_broken = True; last_up_high_value = 0.0
    last_dn_low_broken = True; last_dn_low_value = 0.0
    pending_resistance_level = None; pending_support_level = None

    # 预提取 numpy 数组，避免 per-bar pandas dict lookup
    _phase = result['phase'].values; _is_pending = result['is_pending'].values
    _touch_signal = result['touch_signal'].values; _touch_source = result['touch_source'].values

    for t in range(n):
        # ── 构建 bar t 时刻可见的区间（因果性：只看已确认的 pivot）──
        sp = sorted(pivots, key=lambda x: x[2])
        visible = [(p_idx, p_type) for p_idx, p_type, p_confirm in sp if p_confirm <= t]
        visible.sort(key=lambda x: x[0])
        zones = []
        for vi in range(len(visible) - 1):
            s_idx, s_type = visible[vi]; e_idx, e_type = visible[vi + 1]
            if s_type == 'TROUGH' and e_type == 'PEAK': zones.append((s_idx, e_idx, 'UP'))
            elif s_type == 'PEAK' and e_type == 'TROUGH': zones.append((s_idx, e_idx, 'DOWN'))

        # ────────────────────────────────────────────────
        # +2 强买: BrkHigh — 突破最近3个前高中最高的
        # ────────────────────────────────────────────────
        if zones:
            up_highs = []
            for zs, ze, zp in zones:
                if zp == 'UP' and ze < t: up_highs.append(high[zs:ze + 1].max())
            if up_highs:
                target_high = max(up_highs[-3:])
                if target_high != last_up_high_value:
                    last_up_high_value = target_high; last_up_high_broken = (close[t] > target_high)
                if not last_up_high_broken and close[t] > target_high:
                    bs_signal[t] = 2; bs_reason[t] = f'BrkHigh({target_high:.2f})'
                    last_up_high_broken = True; continue

        # -2 强卖: BrkLow — 跌破最近3个前低中最低的
        if zones:
            dn_lows = []
            for zs, ze, zp in zones:
                if zp == 'DOWN' and ze < t: dn_lows.append(low[zs:ze + 1].min())
            if dn_lows:
                target_low = min(dn_lows[-3:])
                if target_low != last_dn_low_value:
                    last_dn_low_value = target_low; last_dn_low_broken = (close[t] < target_low)
                if not last_dn_low_broken and close[t] < target_low:
                    bs_signal[t] = -2; bs_reason[t] = f'BrkLow({target_low:.2f})'
                    last_dn_low_broken = True; continue

        if bs_signal[t] != 0: continue

        phase = _phase[t]; is_pend = _is_pending[t]
        touch_sig = _touch_signal[t]; touch_src = _touch_source[t]

        # 追踪触碰事件
        if touch_sig == 2: pending_resistance_level = high[t]
        elif touch_sig == -2: pending_support_level = low[t]

        # +1 买入: BrkRes — 突破之前触碰过的压力位
        if pending_resistance_level is not None:
            if close[t] > pending_resistance_level and phase in ('UP', 'NEUTRAL'):
                bs_signal[t] = 1; bs_reason[t] = f'BrkRes({pending_resistance_level:.2f})'
                pending_resistance_level = None; continue
            if phase == 'DOWN' and not is_pend: pending_resistance_level = None

        # +1 买入: PullSup — UP 待定区中回踩支撑
        if is_pend and phase == 'UP' and touch_sig <= -1:
            bs_signal[t] = 1; bs_reason[t] = f'PullSup({touch_src})'; continue

        # -1 卖出: BrkSup — 跌破支撑位
        if pending_support_level is not None:
            if close[t] < pending_support_level and phase in ('DOWN', 'NEUTRAL'):
                bs_signal[t] = -1; bs_reason[t] = f'BrkSup({pending_support_level:.2f})'
                pending_support_level = None; continue
            if phase == 'UP' and not is_pend: pending_support_level = None

        # -1 卖出: BncRes — DOWN 待定区中反弹到阻力
        if is_pend and phase == 'DOWN' and touch_sig >= 1:
            bs_signal[t] = -1; bs_reason[t] = f'BncRes({touch_src})'; continue

    return bs_signal, bs_reason


# ============================================================
# 四、图表绘制 — 4面板
# ============================================================
def plot_price_segmentation_v9(df_ohlc, result, bs_signal, bs_reason,
                               tail_days=200, name="", save_path=None):
    """4面板图表: K线 + 成交量 + 触碰信号 + 买卖信号。"""
    ohlc = df_ohlc.tail(tail_days).copy().reset_index(drop=True)
    n = len(ohlc); x = np.arange(n); offset = len(df_ohlc) - tail_days

    fig, axes = plt.subplots(4, 1, figsize=(22, 14), height_ratios=[4, 1, 0.6, 0.8],
                             sharex=True, gridspec_kw={'hspace': 0.08})
    fig.suptitle(f'{name}  Price Segmentation V9 (Buy/Sell Signals)', fontsize=14, fontweight='bold')

    ax0 = axes[0]; opens = ohlc['open'].values; highs = ohlc['high'].values
    lows = ohlc['low'].values; closes = ohlc['close'].values; vols = ohlc['volume'].values
    bar_w = 0.6

    ph = result['phase'].values[offset:offset + n]
    intervals = []; i = 0
    while i < n:
        j = i
        while j < n and ph[j] == ph[i]: j += 1
        intervals.append((i, j - 1, ph[i], False)); i = j
    if intervals: intervals[-1] = (intervals[-1][0], intervals[-1][1], intervals[-1][2], True)

    for s, e, p, pend in intervals:
        if pend:
            ax0.axvspan(s - 0.5, e + 0.5, alpha=0.04, color='orange' if p == "UP" else 'cyan', zorder=0)
            zl = lows[s:e + 1].min(); zh = highs[s:e + 1].max(); mg = (zh - zl) * 0.03
            ax0.add_patch(plt.Rectangle((s - 0.5, zl - mg), e - s + 1, (zh - zl) + 2 * mg,
                facecolor='none', edgecolor='#FF6F00' if p == "UP" else '#00695C',
                linewidth=1.5, linestyle='--', zorder=4))
            ax0.text((s + e) / 2, zh + mg, f"PENDING {p}", fontsize=7, fontweight='bold',
                     color='#FF6F00' if p == "UP" else '#00695C', ha='center', va='bottom', zorder=5)
        else:
            ax0.axvspan(s - 0.5, e + 0.5, alpha=0.10 if p == "UP" else 0.08,
                        color='red' if p == "UP" else 'green', zorder=0)
            if e > s:
                zl = lows[s:e + 1].min(); zh = highs[s:e + 1].max(); mg = (zh - zl) * 0.03
                ax0.text((s + e) / 2, zh + mg, p, fontsize=7, fontweight='bold',
                         color='#B71C1C' if p == "UP" else '#1B5E20', ha='center', va='bottom', zorder=5)

    for i in range(n):
        c = '#ef5350' if closes[i] >= opens[i] else '#26a69a'
        ax0.plot([x[i], x[i]], [lows[i], highs[i]], color=c, linewidth=0.5)
        bl = min(opens[i], closes[i]); bh = max(opens[i], closes[i])
        ax0.add_patch(plt.Rectangle((x[i] - bar_w / 2, bl), bar_w, bh - bl,
                                    facecolor=c, edgecolor=c, linewidth=0.4))

    fc = df_ohlc['close'].values
    ma120 = pd.Series(fc).rolling(120, min_periods=1).mean().values[-tail_days:]
    ax0.plot(x, ma120, color='#7B1FA2', linewidth=1.2, alpha=0.8, label='MA120')
    sm = result['smooth'].values[offset:offset + n]
    ax0.plot(x, sm, color='#1565C0', linewidth=1.0, alpha=0.6, label='EMA')

    for si, (s, e, p, _) in enumerate(intervals):
        if p == "UP" and e > s:
            ax0.hlines(highs[s:e + 1].max(), s - 0.5, e + 0.5, colors='#B71C1C',
                       linewidths=1.0, linestyles='--', alpha=0.7, zorder=3)
        if p == "DOWN" and e > s:
            ax0.hlines(lows[s:e + 1].min(), s - 0.5, e + 0.5, colors='#1B5E20',
                       linewidths=1.0, linestyles='--', alpha=0.7, zorder=3)

    # 缺口
    GAP_COLOR = '#1B5E20'; GAP_LINE_COLOR = '#2E7D32'; all_gaps = []
    for k in range(1, n):
        if lows[k] > highs[k - 1]: all_gaps.append((k, lows[k], highs[k - 1], True))
        if highs[k] < lows[k - 1]: all_gaps.append((k, lows[k - 1], highs[k], False))
    for gi, gt, gb, iug in all_gaps:
        fb = n
        for k in range(gi + 1, n):
            if iug and lows[k] <= gb: fb = k; break
            if not iug and highs[k] >= gt: fb = k; break
        ax0.add_patch(plt.Rectangle((x[gi - 1] + bar_w / 2, gb), x[gi] - x[gi - 1] - bar_w,
            gt - gb, facecolor=GAP_COLOR, alpha=0.30, edgecolor=GAP_COLOR,
            linewidth=1.0, linestyle='-', zorder=4))
        le = fb - 0.5 if fb < n else n - 0.5
        ax0.add_patch(plt.Rectangle((gi - 0.5, gb), le - gi + 1.0, gt - gb,
            facecolor=GAP_COLOR, alpha=0.10, edgecolor='none', zorder=2))
        ax0.hlines(gt, gi - 0.5, le, colors=GAP_LINE_COLOR, linewidths=0.8, linestyles=':', alpha=0.7, zorder=3)
        ax0.hlines(gb, gi - 0.5, le, colors=GAP_LINE_COLOR, linewidths=0.8, linestyles=':', alpha=0.7, zorder=3)
        lx = le + 0.3; gtype = '▲' if iug else '▼'
        ax0.text(lx, gt, f'{gtype} {gt:.2f}', fontsize=5.5, color=GAP_COLOR, va='center', ha='left', zorder=5,
                 bbox=dict(boxstyle='round,pad=0.08', facecolor='white', edgecolor=GAP_COLOR, alpha=0.75, linewidth=0.4))
        ax0.text(lx, gb, f'{gtype} {gb:.2f}', fontsize=5.5, color=GAP_COLOR, va='center', ha='left', zorder=5,
                 bbox=dict(boxstyle='round,pad=0.08', facecolor='white', edgecolor=GAP_COLOR, alpha=0.75, linewidth=0.4))

    # 关键K线
    def _fkc(s, e, ib):
        bi, bs = None, -1
        for k in range(s, e + 1):
            bp = (closes[k] - opens[k]) / opens[k] * 100 if opens[k] > 0 else 0
            if (ib and bp <= 0) or (not ib and bp >= 0): continue
            vw = vols[k] / max(vols[s:e+1].mean(), 1); sc = abs(bp) * vw
            lp = 20.0 if opens[k] >= 50 else 10.0
            if (ib and bp >= lp * 0.9) or (not ib and bp <= -lp * 0.9): sc *= 10
            if sc > bs: bs = sc; bi = k
        return bi
    def _zhug(s, e, iug, cs):
        for gg, ggt, ggb, ig in all_gaps:
            if gg < s or gg > e or ig != iug: continue
            fbg = n
            for k in range(gg + 1, n):
                if ig and lows[k] <= ggb: fbg = k; break
                if not ig and highs[k] >= ggt: fbg = k; break
            if fbg > cs: return True
        return False
    KCC = '#E65100'
    for si, (s, e, p, _) in enumerate(intervals):
        if si == 0: continue
        ps, pe, pp, _ = intervals[si - 1]
        if pe < ps: continue
        if p == "UP":
            if _zhug(ps, pe, False, s): continue
            ki = _fkc(ps, pe, False)
            if ki is not None:
                kh = highs[ki]; kl = lows[ki]
                ax0.hlines(kh, s - 0.5, e + 0.5, colors=KCC, linewidths=0.8, linestyles=':', alpha=0.6, zorder=3)
                ax0.hlines(kl, s - 0.5, e + 0.5, colors=KCC, linewidths=0.8, linestyles=':', alpha=0.6, zorder=3)
                ax0.text(e + 1.0, kh, f'Key H {kh:.2f}', fontsize=6, color=KCC, va='center', zorder=5)
                bl = min(opens[ki], closes[ki]); bh = max(opens[ki], closes[ki])
                ax0.add_patch(plt.Rectangle((x[ki] - bar_w / 2 - 0.15, bl - 0.1), bar_w + 0.3, bh - bl + 0.2,
                    facecolor='none', edgecolor=KCC, linewidth=2.0, linestyle='-', zorder=5))
        elif p == "DOWN":
            if _zhug(ps, pe, True, s): continue
            ki = _fkc(ps, pe, True)
            if ki is not None:
                kh = highs[ki]; kl = lows[ki]
                ax0.hlines(kh, s - 0.5, e + 0.5, colors=KCC, linewidths=0.8, linestyles=':', alpha=0.6, zorder=3)
                ax0.hlines(kl, s - 0.5, e + 0.5, colors=KCC, linewidths=0.8, linestyles=':', alpha=0.6, zorder=3)
                ax0.text(e + 1.0, kh, f'Key H {kh:.2f}', fontsize=6, color=KCC, va='center', zorder=5)
                bl = min(opens[ki], closes[ki]); bh = max(opens[ki], closes[ki])
                ax0.add_patch(plt.Rectangle((x[ki] - bar_w / 2 - 0.15, bl - 0.1), bar_w + 0.3, bh - bl + 0.2,
                    facecolor='none', edgecolor=KCC, linewidth=2.0, linestyle='-', zorder=5))

    # pivot标记
    for i in range(n):
        gi = offset + i
        if result['is_pivot'].values[gi]:
            pt = result['pivot_type'].values[gi]
            if pt == "PEAK": ax0.plot(x[i], highs[i] * 1.01, 'rv', markersize=6, alpha=0.7)
            elif pt == "TROUGH": ax0.plot(x[i], lows[i] * 0.99, 'g^', markersize=6, alpha=0.7)

    # 阻力线
    uzh = []
    for si2, (s2, e2, p2, pd2) in enumerate(intervals):
        if p2 == "UP" and e2 > s2: uzh.append((e2, highs[s2:e2 + 1].max(), s2))
    for si, (s, e, p, _pd) in enumerate(intervals):
        if p != "UP" or e <= s: continue
        phs = [(zh, s2, e2) for (e2, zh, s2) in uzh if e2 < s]
        if not phs: continue
        pss = sorted(phs, key=lambda x: x[2], reverse=True)[:6]; pss.sort(key=lambda x: x[0])
        for rk, (zh, s2, e2) in enumerate(pss):
            ax0.hlines(zh, e2 + 0.5, e + 0.5, colors='#FF1744', linewidths=0.8,
                       linestyles='-.', alpha=min(0.3 + 0.08 * rk, 0.85), zorder=3)
            ax0.text(e + 1.2, zh, f'R {zh:.2f}', fontsize=6, color='#FF1744', va='center', ha='left', zorder=5,
                     bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='#FF1744', alpha=0.8, linewidth=0.5))

    ax0.set_ylabel('Price', fontsize=10); ax0.grid(True, alpha=0.3); ax0.set_xlim(-1, n)
    ax0.legend(handles=[
        Patch(facecolor='red', alpha=0.15, label='UP (confirmed)'),
        Patch(facecolor='green', alpha=0.12, label='DOWN (confirmed)'),
        Patch(facecolor='orange', alpha=0.15, label='UP (pending)'),
        Patch(facecolor='cyan', alpha=0.15, label='DOWN (pending)'),
        Line2D([0], [0], color='#B71C1C', linewidth=1.0, linestyle='--', label='UP zone high'),
        Line2D([0], [0], color='#1B5E20', linewidth=1.0, linestyle='--', label='DOWN zone low'),
        Line2D([0], [0], color='#7B1FA2', linewidth=1.2, label='MA120'),
        Line2D([0], [0], color='#1565C0', linewidth=1.0, label='EMA'),
        Line2D([0], [0], marker='v', color='r', linestyle='None', markersize=6, label='PEAK'),
        Line2D([0], [0], marker='^', color='g', linestyle='None', markersize=6, label='TROUGH'),
        Patch(facecolor='#1B5E20', alpha=0.30, label='Gap'),
        Line2D([0], [0], color='#2E7D32', linewidth=0.8, linestyle=':', label='Gap line'),
        Line2D([0], [0], color='#E65100', linewidth=0.8, linestyle=':', label='Key candle'),
        Line2D([0], [0], color='#FF1744', linewidth=0.9, linestyle='-.', label='Resistance'),
    ], loc='upper left', fontsize=7, ncol=4)

    # Panel 1: 成交量
    ax1 = axes[1]; vol = ohlc['volume'].values; va = result['vol_annotation'].values[offset:offset + n]
    vc = {"VOL_EXPANDING": "#ef5350", "VOL_SHRINKING": "#26a69a", "NEUTRAL": "#9E9E9E"}
    ax1.bar(x, vol, width=bar_w, color=[vc.get(va[k], '#9E9E9E') for k in range(n)], alpha=0.8)
    ax1.set_ylabel('Volume', fontsize=9); ax1.grid(True, alpha=0.2)
    ax1.set_title('Volume (red=expanding, green=shrinking)', fontsize=9, loc='left', pad=2)

    # Panel 2: 触碰信号
    ax2 = axes[2]; ts = result['touch_signal'].values[offset:offset + n]
    tsrc = result['touch_source'].values[offset:offset + n]
    sc = {2: '#FF6D00', 1: '#FFD600', -1: '#00BCD4', -2: '#1565C0'}
    bv = [ts[k] if ts[k] != 0 else 0 for k in range(n)]
    ax2.bar(x, bv, width=bar_w * 2, color=[sc.get(ts[k], 'none') for k in range(n)], alpha=0.9)
    for i in range(n):
        s = ts[i]
        if s == 0: continue
        ly = s * 1.15 if abs(s) == 2 else s * 1.1
        ax2.text(i, ly, f"{'T:' if abs(s) == 2 else 'A:'}{tsrc[i]}", fontsize=4.5, color=sc.get(s, 'gray'),
                 ha='center', va='bottom' if s > 0 else 'top', rotation=90, zorder=5)
    ax2.set_ylim(-2.8, 2.8); ax2.set_yticks([-2, -1, 0, 1, 2])
    ax2.set_yticklabels(['T:Sup', 'A:Sup', '', 'A:Res', 'T:Res'], fontsize=7)
    ax2.set_ylabel('Touch', fontsize=9); ax2.axhline(0, color='gray', linewidth=0.5, alpha=0.5); ax2.grid(True, alpha=0.2)

    # Panel 3: 买卖信号
    ax3 = axes[3]; bsl = bs_signal[offset:offset + n]; brl = bs_reason[offset:offset + n]
    bc2 = {2: '#0D47A1', 1: '#42A5F5', 0: 'none', -1: '#FF7043', -2: '#B71C1C'}
    ax3.bar(x, [bsl[k] for k in range(n)], width=bar_w * 2,
            color=[bc2.get(bsl[k], 'none') for k in range(n)], alpha=0.9)
    for i in range(n):
        s = bsl[i]
        if s == 0: continue
        ax3.text(i, s * 1.2, brl[i], fontsize=5.5, color=bc2.get(s, 'gray'),
                 ha='center', va='bottom' if s > 0 else 'top', rotation=90, zorder=5)
    ax3.set_ylim(-2.8, 2.8); ax3.set_yticks([-2, -1, 0, 1, 2])
    ax3.set_yticklabels(['Str.Sell', 'Sell', '', 'Buy', 'Str.Buy'], fontsize=8)
    ax3.set_ylabel('Trade', fontsize=9); ax3.axhline(0, color='gray', linewidth=0.5, alpha=0.5); ax3.grid(True, alpha=0.2)
    ax3.set_title('Buy/Sell Signals', fontsize=8, loc='left', pad=2)

    ts2 = max(1, n // 12); dates = ohlc['date'].values
    tp = list(range(0, n, ts2)); tl = [str(dates[i])[:10] for i in tp]
    ax3.set_xticks(tp); ax3.set_xticklabels(tl, rotation=45, fontsize=7)

    plt.tight_layout(); plt.show(); plt.close()


# ============================================================
# 便捷入口
# ============================================================
def run_segmentation(df_ohlc, tail_days=200, name="",
                     lookback=15, min_reversal_pct=0.02, confirm_bars=3,
                     save_path=None, fast_mode=False):
    """
    一键运行：因果分段 → 买卖点检测 → 生成4面板图表。

    参数:
        fast_mode: True=跳过画图，只返回 bool（最后一天是否有买入信号）
    """
    close = df_ohlc['close'].values; volume = df_ohlc['volume'].values
    high = df_ohlc['high'].values; low = df_ohlc['low'].values; opn = df_ohlc['open'].values

    c_seg = CausalIncrementalPriceSegmenter(
        lookback=lookback, min_reversal_pct=min_reversal_pct, confirm_bars=confirm_bars)
    c_result = c_seg.segment(close, volume, high=high, low=low, opn=opn)
    bs_signal, bs_reason = compute_buy_sell_signals(df_ohlc, c_result)

    if fast_mode:
        return bs_signal[-1] > 0

    if save_path is None:
        save_path = f'E:\\\\chip_analyzer_ui\\\\new_algo\\\\result\\\\{name}_price_v9.png'
    plot_price_segmentation_v9(df_ohlc, c_result, bs_signal, bs_reason,
                               tail_days=tail_days, name=name, save_path=save_path)
    return c_result, bs_signal, bs_reason
