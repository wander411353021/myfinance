"""
Price Segmentation V8 - Price-Based Wave Segmentation
=====================================================

Based on V7, adds: Touch Signal indicator (resistance touch detection).

V8 change from V7:
  - New "touch signal": when price touches a resistance level that has
    suppressed it for a long time (unfilled gap, prior UP zone high, or
    key candle high/low), a signal is generated.
  - 3-panel chart: K-line (top) + Volume (mid) + Touch Signal (bottom)
  - Touch signal is computed causally (no future function): at each bar t,
    only resistance levels known at time t are considered.

Resistance sources (all causal, no future function):
  1. Unfilled gap top (upside gap) / gap bottom (downside gap)
  2. Prior UP zone high prices (confirmed zones only)
  3. Key candle high/low from previous zone (only if no unfilled gap)

Signal logic:
  - At bar t, collect all known resistance levels above current price
  - If high[t] >= resistance_level * (1 - touch_threshold), signal = 1
  - Signal shown in a separate panel below volume
  - Each signal annotated with the resistance source type

Future function audit:
  - Resistance levels: only from confirmed zones or unfilled gaps at time t
  - Gap fill detection: purely backward-looking
  - Zone boundaries: only confirmed pivots used (confirm_time <= t)
  - Touch detection: only checks resistances that existed before bar t
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
# Shared utilities
# ============================================================
def _compute_rolling_percentile(log_vol, ground_pct, sky_pct, rolling_window):
    """Rolling percentile thresholds (backward-looking, no future function)."""
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
    """Build standard output DataFrame for price segmentation."""
    n = len(close)
    is_pivot = np.zeros(n, dtype=bool)
    pivot_type = np.array([""] * n, dtype='U8')
    for p in pivots:
        idx = p[0]
        ptype = p[1]
        if idx < n:
            is_pivot[idx] = True
            pivot_type[idx] = ptype

    if is_pending is None:
        is_pending = np.zeros(n, dtype=bool)
    if pending_confidence is None:
        pending_confidence = np.zeros(n, dtype=float)
    if vol_annotation is None:
        vol_annotation = np.array(["NEUTRAL"] * n, dtype='U14')

    result = pd.DataFrame({
        "close": close,
        "smooth": smooth,
        "phase": phase_name,
        "phase_id": phase_id,
        "is_pivot": is_pivot,
        "pivot_type": pivot_type,
        "is_pending": is_pending,
        "pending_confidence": pending_confidence,
        "vol_annotation": vol_annotation,
        "touch_signal": np.zeros(n, dtype=int),
        "touch_source": np.array([""] * n, dtype='U20'),
    })
    result.attrs["pivots"] = list(pivots)
    return result


# ============================================================
# 1. Future-Looking Price Segmenter (God's Eye View)
# ============================================================
class FutureLookingPriceSegmenter:
    """
    Price segmentation using Savgol smoothing + find_peaks.
    Has future function (centered smoothing, global peak detection).
    Used as ground truth for evaluating causal versions.
    """

    def __init__(self, sg_window=11, sg_poly=3,
                 peak_distance=3, min_reversal_pct=0.02):
        self.sg_window = sg_window
        self.sg_poly = sg_poly
        self.peak_distance = peak_distance
        self.min_reversal_pct = min_reversal_pct

    def segment(self, close):
        """
        Segment price into UP/DOWN phases.

        Parameters
        ----------
        close : array-like
            Close prices.

        Returns
        -------
        pd.DataFrame with columns: close, smooth, phase, phase_id,
            is_pivot, pivot_type
        """
        close = np.asarray(close, dtype=float)
        n = len(close)

        # Savgol smoothing
        if n < self.sg_window:
            smooth = close.copy()
        else:
            smooth = savgol_filter(close, self.sg_window, self.sg_poly)

        # Find peaks and troughs
        peaks, peak_props = find_peaks(smooth, distance=self.peak_distance)
        troughs, trough_props = find_peaks(-smooth, distance=self.peak_distance)

        # Merge into sorted pivot list
        pivots = []
        for idx in peaks:
            pivots.append((idx, "PEAK"))
        for idx in troughs:
            pivots.append((idx, "TROUGH"))
        pivots.sort(key=lambda x: x[0])

        # Filter: minimum reversal between adjacent pivots
        filtered = []
        for p in pivots:
            if len(filtered) == 0:
                filtered.append(p)
                continue
            last_idx, last_type = filtered[-1]
            curr_idx, curr_type = p

            # Same type: keep the more extreme one
            if curr_type == last_type:
                if curr_type == "PEAK":
                    if smooth[curr_idx] > smooth[last_idx]:
                        filtered[-1] = p
                else:  # TROUGH
                    if smooth[curr_idx] < smooth[last_idx]:
                        filtered[-1] = p
                continue

            # Different type: check minimum reversal
            reversal = abs(smooth[curr_idx] - smooth[last_idx]) / smooth[last_idx]
            if reversal >= self.min_reversal_pct:
                filtered.append(p)
            else:
                # Too small, keep the more extreme one
                if curr_type == "PEAK" and smooth[curr_idx] > smooth[last_idx]:
                    filtered[-1] = p
                elif curr_type == "TROUGH" and smooth[curr_idx] < smooth[last_idx]:
                    filtered[-1] = p

        pivots = filtered

        # Phase assignment
        phase_id = np.zeros(n, dtype=int)
        phase_name = np.array(["NEUTRAL"] * n, dtype='U8')

        if len(pivots) == 0:
            return _build_price_result(close, smooth, phase_id, phase_name, pivots)

        # Before first pivot: infer from first pivot
        first_type = pivots[0][1]
        if first_type == "PEAK":
            phase_id[:pivots[0][0]] = 1
            phase_name[:pivots[0][0]] = "UP"
        else:
            phase_id[:pivots[0][0]] = -1
            phase_name[:pivots[0][0]] = "DOWN"

        # Between pivots
        for i in range(len(pivots) - 1):
            s_idx = pivots[i][0]
            e_idx = pivots[i + 1][0]
            if pivots[i][1] == "TROUGH" and pivots[i + 1][1] == "PEAK":
                phase_id[s_idx:e_idx + 1] = 1
                phase_name[s_idx:e_idx + 1] = "UP"
            elif pivots[i][1] == "PEAK" and pivots[i + 1][1] == "TROUGH":
                phase_id[s_idx:e_idx + 1] = -1
                phase_name[s_idx:e_idx + 1] = "DOWN"
            else:
                # Same type adjacent (shouldn't happen after filtering)
                # Use slope
                mid = (s_idx + e_idx) // 2
                if smooth[e_idx] > smooth[s_idx]:
                    phase_id[s_idx:mid + 1] = 1
                    phase_name[s_idx:mid + 1] = "UP"
                    phase_id[mid + 1:e_idx + 1] = -1
                    phase_name[mid + 1:e_idx + 1] = "DOWN"
                else:
                    phase_id[s_idx:mid + 1] = -1
                    phase_name[s_idx:mid + 1] = "DOWN"
                    phase_id[mid + 1:e_idx + 1] = 1
                    phase_name[mid + 1:e_idx + 1] = "UP"

        # After last pivot: infer direction
        last_type = pivots[-1][1]
        if last_type == "TROUGH":
            phase_id[pivots[-1][0]:] = 1
            phase_name[pivots[-1][0]:] = "UP"
        else:
            phase_id[pivots[-1][0]:] = -1
            phase_name[pivots[-1][0]:] = "DOWN"

        return _build_price_result(close, smooth, phase_id, phase_name, pivots)


# ============================================================
# 2. Causal Incremental Price Segmenter (No Future Function)
# ============================================================
class CausalIncrementalPriceSegmenter:
    """
    Price segmentation using local high/low detection + retracement confirmation.
    No future function: only looks left for candidates, right for confirmation.

    Algorithm:
      1. Candidate PEAK: current bar is the highest in the past N bars
      2. Candidate TROUGH: current bar is the lowest in the past N bars
      3. Confirmation: PEAK confirmed when price drops X% from candidate;
         TROUGH confirmed when price rises X% from candidate
      4. Phase assignment: between confirmed pivots, direction is locked;
         after last confirmed pivot, pending zone with confidence
      5. Volume annotation: each bar tagged as VOL_EXPANDING or VOL_SHRINKING
         based on EMA of log(volume), but this does NOT affect direction
    """

    def __init__(self, lookback=15, min_reversal_pct=0.02,
                 confirm_bars=3, ema_span=15,
                 ground_pct=20, sky_pct=85, rolling_window=120):
        """
        Parameters
        ----------
        lookback : int
            Number of past bars to check for local high/low.
        min_reversal_pct : float
            Minimum price reversal (as fraction) to confirm a pivot.
            E.g. 0.03 = 3% drop from candidate PEAK to confirm.
        confirm_bars : int
            Minimum number of bars after candidate before confirmation
            is allowed (prevents premature confirmation on noise).
        ema_span : int
            EMA span for volume smoothing (annotation only).
        ground_pct : float
            Percentile for ground-level volume threshold.
        sky_pct : float
            Percentile for sky-level volume threshold.
        rolling_window : int
            Rolling window for volume percentile computation.
        """
        self.lookback = lookback
        self.min_reversal_pct = min_reversal_pct
        self.confirm_bars = confirm_bars
        self.ema_span = ema_span
        self.ground_pct = ground_pct
        self.sky_pct = sky_pct
        self.rolling_window = rolling_window

    def segment(self, close, volume=None, high=None, low=None, opn=None):
        """
        Segment price into UP/DOWN phases.

        Parameters
        ----------
        close : array-like
            Close prices.
        volume : array-like, optional
            Volume data for annotation. If None, no volume annotation.
        high : array-like, optional
            High prices (needed for touch signal). If None, uses close.
        low : array-like, optional
            Low prices (needed for touch signal). If None, uses close.
        opn : array-like, optional
            Open prices (needed for key candle in touch signal). If None, uses close.

        Returns
        -------
        pd.DataFrame with columns: close, smooth, phase, phase_id,
            is_pivot, pivot_type, is_pending, pending_confidence,
            vol_annotation, touch_signal, touch_source
        """
        close = np.asarray(close, dtype=float)
        n = len(close)
        volume = np.asarray(volume, dtype=float) if volume is not None else None
        high = np.asarray(high, dtype=float) if high is not None else close
        low = np.asarray(low, dtype=float) if low is not None else close
        opn = np.asarray(opn, dtype=float) if opn is not None else close

        # ── Step 1: Detect candidate pivots (causal, left-looking only) ──
        candidates = self._detect_candidates(close)

        # ── Step 2: Confirm pivots with retracement ──
        confirmed_pivots = self._confirm_pivots(close, candidates)

        # ── Step 3: Assign phases incrementally ──
        phase_id, phase_name, is_pending, pending_confidence = \
            self._assign_phases(n, confirmed_pivots, close)

        # ── Step 4: Volume annotation (doesn't affect direction) ──
        vol_annotation = np.array(["NEUTRAL"] * n, dtype='U14')
        if volume is not None:
            vol_annotation = self._annotate_volume(volume)

        # Smooth line for chart (simple EMA of close for visual reference)
        smooth = self._ema_close(close)

        # ── Step 5: Touch signal (causal, no future function) ──
        touch_signal, touch_source = self._compute_touch_signal(
            close, high, low, opn, volume, n, confirmed_pivots)

        # Build pivot arrays
        is_pivot = np.zeros(n, dtype=bool)
        pivot_type = np.array([""] * n, dtype='U8')
        for p in confirmed_pivots:
            idx = p[0]
            ptype = p[1]
            if idx < n:
                is_pivot[idx] = True
                pivot_type[idx] = ptype

        result = pd.DataFrame({
            "close": close,
            "smooth": smooth,
            "phase": phase_name,
            "phase_id": phase_id,
            "is_pivot": is_pivot,
            "pivot_type": pivot_type,
            "is_pending": is_pending,
            "pending_confidence": pending_confidence,
            "vol_annotation": vol_annotation,
            "touch_signal": touch_signal,
            "touch_source": touch_source,
        })
        result.attrs["pivots"] = list(confirmed_pivots)
        return result

    def _ema_close(self, close):
        """EMA of close for visual reference (not used for detection)."""
        n = len(close)
        smooth = np.zeros(n)
        alpha = 2.0 / (self.ema_span + 1)
        smooth[0] = close[0]
        for i in range(1, n):
            smooth[i] = alpha * close[i] + (1 - alpha) * smooth[i - 1]
        return smooth

    def _detect_candidates(self, close):
        """
        Detect candidate PEAKs and TROUGHs.
        Causal: only looks at past `lookback` bars.

        Returns list of (idx, type) tuples.
        """
        n = len(close)
        candidates = []

        for t in range(self.lookback, n):
            window = close[t - self.lookback:t + 1]  # t is the rightmost

            # Candidate PEAK: current bar is highest in window
            if close[t] == window.max() and close[t] > close[t - 1]:
                # Avoid duplicate: check if previous bar was also a PEAK candidate
                if len(candidates) == 0 or candidates[-1][1] != "PEAK" or candidates[-1][0] < t - 1:
                    candidates.append((t, "PEAK"))
                else:
                    # Update to the higher one
                    if close[t] >= close[candidates[-1][0]]:
                        candidates[-1] = (t, "PEAK")

            # Candidate TROUGH: current bar is lowest in window
            if close[t] == window.min() and close[t] < close[t - 1]:
                if len(candidates) == 0 or candidates[-1][1] != "TROUGH" or candidates[-1][0] < t - 1:
                    candidates.append((t, "TROUGH"))
                else:
                    if close[t] <= close[candidates[-1][0]]:
                        candidates[-1] = (t, "TROUGH")

        return candidates

    def _confirm_pivots(self, close, candidates):
        """
        Confirm pivots using retracement.

        A candidate PEAK at idx=p is confirmed at time t when:
          - t >= p + confirm_bars
          - close[t] <= close[p] * (1 - min_reversal_pct)

        A candidate TROUGH at idx=p is confirmed at time t when:
          - t >= p + confirm_bars
          - close[t] >= close[p] * (1 + min_reversal_pct)

        Returns list of (idx, type, confirm_time) tuples.
        Only confirmed pivots are returned.
        """
        n = len(close)
        confirmed = []

        for cand_idx, cand_type in candidates:
            confirmed_time = None

            for t in range(cand_idx + self.confirm_bars, n):
                if cand_type == "PEAK":
                    if close[t] <= close[cand_idx] * (1 - self.min_reversal_pct):
                        confirmed_time = t
                        break
                else:  # TROUGH
                    if close[t] >= close[cand_idx] * (1 + self.min_reversal_pct):
                        confirmed_time = t
                        break

            if confirmed_time is not None:
                confirmed.append((cand_idx, cand_type, confirmed_time))

        # Filter: same-type adjacent → keep the more extreme one
        filtered = []
        for p in confirmed:
            if len(filtered) == 0:
                filtered.append(p)
                continue
            last = filtered[-1]
            if p[1] == last[1]:
                # Same type: keep the more extreme
                if p[1] == "PEAK":
                    if close[p[0]] > close[last[0]]:
                        filtered[-1] = p
                else:
                    if close[p[0]] < close[last[0]]:
                        filtered[-1] = p
            else:
                filtered.append(p)

        return filtered

    def _assign_phases(self, n, pivots, close):
        """
        Incremental phase assignment using confirm_time.
        Only uses pivots with confirm_time <= t.

        Between two confirmed pivots: direction is LOCKED.
        After last confirmed pivot: PENDING zone.
        """
        phase_id = np.zeros(n, dtype=int)
        phase_name = np.array(["NEUTRAL"] * n, dtype='U8')
        is_pending = np.zeros(n, dtype=bool)
        pending_confidence = np.zeros(n, dtype=float)

        if len(pivots) == 0:
            return phase_id, phase_name, is_pending, pending_confidence

        # Sort pivots by index for phase assignment
        sorted_by_idx = sorted(pivots, key=lambda x: x[0])

        # Build a timeline: at each bar t, which pivots are "visible"
        # (i.e. confirm_time <= t)
        # A pivot at idx=p with confirm_time=c becomes visible at bar c
        # and affects phase assignment from bar p onwards (retroactively
        # for the range p..c-1 which was previously pending)

        # Simpler approach: process bar by bar
        # At each bar t, collect all pivots with confirm_time <= t
        # These are the "confirmed" pivots known at time t
        # Between confirmed pivots: LOCKED phase
        # After last confirmed pivot: PENDING

        for t in range(n):
            # Collect pivots visible at time t
            visible = [(p_idx, p_type) for p_idx, p_type, p_confirm in sorted_by_idx
                       if p_confirm <= t]
            visible.sort(key=lambda x: x[0])

            if len(visible) == 0:
                continue

            if len(visible) == 1:
                p_idx, p_type = visible[0]
                if p_type == "TROUGH":
                    phase_id[t] = 1
                    phase_name[t] = "UP"
                else:
                    phase_id[t] = -1
                    phase_name[t] = "DOWN"
                # Before first confirmed pivot is also pending
                if t < visible[0][0]:
                    is_pending[t] = True
                continue

            # Find which segment t belongs to
            assigned = False
            for seg_i in range(len(visible) - 1):
                s_idx = visible[seg_i][0]
                e_idx = visible[seg_i + 1][0]
                if s_idx <= t <= e_idx:
                    s_type = visible[seg_i][1]
                    e_type = visible[seg_i + 1][1]
                    if s_type == "TROUGH" and e_type == "PEAK":
                        phase_id[t] = 1
                        phase_name[t] = "UP"
                    elif s_type == "PEAK" and e_type == "TROUGH":
                        phase_id[t] = -1
                        phase_name[t] = "DOWN"
                    else:
                        # Same type adjacent: split at midpoint
                        mid = (s_idx + e_idx) // 2
                        if t <= mid:
                            if close[e_idx] > close[s_idx]:
                                phase_id[t] = 1
                                phase_name[t] = "UP"
                            else:
                                phase_id[t] = -1
                                phase_name[t] = "DOWN"
                        else:
                            if close[e_idx] > close[s_idx]:
                                phase_id[t] = -1
                                phase_name[t] = "DOWN"
                            else:
                                phase_id[t] = 1
                                phase_name[t] = "UP"
                    assigned = True
                    break

            if not assigned:
                # After last visible pivot: PENDING zone
                # V6: direction follows actual price, not just alternating
                last_p = visible[-1]
                p_idx, p_type = last_p
                last_pivot_price = close[p_idx]

                if p_type == "PEAK":
                    # After a PEAK: default DOWN, but flip to UP if price
                    # rises above the PEAK (PEAK invalidated)
                    if close[t] > last_pivot_price:
                        phase_id[t] = 1
                        phase_name[t] = "UP"
                    else:
                        phase_id[t] = -1
                        phase_name[t] = "DOWN"
                else:  # TROUGH
                    # After a TROUGH: default UP, but flip to DOWN if price
                    # falls below the TROUGH (TROUGH invalidated)
                    if close[t] < last_pivot_price:
                        phase_id[t] = -1
                        phase_name[t] = "DOWN"
                    else:
                        phase_id[t] = 1
                        phase_name[t] = "UP"
                is_pending[t] = True

                # Confidence: based on how far price has moved from last pivot
                if phase_name[t] == "UP":
                    move = (close[t] - last_pivot_price) / last_pivot_price
                else:
                    move = (last_pivot_price - close[t]) / last_pivot_price
                pending_confidence[t] = min(1.0, max(0.0, move / self.min_reversal_pct))

        return phase_id, phase_name, is_pending, pending_confidence

    def _annotate_volume(self, volume):
        """
        Annotate each bar with volume status (VOL_EXPANDING / VOL_SHRINKING / NEUTRAL).
        Uses EMA of log(volume) - same as V4 but for annotation only.
        """
        n = len(volume)
        log_vol = np.log1p(volume.astype(float))
        alpha = 2.0 / (self.ema_span + 1)

        smooth = np.zeros(n)
        smooth[0] = log_vol[0]
        for i in range(1, n):
            smooth[i] = alpha * log_vol[i] + (1 - alpha) * smooth[i - 1]

        ground_thresh, sky_thresh = _compute_rolling_percentile(
            log_vol, self.ground_pct, self.sky_pct, self.rolling_window)

        annotation = np.array(["NEUTRAL"] * n, dtype='U14')
        for i in range(1, n):
            if smooth[i] > smooth[i - 1] and log_vol[i] > ground_thresh[i]:
                annotation[i] = "VOL_EXPANDING"
            elif smooth[i] < smooth[i - 1] or log_vol[i] <= ground_thresh[i]:
                annotation[i] = "VOL_SHRINKING"

        return annotation

    def _compute_touch_signal(self, close, high, low, opn, volume, n, confirmed_pivots):
        """
        Compute touch signal: detect when price touches resistance/support.

        Resistance sources (only from immediately preceding zone + ascending
        staircase of earlier UP highs):
          - Previous UP zone: zone high price
          - Previous DOWN zone: unfilled upside gap top (within 10 bars), or key candle high
          - After crossing previous UP high, trace back to earlier UP highs
            that are STRICTLY HIGHER (ascending staircase only)

        Support sources (symmetric):
          - Previous DOWN zone: zone low price
          - Previous UP zone: unfilled downside gap bottom (within 10 bars), or key candle low
          - After crossing previous DOWN low, trace back to earlier DOWN lows
            that are STRICTLY LOWER (descending staircase only)

        Signal levels:
          1 = approach (close within 5% of level)
          2 = touch (high/low within 0.5% of level)

        No future function: only uses data from bars 0..t.
        """
        touch_threshold = 0.005
        approach_threshold = 0.05
        gap_min_age = 10  # gap must exist for >= 10 bars before signal counts

        touch_signal = np.zeros(n, dtype=int)
        touch_source = np.array([""] * n, dtype='U20')

        if len(confirmed_pivots) == 0:
            return touch_signal, touch_source

        sorted_pivots = sorted(confirmed_pivots, key=lambda x: x[0])

        # Pre-compute gaps
        all_gaps = []
        for k in range(1, n):
            if low[k] > high[k - 1]:
                all_gaps.append((k, low[k], high[k - 1], True))
            if high[k] < low[k - 1]:
                all_gaps.append((k, low[k - 1], high[k], False))

        # Pre-compute gap fill bars
        gap_fills = {}
        for gap_idx, gap_top, gap_bottom, is_up in all_gaps:
            fill_bar = n
            for k in range(gap_idx + 1, n):
                if is_up and low[k] <= gap_bottom:
                    fill_bar = k; break
                if not is_up and high[k] >= gap_top:
                    fill_bar = k; break
            gap_fills[gap_idx] = fill_bar

        # Key candle helper
        def _find_key_candle(s, e, is_bullish):
            best_idx, best_score = None, -1
            for k in range(s, e + 1):
                body_pct = (close[k] - opn[k]) / opn[k] * 100 if opn[k] > 0 else 0
                if (is_bullish and body_pct <= 0) or (not is_bullish and body_pct >= 0):
                    continue
                vol_w = volume[k] / max(volume[s:e+1].mean(), 1) if volume is not None else 1.0
                score = abs(body_pct) * vol_w
                limit_pct = 20.0 if opn[k] >= 50 else 10.0
                if (is_bullish and body_pct >= limit_pct * 0.9) or \
                   (not is_bullish and body_pct <= -limit_pct * 0.9):
                    score *= 10
                if score > best_score:
                    best_score = score
                    best_idx = k
            return best_idx

        for t in range(n):
            # Visible pivots at time t
            visible = [(p_idx, p_type) for p_idx, p_type, p_confirm in sorted_pivots
                       if p_confirm <= t]
            visible.sort(key=lambda x: x[0])

            if len(visible) < 2:
                continue

            # Build zones
            zones = []
            for vi in range(len(visible) - 1):
                s_idx, s_type = visible[vi]
                e_idx, e_type = visible[vi + 1]
                if s_type == "TROUGH" and e_type == "PEAK":
                    zones.append((s_idx, e_idx, "UP"))
                elif s_type == "PEAK" and e_type == "TROUGH":
                    zones.append((s_idx, e_idx, "DOWN"))

            if not zones:
                continue

            # Find current zone index
            cur_zi = None
            for zi, (zs, ze, zp) in enumerate(zones):
                if zs <= t <= ze:
                    cur_zi = zi
                    break
            if cur_zi is None:
                last_p_idx, last_p_type = visible[-1]
                if t >= last_p_idx:
                    last_zs, last_ze, last_zp = zones[-1]
                    if last_ze == last_p_idx:
                        cur_zi = len(zones)
                        if last_zp == "UP":
                            zones.append((last_ze, t, "DOWN"))
                        else:
                            zones.append((last_ze, t, "UP"))

            if cur_zi is None or cur_zi == 0:
                continue

            cur_zs, cur_ze, cur_zp = zones[cur_zi]

            # ── Build resistance levels ──
            resistances = []

            # --- Source 1: Previous zone (immediate) ---
            prev_zs, prev_ze, prev_zp = zones[cur_zi - 1]

            if prev_zp == "UP":
                zone_high = high[prev_zs:prev_ze + 1].max()
                cur_max = high[cur_zs:min(t + 1, cur_ze + 1)].max()
                if zone_high > close[t] and cur_max < zone_high:
                    resistances.append((zone_high, "UP_HIGH"))

            elif prev_zp == "DOWN":
                # Gap: must have existed for >= gap_min_age bars
                has_gap = False
                for gap_idx, gap_top, gap_bottom, is_up in all_gaps:
                    if gap_idx < prev_zs or gap_idx > prev_ze:
                        continue
                    if not is_up:
                        continue
                    if t - gap_idx < gap_min_age:  # too young, skip
                        continue
                    fill_bar = gap_fills.get(gap_idx, n)
                    if fill_bar <= t:
                        continue
                    if gap_top > close[t]:
                        cur_max = high[cur_zs:min(t + 1, cur_ze + 1)].max()
                        if cur_max < gap_top:
                            resistances.append((gap_top, "GAP"))
                            has_gap = True
                            break  # one gap is enough

                if not has_gap:
                    key_idx = _find_key_candle(prev_zs, prev_ze, is_bullish=False)
                    if key_idx is not None:
                        key_high = high[key_idx]
                        if key_high > close[t]:
                            cur_max = high[cur_zs:min(t + 1, cur_ze + 1)].max()
                            if cur_max < key_high:
                                resistances.append((key_high, "KEY"))

            # --- Source 2: Ascending staircase of earlier UP highs ---
            # Trace back through earlier zones, finding UP zone highs that are
            # STRICTLY HIGHER than any already collected resistance.
            # Only add if current zone hasn't already exceeded them.
            cur_max = high[cur_zs:min(t + 1, cur_ze + 1)].max()
            existing_max = max((lv for lv, _ in resistances), default=cur_max)

            for zi in range(cur_zi - 2, -1, -1):
                zs, ze, zp = zones[zi]
                if zp != "UP":
                    continue
                zh = high[zs:ze + 1].max()
                # Must be higher than current highest resistance
                if zh <= existing_max:
                    continue
                # Current zone must not have exceeded it yet
                if cur_max >= zh:
                    continue
                # Price must be below it
                if zh <= close[t]:
                    continue
                resistances.append((zh, f"UP_HIGH+{cur_zi - zi}"))
                existing_max = zh  # raise the bar for next iteration

            # Sort resistances: nearest first (smallest level)
            resistances.sort(key=lambda x: x[0])

            # ── Build support levels (symmetric) ──
            supports = []

            if prev_zp == "DOWN":
                zone_low = low[prev_zs:prev_ze + 1].min()
                cur_min = low[cur_zs:min(t + 1, cur_ze + 1)].min()
                if zone_low < close[t] and cur_min > zone_low:
                    supports.append((zone_low, "DN_LOW"))

            elif prev_zp == "UP":
                # Gap: must have existed for >= gap_min_age bars
                has_gap = False
                for gap_idx, gap_top, gap_bottom, is_up in all_gaps:
                    if gap_idx < prev_zs or gap_idx > prev_ze:
                        continue
                    if is_up:
                        continue
                    if t - gap_idx < gap_min_age:
                        continue
                    fill_bar = gap_fills.get(gap_idx, n)
                    if fill_bar <= t:
                        continue
                    if gap_bottom < close[t]:
                        cur_min = low[cur_zs:min(t + 1, cur_ze + 1)].min()
                        if cur_min > gap_bottom:
                            supports.append((gap_bottom, "GAP"))
                            has_gap = True
                            break

                if not has_gap:
                    key_idx = _find_key_candle(prev_zs, prev_ze, is_bullish=True)
                    if key_idx is not None:
                        key_low = low[key_idx]
                        if key_low < close[t]:
                            cur_min = low[cur_zs:min(t + 1, cur_ze + 1)].min()
                            if cur_min > key_low:
                                supports.append((key_low, "KEY"))

            # Descending staircase of earlier DOWN lows
            cur_min = low[cur_zs:min(t + 1, cur_ze + 1)].min()
            existing_min = min((lv for lv, _ in supports), default=cur_min)

            for zi in range(cur_zi - 2, -1, -1):
                zs, ze, zp = zones[zi]
                if zp != "DOWN":
                    continue
                zl = low[zs:ze + 1].min()
                if zl >= existing_min:
                    continue
                if cur_min <= zl:
                    continue
                if zl >= close[t]:
                    continue
                supports.append((zl, f"DN_LOW+{cur_zi - zi}"))
                existing_min = zl

            # Sort supports: nearest first (largest level)
            supports.sort(key=lambda x: -x[0])

            # ── Check for touches ──
            for level, source in resistances:
                if high[t] >= level * (1 - touch_threshold):
                    touch_signal[t] = 2
                    touch_source[t] = source
                    break
                elif close[t] >= level * (1 - approach_threshold):
                    touch_signal[t] = 1
                    touch_source[t] = source
                    break

            if touch_signal[t] == 0:
                for level, source in supports:
                    if low[t] <= level * (1 + touch_threshold):
                        touch_signal[t] = -2
                        touch_source[t] = source
                        break
                    elif close[t] <= level * (1 + approach_threshold):
                        touch_signal[t] = -1
                        touch_source[t] = source
                        break

        return touch_signal, touch_source


# ============================================================
# Chart
# ============================================================
def plot_price_segmentation(df_ohlc, result,
                            tail_days=200, name="", save_path=None):
    """
    3-panel chart: K-line (top) + Volume (mid) + Touch Signal (bottom).

    Features:
      - Phase backgrounds: UP (red) / DOWN (green), confirmed vs pending
      - UP zone: highest price horizontal line
      - Key candle from previous zone displayed in current zone:
          UP zone shows key bearish candle from previous DOWN zone
          DOWN zone shows key bullish candle from previous UP zone
      - Persistent gaps: dark green, persist until filled
      - Resistance levels: prior UP zone highs
      - Touch signal panel: shows when price touches resistance/support

    Parameters
    ----------
    df_ohlc : pd.DataFrame
        OHLC data with columns: date, open, high, low, close, volume
    result : pd.DataFrame
        Result from CausalIncrementalPriceSegmenter
    tail_days : int
        Number of recent bars to show
    name : str
        Stock code for title
    save_path : str
        Path to save chart
    """
    ohlc = df_ohlc.tail(tail_days).copy().reset_index(drop=True)
    n = len(ohlc)
    x = np.arange(n)
    offset = len(df_ohlc) - tail_days

    fig, axes = plt.subplots(3, 1, figsize=(22, 12),
                             height_ratios=[4, 1, 0.6],
                             sharex=True, gridspec_kw={'hspace': 0.08})
    fig.suptitle(f'{name}  Price Segmentation V8 (Touch Signal)', fontsize=14, fontweight='bold')

    # ── Panel 0: K-line with phase backgrounds ──
    ax0 = axes[0]
    opens = ohlc['open'].values
    highs = ohlc['high'].values
    lows = ohlc['low'].values
    closes = ohlc['close'].values
    vols = ohlc['volume'].values
    bar_w = 0.6

    # Build phase intervals (merge same-direction segments regardless of pending)
    ph = result['phase'].values[offset:offset + n]

    intervals = []
    i = 0
    while i < n:
        j = i
        while j < n and ph[j] == ph[i]:
            j += 1
        intervals.append((i, j - 1, ph[i], False))
        i = j

    # Only the last interval can be pending
    if intervals:
        intervals[-1] = (intervals[-1][0], intervals[-1][1], intervals[-1][2], True)

    # Draw phase backgrounds + pending dashed boxes
    for s, e, p, pend in intervals:
        if pend:
            if p == "UP":
                ax0.axvspan(s - 0.5, e + 0.5, alpha=0.04, color='orange', zorder=0)
            elif p == "DOWN":
                ax0.axvspan(s - 0.5, e + 0.5, alpha=0.04, color='cyan', zorder=0)
            # Dashed border box
            zone_lo = lows[s:e + 1].min()
            zone_hi = highs[s:e + 1].max()
            margin = (zone_hi - zone_lo) * 0.03
            ax0.add_patch(plt.Rectangle(
                (s - 0.5, zone_lo - margin), e - s + 1, (zone_hi - zone_lo) + 2 * margin,
                facecolor='none', edgecolor='#FF6F00' if p == "UP" else '#00695C',
                linewidth=1.5, linestyle='--', zorder=4))
            # Text label
            label = f"PENDING {'UP' if p == 'UP' else 'DOWN'}"
            ax0.text((s + e) / 2, zone_hi + margin, label,
                     fontsize=7, fontweight='bold',
                     color='#FF6F00' if p == "UP" else '#00695C',
                     ha='center', va='bottom', zorder=5)
        else:
            if p == "UP":
                ax0.axvspan(s - 0.5, e + 0.5, alpha=0.10, color='red', zorder=0)
            elif p == "DOWN":
                ax0.axvspan(s - 0.5, e + 0.5, alpha=0.08, color='green', zorder=0)
            # Text label for confirmed zones
            if e > s and p in ("UP", "DOWN"):
                zone_lo = lows[s:e + 1].min()
                zone_hi = highs[s:e + 1].max()
                margin = (zone_hi - zone_lo) * 0.03
                label = p
                ax0.text((s + e) / 2, zone_hi + margin, label,
                         fontsize=7, fontweight='bold',
                         color='#B71C1C' if p == "UP" else '#1B5E20',
                         ha='center', va='bottom', zorder=5)

    # K-line bars
    for i in range(n):
        color = '#ef5350' if closes[i] >= opens[i] else '#26a69a'
        ax0.plot([x[i], x[i]], [lows[i], highs[i]], color=color, linewidth=0.5)
        body_lo = min(opens[i], closes[i])
        body_hi = max(opens[i], closes[i])
        ax0.add_patch(plt.Rectangle((x[i] - bar_w / 2, body_lo), bar_w,
                                     body_hi - body_lo,
                                     facecolor=color, edgecolor=color, linewidth=0.4))

    # MA120
    full_close = df_ohlc['close'].values
    ma120 = pd.Series(full_close).rolling(120, min_periods=1).mean().values[-tail_days:]
    ax0.plot(x, ma120, color='#7B1FA2', linewidth=1.2, alpha=0.8, label='MA120')

    # EMA smooth line (from result)
    smooth = result['smooth'].values[offset:offset + n]
    ax0.plot(x, smooth, color='#1565C0', linewidth=1.0, alpha=0.6, label='EMA')

    # ── UP zone: highest price line ──
    # ── DOWN zone: lowest price line ──
    for seg_i, (s, e, p, pend) in enumerate(intervals):
        if p == "UP" and e > s:
            zone_high = highs[s:e + 1].max()
            ax0.hlines(zone_high, s - 0.5, e + 0.5,
                       colors='#B71C1C', linewidths=1.0, linestyles='--', alpha=0.7, zorder=3)
        if p == "DOWN" and e > s:
            zone_low = lows[s:e + 1].min()
            ax0.hlines(zone_low, s - 0.5, e + 0.5,
                       colors='#1B5E20', linewidths=1.0, linestyles='--', alpha=0.7, zorder=3)

    # ── Persistent gaps: gaps persist until filled, dark green color ──
    # Scan ALL bars for gaps (both upside and downside).
    # A gap persists from its creation bar until the first bar where price
    # fully crosses back through it (no future function: fill detection is
    # purely backward-looking).
    # Upside gap: filled when low <= gap_bottom (price drops back through)
    # Downside gap: filled when high >= gap_top (price rises back through)

    GAP_COLOR = '#1B5E20'  # dark green
    GAP_LINE_COLOR = '#2E7D32'  # slightly lighter dark green for lines

    # Collect all gaps
    all_gaps = []  # list of (gap_idx, gap_top, gap_bottom, is_up_gap)
    for k in range(1, n):
        # Upside gap: today's low > yesterday's high
        if lows[k] > highs[k - 1]:
            gap_bottom = highs[k - 1]
            gap_top = lows[k]
            all_gaps.append((k, gap_top, gap_bottom, True))
        # Downside gap: today's high < yesterday's low
        if highs[k] < lows[k - 1]:
            gap_top = lows[k - 1]
            gap_bottom = highs[k]
            all_gaps.append((k, gap_top, gap_bottom, False))

    # Build a set of bar indices that have a gap, for key-candle fallback check
    gap_bar_set = set()
    for gap_idx, gap_top, gap_bottom, is_up_gap in all_gaps:
        gap_bar_set.add(gap_idx)

    # For each gap, find fill bar (first bar after gap where price crosses back)
    for gap_idx, gap_top, gap_bottom, is_up_gap in all_gaps:
        fill_bar = n  # default: never filled, persists to end
        for k in range(gap_idx + 1, n):
            if is_up_gap:
                # Upside gap filled when low drops back to or below gap_bottom
                if lows[k] <= gap_bottom:
                    fill_bar = k
                    break
            else:
                # Downside gap filled when high rises back to or above gap_top
                if highs[k] >= gap_top:
                    fill_bar = k
                    break

        # Draw gap rectangle at creation point
        ax0.add_patch(plt.Rectangle(
            (x[gap_idx - 1] + bar_w / 2, gap_bottom),
            x[gap_idx] - x[gap_idx - 1] - bar_w,
            gap_top - gap_bottom,
            facecolor=GAP_COLOR, alpha=0.30, edgecolor=GAP_COLOR,
            linewidth=1.0, linestyle='-', zorder=4))

        # Draw persistent filled band from gap creation to fill bar (or end)
        line_end = fill_bar - 0.5 if fill_bar < n else n - 0.5
        ax0.add_patch(plt.Rectangle(
            (gap_idx - 0.5, gap_bottom),
            line_end - gap_idx + 1.0,
            gap_top - gap_bottom,
            facecolor=GAP_COLOR, alpha=0.10, edgecolor='none',
            zorder=2))

        # Draw persistent horizontal lines from gap creation to fill bar (or end)
        ax0.hlines(gap_top, gap_idx - 0.5, line_end,
                   colors=GAP_LINE_COLOR, linewidths=0.8, linestyles=':', alpha=0.7, zorder=3)
        ax0.hlines(gap_bottom, gap_idx - 0.5, line_end,
                   colors=GAP_LINE_COLOR, linewidths=0.8, linestyles=':', alpha=0.7, zorder=3)

        # Label at the end of the persistent line
        label_x = line_end + 0.3
        gap_type = '▲' if is_up_gap else '▼'
        ax0.text(label_x, gap_top, f'{gap_type} {gap_top:.2f}',
                 fontsize=5.5, color=GAP_COLOR, va='center', ha='left', zorder=5,
                 bbox=dict(boxstyle='round,pad=0.08', facecolor='white',
                           edgecolor=GAP_COLOR, alpha=0.75, linewidth=0.4))
        ax0.text(label_x, gap_bottom, f'{gap_type} {gap_bottom:.2f}',
                 fontsize=5.5, color=GAP_COLOR, va='center', ha='left', zorder=5,
                 bbox=dict(boxstyle='round,pad=0.08', facecolor='white',
                           edgecolor=GAP_COLOR, alpha=0.75, linewidth=0.4))

    # ── Key candle fallback: when a zone has no gap in the previous zone ──
    def _find_key_candle(s, e, is_bullish):
        """Find key candle in range [s, e].
        is_bullish=True: find key bullish candle (biggest positive body)
        is_bullish=False: find key bearish candle (biggest negative body)
        """
        best_idx = None
        best_score = -1

        for k in range(s, e + 1):
            body_pct = (closes[k] - opens[k]) / opens[k] * 100 if opens[k] > 0 else 0

            if is_bullish and body_pct <= 0:
                continue
            if not is_bullish and body_pct >= 0:
                continue

            # Score: body_pct * volume weight
            vol_w = vols[k] / max(vols[s:e+1].mean(), 1)
            score = abs(body_pct) * vol_w

            # Limit-up bonus
            limit_pct = 20.0 if (opens[k] >= 50) else 10.0  # rough estimate
            if is_bullish and body_pct >= limit_pct * 0.9:
                score *= 10  # strong bonus for limit-up
            if not is_bullish and body_pct <= -limit_pct * 0.9:
                score *= 10

            if score > best_score:
                best_score = score
                best_idx = k

        return best_idx

    def _zone_has_unfilled_gap(s, e, is_up_gap, current_start):
        """Check if zone [s, e] contains any gap of the specified type
        that is STILL UNFILLED at current_start (i.e. gap extends into current zone).
        If the gap was filled before current_start, it doesn't replace key candle."""
        for gap_idx_g, gap_top_g, gap_bottom_g, is_up_g in all_gaps:
            if gap_idx_g < s or gap_idx_g > e:
                continue
            if is_up_g != is_up_gap:
                continue
            # Find fill bar for this gap
            fill_bar_g = n
            for k in range(gap_idx_g + 1, n):
                if is_up_g:
                    if lows[k] <= gap_bottom_g:
                        fill_bar_g = k
                        break
                else:
                    if highs[k] >= gap_top_g:
                        fill_bar_g = k
                        break
            # Gap is still unfilled at current zone start → it replaces key candle
            if fill_bar_g > current_start:
                return True
        return False

    KEY_CANDLE_COLOR = '#E65100'  # orange for key candle lines
    for seg_i, (s, e, p, pend) in enumerate(intervals):
        if seg_i == 0:
            continue
        prev_s, prev_e, prev_p, prev_pend = intervals[seg_i - 1]
        if prev_e < prev_s:
            continue

        if p == "UP":
            # Current is UP → check if previous DOWN zone has downside gaps
            # that are still unfilled at current zone start
            if _zone_has_unfilled_gap(prev_s, prev_e, is_up_gap=False, current_start=s):
                continue  # gap already drawn globally, skip key candle
            # No gap → fall back to key bearish candle
            key_idx = _find_key_candle(prev_s, prev_e, is_bullish=False)
            if key_idx is not None:
                key_high = highs[key_idx]
                key_low = lows[key_idx]
                ax0.hlines(key_high, s - 0.5, e + 0.5,
                           colors=KEY_CANDLE_COLOR, linewidths=0.8, linestyles=':', alpha=0.6, zorder=3)
                ax0.hlines(key_low, s - 0.5, e + 0.5,
                           colors=KEY_CANDLE_COLOR, linewidths=0.8, linestyles=':', alpha=0.6, zorder=3)
                ax0.text(e + 1.0, key_high, f'Key H {key_high:.2f}',
                         fontsize=6, color=KEY_CANDLE_COLOR, va='center', zorder=5)
                ax0.text(e + 1.0, key_low, f'Key L {key_low:.2f}',
                         fontsize=6, color=KEY_CANDLE_COLOR, va='center', zorder=5)
                body_lo = min(opens[key_idx], closes[key_idx])
                body_hi = max(opens[key_idx], closes[key_idx])
                ax0.add_patch(plt.Rectangle(
                    (x[key_idx] - bar_w / 2 - 0.15, body_lo - 0.1),
                    bar_w + 0.3, body_hi - body_lo + 0.2,
                    facecolor='none', edgecolor=KEY_CANDLE_COLOR,
                    linewidth=2.0, linestyle='-', zorder=5))

        elif p == "DOWN":
            # Current is DOWN → check if previous UP zone has upside gaps
            # that are still unfilled at current zone start
            if _zone_has_unfilled_gap(prev_s, prev_e, is_up_gap=True, current_start=s):
                continue  # gap already drawn globally, skip key candle
            # No gap → fall back to key bullish candle
            key_idx = _find_key_candle(prev_s, prev_e, is_bullish=True)
            if key_idx is not None:
                key_high = highs[key_idx]
                key_low = lows[key_idx]
                ax0.hlines(key_high, s - 0.5, e + 0.5,
                           colors=KEY_CANDLE_COLOR, linewidths=0.8, linestyles=':', alpha=0.6, zorder=3)
                ax0.hlines(key_low, s - 0.5, e + 0.5,
                           colors=KEY_CANDLE_COLOR, linewidths=0.8, linestyles=':', alpha=0.6, zorder=3)
                ax0.text(e + 1.0, key_high, f'Key H {key_high:.2f}',
                         fontsize=6, color=KEY_CANDLE_COLOR, va='center', zorder=5)
                ax0.text(e + 1.0, key_low, f'Key L {key_low:.2f}',
                         fontsize=6, color=KEY_CANDLE_COLOR, va='center', zorder=5)
                body_lo = min(opens[key_idx], closes[key_idx])
                body_hi = max(opens[key_idx], closes[key_idx])
                ax0.add_patch(plt.Rectangle(
                    (x[key_idx] - bar_w / 2 - 0.15, body_lo - 0.1),
                    bar_w + 0.3, body_hi - body_lo + 0.2,
                    facecolor='none', edgecolor=KEY_CANDLE_COLOR,
                    linewidth=2.0, linestyle='-', zorder=5))

    # Mark pivots
    for i in range(n):
        global_i = offset + i
        if result['is_pivot'].values[global_i]:
            pt = result['pivot_type'].values[global_i]
            if pt == "PEAK":
                ax0.plot(x[i], highs[i] * 1.01, 'rv', markersize=6, alpha=0.7)
            elif pt == "TROUGH":
                ax0.plot(x[i], lows[i] * 0.99, 'g^', markersize=6, alpha=0.7)

    # ── Resistance levels: show prior UP zone highs on every UP/pending zone ──
    # Pre-compute all UP zone highs (no future function: only use zones before current)
    up_zone_highs = []  # list of (zone_end_idx, high_price)
    for seg_i2, (s2, e2, p2, pend2) in enumerate(intervals):
        if p2 == "UP" and e2 > s2:
            zh = highs[s2:e2 + 1].max()
            up_zone_highs.append((e2, zh, s2, seg_i2, pend2))

    for seg_i, (s, e, p, pend) in enumerate(intervals):
        if p != "UP" or e <= s:
            continue
        # Collect UP highs from zones BEFORE this one (no future function)
        prior_highs = [(zh, s2, e2, pend2) for (e2, zh, s2, _, pend2) in up_zone_highs if e2 < s]
        if not prior_highs:
            continue
        # Take up to 6 most recent prior UP highs
        prior_highs_sorted = sorted(prior_highs, key=lambda x: x[2], reverse=True)[:6]
        # Sort by price ascending for label stacking
        prior_highs_sorted.sort(key=lambda x: x[0])
        for rank, (zh, s2, e2, pend2) in enumerate(prior_highs_sorted):
            # Line from prior zone end into current zone
            line_start = e2 + 0.5
            line_end = e + 0.5
            ax0.hlines(zh, line_start, line_end,
                       colors='#FF1744', linewidths=0.8, linestyles='-.',
                       alpha=min(0.3 + 0.08 * rank, 0.85), zorder=3)
            # Label at right edge of current zone
            ax0.text(e + 1.2, zh, f'R {zh:.2f}',
                     fontsize=6, color='#FF1744',
                     va='center', ha='left', zorder=5,
                     bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                               edgecolor='#FF1744', alpha=0.8, linewidth=0.5))

    ax0.set_ylabel('Price', fontsize=10)
    ax0.grid(True, alpha=0.3)
    ax0.set_xlim(-1, n)

    # Legend
    legend_elements = [
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
        Patch(facecolor='#1B5E20', alpha=0.30, label='Gap (persistent)'),
        Line2D([0], [0], color='#2E7D32', linewidth=0.8, linestyle=':', label='Gap line (until filled)'),
        Patch(facecolor='#1B5E20', alpha=0.10, label='Gap band'),
        Line2D([0], [0], color='#E65100', linewidth=0.8, linestyle=':', label='Key candle line'),
        Line2D([0], [0], color='#FF1744', linewidth=0.9, linestyle='-.', label='Resistance (UP high)'),
    ]
    ax0.legend(handles=legend_elements, loc='upper left', fontsize=7, ncol=5)

    # ── Panel 1: Volume with annotation ──
    ax1 = axes[1]
    vol = ohlc['volume'].values
    vol_ann = result['vol_annotation'].values[offset:offset + n]

    vol_colors = {
        "VOL_EXPANDING": "#ef5350",
        "VOL_SHRINKING": "#26a69a",
        "NEUTRAL": "#9E9E9E",
    }
    vcolors = [vol_colors.get(vol_ann[k], '#9E9E9E') for k in range(n)]
    ax1.bar(x, vol, width=bar_w, color=vcolors, alpha=0.8)
    ax1.set_ylabel('Volume', fontsize=9)
    ax1.set_title('Volume (red=expanding, green=shrinking)', fontsize=9, loc='left', pad=2)
    ax1.grid(True, alpha=0.2)

    # ── Panel 2: Touch Signal ──
    ax2 = axes[2]
    touch_sig = result['touch_signal'].values[offset:offset + n]
    touch_src = result['touch_source'].values[offset:offset + n]

    # Signal bars: 2=touch resist(orange), 1=approach resist(yellow),
    #             -1=approach support(cyan), -2=touch support(blue)
    sig_colors = {
        2: '#FF6D00',   # touch resistance: deep orange
        1: '#FFD600',   # approach resistance: yellow
        -1: '#00BCD4',  # approach support: cyan
        -2: '#1565C0',  # touch support: deep blue
    }
    bar_colors = [sig_colors.get(touch_sig[k], 'none') for k in range(n)]
    bar_vals = [touch_sig[k] if touch_sig[k] != 0 else 0 for k in range(n)]

    ax2.bar(x, bar_vals, width=bar_w * 2, color=bar_colors, alpha=0.9)

    # Add source labels for signals
    for i in range(n):
        s = touch_sig[i]
        if s == 0:
            continue
        label_y = s * 1.15 if abs(s) == 2 else s * 1.1
        color = sig_colors.get(s, 'gray')
        label_prefix = "T:" if abs(s) == 2 else "A:"
        ax2.text(i, label_y, f"{label_prefix}{touch_src[i]}", fontsize=4.5, color=color,
                 ha='center', va='bottom' if s > 0 else 'top',
                 rotation=90, zorder=5)

    ax2.set_ylim(-2.8, 2.8)
    ax2.set_yticks([-2, -1, 0, 1, 2])
    ax2.set_yticklabels(['T:Sup', 'A:Sup', '', 'A:Res', 'T:Res'], fontsize=7)
    ax2.set_ylabel('Touch', fontsize=9)
    ax2.set_title('Touch Signal (T=touch, A=approach | orange/yellow=resist, blue/cyan=support)',
                  fontsize=8, loc='left', pad=2)
    ax2.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax2.grid(True, alpha=0.2)

    # X-axis
    tick_step = max(1, n // 12)
    dates = ohlc['date'].values
    tick_pos = list(range(0, n, tick_step))
    tick_lbl = [str(dates[i])[:10] for i in tick_pos]
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_lbl, rotation=45, fontsize=7)

    plt.tight_layout()
    # if save_path:
    #     fig.savefig(save_path, dpi=150, bbox_inches='tight')
    #     print(f"Chart saved: {save_path}")
    plt.show()
    plt.close()

# ============================================================
# Convenience: run causal segmenter and generate chart
# ============================================================
def run_segmentation(df_ohlc, tail_days=200, name="",
                     lookback=15, min_reversal_pct=0.02, confirm_bars=3,
                     save_path=None):
    """
    Run Causal Incremental segmenter and generate chart.
    """
    close = df_ohlc['close'].values
    volume = df_ohlc['volume'].values
    high = df_ohlc['high'].values
    low = df_ohlc['low'].values
    opn = df_ohlc['open'].values

    c_seg = CausalIncrementalPriceSegmenter(
        lookback=lookback, min_reversal_pct=min_reversal_pct,
        confirm_bars=confirm_bars)
    c_result = c_seg.segment(close, volume, high=high, low=low, opn=opn)

    # Chart
    if save_path is None:
        save_path = f'E:\\chip_analyzer_ui\\new_algo\\result\\{name}_price_v8.png'
    plot_price_segmentation(df_ohlc, c_result,
                            tail_days=tail_days, name=name,
                            save_path=save_path)

    return c_result


# # ============================================================
# # Main
# # ============================================================
# def main():
#     import finshare

#     stocks = [
#         ('688017', '2025-12-23', 250),
#         ('600550', '2020-02-27', 200),
#         ('300437', '2020-08-25', 200),
#         ('600199', '2025-06-11', 200),
#         ('300204', '2025-06-11', 200),
#     ]

#     for code, end_date, tail in stocks:
#         print(f"\n{'='*60}")
#         print(f"  {code}  end={end_date}  tail={tail}")
#         print(f"{'='*60}")

#         df = finshare.get_historical_data(code, start='2015-01-01', end=end_date, adjust='qfq')
#         df = df.rename(columns={'trade_date': 'date', 'open_price': 'open',
#                                 'high_price': 'high', 'low_price': 'low',
#                                 'close_price': 'close', 'volume': 'volume'})
#         df = df[df['close'] > 0].reset_index(drop=True)
#         df['date'] = pd.to_datetime(df['date'])

#         run_segmentation(df, tail_days=tail, name=code)


# if __name__ == "__main__":
#     main()
