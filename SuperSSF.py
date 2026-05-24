import numpy as np
import pandas as pd
from numba import njit


# ============================================================
# SSF (Super Smoother Filter) 指标
# ============================================================

@njit
def _ssf_core(close, length, poles):
    m = close.shape[0]
    ssf = close.copy()
    if poles == 3:
        x = np.pi / length
        a0 = np.exp(-x)
        b0 = 2 * a0 * np.cos(np.sqrt(3) * x)
        c0 = a0 * a0
        c4 = c0 * c0
        c3 = -c0 * (1 + b0)
        c2 = c0 + b0
        c1 = 1 - c2 - c3 - c4
        for i in range(3, m):
            ssf[i] = c1 * close[i] + c2 * ssf[i-1] + c3 * ssf[i-2] + c4 * ssf[i-3]
    else:
        x = np.pi * np.sqrt(2) / length
        a0 = np.exp(-x)
        a1 = -a0 * a0
        b1 = 2 * a0 * np.cos(x)
        c1 = 1 - a1 - b1
        for i in range(2, m):
            ssf[i] = c1 * close[i] + b1 * ssf[i-1] + a1 * ssf[i-2]
    return ssf


def ssf(close, length=None, poles=None, offset=None, **kwargs):
    length = int(length) if length and length > 0 else 10
    poles = int(poles) if poles in [2, 3] else 2
    offset = int(offset) if offset is not None else 0
    close_arr = close.values
    ssf_arr = _ssf_core(close_arr, length, poles)
    ssf_res = pd.Series(ssf_arr, index=close.index)
    if offset != 0:
        ssf_res = ssf_res.shift(offset)
    if "fillna" in kwargs:
        ssf_res.fillna(kwargs["fillna"], inplace=True)
    return ssf_res


# ============================================================
# 格栅线耦合算法 v16 - 优先级: ssf_l>ssf_m>ssf_s, 长窗口优先, 高于ssf_m时上浮5%
# ============================================================

@njit
def _grid_coupling_core(close_arr, ma1, ma2, ma3, volume_arr, n_above, n_below, step_pct,
                         min_window, max_window, tight_pct, min_ratio, track_pct, track_ratio):
    """
    格栅线耦合核心算法(numba加速)
    优先级: ma_idx越小越高(0=ssf_l,1=ssf_m,2=ssf_s) → 窗口越长越高 → 格栅值越高越好
    后处理: 如果价格在ssf_m下方运行，耦合力值封顶为ssf_m
           如果价格不在ssf_m下方，耦合力值上浮5%
    
    新增: 
    1. 高位区域过滤 - 当价格 > ssf_m*1.2 时，直接忽略信号
    2. 成交量分析的走平过滤 - 只有成交量放大的走平才被视为风险信号
    """
    n = close_arr.shape[0]
    result = np.full(n, np.nan)

    for t in range(min_window - 1, n):
        best_grid_val = np.nan
        found = False
        best_ma_idx = 99       # 越小优先级越高
        best_w = 0             # 越大优先级越高
        best_sub_val = -1e18   # 格栅值越高越好(同优先级时)

        # --- 高位区域检测 ---
        ma_m_t = ma2[t]
        is_high_zone = (ma_m_t == ma_m_t and ma_m_t > 0 and close_arr[t] > ma_m_t * 1.2)
        
        # 高位区域直接忽略信号
        if is_high_zone:
            result[t] = np.nan
            continue
        
        # --- 上涨后走平检测 + 成交量分析 ---
        # 检查是否出现"上涨后走平"的弱势信号（需要成交量配合）
        is_flat_after_rise = False
        check_days = min(10, t)  # 检查最近10天
        if t >= check_days:
            recent_prices = close_arr[t-check_days+1:t+1]
            recent_changes = np.diff(recent_prices)
            
            # 判断走平：最近3天的变化都很小
            flat_window = min(3, len(recent_changes))
            is_flat = np.all(np.abs(recent_changes[-flat_window:]) / close_arr[t-1] < 0.01)  # 1%内波动视为走平
            
            # 判断前期上涨：前半段有明显上涨
            if len(recent_changes) >= 5:
                early_changes = recent_changes[:len(recent_changes)//2]
                has_rise = np.sum(early_changes > 0) / len(early_changes) > 0.6  # 60%的日子在上涨
                
                if is_flat and has_rise:
                    # 进一步分析成交量
                    # 走平期间的成交量（最近flat_window天）
                    flat_volumes = volume_arr[t-flat_window+1:t+1]
                    avg_flat_volume = np.mean(flat_volumes)
                    
                    # 上涨期间的成交量（前半段）
                    rise_volumes = volume_arr[t-len(recent_changes)+1:t-flat_window+1]
                    if len(rise_volumes) > 0:
                        avg_rise_volume = np.mean(rise_volumes)
                        
                        # 成交量放大判断：走平期间成交量 > 上涨期间成交量 * 1.3
                        # 说明有出货嫌疑，这是风险信号
                        is_volume_expanding = (avg_flat_volume > avg_rise_volume * 1.3)
                        
                        if is_volume_expanding:
                            is_flat_after_rise = True
                        # 如果成交量衰减，说明是惜售/蓄势，不忽略信号
        
        # 上涨后走平且成交量放大的情况才忽略信号
        if is_flat_after_rise:
            result[t] = np.nan
            continue
        
        # 正常区域使用标准参数
        high_zone_min_ratio = min_ratio
        high_zone_tight_pct = tight_pct

        for ma_idx in range(3):
            # 如果当前ma优先级已经低于已有最优，跳过
            if found and ma_idx > best_ma_idx:
                continue

            if ma_idx == 0:
                ma_val_t = ma1[t]
            elif ma_idx == 1:
                ma_val_t = ma2[t]
            else:
                ma_val_t = ma3[t]

            if ma_val_t != ma_val_t or ma_val_t <= 0:
                continue

            # --- 模式1: 格栅线耦合 (gi = -n_below ... n_above) ---
            for gi in range(-n_below, n_above + 1):
                G_ratio = 1.0 + step_pct * gi
                if G_ratio <= 0:
                    continue

                G_t = ma_val_t * G_ratio
                if abs(close_arr[t] - G_t) / G_t >= high_zone_tight_pct:
                    continue

                for w in range(min(max_window, t + 1), min_window - 1, -1):
                    # 如果同ma但窗口不够长，不必继续缩短
                    if found and ma_idx == best_ma_idx and w <= best_w:
                        break

                    tight_count = 0
                    for j in range(t - w + 1, t + 1):
                        if ma_idx == 0:
                            ma_j = ma1[j]
                        elif ma_idx == 1:
                            ma_j = ma2[j]
                        else:
                            ma_j = ma3[j]
                        if ma_j != ma_j or ma_j <= 0:
                            continue
                        G_j = ma_j * G_ratio
                        if G_j > 0 and abs(close_arr[j] - G_j) / G_j < high_zone_tight_pct:
                            tight_count += 1

                    ratio = tight_count / w
                    # 高位区域需要更高的紧密比例
                    if ratio >= high_zone_min_ratio:
                        # 优先级比较: ma_idx小优先 → w大优先 → G_t大优先
                        if (not found) or \
                           (ma_idx < best_ma_idx) or \
                           (ma_idx == best_ma_idx and w > best_w) or \
                           (ma_idx == best_ma_idx and w == best_w and G_t > best_sub_val):
                            best_grid_val = G_t
                            best_ma_idx = ma_idx
                            best_w = w
                            best_sub_val = G_t
                        found = True
                        break  # 最长窗口已满足

            # --- 模式2: MA跟踪 ---
            if abs(close_arr[t] - ma_val_t) / ma_val_t >= track_pct:
                continue

            for w in range(min(max_window, t + 1), min_window - 1, -1):
                if found and ma_idx == best_ma_idx and w <= best_w:
                    break

                track_count = 0
                for j in range(t - w + 1, t + 1):
                    if ma_idx == 0:
                        ma_j = ma1[j]
                    elif ma_idx == 1:
                        ma_j = ma2[j]
                    else:
                        ma_j = ma3[j]
                    if ma_j != ma_j or ma_j <= 0:
                        continue
                    if abs(close_arr[j] - ma_j) / ma_j < track_pct:
                        track_count += 1

                ratio = track_count / w
                # 高位区域需要更高的跟踪比例
                if ratio >= high_zone_min_ratio:
                    # MA跟踪输出MA值本身，格栅值=ma_val_t
                    if (not found) or \
                       (ma_idx < best_ma_idx) or \
                       (ma_idx == best_ma_idx and w > best_w) or \
                       (ma_idx == best_ma_idx and w == best_w and ma_val_t > best_sub_val):
                        best_grid_val = ma_val_t
                        best_ma_idx = ma_idx
                        best_w = w
                        best_sub_val = ma_val_t
                    found = True
                    break

        if found:
            # --- 后处理: 价格高于ssf_m时上浮5% ---
            # 取消低于ssf_m时强制改成ssf_m的限制
            
            # 计算上浮5%后的值
            adjusted_val = best_grid_val * 1.05
            
            # 与ssf_m比较：只有上浮后的值高于ssf_m时才使用上浮值
            if ma_m_t == ma_m_t and ma_m_t > 0 and adjusted_val > ma_m_t:
                # 上浮后的值高于ssf_m，使用上浮值
                result[t] = adjusted_val
            else:
                # 否则直接使用原始耦合值
                result[t] = best_grid_val
        # 如果 found=False，result[t] 保持为 NaN

    return result


@njit
def _apply_stickiness(close_arr, raw_result, ssf_m_arr, stick_pct, min_hold):
    """
    粘性后处理: 确保耦合段至少持续min_hold天
    强制延续机制:
    - 一旦检测到非NaN的耦合值，接下来min_hold天强制保持该值不变
    - 即使在ssf_m下方也保持（不应用封顶规则）
    - min_hold天后才允许切换到新值或变为NaN
    """
    n = close_arr.shape[0]
    result = raw_result.copy()
    hold_days = np.zeros(n, dtype=np.int64)  # 当前值已持续天数
    forced_value = np.nan  # 当前强制保持的值
    forced_start_day = -1  # 强制保持开始的日期

    for t in range(n):
        if t == 0:
            # 第一天
            if raw_result[t] == raw_result[t]:  # 非NaN
                forced_value = raw_result[t]
                forced_start_day = t
                hold_days[t] = 1
                result[t] = forced_value
            else:
                hold_days[t] = 0
            continue
        
        # 检查是否在强制保持期内
        in_forced_period = (forced_value == forced_value and 
                           forced_start_day >= 0 and 
                           (t - forced_start_day) < min_hold)
        
        if in_forced_period:
            # 强制保持期内：无论什么情况都保持原值
            result[t] = forced_value
            hold_days[t] = hold_days[t - 1] + 1
            continue
        
        # 不在强制保持期内
        if raw_result[t] != raw_result[t]:  # 当前无耦合
            # 重置
            forced_value = np.nan
            forced_start_day = -1
            hold_days[t] = 0
            continue
        
        # 当前有耦合值
        if result[t - 1] != result[t - 1]:  # 前一天无值，新耦合段开始
            forced_value = raw_result[t]
            forced_start_day = t
            hold_days[t] = 1
            result[t] = forced_value
            continue
        
        # 前一天有值，判断是否需要切换
        prev = result[t - 1]
        prev_hold = hold_days[t - 1]
        
        # 两个条件满足任意一个 → 保持前值
        price_close = (prev > 0 and abs(close_arr[t] - prev) / prev < stick_pct)
        not_enough_hold = (prev_hold < min_hold)
        
        if price_close or not_enough_hold:
            result[t] = prev
            hold_days[t] = prev_hold + 1
        else:
            # 价格已明显偏离且已保持足够久 → 切换到新值
            forced_value = raw_result[t]
            forced_start_day = t
            result[t] = raw_result[t]
            hold_days[t] = 1

    return result


def compute_grid_coupling(df, ma_cols=None,
                          n_above=30, n_below=10, step_pct=0.02,
                          min_window=5, max_window=40,
                          tight_pct=0.008, min_ratio=0.8,
                          track_pct=0.03, track_ratio=0.8,
                          stick_pct=0.06, min_hold=20,
                          extend_days=5):
    """
    格栅线耦合算法 v16
    优先级: ssf_l > ssf_m > ssf_s (长周期MA优先)
           窗口越长优先级越高
    后处理1: 耦合值上浮5%（仅当上浮后高于ssf_m时）
    后处理2: 粘性(双重): 偏差<stick_pct 或 持有不足 min_hold 天 → 保持前值
    
    新增过滤机制:
    1. 高位过滤: 当价格 > ssf_m*1.2 时，直接忽略信号（高位追涨风险大）
    2. 成交量分析的走平过滤: 区分不同性质的横盘
       - 走平定义: 最近3天价格变化在1%以内
       - 前期上涨: 最近10天前半段中60%的日子在上涨
       - 成交量放大判断: 走平期间成交量 > 上涨期间成交量 * 1.3
       - 只有"上涨后走平且成交量放大"才忽略信号（出货嫌疑）
       - "上涨后走平但成交量衰减"不忽略信号（惜售/蓄势，是好信号）

    参数:
    - df: DataFrame，需包含close列、volume列和均线列
    - ma_cols: 均线列名列表，默认['ssf_l','ssf_m','ssf_s']（顺序=优先级）
    - n_above/n_below: 上下方格栅线数量
    - step_pct: 格栅线间距(默认2%)
    - min_window/max_window: 耦合窗口范围(默认10~40，提高最小窗口减少噪音)
    - tight_pct: 格栅线紧密阈值(默认0.8%，更严格)
    - min_ratio: 格栅线紧密占比阈值(默认80%，提高质量要求)
    - track_pct: MA跟踪范围(默认3%)
    - track_ratio: MA跟踪占比阈值(默认80%，更严格)
    - stick_pct: 粘性偏差阈值(默认6%)
    - min_hold: 最小持有天数(默认10天，减少频繁切换)

    返回: list，每天的耦合格栅线值(或MA值)，无耦合为NaN
    """
    if ma_cols is None:
        ma_cols = ['ssf_l', 'ssf_m', 'ssf_s']

    close_arr = df['close'].values.astype(np.float64)
    volume_arr = df['volume'].values.astype(np.float64)
    ma_arrays = [df[col].values.astype(np.float64) for col in ma_cols]
    ssf_m_arr = df[ma_cols[1]].values.astype(np.float64)

    result_arr = _grid_coupling_core(
        close_arr, ma_arrays[0], ma_arrays[1], ma_arrays[2], volume_arr,
        n_above, n_below, step_pct, min_window, max_window,
        tight_pct, min_ratio, track_pct, track_ratio
    )

    # 后处理3: 粘性
    result_arr = _apply_stickiness(close_arr, result_arr, ssf_m_arr, stick_pct, min_hold)

    return result_arr.tolist()
