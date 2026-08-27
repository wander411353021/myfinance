# -*- coding: utf-8 -*-
"""
黄金坑算法 v2 优化 + 真实数据回测验证
针对4个缺点的针对性优化：
1. 多窗口自适应回归 → 解决次新股/数据长度依赖
2. 双时间框架+残差加速度 → 解决慢跌/阴跌检测不到
3. 双轨信号体系 → 解决快启动条件错过慢牛坑
4. 坑性质多维度评分 → 解决高位坑与上涨中继坑不可分
"""
import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import urllib.request

warnings.filterwarnings('ignore')

# ============================================================
# 1. 数据获取模块（新浪财经，无需代理）
# ============================================================
def fetch_kline_sina(symbol, datalen=1023):
    """从新浪财经获取日K线数据。symbol格式: sz000001 / sh600000"""
    url = (f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen={datalen}")
    # 清除代理
    for k in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','all_proxy','ALL_PROXY']:
        os.environ.pop(k, None)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode('utf-8')
        data = json.loads(raw)
        if not data:
            return None
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['day'])
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df = df.sort_values('date').reset_index(drop=True)
        return df[['date','open','high','low','close','volume']]
    except Exception as e:
        print(f"  [WARN] {symbol} 数据获取失败: {e}")
        return None

# ============================================================
# 2. 基础工具函数
# ============================================================
def compute_rolling_regression(closes, window=250, use_log=True):
    """滚动对数线性回归，返回(preds, slopes)"""
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

def rolling_std(arr, window=60):
    """滚动标准差"""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = np.std(arr[i - window + 1:i + 1])
    return out

# ============================================================
# 3. 原始算法 v1（从 panic_reversal.py 提取，作为对照基准）
# ============================================================
def detect_golden_pit_v1(closes, reg250, z_thr=-1.5, merge_gap=15,
                          launch_gate=0.9, use_pre_std=True):
    """原始黄金坑检测：250日回归 + 60日std z-score"""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    if n < 310:
        return []  # v1硬性要求310天以上
    resid = closes - np.asarray(reg250, dtype=float)
    rstd = rolling_std(resid, 60)
    z = np.full(n, np.nan)
    for i in range(59, n):
        z[i] = resid[i] / rstd[i] if rstd[i] > 0 else 0.0
    gate_line = np.asarray(reg250, dtype=float) * launch_gate

    # pass1
    pits = []
    st = None
    for i in range(59, n):
        in_pit = np.isfinite(z[i]) and z[i] < z_thr
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

    # pass2
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
            in_pit2 = zz < z_thr
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
# 4. 优化算法 v2（4个针对性优化）
# ============================================================
def detect_golden_pit_v2(closes, volumes, z_thr=-1.5, merge_gap=15, launch_gate=0.9):
    """
    优化版黄金坑检测 v2：
    优化1: 多窗口自适应回归（数据不足时自动降级短窗口）
    优化2: 双时间框架z-score + 残差加速度（捕捉慢跌/阴跌）
    优化3: 双轨信号（快启动坑A类 + 慢坑确认B类）
    优化4: 坑性质评分（区分上涨中继vs高位见顶）
    返回: [(s, b, lch, signal_type, nature_score, confidence), ...]
    """
    closes = np.asarray(closes, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    n = len(closes)

    # ===== 优化1: 多窗口自适应回归 =====
    if n >= 310:
        reg_win_l, std_win_l, z_thr_l, conf = 250, 60, -1.5, 'high'
    elif n >= 180:
        reg_win_l, std_win_l, z_thr_l, conf = 120, 30, -1.7, 'medium'
    elif n >= 100:
        reg_win_l, std_win_l, z_thr_l, conf = 60, 15, -1.9, 'low'
    else:
        return []  # 数据太少，无法可靠检测

    # 长期回归（主框架）
    reg_l, slopes_l = compute_rolling_regression(closes, window=reg_win_l)
    resid_l = closes - reg_l
    rstd_l = rolling_std(resid_l, std_win_l)
    z_l = np.full(n, np.nan)
    for i in range(std_win_l - 1, n):
        z_l[i] = resid_l[i] / rstd_l[i] if rstd_l[i] > 0 else 0.0

    # ===== 优化2: 短期回归（辅助框架，捕捉慢跌）+ 残差加速度 =====
    reg_win_s, std_win_s = min(60, n // 3), min(20, n // 6)
    if reg_win_s >= 30 and std_win_s >= 10:
        reg_s, _ = compute_rolling_regression(closes, window=reg_win_s)
        resid_s = closes - reg_s
        rstd_s = rolling_std(resid_s, std_win_s)
        z_s = np.full(n, np.nan)
        for i in range(std_win_s - 1, n):
            z_s[i] = resid_s[i] / rstd_s[i] if rstd_s[i] > 0 else 0.0
    else:
        z_s = z_l.copy()

    # 残差加速度（5日变化率，平滑后）
    resid_accel = np.full(n, np.nan)
    for i in range(10, n):
        if np.isfinite(resid_l[i]) and np.isfinite(resid_l[i-5]):
            resid_accel[i] = (resid_l[i] - resid_l[i-5]) / 5.0

    # 双框架坑判定：长期z<-阈值 OR (短期z<-2.0 AND 残差加速度<0即持续下行)
    in_pit_combined = np.zeros(n, dtype=bool)
    for i in range(n):
        long_ok = np.isfinite(z_l[i]) and z_l[i] < z_thr_l
        short_ok = (np.isfinite(z_s[i]) and z_s[i] < -2.0 and
                     np.isfinite(resid_accel[i]) and resid_accel[i] < 0)
        in_pit_combined[i] = long_ok or short_ok

    gate_line = reg_l * launch_gate

    # 合并坑段
    pits = []
    st = None
    for i in range(n):
        if in_pit_combined[i] and st is None:
            st = i
        elif (not in_pit_combined[i]) and st is not None:
            pits.append([st, i - 1]); st = None
    if st is not None:
        pits.append([st, n - 1])
    merged = []
    for p in pits:
        if merged and p[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = p[1]
        else:
            merged.append(list(p))

    # 坑底 + 出坑日
    refined = []
    for s0, e0 in merged:
        # 坑前std精修（沿用v1的pass2逻辑）
        if s0 >= std_win_l:
            pre_std = np.std(resid_l[max(0,s0-std_win_l):s0])
            if pre_std <= 0:
                pre_std = rstd_l[s0] if np.isfinite(rstd_l[s0]) else 1.0
        else:
            pre_std = rstd_l[s0] if np.isfinite(rstd_l[s0]) else 1.0

        s_a = max(0, s0 - 10); e_a = min(n - 1, e0 + 10)
        seg = []
        st2 = None
        for i in range(s_a, e_a + 1):
            zz = resid_l[i] / pre_std if pre_std > 0 else 0.0
            in2 = zz < z_thr_l or (np.isfinite(z_s[i]) and z_s[i] < -2.0)
            if in2 and st2 is None:
                st2 = i
            elif (not in2) and st2 is not None:
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

        # 出坑日
        lch = None
        for i in range(b + 1, n):
            if (np.isfinite(reg_l[i]) and closes[i] >= reg_l[i] * launch_gate
                    and closes[i] > closes[b] * 1.02):
                lch = i
                break
        if lch is None:
            continue  # 未出坑的不纳入回测

        # ===== 优化3: 双轨信号分类 =====
        days_to_launch = lch - b
        if days_to_launch <= 5:
            signal_type = 'A_fast'  # 快启动坑（原策略，高胜率）
        else:
            # 慢坑：检查确认信号
            signal_type = classify_slow_pit(closes, volumes, b, lch)

        # ===== 优化4: 坑性质评分 =====
        nature_score = pit_nature_score(closes, volumes, b, reg_l, slopes_l)

        refined.append((s, b, lch, signal_type, nature_score, conf))
    return refined

def classify_slow_pit(closes, volumes, b, lch):
    """优化3: 慢坑确认信号分类
    B_confirmed: 慢坑但有确认信号（MA金叉/放量突破）
    C_unconfirmed: 慢坑无确认信号（过滤）
    """
    if lch - b <= 5:
        return 'A_fast'
    # MA5/MA20金叉
    ma5 = pd.Series(closes).rolling(5).mean().values
    ma20 = pd.Series(closes).rolling(20).mean().values
    golden_cross = False
    for i in range(b, min(lch + 1, len(closes))):
        if (np.isfinite(ma5[i]) and np.isfinite(ma20[i])
                and ma5[i] > ma20[i]
                and (i == 0 or ma5[i-1] <= ma20[i-1] if np.isfinite(ma5[i-1]) and np.isfinite(ma20[i-1]) else True)):
            golden_cross = True
            break
    # 放量突破坑底震荡区间上沿
    pit_high = np.max(closes[b:lch+1])
    vol_mean = np.mean(volumes[b:lch+1]) if lch > b else volumes[b]
    breakout = (closes[lch] > pit_high * 0.97) and (volumes[lch] > vol_mean * 1.3)

    if golden_cross or breakout:
        return 'B_slow_confirmed'
    return 'C_slow_unconfirmed'

def pit_nature_score(closes, volumes, b, reg, slopes):
    """优化4: 坑性质多维度评分
    高分=上涨中继坑概率高，低分=高位见顶坑概率高
    维度: 回归斜率方向(+2) / 坑底缩量程度(+1) / 前期涨幅(-1) / 坑深适度(+1)
    """
    score = 0
    n = len(closes)
    # 1. 回归斜率方向（坑底前20日斜率均值）
    if b >= 20 and np.isfinite(slopes[b-20:b+1]).any():
        recent_slope = np.nanmean(slopes[max(0,b-20):b+1])
        if recent_slope > 0:
            score += 2  # 上升趋势中的坑=中继
        elif recent_slope < -0.001:
            score -= 1  # 下降趋势中的坑=风险
    # 2. 坑底缩量程度
    if b >= 30:
        vol_pit = np.mean(volumes[max(0,b-3):b+3])
        vol_pre = np.mean(volumes[max(0,b-30):b-5])
        if vol_pre > 0 and vol_pit / vol_pre < 0.6:
            score += 1  # 缩量充分=洗盘
    # 3. 前期涨幅（60日）
    if b >= 60:
        pre_gain = closes[b] / closes[b-60] - 1
        if pre_gain > 0.5:
            score -= 1  # 前期涨幅过大=高位风险
    # 4. 坑深（适度深坑更好，过浅可能是假坑）
    if b >= 20:
        pre_high = np.max(closes[max(0,b-60):b])
        pit_depth = 1 - closes[b] / pre_high if pre_high > 0 else 0
        if 0.15 < pit_depth < 0.5:
            score += 1  # 适度深坑
    return score

# ============================================================
# 5. 回测框架
# ============================================================
def backtest_pits(closes, pits_v1, pits_v2, horizon=60):
    """
    回测：出坑日买入，持有horizon天，计算收益
    返回 v1结果, v2结果
    """
    n = len(closes)
    results_v1 = []
    for s, b, lch in pits_v1:
        if lch is None or lch + horizon >= n:
            continue
        # v1策略：快启动(≤5天) + 坑长≥8
        if lch - b > 5 or b - s + 1 < 8:
            continue
        buy_px = closes[lch]
        sell_px = closes[min(lch + horizon, n - 1)]
        ret = sell_px / buy_px - 1
        results_v1.append({'buy_idx': lch, 'ret': ret, 'pit_len': b-s+1,
                           'launch_days': lch-b})

    results_v2 = []
    for s, b, lch, sig_type, nature, conf in pits_v2:
        if lch is None or lch + horizon >= n:
            continue
        buy_px = closes[lch]
        sell_px = closes[min(lch + horizon, n - 1)]
        ret = sell_px / buy_px - 1
        results_v2.append({'buy_idx': lch, 'ret': ret, 'pit_len': b-s+1,
                           'launch_days': lch-b, 'signal': sig_type,
                           'nature_score': nature, 'confidence': conf})
    return results_v1, results_v2

def stats(results, label=''):
    """统计回测结果"""
    if not results:
        return {'label': label, 'n': 0, 'win_rate': 0, 'mean_ret': 0,
                'median_ret': 0, 'max_ret': 0, 'min_ret': 0}
    rets = [r['ret'] for r in results]
    return {
        'label': label,
        'n': len(rets),
        'win_rate': sum(1 for r in rets if r > 0) / len(rets),
        'mean_ret': np.mean(rets),
        'median_ret': np.median(rets),
        'max_ret': np.max(rets),
        'min_ret': np.min(rets),
    }

# ============================================================
# 6. 股票池与主回测
# ============================================================
STOCK_POOL = [
    # 大盘蓝筹
    ('sh600000', '浦发银行'), ('sz000001', '平安银行'),
    ('sh601318', '中国平安'), ('sh600519', '贵州茅台'),
    # 科技成长
    ('sz002415', '海康威视'), ('sz300750', '宁德时代'),
    ('sz002594', '比亚迪'), ('sh688981', '中芯国际'),
    # 消费医药
    ('sz000858', '五粮液'), ('sh600276', '恒瑞医药'),
    ('sz000568', '泸州老窖'), ('sh600887', '伊利股份'),
    # 周期/新能源
    ('sh601012', '隆基绿能'), ('sz002460', '赣锋锂业'),
    ('sh600030', '中信证券'), ('sz000725', '京东方A'),
    # 中小盘/次新
    ('sz002938', '鹏鼎控股'), ('sh603259', '药明康德'),
    ('sz300059', '东方财富'), ('sh601899', '紫金矿业'),
]

def main():
    print("=" * 80)
    print("黄金坑算法 v1(原始) vs v2(优化) — 真实数据回测对比")
    print("=" * 80)

    all_v1 = []
    all_v2 = []
    v2_by_signal = {'A_fast': [], 'B_slow_confirmed': [], 'C_slow_unconfirmed': []}
    v2_by_nature = {'high_score(>=3)': [], 'low_score(<3)': []}
    v1_skipped_short = 0  # v1因数据不足跳过的股票数
    v2_caught_short = 0   # v2在数据不足时仍能检测的坑数

    for symbol, name in STOCK_POOL:
        print(f"\n--- {name} ({symbol}) ---")
        df = fetch_kline_sina(symbol, datalen=1023)
        if df is None or len(df) < 100:
            print(f"  数据不足，跳过")
            continue
        closes = df['close'].values.astype(float)
        volumes = df['volume'].values.astype(float)
        n = len(closes)
        print(f"  数据: {n}天, {df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')}")

        # v1（需要250日回归）
        if n >= 310:
            reg250, _ = compute_rolling_regression(closes, window=250)
            pits_v1 = detect_golden_pit_v1(closes, reg250)
        else:
            pits_v1 = []
            v1_skipped_short += 1
            print(f"  [v1] 数据不足310天，跳过")

        # v2（自适应窗口）
        pits_v2 = detect_golden_pit_v2(closes, volumes)
        if n < 310 and len(pits_v2) > 0:
            v2_caught_short += len(pits_v2)

        print(f"  [v1] 检测到坑: {len(pits_v1)}个")
        print(f"  [v2] 检测到坑: {len(pits_v2)}个")
        for s, b, lch, sig, nat, conf in pits_v2:
            print(f"       坑底{df['date'].iloc[b].strftime('%Y-%m-%d')} "
                  f"出坑{df['date'].iloc[lch].strftime('%Y-%m-%d')} "
                  f"信号={sig} 性质分={nat} 置信={conf}")

        # 回测
        r1, r2 = backtest_pits(closes, pits_v1, pits_v2, horizon=60)
        all_v1.extend(r1)
        all_v2.extend(r2)
        for r in r2:
            v2_by_signal[r['signal']].append(r)
            if r['nature_score'] >= 3:
                v2_by_nature['high_score(>=3)'].append(r)
            else:
                v2_by_nature['low_score(<3)'].append(r)

        s1 = stats(r1, 'v1')
        s2 = stats(r2, 'v2')
        print(f"  回测(60天): v1 n={s1['n']} 胜率={s1['win_rate']:.1%} 均值={s1['mean_ret']:.1%} | "
              f"v2 n={s2['n']} 胜率={s2['win_rate']:.1%} 均值={s2['mean_ret']:.1%}")

    # ============================================================
    # 7. 汇总对比
    # ============================================================
    print("\n" + "=" * 80)
    print("汇总对比（全股票池，60天持有期）")
    print("=" * 80)

    s_all_v1 = stats(all_v1, 'v1原始')
    s_all_v2 = stats(all_v2, 'v2优化')
    print(f"\n{'指标':<20} {'v1原始':>12} {'v2优化':>12} {'变化':>12}")
    print("-" * 60)
    print(f"{'信号数':<20} {s_all_v1['n']:>12} {s_all_v2['n']:>12} {s_all_v2['n']-s_all_v1['n']:>+12}")
    print(f"{'胜率':<20} {s_all_v1['win_rate']:>11.1%} {s_all_v2['win_rate']:>11.1%} {s_all_v2['win_rate']-s_all_v1['win_rate']:>+11.1%}")
    print(f"{'平均收益':<20} {s_all_v1['mean_ret']:>11.1%} {s_all_v2['mean_ret']:>11.1%} {s_all_v2['mean_ret']-s_all_v1['mean_ret']:>+11.1%}")
    print(f"{'中位收益':<20} {s_all_v1['median_ret']:>11.1%} {s_all_v2['median_ret']:>11.1%} {s_all_v2['median_ret']-s_all_v1['median_ret']:>+11.1%}")
    print(f"{'最大收益':<20} {s_all_v1['max_ret']:>11.1%} {s_all_v2['max_ret']:>11.1%}")
    print(f"{'最小收益':<20} {s_all_v1['min_ret']:>11.1%} {s_all_v2['min_ret']:>11.1%}")

    # 优化1验证：数据长度
    print(f"\n--- 优化1: 多窗口自适应（数据长度）---")
    print(f"  v1因数据不足跳过股票: {v1_skipped_short}只")
    print(f"  v2在短数据中检测到坑: {v2_caught_short}个")

    # 优化3验证：信号类型细分
    print(f"\n--- 优化3: 双轨信号体系（按信号类型）---")
    for sig, rets in v2_by_signal.items():
        s = stats(rets, sig)
        print(f"  {sig:<22} n={s['n']:>4}  胜率={s['win_rate']:.1%}  均值={s['mean_ret']:.1%}  中位={s['median_ret']:.1%}")

    # 优化4验证：坑性质评分
    print(f"\n--- 优化4: 坑性质评分（上涨中继vs高位见顶）---")
    for label, rets in v2_by_nature.items():
        s = stats(rets, label)
        print(f"  {label:<22} n={s['n']:>4}  胜率={s['win_rate']:.1%}  均值={s['mean_ret']:.1%}  中位={s['median_ret']:.1%}")

    # 优化2验证：v2新增的坑（v1没检测到的）
    v1_buy_dates = set(r['buy_idx'] for r in all_v1)
    v2_new = [r for r in all_v2 if r['buy_idx'] not in v1_buy_dates]
    s_new = stats(v2_new, 'v2新增信号')
    print(f"\n--- 优化2: 双时间框架（v2新增但v1漏检的信号）---")
    print(f"  v2新增信号数: {s_new['n']}个")
    if s_new['n'] > 0:
        print(f"  新增信号胜率: {s_new['win_rate']:.1%}  均值: {s_new['mean_ret']:.1%}")

    print("\n" + "=" * 80)
    print("回测完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
