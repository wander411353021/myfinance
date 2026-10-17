# -*- coding: utf-8 -*-
"""全市场"压制→主升突破"信号扫描(2026-10-14 polo4111 定版信号, 严格无未来函数)。

信号定义(所有判定只用 <= 当天的数据):
  ① 压制期: 最近 LOOKBACK 天窗口内, 格栅压制线(compute_grid_target_price, min_level=4)
     有效天数 >= PRESS_VALID_MIN(70%), 价格>线天数占比 <= PRESS_ABOVE_MAX(5%),
     线高于价格中位 >= PRESS_GAP_MIN(10%)  —— 价格被格栅线长期压制
  ② 突破确认: 近 BREAK_DAYS(20) 天内出现 价格偏离 base > max_dev(13%) 的"置空"日
     (线消失 = 价格冲出压制半径, 主升确认信号)
  ③ 动量确认: 当前收盘价 > MA20 —— 过滤置空后立即回落的假突破

用法:
    python scan_pressure_breakout.py [YYYYMMDD] [--fresh]
    不传日期默认取数据末端。结果写入 result/pressure_breakout_{date}.csv
    本地缓存 result/scan_cache/{code}.csv, 重跑秒级; --fresh 强制重拉。
"""
import sys, os, time, glob
import numpy as np
import pandas as pd
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panic_reversal as pr
import tdx_quant as tq
from mean_reversion.signal_residual import compute_rolling_regression

LOOKBACK = 150        # 压制期窗口
PRESS_VALID_MIN = 0.70  # 压制期线有效占比下限
PRESS_ABOVE_MAX = 15    # 压制期价格>线占比上限(%)
PRESS_GAP_MIN = 10.0    # 压制期线-价中位下限(%)
BREAK_DAYS = 20         # 突破确认回溯窗口(void 模式)
# ── 早期模式(2026-10-15 reasonix, 用户: "要刚刚要进入主升的, 不是已经猛涨的")──
# 历史验证(1000只池/79551样本日, 因果, 全部满足压制期+收>MA20):
#   dev 分档后 60 日收益单调 —— 越早越好:
#     dev < -10%(仍在压制线下方): 信号前20日+2.9% → 后20日+4.5%(胜62.0%) 后60日+7.0%(胜55.5%)
#     dev  0~+3%(刚站上):        前+9.2% → 后20日+1.9%(50.6%) 后60日+4.8%
#     dev +8~13%(已涨):          前+17.3% → 后20日+1.5%(44.7%) 后60日+2.4%(胜43.0%)  ← 最差
#   现"置空突破"条件(dev>13%)信号前已涨 26.5%, 后续 60日仅 +4.5%(胜43.3%) = 买在猛涨之后。
#   最优组合: dev<-10% + gap 15~25% + 收>MA20 → 前20日+2.7%, 后20日+4.7%(胜64.7%), 后60日+8.8%(胜59.3%)
#   基准(全体满足压制期样本): 后20日+3.0%(54.3%) / 后60日+5.72%(51.9%)
EARLY_DEV_MAX = -0.10   # 早期模式: 收盘仍低于 base 10% 以上(还没启动主升)
EARLY_GAP_MIN = 10.0    # 压制线高度下限(%)
EARLY_GAP_MAX = 25.0    # 压制线高度上限(>25% 反而差: 后60日 +4.5%/胜47.6%)
WORKERS = 8             # 并发取数线程数
DATALEN = 450           # 只取450天(reg250预热250+窗口150+缓冲50), 减半取数耗时
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', 'scan_cache')
MODE = 'early'          # 'early'(默认, 刚启动前) | 'void'(旧: 置空突破后)


def cached_kline(code, end_date, fresh=False):
    """带本地缓存的日线(450天, 无换手率——本信号不依赖)。断点续跑秒级。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    fp = os.path.join(CACHE_DIR, f'{code}.csv')
    if not fresh and os.path.exists(fp):
        df = pd.read_csv(fp)
        # 校验: 数据末端 >= 需要的数据末端才算命中缓存 (date列ISO格式, 去'-'后比较)
        # 2026-10-18 reasonix: 通达信数据可能滞后于请求日期(请求 20261014 实际只到 20261012),
        #   仅按"末端日期 >= end_date"永远不命中 → 每次网络重拉。改为"日期够新 或 缓存 12 小时内拉过"。
        _date_ok = str(df['date'].iloc[-1])[:10].replace('-', '') >= end_date
        _recent = (time.time() - os.path.getmtime(fp)) < 12 * 3600
        if (_date_ok or _recent) and len(df) >= 260:
            return df
    df = tq.get_daily_kline_from_tdx(code, end_date, DATALEN, with_turnover=False)
    if df is not None and len(df):
        df.to_csv(fp, index=False)
    return df


def detect(code, end_date, fresh=False, mode=None):
    """返回压制→主升信号 dict 或 None。全部判定只用 <=i 数据。

    mode: 'early'(默认, 2026-10-15 用户定版) = 刚启动前(仍在压制线下 + 压制线在头上);
          'void' = 旧逻辑(近20天置空突破, 信号点已在猛涨之后)。
    """
    if mode is None:
        mode = MODE
    df = cached_kline(code, end_date, fresh)
    if df is None or len(df) < 260:
        return None
    closes = df['close'].values.astype(float)
    dates = df['date'].values
    reg120, _ = compute_rolling_regression(closes, 120)
    reg250, _ = compute_rolling_regression(closes, 250)
    r120s = pr.double_smooth_reg(reg120, 5, 5)
    r250s = pr.double_smooth_reg(reg250, 5, 5)
    base = np.maximum(r120s, r250s)
    gt_p, gl_p = pr.compute_grid_target_price(closes, r120s, r250s)  # min_level=4 定版

    n = len(closes)
    s = n - LOOKBACK
    win = slice(s, n)
    m = np.isfinite(gt_p[win])
    if m.sum() < LOOKBACK * PRESS_VALID_MIN:
        return None
    c = closes[win]
    above = np.sum(c[m] > gt_p[win][m])
    abr = above * 100 / m.sum()
    if abr > PRESS_ABOVE_MAX:
        return None
    gap = np.median((gt_p[win][m] / c[m] - 1) * 100)
    if gap < PRESS_GAP_MIN:
        return None
    sw = np.sum(np.diff(gl_p[win][gl_p[win] >= 0]) != 0)

    dev = closes / base - 1
    if mode == 'early':
        # ── 早期模式: 股价**仍被压在压制线下方**(还没启动主升) + 压制线高度适中 ──
        # 触发: dev < -10%(离压制线还有空间) 且 gap 在 10~25%(线在头上但不过远)
        if not np.isfinite(dev[-1]) or dev[-1] >= EARLY_DEV_MAX:
            return None
        if not (EARLY_GAP_MIN <= gap <= EARLY_GAP_MAX):
            return None
        # 启动确认: **刚站上 MA20**(今日收>MA20 且昨日收<=MA20) = "刚要进入主升"的那一天。
        # 实测(989只快照): 仅 dev/gap 条件命中 6.1%(全市场约335只/日, 池子会爆);
        #   加"刚站上 MA20"后 0.4%(约20只/日) —— 规模合理且对应启动点。
        _ma20 = pd.Series(closes).rolling(20).mean().values
        if not (np.isfinite(_ma20[-1]) and np.isfinite(_ma20[-2])):
            return None
        if not (closes[-1] > _ma20[-1] and closes[-2] <= _ma20[-2]):
            return None
        vd = str(dates[-1])[:10]
        void_idx = [n - 1]
    else:
        # 突破确认(旧): 近 BREAK_DAYS 天出现置空(dev > 13%)
        void_idx = np.where(np.isfinite(dev) & (dev > 0.13))[0]
        void_idx = void_idx[void_idx >= n - BREAK_DAYS]
        if len(void_idx) == 0:
            return None
        vd = str(dates[void_idx[0]])[:10]

    # 动量确认: 当前收盘 > MA20
    ma20 = pd.Series(closes).rolling(20).mean().values
    if not np.isfinite(ma20[-1]) or closes[-1] <= ma20[-1]:
        return None

    # 突破后表现(信息性): 置空首日以来收盘涨幅
    post_ret = (closes[-1] / closes[void_idx[0]] - 1) * 100
    return dict(code=code, press_abr=round(abr, 1), press_gap=round(gap, 1),
                switch=sw, void_date=vd, void_dev=round(dev[void_idx[0]] * 100, 1),
                void_close=round(float(closes[void_idx[0]]), 3),   # 入池基准价(2026-10-15 reasonix: 防基准漂移)
                cur=round(closes[-1], 2), ma20=round(ma20[-1], 2), post_ret=round(post_ret, 1))


def main():
    t0 = time.time()
    END = sys.argv[1] if len(sys.argv) > 1 else time.strftime('%Y%m%d')
    fresh = '--fresh' in sys.argv
    global MODE
    if '--void-mode' in sys.argv:
        MODE = 'void'
    print(f'模式: {MODE} (' + ('刚启动前 dev<-10%% + gap 10~25%%' if MODE == 'early' else '置空突破后') + ')', flush=True)
    # 多进程分片: --parts N --part K (0-based), 每进程独立连接串行取数
    parts = int(sys.argv[sys.argv.index('--parts') + 1]) if '--parts' in sys.argv else 1
    part = int(sys.argv[sys.argv.index('--part') + 1]) if '--part' in sys.argv else 0

    all_df = tq.load_stock_list()
    codes = all_df['code'].astype(str).str.zfill(6).tolist()
    names = dict(zip(codes, all_df['name']))
    if parts > 1:
        codes = codes[part::parts]
    print(f'分片{part+1}/{parts}: {len(codes)} 只, END={END}, 阈值: 上穿<={PRESS_ABOVE_MAX}% gap>={PRESS_GAP_MIN}% 近{BREAK_DAYS}天置空突破', flush=True)

    hits, n_ok = [], 0
    for i, code in enumerate(codes):
        try:
            r = detect(code, END, fresh)
            if r is not None:
                r['name'] = names.get(code, '')
                hits.append(r)
            n_ok += 1
        except Exception:
            pass
        if (i + 1) % 250 == 0:
            print(f'  [{part+1}/{parts}] 进度 {i+1}/{len(codes)} 命中{len(hits)} 耗时{time.time()-t0:.0f}s', flush=True)
    # 保存
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', f'pressure_breakout_{END}_part{part}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if hits:
        df = pd.DataFrame(hits).sort_values('post_ret', ascending=False)
        df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'[分片{part+1}/{parts}] 完成: 有效{n_ok}只 命中{len(hits)}只 耗时{time.time()-t0:.0f}s -> {out}', flush=True)


if __name__ == '__main__':
    main()
