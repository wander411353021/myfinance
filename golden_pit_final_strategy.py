# -*- coding: utf-8 -*-
"""
黄金坑策略 v2.0 最终版（穿越牛熊版）
====================================
基于 1000 只 A 股大规模回测（1206 个信号）验证的最终策略：

【信号层】v1 黄金坑检测（250 日对数回归 + 60 日 std z-score）
  - 坑深 z < -1.5
  - 快启动：坑底到出坑 <= 5 个交易日
  - 坑长 >= 8 个交易日
  - 出坑条件：收盘收复回归线 * 0.9 且 > 坑底 * 1.02

【评分层】6 维度质量评分（0-100）
  1. 缩量挖坑（25 分）：坑内均量 vs 坑前 20 日均量
  2. 放量出坑（25 分）：出坑日量 vs 出坑前 5 日均量
  3. 坑深适度（15 分）：坑底相对坑前 60 日最高跌幅 15%-50% 最佳
  4. 个股相对强度（15 分）：个股 20 日涨幅 - 沪深300 20 日涨幅
  5. 回归线方向（10 分）：250 日回归线斜率向上
  6. 坑底缩量企稳（10 分）：坑底附近缩量洗盘

【持有层】20 个交易日固定持有
  - 全周期胜率 81.2%（2023-2026 各年份 78%-87%，穿越牛熊）
  - 相比 60 天持有（胜率 70.4%）显著提升胜率稳定性

【执行规则】
  - 出坑日收盘买入，持有 20 个交易日卖出
  - 评分 >= 30：正常仓位
  - 评分 42-51（甜点区）：加仓（该区胜率 76%、均值 20%）
  - 评分 >= 52：放弃（高分化 = 收益弹性大但胜率崩塌）

依赖：golden_pit_v2_backtest（提供数据获取与信号检测）
用法：
  python golden_pit_final_strategy.py scan --symbol sh600000   # 单股扫描
  python golden_pit_final_strategy.py scan --pool stock_pool_300.txt   # 全池扫描
  python golden_pit_final_strategy.py signals --pool stock_pool_300.txt --date 2026-08-27  # 指定日信号
"""
import os
import sys
import time
import random
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from golden_pit_v2_backtest import (
    fetch_kline_sina,
    compute_rolling_regression,
    detect_golden_pit_v1,
)

# ============================================================
# 1. 沪深300 大盘数据（用于相对强度）
# ============================================================
_HS300_CACHE = None

def get_hs300():
    """获取沪深300日线并缓存，返回 (date_str_list, close_array)"""
    global _HS300_CACHE
    if _HS300_CACHE is None:
        df = fetch_kline_sina('sh000300', datalen=1023)
        if df is None:
            return [], np.array([])
        dates = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in df['date'].values]
        closes = df['close'].values.astype(float)
        _HS300_CACHE = (dates, closes)
    return _HS300_CACHE

def hs300_ret(date_str, days=20):
    """沪深300 在指定日期的 N 日收益率"""
    dates, closes = get_hs300()
    if not dates:
        return None
    idx_map = {d: i for i, d in enumerate(dates)}
    i = idx_map.get(date_str)
    if i is None or i < days:
        return None
    return closes[i] / closes[i - days] - 1

# ============================================================
# 2. 6 维度质量评分
# ============================================================
def score_signal(closes, volumes, reg250, s, b, lch):
    """
    黄金坑信号 6 维度质量评分 (0-100)
    closes: 收盘价数组
    volumes: 成交量数组
    reg250: 250 日回归线数组
    s: 坑段起点, b: 坑底, lch: 出坑日
    """
    n = len(closes)
    score = 0.0

    # 个股相对强度（20日）vs 沪深300
    buy_date_str = 'unused'
    rs20 = None

    # 1. 缩量挖坑 (25分)
    vol_pit = np.mean(volumes[s:b+1])
    vol_pre = np.mean(volumes[max(0, s-20):s]) if s >= 20 else vol_pit
    if vol_pre > 0:
        shrink = vol_pit / vol_pre
        if shrink <= 0.3:
            score += 25
        elif shrink <= 0.7:
            score += 25 - (shrink - 0.3) / 0.4 * 15
        elif shrink <= 1.0:
            score += 10 - (shrink - 0.7) / 0.3 * 10
        # 放量挖坑 = 恐慌砸盘，不加分

    # 2. 放量出坑 (25分)
    vol_lch = volumes[lch]
    vol_lch_pre = np.mean(volumes[max(0, lch-5):lch]) if lch >= 5 else vol_lch
    if vol_lch_pre > 0:
        fill = vol_lch / vol_lch_pre
        if fill >= 2.0:
            score += 25
        elif fill >= 1.2:
            score += 10 + (fill - 1.2) / 0.8 * 15
        elif fill >= 0.8:
            score += 5 + (fill - 0.8) / 0.4 * 5
        # 无量出坑 = 弱反弹，不加分

    # 3. 坑深适度 (15分)
    if b >= 60:
        pre_high = np.max(closes[b-60:b])
    else:
        pre_high = np.max(closes[max(0, b-20):b+1])
    if pre_high > 0:
        depth = 1 - closes[b] / pre_high
        if 0.15 <= depth <= 0.50:
            score += 15
        elif 0.05 <= depth < 0.15:
            score += 8
        elif depth > 0.50:
            score += 5  # 太深可能是基本面恶化
        # 太浅 = 假坑，不加分

    # 4. 个股相对强度 (15分) —— 由外部传入补分
    # 5. 回归线方向 (10分)
    if b >= 20 and np.isfinite(reg250[b]) and np.isfinite(reg250[b-20]):
        slope = reg250[b] / reg250[b-20] - 1
        if slope > 0.02:
            score += 10
        elif slope > 0:
            score += 6
        elif slope > -0.02:
            score += 3
        # 回归线向下 = 下跌趋势中的坑，不加分

    # 6. 坑底缩量企稳 (10分)
    vol_bottom = np.mean(volumes[max(0, b-2):b+3])
    vol_pre2 = np.mean(volumes[max(0, s-20):s]) if s >= 20 else vol_bottom
    if vol_pre2 > 0:
        bottom_shrink = vol_bottom / vol_pre2
        if bottom_shrink < 0.5:
            score += 10
        elif bottom_shrink < 0.8:
            score += 6
        elif bottom_shrink < 1.0:
            score += 3
        # 坑底放量 = 恐慌未止，不加分

    return score, shrink, fill, depth

def add_rs_score(score, rs20):
    """叠加个股相对强度维度（15分）"""
    if rs20 is not None:
        if rs20 > 0.10:
            score += 15
        elif rs20 > 0.05:
            score += 12
        elif rs20 > 0:
            score += 8
        elif rs20 > -0.05:
            score += 4
        else:
            score += 0
    return score

def classify_score(score):
    """
    评分分类（执行规则）
    - >=30: 正常仓位
    - 42-51: 甜点区，加仓
    - >=52: 放弃（胜率崩塌区）
    - <30: 观望（低质量）
    """
    if score >= 52:
        return '放弃(高分化胜率崩塌)'
    if 42 <= score <= 51:
        return '加仓(甜点区)'
    if score >= 30:
        return '正常仓位'
    return '观望(低质量)'

# ============================================================
# 3. 单股黄金坑信号扫描
# ============================================================
def scan_stock(symbol, name='', need_data_len=310):
    """
    扫描单只股票的黄金坑信号，返回信号列表
    每个信号: {
      symbol, name, buy_date, buy_price, score, level,
      s, b, lch, shrink, fill, depth, rs20, buy_year
    }
    """
    df = fetch_kline_sina(symbol, datalen=1023)
    if df is None or len(df) < need_data_len:
        return []

    closes = df['close'].values.astype(float)
    volumes = df['volume'].values.astype(float)
    dates = df['date'].values
    n = len(closes)

    reg250, _ = compute_rolling_regression(closes, window=250)
    pits_v1 = detect_golden_pit_v1(closes, reg250)

    signals = []
    for pit in pits_v1:
        s, b, lch = pit
        if lch is None or lch + 20 >= n:
            continue
        if lch - b > 5:      # 快启动条件(坑底后5天内收复)
            continue
        # 规则F(2026-08-28): 坑长≥8约束已移除。坑长<8的急跌急涨V型短坑胜率反而更高:
        # 1000池验证 坑长5-7天151个胜率80.1% / <5天756个胜率75.9% vs 现版74.1%(660个)。
        # 快启动≤5仍是核心约束,去掉后胜率掉到65.8%。详见 golden_pit_rule_compare.py

        buy_date = pd.Timestamp(dates[lch]).strftime('%Y-%m-%d')
        buy_price = closes[lch]
        buy_year = pd.Timestamp(dates[lch]).year

        # 相对强度
        rs20 = None
        if b >= 20:
            stock_ret20 = closes[b] / closes[b-20] - 1
            mkt_ret20 = hs300_ret(buy_date, 20)
            if mkt_ret20 is not None:
                rs20 = stock_ret20 - mkt_ret20

        # 评分
        score, shrink, fill, depth = score_signal(closes, volumes, reg250, s, b, lch)
        score = add_rs_score(score, rs20)

        signals.append({
            'symbol': symbol, 'name': name,
            'buy_date': buy_date, 'buy_year': buy_year,
            'buy_price': round(buy_price, 2),
            'score': round(score, 1),
            'level': classify_score(score),
            'shrink': round(shrink, 2) if shrink else None,
            'fill': round(fill, 2) if fill else None,
            'depth': round(depth, 2) if depth else None,
            'rs20': round(rs20, 3) if rs20 is not None else None,
        })
    return signals

# ============================================================
# 4. 全池扫描
# ============================================================
def load_pool(pool_file):
    """加载股票池文件（每行: code,name）"""
    stocks = []
    with open(pool_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                if ',' in line:
                    code, name = line.split(',', 1)
                else:
                    code, name = line, line
                stocks.append((code.strip(), name.strip()))
    return stocks

def scan_pool(pool_file, top_n=None, sleep_range=(0.02, 0.08)):
    """扫描整个股票池，返回所有信号（按评分降序）"""
    stocks = load_pool(pool_file)
    if top_n:
        stocks = stocks[:top_n]

    print(f"扫描股票池: {len(stocks)}只")
    print("-" * 60)
    all_signals = []
    success = 0
    start = time.time()

    for idx, (symbol, name) in enumerate(stocks):
        if (idx + 1) % 50 == 0:
            print(f"  进度: {idx+1}/{len(stocks)} ({time.time()-start:.0f}s) 信号{len(all_signals)}")
        sigs = scan_stock(symbol, name)
        if sigs:
            success += 1
        all_signals.extend(sigs)
        time.sleep(random.uniform(*sleep_range))

    print(f"\n完成: {success}只出现信号, 共{len(all_signals)}个信号, 耗时{time.time()-start:.0f}s")
    return all_signals

# ============================================================
# 5. 输出模块
# ============================================================
def print_signals(signals, date_filter=None, min_score=0):
    """打印信号列表，可按买入日期过滤"""
    if date_filter:
        signals = [s for s in signals if s['buy_date'] == date_filter]
    if min_score:
        signals = [s for s in signals if s['score'] >= min_score]

    if not signals:
        print("  无符合条件的信号")
        return

    # 按日期分组
    from collections import defaultdict
    by_date = defaultdict(list)
    for s in signals:
        by_date[s['buy_date']].append(s)

    total_score = np.mean([s['score'] for s in signals]) if signals else 0
    print(f"\n信号总数: {len(signals)}  平均评分: {total_score:.1f}")
    print("=" * 100)
    print(f"{'买入日期':<12} {'代码':<10} {'名称':<10} {'买入价':>8} {'评分':>6} {'级别':<20} {'缩量比':>7} {'放量比':>7} {'坑深':>6} {'RS20':>7}")
    print("-" * 100)
    for date in sorted(by_date.keys()):
        for s in sorted(by_date[date], key=lambda x: -x['score']):
            print(f"{s['buy_date']:<12} {s['symbol']:<10} {s['name']:<10} "
                  f"{s['buy_price']:>8.2f} {s['score']:>6.1f} {s['level']:<20} "
                  f"{str(s['shrink']):>7} {str(s['fill']):>7} {str(s['depth']):>6} {str(s['rs20']):>7}")

def backtest_signals(signals, horizon=20):
    """
    对信号做固定持有期回测（需要信号含持仓收益，用于验证）
    """
    print("\n" + "=" * 60)
    print(f"持有 {horizon} 日回测（信号数 {len(signals)}）")
    print("=" * 60)
    # 该函数需要在扫描时附带收益，简化版：仅统计评分分布
    scores = [s['score'] for s in signals]
    if scores:
        print(f"评分分布: min={min(scores):.1f} max={max(scores):.1f} 均值={np.mean(scores):.1f}")
        for lo, hi, label in [(30, 42, '30-42'), (42, 52, '42-51甜点区'), (52, 200, '52+')]:
            grp = [s for s in signals if lo <= s['score'] < hi]
            print(f"  {label}: {len(grp)}个信号")

# ============================================================
# 6. main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='黄金坑策略 v2.0 最终版')
    sub = parser.add_subparsers(dest='cmd')

    p_scan = sub.add_parser('scan', help='扫描黄金坑信号')
    p_scan.add_argument('--symbol', help='单只股票代码（如 sh600000）')
    p_scan.add_argument('--pool', help='股票池文件（每行 code,name）')
    p_scan.add_argument('--top', type=int, default=None, help='只扫描前 N 只（调试用）')

    p_sig = sub.add_parser('signals', help='输出指定日期的信号')
    p_sig.add_argument('--pool', required=True, help='股票池文件')
    p_sig.add_argument('--date', required=True, help='买入日期 YYYY-MM-DD')
    p_sig.add_argument('--min-score', type=float, default=0, help='最低评分过滤')
    p_sig.add_argument('--top', type=int, default=None, help='只扫描前 N 只')

    args = parser.parse_args()

    if args.cmd == 'scan':
        if args.symbol:
            print(f"扫描 {args.symbol} ...")
            sigs = scan_stock(args.symbol)
            print_signals(sigs)
        elif args.pool:
            sigs = scan_pool(args.pool, top_n=args.top)
            print_signals(sigs)
        else:
            print("请指定 --symbol 或 --pool")
    elif args.cmd == 'signals':
        sigs = scan_pool(args.pool, top_n=args.top)
        print_signals(sigs, date_filter=args.date, min_score=args.min_score)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
