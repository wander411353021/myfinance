# -*- coding: utf-8 -*-
"""V10 画图 + 最新黄金坑算法 + 筹码色带(fengwo) 3面板。
数据走 tdx_quant.get_daily_kline_from_tdx(带 turnover/circ_shares 逐日历史股本, 无未来函数),
调 run_segmentation(show_chip=True) 画 K线+量+筹码 3 面板(隐藏恐慌反转/黄金坑方波, K线保留坑标记)。
用法: python3 golden_pit_plot_v10.py sh600234 [tail_days] [end_date]
"""
import os, sys
import pandas as pd
import matplotlib; matplotlib.use('Agg')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from price_segmenter_v10 import run_segmentation
from tdx_quant import get_daily_kline_from_tdx


def fetch_tdx(symbol, end_date='20260928', datalen=1800):
    """优先读信号缓存(scan_cache_v3b, 1800天与信号同源), 无缓存回退 tdx 拉取。
    2026-10-23: datalen 默认 1023→1800, 与信号体系 COST 递推长度统一(用户定版)。"""
    code6 = symbol[2:] if symbol[:2] in ('sh', 'sz', 'bj') else symbol
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'result', 'scan_cache_v3b', f'{code6}.csv')
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
        end = pd.Timestamp(end_date)
        df = df[df['date'] <= end].reset_index(drop=True)
        if len(df) >= 300:
            print(f'[fetch] 用信号缓存 scan_cache_v3b/{code6}.csv ({len(df)}行 ≤ {end_date}), 与信号同源')
            return df
        print(f'[fetch] 缓存 {len(df)} 行不足(需≥300), 回退 tdx datalen={datalen}')
    df = get_daily_kline_from_tdx(symbol, end_date, datalen=datalen, with_turnover=True)
    if df is None or len(df) == 0:
        return None
    return df.reset_index(drop=True)


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'sh600234'
    tail_days = int(sys.argv[2]) if len(sys.argv) > 2 else 260
    end_date = sys.argv[3] if len(sys.argv) > 3 else '20260928'
    mark_date = None
    if '--mark' in sys.argv:
        mark_date = sys.argv[sys.argv.index('--mark') + 1]
    df = fetch_tdx(symbol, end_date=end_date)
    if df is None or len(df) < 300:
        print(f'{symbol} 数据不足'); return
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result',
                       f'v10_{symbol}_{end_date}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    run_segmentation(df, tail_days=tail_days, name=symbol,
                     save_path=out, code=symbol, end_date=end_date,
                     show_chip=True, hide_overlay_lines=True, mark_date=mark_date,
                     show_v3b=True)
    print('已生成:', out)


if __name__ == '__main__':
    main()
