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
from challenge_scan import chip_warmup_window
from tdx_quant import get_daily_kline_from_tdx


def fetch_tdx(symbol, end_date='20260928', datalen=1800, min_rows=0):
    """缓存策略与 Streamlit 统一(2026-10-23 reasonix 定版口径):
      信号缓存(锚定最近1800天)仅当 截取后≥1600行 且 末日距画图日≤3天 时使用(近期图完整且新鲜);
      否则回退 tdx datalen=1800(从画图日往前, COST递推起点由换手预热驱动, 不人为锚定)。
      历史图(如600550@20200515, 缓存截取仅244行)一律回退 tdx → 与信号实时扫描口径一致。"""
    code6 = symbol[2:] if symbol[:2] in ('sh', 'sz', 'bj') else symbol
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'result', 'scan_cache_v3b', f'{code6}.csv')
    if os.path.exists(cache):
        try:
            df = pd.read_csv(cache)
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.normalize()
            end_ts = pd.Timestamp(end_date)
            df = df[df['date'] <= end_ts].reset_index(drop=True)
            _fresh = len(df) > 0 and df['date'].max() >= end_ts - pd.Timedelta(days=3)
            if len(df) >= 1600 and _fresh:
                print(f'[fetch] 用信号缓存 scan_cache_v3b/{code6}.csv ({len(df)}行, 末日距画图日≤3天)')
                return df
        except Exception:
            pass
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
    df = fetch_tdx(symbol, end_date=end_date, min_rows=tail_days)
    if df is None or len(df) < tail_days:
        print(f'{symbol} 数据不足({0 if df is None else len(df)}行 < 窗口{tail_days})'); return
    _ws = chip_warmup_window(df, tail_days)
    if _ws['start'] > 0:
        df = df.iloc[_ws['start']:].reset_index(drop=True)
    if not _ws['ok']:
        print(f'[warmup] {symbol} 筹码预热不足(仅提示): 计算窗口{_ws["total_days"]}天,'
              f'显示前累计换手{_ws["warmup_turnover"]*100:.0f}%'
              f' < {_ws["required_turnover"]*100:.0f}% (总窗口上限{_ws["required_days"]}天)')
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result',
                       f'v10_{symbol}_{end_date}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    run_segmentation(df, tail_days=tail_days, name=symbol,
                     save_path=out, code=symbol, end_date=end_date,
                     show_chip=True, hide_overlay_lines=True, mark_date=mark_date)
    print('已生成:', out)


if __name__ == '__main__':
    main()
