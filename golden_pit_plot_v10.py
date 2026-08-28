# -*- coding: utf-8 -*-
"""V10 画图 + 最新黄金坑算法(GOLD PIT面板)。
用 tdx 拉完整数据(含open), 调 run_segmentation 画 V10 图(6面板含 GOLD PIT)。
用法: python3 golden_pit_plot_v10.py sh600234 [tail_days]
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from price_segmenter_v10 import run_segmentation


def fetch_tdx(symbol, end_date='20260828'):
    from eltdx import TdxClient
    with TdxClient() as client:
        data = client.bars.get(symbol, period='day', count=1023,
                               adjust='qfq', anchor_date=end_date, all_pages=True)
    bars = getattr(data, 'bars', None)
    if not bars:
        return None
    rows = []
    for b in bars:
        if float(b.close) > 0:
            rows.append({
                'date': b.time, 'open': float(b.open), 'high': float(b.high),
                'low': float(b.low), 'close': float(b.close), 'volume': float(b.volume_lots),
            })
    rows.sort(key=lambda r: r['date'])  # all_pages 顺序可能乱,必须排序
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    return df.reset_index(drop=True)


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'sh600234'
    tail_days = int(sys.argv[2]) if len(sys.argv) > 2 else 260
    df = fetch_tdx(symbol)
    if df is None or len(df) < 300:
        print(f'{symbol} 数据不足'); return
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result',
                       f'v10_{symbol}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # tail_days 足够大, 让 2026 年的坑完整显示
    run_segmentation(df, tail_days=tail_days, name=symbol,
                     save_path=out, code=symbol)
    print('已生成:', out)


if __name__ == '__main__':
    main()
