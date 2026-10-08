# -*- coding: utf-8 -*-
"""指数筹码 V10 风格画图(方案A: 换手率=量比代理, 临时脚本不入库)。

用法: python3 index_v10_plot.py <code> <tail> [end]
复用 V10 画图管线(run_segmentation + plot_price_segmentation_v10, show_chip=True):
K线 + 分段背景 + COST筹码带 + COST线。指数无换手率 → turnover = 0.05×量比。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from eltdx import Client
from price_segmenter_v10 import run_segmentation
from tdx_index_names import display_label

TURN_BASE = 0.05


def fetch_index(code, end, datalen=1600):
    cli = Client(timeout=8.0)
    try:
        ks = cli.bars.get(code, period='day', count=datalen,
                          anchor_date=end, all_pages=True)
        bars = list(ks.bars)
        rows = []
        for b in bars:
            rows.append(dict(date=pd.Timestamp(b.time.date()), open=b.open, close=b.close,
                             high=b.high, low=b.low,
                             volume=b.volume_lots if b.volume_lots else 0.0,
                             amount=b.amount if b.amount else 0.0))
        df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
        if end:
            df = df[df['date'].astype(str) <= end]
        return df
    finally:
        cli.close()


def main():
    code = sys.argv[1] if len(sys.argv) > 1 else 'sh881478'
    tail = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    end = sys.argv[3] if len(sys.argv) > 3 else '20261008'

    df = fetch_index(code, end)
    if df is None or len(df) < 400:
        print('数据不足:', code, len(df) if df is not None else 0)
        return

    V = df['volume'].values.astype(float)
    ma20 = np.convolve(V, np.ones(20) / 20, mode='valid')
    ma20 = np.concatenate([np.full(19, np.nan), ma20])
    ratio = np.where(ma20 > 0, V / ma20, 1.0)
    turn = (TURN_BASE * ratio).clip(0.005, 0.30)
    turn[:20] = TURN_BASE
    df['turnover'] = turn

    out = 'result/v10_index_%s_%d.png' % (code.replace('/', '_'), tail)
    try:
        run_segmentation(df, tail_days=tail, name=display_label(code), code=None, end_date=end,
                         show_chip=True, hide_overlay_lines=True,
                         save_path=out)
        print('已输出:', out)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('FAIL:', str(e)[:200])


if __name__ == '__main__':
    main()
