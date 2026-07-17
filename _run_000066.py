import sys, os
sys.path.insert(0, r'E:\chip_analyzer_ui\new_algo')

import matplotlib
matplotlib.use('Agg')

from price_segmenter_v10 import run_segmentation, plot_price_segmentation_v10
import price_segmenter_v10 as psv10
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from juejing import get_stock_klines_from_juejing

_orig = psv10.plot_price_segmentation_v10
_orig_close, _orig_show = plt.close, plt.show

def _patched(df_ohlc, result, bs_signal, bs_reason, bs_strength=None, all_levels=None,
             tail_days=200, name="", save_path=None):
    if save_path is None:
        save_path = os.path.join(r'E:\chip_analyzer_ui\new_algo\result', f'{name}_price_v10.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.show, plt.close = lambda *a, **k: None, lambda *a, **k: None
    _orig(df_ohlc, result, bs_signal, bs_reason, tail_days=tail_days, name=name,
          save_path=save_path, bs_strength=bs_strength, all_levels=all_levels)
    fig = plt.gcf()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved: {save_path}")
    plt.show, plt.close = _orig_show, _orig_close
    plt.close('all')

psv10.plot_price_segmentation_v10 = _patched

code, end_date, tail = '000066', '20241228', 250
df = get_stock_klines_from_juejing(code, end_date, fqt=1)
if df is None:
    print("ERROR: failed to fetch data for", code)
    sys.exit(1)
print(f"Fetched {code}: {len(df)} rows")
df = df[df['close'] > 0].reset_index(drop=True)
df['date'] = pd.to_datetime(df['date'])

result, bs_signal, bs_reason, bs_strength, all_levels = run_segmentation(df, tail_days=tail, name=code)

n_pivots = len(result.attrs.get('pivots', []))
n_buy = int(((bs_signal == 1).sum() + (bs_signal == 2).sum()))
n_sell = int(((bs_signal == -1).sum() + (bs_signal == -2).sum()))
print(f"Pivots: {n_pivots} | Buy signals: {n_buy} | Sell signals: {n_sell}")

idx = np.where(bs_signal != 0)[0]
print("--- last 15 trade signals ---")
for i in idx[-15:]:
    print(f"  bar {i} ({df['date'].iloc[i].date()}): {int(bs_signal[i]):+d} {bs_reason[i]}")
