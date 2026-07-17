import sys, os
sys.path.insert(0, r'E:\chip_analyzer_ui\new_algo')

import matplotlib
matplotlib.use('Agg')

from price_segmenter_v11 import run_segmentation, plot_price_segmentation_v11
import price_segmenter_v11 as psv11
import matplotlib.pyplot as plt
import pandas as pd
from juejing import get_stock_klines_from_juejing

_orig = psv11.plot_price_segmentation_v11
_orig_close, _orig_show = plt.close, plt.show

def _patched(df_ohlc, result, bs_signal, bs_reason, tail_days=200, name="", save_path=None,
             position=None, action=None):
    if save_path is None:
        save_path = os.path.join(r'E:\chip_analyzer_ui\new_algo\result', f'{name}_price_v11.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.show, plt.close = lambda *a, **k: None, lambda *a, **k: None
    _orig(df_ohlc, result, bs_signal, bs_reason, tail_days=tail_days, name=name, save_path=save_path,
          position=position, action=action)
    fig = plt.gcf()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved: {save_path}")
    plt.show, plt.close = _orig_show, _orig_close
    plt.close('all')

psv11.plot_price_segmentation_v11 = _patched

code, end_date, tail = '000066', '20241228', 250
df = get_stock_klines_from_juejing(code, end_date, fqt=1)
if df is None:
    print("ERROR: failed to fetch data for", code); sys.exit(1)
print(f"Fetched {code}: {len(df)} rows")
df = df[df['close'] > 0].reset_index(drop=True)
df['date'] = pd.to_datetime(df['date'])

res = run_segmentation(df, tail_days=tail, name=code, with_trades=True,
                       vol_confirm_mult=1.2, atr_mult=0.5, stop_mult=3.0, risk_pct=0.10)
c_result, bs_signal, bs_reason, position, action, trades = res

print(f"Pivots: {len(c_result.attrs.get('pivots', []))}")
print(f"Raw buy signals: {int((bs_signal == 1).sum())}")
print(f"Raw sell signals: {int((bs_signal == -1).sum())}")
print(f"Trades: {len(trades)}  stops: {sum(1 for _,_,_,_,r in trades if r == 'STOP')}")
print("--- last 10 trades ---")
for ei, xi, ep, xp, r in trades[-10:]:
    print(f"  {df['date'].iloc[ei].date()} -> {df['date'].iloc[xi].date()}  "
          f"{ep:.2f} -> {xp:.2f} ({r})")
