"""剖析 000066 在 2024-10-21 前后的'接近/突破前高后缩量回调止跌'形态。"""
import sys, os, pandas as pd, numpy as np
sys.path.insert(0, r'E:\chip_analyzer_ui\new_algo')
WORK = r'E:\chip_analyzer_ui\new_algo'
CACHE = os.path.join(WORK, 'result', '_cache')
code, end = '000066', '20241228'
df = pd.read_pickle(os.path.join(CACHE, f'{code}_{end}.pkl'))
df['date'] = pd.to_datetime(df['date'])

# 全局 20 日均量
vma20 = df['volume'].rolling(20).mean()

# 窗口（保留原始 index 以对齐 vma20）
mask = (df['date'] >= '2024-09-18') & (df['date'] <= '2024-10-25')
sub = df[mask].copy()
sub['vratio'] = sub['volume'] / vma20[sub.index]
sub['ret'] = sub['close'].pct_change()
sub['range_pct'] = (sub['high'] - sub['low']) / sub['close'] * 100
disp = sub.reset_index(drop=True)

cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'vratio', 'ret', 'range_pct']
print(disp[cols].to_string(index=False,
      formatters={'date': lambda d: d.strftime('%m-%d'),
                  'open': '{:.2f}'.format, 'high': '{:.2f}'.format, 'low': '{:.2f}'.format,
                  'close': '{:.2f}'.format, 'volume': '{:.0f}'.format,
                  'vratio': '{:.2f}'.format, 'ret': '{:+.2%}'.format, 'range_pct': '{:.1f}'.format}))

# 前高（窗口前段，即被接近/突破的那个高点）
pre = df[(df['date'] >= '2024-09-18') & (df['date'] <= '2024-10-10')]
hi_idx = pre['high'].idxmax()
print(f"\n前高(被接近/突破): {pre.loc[hi_idx,'date'].strftime('%Y-%m-%d')} high={pre['high'].max():.2f} close={pre.loc[hi_idx,'close']:.2f}")

# 10-21 附近的回撤低位
post = df[(df['date'] >= '2024-10-10') & (df['date'] <= '2024-10-21')]
lo_idx = post['low'].idxmin()
print(f"回撤低位: {post.loc[lo_idx,'date'].strftime('%Y-%m-%d')} low={post['low'].min():.2f} "
      f"(较前高回落 {(post['low'].min()/pre['high'].max()-1)*100:+.1f}%)")
print(f"突破时点(close首超前高): ", end='')
brk = df[(df['date'] >= '2024-10-10') & (df['close'] > pre['high'].max())]
if len(brk):
    print(brk.iloc[0]['date'].strftime('%Y-%m-%d'), f"close={brk.iloc[0]['close']:.2f}")
else:
    print("窗口内未突破(仅为接近)")
