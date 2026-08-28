# -*- coding: utf-8 -*-
"""黄金坑组合模拟 资金曲线图（口径A 固定份额 vs 口径B 子账户复利 + 沪深300基准）

用法: python3 golden_pit_portfolio_chart.py
输出: golden_pit_portfolio_curve.png
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden_pit_portfolio_sim as sim

# 中文字体
for f in ['Noto Serif CJK SC', 'Noto Sans CJK SC', 'SimHei', 'AR PL UMing CN']:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [f]
        plt.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue

START = pd.Timestamp('2023-01-01')
END = pd.Timestamp('2026-08-28')
INIT = 1000000.0


def run_fixed_sim(stocks, buy_delay=0):
    """口径A: 固定每份10万, 收益落袋不复投。返回 (date_series, asset_series)"""
    events = []
    for sym in stocks:
        events += sim.collect_events(sym, buy_delay=buy_delay)
    events.sort(key=lambda e: (e[1] is not None, e[1] if e[1] is not None else pd.Timestamp.max))
    cash = INIT
    positions = {}  # (sym,lch)-> dict(qty, price, close_idx)
    trades = []
    snap = []
    for ev in events:
        typ, date, sym, lch, idx = ev
        if typ == 'buy':
            if date < START:
                continue
            if len(positions) >= 10:
                continue
            d = sim.load(sym); price = float(d['close'][idx])
            qty = int(100000 / price / 100) * 100
            if qty <= 0: continue
            comm = max(price * qty * sim.COMMISSION, sim.MIN_COMM)
            if price * qty + comm > cash: continue
            cash -= price * qty + comm
            positions[(sym, lch)] = {'qty': qty, 'price': price, 'close_idx': idx}
        elif typ == 'sell':
            key = (sym, lch)
            if key not in positions: continue
            pos = positions.pop(key)
            d = sim.load(sym); price = float(d['close'][idx])
            proceeds = price * pos['qty'] * (1 - sim.COMMISSION - sim.STAMP)
            cash += proceeds
            snap.append((date, cash + _mv(positions)))
    snap.append((END, cash + _mv(positions)))
    s = pd.DataFrame(snap, columns=['date', 'asset']).sort_values('date').reset_index(drop=True)
    return s


def _mv(positions):
    mv = 0.0
    for (sym, lch), pos in positions.items():
        d = sim.load(sym)
        mv += pos['qty'] * float(d['close'][pos['close_idx']])
    return mv


def main():
    buy_delay = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    delay_txt = '次日买入(严格实盘)' if buy_delay else '出坑日收盘买入(理论)'
    stocks = [l.strip().split(',')[0] for l in open(os.path.join(sim.WORKDIR, 'stock_pool_1000.txt')) if l.strip()]
    print(f'加载 {len(stocks)} 只, 口径: {delay_txt}')
    r = sim.run_sim(stocks, buy_delay=buy_delay)
    print(f'  口径B: 成交{r["n_trades"]}笔 胜率{r["win_rate"]:.1f}% 总收益{r["total_ret"]*100:+.1f}%')
    print('运行口径A(固定份额)...')
    sA = run_fixed_sim(stocks, buy_delay=buy_delay)
    finalA = sA['asset'].iloc[-1]
    print(f'  口径A: 总收益{finalA/INIT*100-100:+.1f}%')

    # 沪深300 基准（归一化 100万）
    idx = sim.load('sh000300')
    ts300 = pd.to_datetime(idx['ts'].astype('int64'), unit='s')
    c300 = idx['close'].astype(float)
    m = (ts300 >= START) & (ts300 <= END)
    c300_arr = c300[m]
    base = INIT * c300_arr / c300_arr[0]

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(14, 7.5))
    sB = r['snapshot']
    ax.plot(sB['date'], sB['asset'] / 1e4, color='#d62728', lw=1.8, label=f"口径B 子账户复利（最终 {r['total_ret']*100:+.0f}%）")
    ax.plot(sA['date'], sA['asset'] / 1e4, color='#1f77b4', lw=1.8, label=f"口径A 固定每份10万（最终 {finalA/INIT*100-100:+.0f}%）")
    ax.plot(ts300[m], base / 1e4, color='#7f7f7f', lw=1.2, ls='--', label=f"沪深300（同期 {base[-1]/INIT*100-100:+.0f}%）")

    # 年度分界 + 标注
    for y in [2023, 2024, 2025, 2026]:
        d0 = pd.Timestamp(f'{y}-01-01')
        if START <= d0 <= END:
            ax.axvline(d0, color='#999', lw=0.8, ls=':', alpha=0.6)
        # 年度收益率标注（口径B）
        if y in r['year_assets']:
            ya = r['year_assets'][y]
            ax.annotate(f"{y}: {ya['ret']*100:+.0f}%",
                        xy=(pd.Timestamp(f'{y}-06-15'), ya['asset'] / 1e4),
                        fontsize=10, color='#d62728', alpha=0.85)

    ax.set_title('黄金坑组合模拟资金曲线（100万 / 10份 / 2023-01-01 ~ 2026-08-28 / ' + delay_txt + '，单位：万元）', fontsize=14)
    ax.set_xlabel('日期'); ax.set_ylabel('总资产（万元）')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(sim.WORKDIR, 'golden_pit_portfolio_curve.png')
    fig.savefig(out, dpi=130)
    print(f'\n已保存: {out}')
    print(f'口径B 最终 {r["final_asset"]/1e4:.0f}万 (+{r["total_ret"]*100:.0f}%)  口径A 最终 {finalA/1e4:.0f}万')
    print(f'沪深300 同期 {base[-1]/1e4:.0f}万')
    return out


if __name__ == '__main__':
    main()
