# -*- coding: utf-8 -*-
"""黄金坑组合模拟（100万 / 10份 / 2023-01-01 起 / 持有20日）

规则（与定版策略一致）：
- 信号 = panic_reversal.detect_golden_pit 规则F（z<-1.5 + 快启动 lch-b<=5 + 出坑确认）
- 买入 = 出坑日(lch)收盘价，每份固定 10 万（100万/10份）
- 卖出 = lch+20 交易日收盘价（到期强制），数据末尾未到期按最后收盘估值
- 资金管理 = 最多同时持有 10 份（无空闲份额则跳过新信号，计入 skipped）
- 手续费 = 佣金万2.5(最低5元) + 卖出印花税 0.05%
- 数据 = .cache_kline（tdx 前复权，2022-02 至 2026-08-28）
用法: python3 golden_pit_portfolio_sim.py [pool_file] [top_n]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression

WORKDIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(WORKDIR, '.cache_kline')
COMMISSION = 0.00025   # 佣金 万2.5
MIN_COMM = 5.0
STAMP = 0.0005         # 印花税 0.05%（卖出）
UNIT = 100000.0        # 每份 10 万
NFUNDS = 10            # 10 份
START_DATE = pd.Timestamp('2023-01-01')


def load(sym):
    f = os.path.join(CACHE, f'{sym}.npy')
    if not os.path.exists(f):
        return None
    return np.load(f, allow_pickle=True).item()


def collect_events(sym):
    """返回该股票所有规则F信号的 买入/卖出事件"""
    d = load(sym)
    if d is None:
        return []
    c = d['close'].astype(float); ts = d['ts']; n = len(c)
    reg, _ = compute_rolling_regression(c, window=250, use_log=True)
    pits = pr.detect_golden_pit(c, reg)
    evs = []
    for s, b, lch in pits:
        if lch is None or lch - b > 5:
            continue  # 规则F: 快启动<=5
        bdate = pd.to_datetime(int(ts[lch]), unit='s')
        evs.append(('buy', bdate, sym, lch, lch))
        if lch + 20 < n:
            sdate = pd.to_datetime(int(ts[lch + 20]), unit='s')
            evs.append(('sell', sdate, sym, lch, lch + 20))
        else:
            evs.append(('sell_eod', None, sym, lch, n - 1))
    return evs


def main():
    pool_file = sys.argv[1] if len(sys.argv) > 1 else 'stock_pool_1000.txt'
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    stocks = [l.strip().split(',')[0] for l in open(os.path.join(WORKDIR, pool_file)) if l.strip()][:top]

    t0 = time.time()
    events = []
    for i, sym in enumerate(stocks):
        events += collect_events(sym)
        if (i + 1) % 200 == 0:
            print(f'  扫描 {i+1}/{len(stocks)} 事件{len(events)} {time.time()-t0:.0f}s', flush=True)
    print(f'扫描完成: {len(stocks)}只, 规则F信号事件 {len(events)} 个 ({time.time()-t0:.0f}s)')

    # 按日期排序；sell_eod 排最后
    events.sort(key=lambda e: (e[1] is not None, e[1] if e[1] is not None else pd.Timestamp.max))
    buy_evs = [e for e in events if e[0] == 'buy']
    buy_evs = [e for e in buy_evs if e[1] >= START_DATE]
    print(f'2023-01-01 后买入信号: {len(buy_evs)} 个')

    # ---- 模拟：10 个独立子账户复利模型 ----
    # 每个子账户 10 万；空仓时用该账户全部可用资金买入单票，20日后卖出资金回到该账户（钱滚钱）
    funds = [UNIT] * NFUNDS
    positions = {}       # key=(sym,lch) -> dict(fund_id, qty, cost, buy_price, buy_date, close_idx)
    trades = []          # 已实现交易
    skipped = 0
    snapshot = []        # (date, total_asset)

    def total_asset():
        mv = 0.0
        for (sym, lch), pos in positions.items():
            d = load(sym)
            mv += pos['qty'] * float(d['close'][pos['close_idx']])
        return sum(funds) + mv

    # 处理事件
    for ev in events:
        typ, date, sym, lch, idx = ev
        if typ == 'buy':
            if date < START_DATE:
                continue
            if len(positions) >= NFUNDS:
                skipped += 1
                continue
            # 选一个空仓且资金最多的子账户
            busy = {p['fund_id'] for p in positions.values()}
            free = [i for i in range(NFUNDS) if i not in busy]
            if not free:
                skipped += 1
                continue
            fid = max(free, key=lambda i: funds[i])
            d = load(sym)
            price = float(d['close'][idx])
            amount = funds[fid] * 0.99  # 留手续费余量
            qty = int(amount / price / 100) * 100
            if qty <= 0:
                continue
            comm = max(price * qty * COMMISSION, MIN_COMM)
            cost = price * qty + comm
            if cost > funds[fid]:
                continue
            funds[fid] -= cost
            positions[(sym, lch)] = {'fund_id': fid, 'qty': qty, 'cost': cost,
                                     'buy_price': price, 'buy_date': date, 'close_idx': idx}
        elif typ == 'sell':
            key = (sym, lch)
            if key not in positions:
                continue
            pos = positions.pop(key)
            d = load(sym)
            price = float(d['close'][idx])
            proceeds = price * pos['qty'] * (1 - COMMISSION - STAMP)
            funds[pos['fund_id']] += proceeds
            trades.append({'sym': sym, 'buy_date': pos['buy_date'], 'sell_date': date,
                           'buy_price': pos['buy_price'], 'sell_price': price,
                           'ret': price / pos['buy_price'] - 1,
                           'invest': pos['cost'] / (1 + COMMISSION)})
            snapshot.append((date, total_asset()))
        # sell_eod: 数据末尾未到期, 保持持仓, 最后统一估值

    # 未平仓持仓: 按最后收盘估值
    open_pos = list(positions.items())
    final_asset = total_asset()

    # 已实现统计
    rets = np.array([t['ret'] for t in trades])
    print('\n' + '=' * 60)
    print(f'黄金坑组合模拟: 100万 / {NFUNDS}份子账户复利 / {START_DATE.date()} ~ 2026-08-28')
    print('=' * 60)
    print(f'完成交易: {len(trades)} 笔   胜率: {(rets > 0).mean() * 100:.1f}%')
    # 已实现收益(按实际投入,子账户已复利)
    realized_pnl = sum(t['ret'] * t['invest'] for t in trades)
    print(f'已实现收益(实际投入): {realized_pnl:+,.0f} 元')
    open_mv = sum(pos['qty'] * float(load(sym)['close'][pos['close_idx']]) for (sym, lch), pos in open_pos)
    print(f'未平仓持仓: {len(open_pos)} 份  市值: {open_mv:,.0f} 元')
    print(f'各子账户资金: {[int(f) for f in funds]} 元')
    print(f'最终总资产: {final_asset:,.0f} 元')
    print(f'总收益率: {final_asset / 1000000 - 1:+.2%}')
    print(f'满仓跳过信号: {skipped} 个')

    # 年度已实现
    if trades:
        df = pd.DataFrame(trades)
        df['year'] = df['sell_date'].dt.year
        print('\n分年度已实现:')
        for y, g in df.groupby('year'):
            r = g['ret'].values
            print(f'  {y}: {len(g)}笔 胜率{(r > 0).mean()*100:.0f}% 均值{r.mean()*100:+.1f}% 收益{(g["ret"]*g["invest"]).sum():+,.0f}元')

    # 最大回撤（快照法）
    if snapshot:
        snap = pd.DataFrame(snapshot, columns=['date', 'asset'])
        snap = snap.sort_values('date').reset_index(drop=True)
        peak = snap['asset'].cummax()
        dd = (snap['asset'] - peak) / peak
        print(f'\n最大回撤(按已实现快照): {dd.min()*100:.1f}%')

    # 沪深300 基准
    try:
        idx = load('sh000300')
        if idx is not None:
            ts = pd.to_datetime(idx['ts'].astype('int64'), unit='s')
            m = (ts >= START_DATE) & (ts <= pd.Timestamp('2026-08-28'))
            c300 = idx['close'][m]
            if len(c300) > 1:
                print(f'沪深300 同期: {c300[-1]/c300[0]-1:+.2%}')
    except Exception as e:
        print(f'沪深300基准不可用: {e}')


if __name__ == '__main__':
    main()
