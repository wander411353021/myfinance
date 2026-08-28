# -*- coding: utf-8 -*-
"""黄金坑组合模拟（100万 / 10份 / 2023-01-01 起 / 持有20日）

规则（与定版策略一致）：
- 信号 = panic_reversal.detect_golden_pit 规则F（z<-1.5 + 快启动 lch-b<=5 + 出坑确认）
- 买入 = 出坑日(lch)收盘价；资金管理 = 10 个独立子账户，空仓时用该账户全部资金买入单票
- 卖出 = lch+20 交易日收盘价（到期强制），数据末尾未到期按最后收盘估值
- 手续费 = 佣金万2.5(最低5元) + 卖出印花税 0.05%
- 数据 = .cache_kline（tdx 前复权，2022-02 至 2026-08-28）

本模块提供可复用 run_sim(stocks) 供组合/压力测试调用。
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
END_DATE = pd.Timestamp('2026-08-28')


def load(sym):
    f = os.path.join(CACHE, f'{sym}.npy')
    if not os.path.exists(f):
        return None
    return np.load(f, allow_pickle=True).item()


def collect_events(sym, buy_delay=0):
    """返回该股票所有规则F信号的 买入/卖出事件。
    buy_delay=0: 出坑日(lch)收盘买入(理论口径,实盘买不到)
    buy_delay=1: 次日(lch+1)收盘买入(严格实盘口径), 卖出=lch+1+20(lch+21)"""
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
        bi = lch + buy_delay
        if bi >= n:
            continue
        bdate = pd.to_datetime(int(ts[bi]), unit='s')
        evs.append(('buy', bdate, sym, lch, bi))
        si = bi + 20
        if si < n:
            sdate = pd.to_datetime(int(ts[si]), unit='s')
            evs.append(('sell', sdate, sym, lch, si))
        else:
            evs.append(('sell_eod', None, sym, lch, n - 1))
    return evs


def run_sim(stocks, nfunds=NFUNDS, unit=UNIT, start=START_DATE, end=END_DATE, buy_delay=0,
            stop_loss=0.0, pos_ratio=1.0, market_filter=False):
    """对给定股票列表执行完整模拟，返回结果字典。
    模型：10 个独立子账户复利——空仓时用该账户全部资金买入单票，20日后卖出资金回到该账户。
    buy_delay: 0=出坑日收盘买入(理论) / 1=次日收盘买入(严格实盘), 卖出=买入后20交易日收盘。
    stop_loss: 持有期内跌破买入价*(1-stop_loss) 提前止损离场(0=不止损)
    pos_ratio: 每次买入用子账户资金比例(1.0=满仓, 0.7=7成仓)
    market_filter: 沪深300 20日线下方(空头)时跳过买入信号(大盘过滤)"""
    """对给定股票列表执行完整模拟，返回结果字典。
    模型：10 个独立子账户复利——空仓时用该账户全部资金买入单票，20日后卖出资金回到该账户。
    buy_delay: 0=出坑日收盘买入(理论) / 1=次日收盘买入(严格实盘), 卖出=买入后20交易日收盘。"""
    events = []
    for sym in stocks:
        events += collect_events(sym, buy_delay=buy_delay)
    events.sort(key=lambda e: (e[1] is not None, e[1] if e[1] is not None else pd.Timestamp.max))

    funds = [unit] * nfunds
    positions = {}
    trades = []
    skipped = 0
    snapshot = []

    # 缓存每只股票 (ts_np, close) 用于按日期市价估值（避免反复读文件）
    _kl = {}

    def _kline(sym):
        if sym not in _kl:
            d = load(sym)
            if d is None:
                return None
            ts = pd.to_datetime(d['ts'].astype('int64'), unit='s').values
            _kl[sym] = (ts, d['close'].astype(float))
        return _kl[sym]

    def price_at(sym, cur_date):
        """sym 在 cur_date（含）之前的最后收盘价（无未来）"""
        kl = _kline(sym)
        if kl is None:
            return 0.0
        ts, close = kl
        i = int(np.searchsorted(ts, np.datetime64(pd.Timestamp(cur_date)), side='right') - 1)
        if i < 0:
            i = 0
        return float(close[i])

    def total_asset(cur_date):
        """按当前日期市价估值全部持仓（反映真实浮盈浮亏）"""
        mv = 0.0
        for (sym, lch), pos in positions.items():
            mv += pos['qty'] * price_at(sym, cur_date)
        return sum(funds) + mv

    # 大盘过滤数据(沪深300 20日线)
    _hs = None
    if market_filter:
        d = load('sh000300')
        if d is not None:
            ts300 = pd.to_datetime(d['ts'].astype('int64'), unit='s').values
            cl300 = d['close'].astype(float)
            ma20 = pd.Series(cl300).rolling(20).mean().values
            _hs = (ts300, cl300, ma20)

    def hs_bear(date):
        if _hs is None:
            return False
        ts300, cl300, ma20 = _hs
        i = int(np.searchsorted(ts300, np.datetime64(pd.Timestamp(date)), side='right') - 1)
        if i < 20 or i >= len(cl300):
            return False
        return bool(cl300[i] < ma20[i])

    n_buy_signals = 0
    for ev in events:
        typ, date, sym, lch, idx = ev

        # 止损检查(每次事件前, 用当前日期市价判定; 不早于买入日)
        if stop_loss > 0 and positions:
            for key in list(positions.keys()):
                pos = positions[key]
                if date <= pos['buy_date']:
                    continue
                cur = price_at(key[0], date)
                if cur > 0 and cur <= pos['buy_price'] * (1 - stop_loss):
                    proceeds = cur * pos['qty'] * (1 - COMMISSION - STAMP)
                    funds[pos['fund_id']] += proceeds
                    trades.append({'sym': key[0], 'buy_date': pos['buy_date'], 'sell_date': date,
                                   'buy_price': pos['buy_price'], 'sell_price': cur,
                                   'ret': cur / pos['buy_price'] - 1,
                                   'invest': pos['cost'] / (1 + COMMISSION)})
                    del positions[key]
                    snapshot.append((date, total_asset(date)))

        if typ == 'buy':
            if date < start:
                continue
            if market_filter and hs_bear(date):
                continue
            n_buy_signals += 1
            if len(positions) >= nfunds:
                skipped += 1
                continue
            busy = {p['fund_id'] for p in positions.values()}
            free = [i for i in range(nfunds) if i not in busy]
            if not free:
                skipped += 1
                continue
            fid = max(free, key=lambda i: funds[i])
            d = load(sym)
            price = float(d['close'][idx])
            amount = funds[fid] * 0.99 * pos_ratio
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
            snapshot.append((date, total_asset(date)))
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
            snapshot.append((date, total_asset(date)))

    snapshot.append((end, total_asset(end)))
    open_pos = list(positions.items())
    final_asset = total_asset(end)
    rets = np.array([t['ret'] for t in trades]) if trades else np.array([])
    snap_df = pd.DataFrame(snapshot, columns=['date', 'asset']).reset_index(drop=True)  # 保持事件顺序, 同日不乱序

    # 年度资产曲线
    year_assets = {}
    snap_df['year'] = snap_df['date'].dt.year
    prev = float(nfunds * unit)
    for y in sorted(snap_df['year'].unique()):
        ye = snap_df[snap_df['year'] == y]['asset'].iloc[-1]
        year_assets[int(y)] = {'asset': float(ye), 'ret': float(ye / prev - 1)}
        prev = float(ye)

    # 最大回撤
    peak = snap_df['asset'].cummax()
    max_dd = float(((snap_df['asset'] - peak) / peak).min())

    return {
        'n_stocks': len(stocks), 'n_signals': n_buy_signals, 'n_trades': len(trades),
        'n_open': len(open_pos), 'skipped': skipped,
        'win_rate': float((rets > 0).mean() * 100) if len(rets) else 0.0,
        'final_asset': float(final_asset),
        'total_ret': float(final_asset / (nfunds * unit) - 1),
        'funds': [float(f) for f in funds],
        'trades': trades, 'snapshot': snap_df, 'year_assets': year_assets,
        'max_dd': max_dd,
    }


def main():
    pool_file = sys.argv[1] if len(sys.argv) > 1 else 'stock_pool_1000.txt'
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    buy_delay = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    delay_txt = '次日买入(严格实盘口径)' if buy_delay else '出坑日收盘买入(理论口径)'
    stocks = [l.strip().split(',')[0] for l in open(os.path.join(WORKDIR, pool_file)) if l.strip()][:top]
    t0 = time.time()
    # 预扫描显示进度
    n_ev = 0
    for i, sym in enumerate(stocks):
        n_ev += len(collect_events(sym, buy_delay=buy_delay))
        if (i + 1) % 200 == 0:
            print(f'  扫描 {i+1}/{len(stocks)} 事件{n_ev} {time.time()-t0:.0f}s', flush=True)
    print(f'扫描完成: {len(stocks)}只, 规则F信号事件 {n_ev} 个 ({time.time()-t0:.0f}s)')
    r = run_sim(stocks, buy_delay=buy_delay)
    print(f'2023-01-01 后买入信号: {r["n_signals"]} 个')
    print('\n' + '=' * 60)
    print(f'黄金坑组合模拟: 100万 / {NFUNDS}份子账户复利 / {START_DATE.date()} ~ 2026-08-28')
    print(f'买卖口径: {delay_txt} + 持有20交易日收盘卖出')
    print('=' * 60)
    print(f'完成交易: {r["n_trades"]} 笔   胜率: {r["win_rate"]:.1f}%')
    realized_pnl = sum(t['ret'] * t['invest'] for t in r['trades'])
    print(f'已实现收益(实际投入): {realized_pnl:+,.0f} 元')
    open_mv = r['final_asset'] - sum(r['funds'])  # 未平仓市值 = 总资产 - 各子账户现金
    print(f'未平仓持仓: {r["n_open"]} 份  市值: {open_mv:,.0f} 元')
    print(f'各子账户资金: {[int(f) for f in r["funds"]]} 元')
    print(f'最终总资产: {r["final_asset"]:,.0f} 元')
    print(f'总收益率: {r["total_ret"]:+.2%}')
    print(f'满仓跳过信号: {r["skipped"]} 个')
    print(f'最大回撤: {r["max_dd"]*100:.1f}%')
    if r['trades']:
        df = pd.DataFrame(r['trades'])
        df['year'] = df['sell_date'].dt.year
        print('\n分年度已实现:')
        for y, g in df.groupby('year'):
            rr = g['ret'].values
            print(f'  {y}: {len(g)}笔 胜率{(rr > 0).mean()*100:.0f}% 均值{rr.mean()*100:+.1f}% 收益{(g["ret"]*g["invest"]).sum():+,.0f}元')
    print('\n分年度总资产(口径B 子账户复利):')
    print(f'  期初 {START_DATE.date()}: {1000000:,} 元')
    for y in sorted(r['year_assets']):
        ya = r['year_assets'][y]
        print(f'  {y}年末: {ya["asset"]:>12,.0f} 元   年度收益 {ya["ret"]:+.1%}')
    try:
        idx = load('sh000300')
        if idx is not None:
            ts = pd.to_datetime(idx['ts'].astype('int64'), unit='s')
            m = (ts >= START_DATE) & (ts <= END_DATE)
            c300 = idx['close'][m]
            if len(c300) > 1:
                print(f'沪深300 同期: {c300[-1]/c300[0]-1:+.2%}')
    except Exception as e:
        print(f'沪深300基准不可用: {e}')


if __name__ == '__main__':
    main()
