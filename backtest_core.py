# -*- coding: utf-8 -*-
"""通用回测框架核心 — Backtest Engine(显示英文,注释中文)。

设计:
  - 引擎通用:任何策略插件生成"信号列表",引擎负责买卖模拟/资金曲线/指标/可视化。
  - 策略插件协议:实现 generate_signals(df) -> [Signal, ...]
    Signal = {idx: 信号在日线中的位置, date: 'YYYYMMDD', meta: {...}}
  - 卖出规则(可配):止损 stop / 止盈 take / 到期 horizon / 移动止盈 trailing
  - 指标:总收益/年化/最大回撤/胜率/盈亏比/夏普/平均持仓
  - 可视化:买卖点K线 + 收益曲线 + 回撤曲线 + 月度收益热图

用法:
    from backtest_core import BacktestEngine
    from strategies.golden_pit import GoldenPitStrategy
    eng = BacktestEngine(stop=-0.15, take=0.30, horizon=60)
    sigs = GoldenPitStrategy().generate_signals(df)
    trades, equity, eq_dates = eng.run(df, sigs)
    eng.report(df, trades, equity, eq_dates, name='600016', save_path='result/bt.png')
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class BacktestEngine:
    """通用回测引擎:买卖模拟 + 指标 + 可视化(所有文案英文,注释中文)。"""

    def __init__(self, stop=-0.15, take=0.30, horizon=60, trailing=None,
                 commission=0.0, slippage=0.0, capital=1.0):
        # 卖出规则:stop 止损比例(负),take 止盈比例(正),horizon 到期交易日,trailing 移动止盈比例(如 0.10)
        self.stop = stop
        self.take = take
        self.horizon = horizon
        self.trailing = trailing
        self.commission = commission  # 单边费率
        self.slippage = slippage      # 滑点(比例)
        self.capital = capital

    # ── 核心:单只交易模拟 ──
    def _simulate(self, closes, buy_idx, n):
        """从 buy_idx 买入,按卖出规则返回 (sell_idx, sell_price, reason)。"""
        buy_px = closes[buy_idx] * (1 + self.slippage)
        peak = buy_px
        for i in range(buy_idx + 1, min(buy_idx + self.horizon + 1, n)):
            px = closes[i]
            peak = max(peak, px)
            # 移动止盈:从峰值回落 trailing 比例离场
            if self.trailing and (peak - px) / peak >= self.trailing:
                return i, px, 'trail'
            if px <= buy_px * (1 + self.stop):
                return i, px, 'stop'
            if self.take and px >= buy_px * (1 + self.take):
                return i, px, 'take'
        si = min(buy_idx + self.horizon, n - 1)
        return si, closes[si], 'horizon'

    def run(self, df, signals):
        """执行回测:对每个信号买入→卖出,构建交易列表与逐日资金曲线。

        trades: [{code, buy_idx, buy_date, buy_price, sell_idx, sell_date,
                  sell_price, ret, reason, meta}]
        equity: 逐日累计资金(等权,按买入日投入/卖出日结算)
        """
        c = df['close'].values.astype(float)
        dates = df['date'].dt.strftime('%Y%m%d').values
        n = len(c)
        sigs = sorted(signals, key=lambda s: s['idx'])
        trades = []
        eq = np.ones(n) * self.capital
        active = []  # 持仓中的信号(等权分配)
        for sig in sigs:
            bi = sig['idx']
            if bi >= n - 2:
                continue
            si, sp, reason = self._simulate(c, bi, n)
            if si <= bi:
                continue
            ret = sp / (c[bi] * (1 + self.slippage)) - 1 - self.commission
            trades.append({
                'code': sig.get('code', ''), 'buy_idx': bi, 'buy_date': dates[bi],
                'buy_price': c[bi], 'sell_idx': si, 'sell_date': dates[si],
                'sell_price': sp, 'ret': ret, 'reason': reason, 'meta': sig.get('meta', {}),
            })
            # 资金:该笔在 [bi, si] 期间生效(简化:结算日计入)
            eq[si:] = eq[si:] * (1 + ret)
        # 归一(避免 eq[0] 之前被乘)
        eq = eq / eq[0] if eq[0] > 0 else eq
        return trades, eq, dates

    # ── 指标 ──
    def metrics(self, trades, equity, dates):
        """关键指标:收益/年化/回撤/胜率/盈亏比/夏普/持仓。"""
        m = {}
        if not trades:
            return m
        rets = np.array([t['ret'] for t in trades])
        m['n_trades'] = len(trades)
        m['win_rate'] = float(np.mean(rets > 0))
        m['avg_win'] = float(np.mean(rets[rets > 0])) if (rets > 0).any() else 0.0
        m['avg_loss'] = float(np.mean(rets[rets < 0])) if (rets < 0).any() else 0.0
        m['profit_factor'] = float(rets[rets > 0].sum() / abs(rets[rets < 0].sum())) if (rets < 0).any() else float('inf')
        m['total_return'] = float(np.prod(1 + rets) - 1)
        m['avg_hold'] = float(np.mean([t['sell_idx'] - t['buy_idx'] for t in trades]))
        if equity is not None and len(equity) > 1:
            eq = equity
            peak = np.maximum.accumulate(eq)
            mdd = float(np.max((peak - eq) / peak))
            m['max_drawdown'] = mdd
            days = len(eq)
            years = days / 244.0
            m['annual_return'] = float(eq[-1] ** (1 / years) - 1) if years > 0 and eq[-1] > 0 else 0.0
            daily = np.diff(eq) / eq[:-1]
            if daily.std() > 0 and daily.std() > 0:
                m['sharpe'] = float(daily.mean() / daily.std() * np.sqrt(244))
            else:
                m['sharpe'] = 0.0
        return m

    # ── 可视化(英文显示) ──
    def plot(self, df, trades, equity, eq_dates, name='', save_path=None, tail_days=500):
        """四子图:K线买卖点 / 收益曲线 / 回撤曲线 / 月度收益热图。"""
        c = df['close'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        o = df['open'].values.astype(float)
        dates = df['date'].dt.strftime('%Y%m%d').values
        n = len(c)
        off = max(0, n - tail_days)
        x = np.arange(n - off)
        dtime = pd.to_datetime(dates[off:])
        xnum = mdates.date2num(dtime)  # 真实日期数值(修复:整数 x + DateFormatter → 1970)

        fig = plt.figure(figsize=(22, 16))
        gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 1, 1], width_ratios=[3, 1])
        axk = fig.add_subplot(gs[0, :])
        axeq = fig.add_subplot(gs[1, 0])
        axdd = fig.add_subplot(gs[2, 0])
        axmo = fig.add_subplot(gs[3, :])
        axsum = fig.add_subplot(gs[1:, 1])

        # K线
        for i in range(n - off):
            color = '#E53935' if c[off + i] >= o[off + i] else '#2E7D32'
            xi = xnum[i]
            axk.plot([xi, xi], [l[off + i], h[off + i]], color=color, linewidth=0.6)
            axk.add_patch(plt.Rectangle((xi - 0.3, min(o[off + i], c[off + i])), 0.6,
                                        abs(c[off + i] - o[off + i]) or 1e-6, color=color))
        for t in trades:
            bi, si = t['buy_idx'] - off, t['sell_idx'] - off
            if si < 0 or bi >= n - off:
                continue
            bi = max(0, bi)
            col = '#FF8F00' if t['meta'].get('super') else '#1976D2'
            axk.plot(xnum[bi], t['buy_price'], '^', color=col, markersize=11, zorder=8)
            axk.plot(xnum[min(si, n - off - 1)], t['sell_price'], 'v', color='#7B1FA2', markersize=11, zorder=8)
            axk.annotate(f"{t['ret']:+.0%}({t['reason']})" + ('★' if t['meta'].get('super') else ''),
                         (min(si, n - off - 1), t['sell_price']), textcoords='offset points',
                         xytext=(2, 8), fontsize=7, color='#7B1FA2')
        axk.set_title(f'{name} - Buy/Sell Marks (▲buy ▼sell, orange=super)', fontsize=11)
        axk.grid(True, alpha=0.2)

        # 收益曲线 + 回撤
        if equity is not None:
            eqw = equity[off:]
            axeq.plot(xnum, eqw, color='#1565C0', lw=1.8)
            axeq.axhline(1.0, color='#999', lw=0.8, ls='--')
            axeq.fill_between(xnum, 1.0, eqw, where=eqw >= 1.0, color='#E53935', alpha=0.25)
            axeq.fill_between(xnum, 1.0, eqw, where=eqw < 1.0, color='#2E7D32', alpha=0.25)
            axeq.set_title(f'Equity Curve - final {eqw[-1]-1:+.1%}', fontsize=10)
            axeq.grid(True, alpha=0.2)
            peak = np.maximum.accumulate(eqw)
            dd = (eqw - peak) / peak
            axdd.fill_between(xnum, 0, dd * 100, color='#8E24AA', alpha=0.4)
            axdd.set_title(f'Drawdown - max {np.min(dd)*100:.1f}%', fontsize=10)
            axdd.grid(True, alpha=0.2)
            axdd.set_ylabel('%')
            # 月度收益热图
            eqs = pd.Series(equity, index=pd.to_datetime(dates))
            monthly = eqs.resample('ME').last().pct_change().dropna()
            if len(monthly) > 0:
                mm = pd.DataFrame({'ret': monthly.values * 100}, index=monthly.index)
                mm['year'] = mm.index.year
                mm['month'] = mm.index.month
                pivot = mm.pivot(index='year', columns='month', values='ret')
                im = axmo.imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=-20, vmax=20)
                axmo.set_yticks(range(len(pivot.index)), pivot.index)
                axmo.set_xticks(range(12), [f'{i+1}M' for i in range(12)], fontsize=7)
                for (jj, ii), val in np.ndenumerate(pivot.values):
                    if np.isfinite(val):
                        axmo.text(ii, jj, f'{val:.0f}', ha='center', va='center', fontsize=6)
                axmo.set_title('Monthly Returns (%)', fontsize=10)
                fig.colorbar(im, ax=axmo, fraction=0.02)

        # 摘要卡片
        m = self.metrics(trades, equity, dates)
        axsum.axis('off')
        lines = [f'=== Summary ===', f'Trades: {m.get("n_trades", 0)}',
                 f'Win rate: {m.get("win_rate", 0):.1%}',
                 f'Profit factor: {m.get("profit_factor", 0):.2f}',
                 f'Total return: {m.get("total_return", 0):+.1%}',
                 f'Annual return: {m.get("annual_return", 0):+.1%}',
                 f'Max drawdown: {m.get("max_drawdown", 0):.1%}',
                 f'Sharpe: {m.get("sharpe", 0):.2f}',
                 f'Avg win: {m.get("avg_win", 0):+.1%}',
                 f'Avg loss: {m.get("avg_loss", 0):+.1%}',
                 f'Avg hold: {m.get("avg_hold", 0):.0f}d',
                 ]
        axsum.text(0.05, 0.95, '\n'.join(lines), va='top', ha='left', fontsize=10,
                   family='monospace', bbox=dict(boxstyle='round', facecolor='#F5F5F5'))

        for ax in [axk, axeq, axdd]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=30)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=110)
            plt.close(fig)
            return save_path
        return fig
