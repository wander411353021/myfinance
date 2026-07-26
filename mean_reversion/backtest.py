"""事件检测回测框架（mean_reversion）

设计原则（按"区间信号"公平验收范式）：
  - 信号判定只用「当日及以前」数据（compute_signal 本身纯因果）。
  - 评估标签只用「事后」数据（未来 N 日涨跌），不偷看未来。
  - 以「随机基准对照」衡量信号是否真有信息量：同条件下瞎挑买点，
    命中 good_zone 的概率多少？信号命中率必须显著高于随机才算有效。

指标：
  - good_zone 命中率：买入信号后 hold_days 内，价格相对入场价上涨 >= good_zone_pct 的比例。
  - 召回率(recall)：全样本中所有 good_zone 区间，有多少被某个前置买入信号覆盖。
  - 随机基准：随机抽取同等数量买点，good_zone 命中率与平均收益。
  - 交易视角(参考)：T+1 开盘入场、止损/止盈/持有退出，胜率、平均收益、最大单笔回撤。

用法：
  from mean_reversion.backtest import run_backtest, run_backtest_many, summarize
  stats = run_backtest(df, code="sz000001", hold_days=10)
  print(summarize(stats))
"""

import numpy as np
import pandas as pd

from .fuser import compute_signal


def run_backtest(
    df: pd.DataFrame,
    code: str = "",
    reg_window: int = 120,
    energy_window: int = 10,
    hold_days: int = 10,
    take_target: float = 0.08,
    stop_pct: float = -0.08,
    good_zone_pct: float = 0.05,
    n_random: int = 300,
    min_confidence: int = 2,
    verbose: bool = False,
    **kwargs,
):
    """对单只股票日线做事件检测回测。

    Parameters
    ----------
    df : pd.DataFrame  必须含 close/high/low/open/volume（open 用于 T+1 入场）
    hold_days : int    持有窗口（交易日），good_zone 与退出都看这个窗口
    take_target / stop_pct : 止盈/止损比例（交易视角参考）
    good_zone_pct : good_zone 阈值（区间信号命中标准）
    n_random : 随机基准抽样的买点数量
    min_confidence : 仅对 confidence >= 该值的买点回测（过滤噪声关注）
    """
    closes = df["close"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    opens = df["open"].values.astype(float) if "open" in df.columns else closes
    n = len(df)
    min_i = max(reg_window, 21) + 1
    if n < min_i + 2:
        return {"code": code, "n_signals": 0, "error": "数据不足"}

    signal_days = set()
    trades = []
    for i in range(min_i, n - 1):
        res = compute_signal(df.iloc[: i + 1], code=code,
                             reg_window=reg_window, energy_window=energy_window, **kwargs)
        if res.signal != "buy" or res.confidence < min_confidence:
            continue
        signal_days.add(i)
        entry_idx = i + 1
        entry = opens[entry_idx]
        if entry <= 0:
            continue
        end = min(entry_idx + hold_days, n)
        win_high = highs[entry_idx:end]
        win_low = lows[entry_idx:end]

        gz_hit = bool(np.any(win_high >= entry * (1 + good_zone_pct)))
        stop_trig = bool(np.any(win_low <= entry * (1 + stop_pct)))
        tgt_trig = bool(np.any(win_high >= entry * (1 + take_target)))

        if stop_trig and not tgt_trig:
            exit_price = entry * (1 + stop_pct)
            exit_reason = "stop"
        elif tgt_trig:
            exit_price = entry * (1 + take_target)
            exit_reason = "target"
        else:
            exit_price = closes[end - 1]
            exit_reason = "hold"
        ret = exit_price / entry - 1.0

        trades.append({
            "code": code, "signal_day": i, "entry_day": entry_idx,
            "entry": entry, "exit": exit_price, "ret": ret,
            "gz_hit": gz_hit, "exit_reason": exit_reason,
            "confidence": res.confidence, "z": res.z_residual,
            "energy": res.energy_score,
        })

    # ---- 随机基准 ----
    rng = np.random.default_rng(20240724)
    cand = np.arange(min_i, n - 1)
    if len(cand) > 0:
        k = min(n_random, len(cand))
        rand_entries = rng.choice(cand, size=k, replace=False)
    else:
        rand_entries = np.array([], dtype=int)
    rand_hits = 0
    rand_rets = []
    for entry_idx in rand_entries:
        entry = opens[entry_idx]
        if entry <= 0:
            continue
        end = min(entry_idx + hold_days, n)
        gz = bool(np.any(highs[entry_idx:end] >= entry * (1 + good_zone_pct)))
        rand_hits += int(gz)
        rand_rets.append(closes[end - 1] / entry - 1.0)

    # ---- 召回率：所有 good_zone 区间有多少被前置信号覆盖 ----
    gz_days = []
    for j in range(min_i, n - hold_days):
        lo = lows[j]
        if lo <= 0:
            continue
        if np.any(highs[j + 1: j + 1 + hold_days] >= lo * (1 + good_zone_pct)):
            gz_days.append(j)
    recall = 0.0
    if gz_days:
        covered = sum(
            1 for j in gz_days
            if any(s in signal_days for s in range(max(min_i, j - hold_days), j + 1))
        )
        recall = covered / len(gz_days)

    # ---- 汇总 ----
    rets = [t["ret"] for t in trades]
    gz = [t["gz_hit"] for t in trades]
    wins = [r for r in rets if r > 0]
    max_dd = min(rets) if rets else 0.0

    stats = {
        "code": code,
        "n": n,
        "n_signals": len(trades),
        "min_confidence": min_confidence,
        "hold_days": hold_days,
        "good_zone_pct": good_zone_pct,
        # 信号表现
        "gz_hit_rate": (np.mean(gz) if gz else 0.0),
        "win_rate": (len(wins) / len(rets) if rets else 0.0),
        "avg_ret": (np.mean(rets) if rets else 0.0),
        "median_ret": (np.median(rets) if rets else 0.0),
        "max_drawdown_trade": max_dd,
        "recall": recall,
        "n_good_zones": len(gz_days),
        # 随机基准
        "rand_n": len(rand_entries),
        "rand_gz_hit_rate": (rand_hits / len(rand_entries) if len(rand_entries) else 0.0),
        "rand_avg_ret": (np.mean(rand_rets) if rand_rets else 0.0),
        # 信息量增量
        "gz_excess": ((np.mean(gz) if gz else 0.0) -
                      (rand_hits / len(rand_entries) if len(rand_entries) else 0.0)),
        "ret_excess": ((np.mean(rets) if rets else 0.0) -
                       (np.mean(rand_rets) if rand_rets else 0.0)),
        "trades": trades,
    }
    if verbose:
        print(summarize(stats))
    return stats


def run_backtest_many(results: list, **kwargs):
    """对多只股票结果(list of stats dict)做聚合。

    results 来自多次 run_backtest 的返回值（取不带 trades 的轻量版也可）。
    返回聚合 stats：总信号数、加权 good_zone 命中率、随机基准、召回率。
    """
    valid = [r for r in results if r and r.get("n_signals", 0) > 0]
    if not valid:
        return {"n_stocks": len(results), "n_signals": 0}
    tot_sig = sum(r["n_signals"] for r in valid)
    w = tot_sig
    gz = sum(r["gz_hit_rate"] * r["n_signals"] for r in valid) / w if w else 0
    ret = sum(r["avg_ret"] * r["n_signals"] for r in valid) / w if w else 0
    win = sum(r["win_rate"] * r["n_signals"] for r in valid) / w if w else 0
    rand_gz = np.mean([r["rand_gz_hit_rate"] for r in valid])
    rand_ret = np.mean([r["rand_avg_ret"] for r in valid])
    recall = np.mean([r["recall"] for r in valid if r.get("n_good_zones")])
    return {
        "n_stocks": len(results),
        "n_valid_stocks": len(valid),
        "n_signals": tot_sig,
        "gz_hit_rate": gz,
        "win_rate": win,
        "avg_ret": ret,
        "recall": recall,
        "rand_gz_hit_rate": rand_gz,
        "rand_avg_ret": rand_ret,
        "gz_excess": gz - rand_gz,
        "ret_excess": ret - rand_ret,
    }


def summarize(stats: dict) -> str:
    """人类可读汇总。"""
    if stats.get("error"):
        return f"[{stats.get('code','?')}] 跳过：{stats['error']}"
    lines = []
    lines.append(f"=== {stats['code']}  样本 {stats['n']} 日 / 信号 {stats['n_signals']} 个 "
                 f"(conf>={stats['min_confidence']}, 持有 {stats['hold_days']} 日) ===")
    lines.append(f"  good_zone 命中率 : {stats['gz_hit_rate']*100:5.1f}%   "
                 f"(随机基准 {stats['rand_gz_hit_rate']*100:5.1f}%)   "
                 f"超额 {stats['gz_excess']*100:+5.1f}%")
    lines.append(f"  交易胜率         : {stats['win_rate']*100:5.1f}%   "
                 f"平均收益 {stats['avg_ret']*100:+5.2f}%   "
                 f"(随机 {stats['rand_avg_ret']*100:+5.2f}%)")
    lines.append(f"  召回率           : {stats['recall']*100:5.1f}%   "
                 f"(全样本 good_zone 区间 {stats['n_good_zones']} 个)")
    lines.append(f"  最大单笔回撤     : {stats['max_drawdown_trade']*100:5.2f}%")
    return "\n".join(lines)
