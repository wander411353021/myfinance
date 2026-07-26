"""回测框架自测：合成含多次「急跌→反弹」good_zone 的序列，验证事件检测回测可跑通，
且信号 good_zone 命中率应高于随机基准（信息量验证）。"""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mean_reversion.backtest import run_backtest, summarize


def make_df(closes, vols):
    n = len(closes)
    opens = np.empty(n); highs = np.empty(n); lows = np.empty(n)
    opens[0] = closes[0]; opens[1:] = closes[:-1]
    for i in range(n):
        o, c = opens[i], closes[i]
        hi = max(o, c) * 1.015
        lo = min(o, c) * 0.985
        highs[i], lows[i] = hi, lo
    # 让反弹日 high 足够高以触发 good_zone（>=+5%）
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame({"date": dates, "open": opens, "high": highs,
                         "low": lows, "close": closes, "volume": vols})


def build_episode_series(seed=7, n=620):
    np.random.seed(seed)
    closes = 20.0 + np.random.normal(0, 0.1, n)   # 平稳基准（回归线稳定）
    vols = np.random.uniform(1e5, 2e5, n)
    for start in [70, 160, 250, 340, 430, 520]:
        if start + 15 >= n:
            continue
        bot = closes[start] * 0.85                  # 急跌 15%
        closes[start:start + 5] = np.linspace(closes[start], bot, 5)
        closes[start + 5:start + 13] = np.linspace(bot, bot * 1.18, 8)  # 反弹 +18%
        vols[start + 3:start + 6] *= 0.4            # 跌尾缩量 → 能量衰竭确认
    return make_df(closes, vols)


def main():
    df = build_episode_series()
    stats = run_backtest(df, code="SYN", hold_days=10, good_zone_pct=0.05,
                         take_target=0.08, stop_pct=-0.08, n_random=300, verbose=True)
    print()
    print(f"信号 good_zone 命中率 = {stats['gz_hit_rate']*100:.1f}%  "
          f"随机基准 = {stats['rand_gz_hit_rate']*100:.1f}%  "
          f"超额 = {stats['gz_excess']*100:+.1f}%")
    print(f"召回率 = {stats['recall']*100:.1f}%  (good_zone 区间 {stats['n_good_zones']} 个)")
    if stats["n_signals"] == 0:
        print("⚠️ 未产生任何信号，请检查合成序列/参数")
        sys.exit(1)
    if stats["gz_excess"] <= 0:
        print("⚠️ 信号未跑赢随机基准（合成数据下预期应跑赢；若失败需复查算法）")
        sys.exit(1)
    print("\n✅ 回测框架自测通过（信号命中率 > 随机基准）")


if __name__ == "__main__":
    main()
