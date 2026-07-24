"""
均值回归算法 — 单股验证 / 板块扫描演示

用法：
  # 单股测试
  "/d/ProgramData/miniconda3/envs/chip_analyzer/python.exe" demo_mean_reversion.py --code sz000032 --date 20260723

  # 板块扫描
  "/d/ProgramData/miniconda3/envs/chip_analyzer/python.exe" demo_mean_reversion.py --block 人工智能 --date 20260723
"""

import sys, os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdx_quant import get_daily_kline_from_tdx
from mean_reversion.fuser import compute_signal


def plot_diagnostic(df: pd.DataFrame, result, code: str, save_path: str = None):
    """绘制诊断图：价格 + 回归线 + 残差 + 能量"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

    dates = df["date"].values
    closes = df["close"].values
    ax1, ax2, ax3 = axes

    # --- 子图1：价格 + 回归线 ---
    ax1.plot(dates, closes, "b-", linewidth=1.2, label="Close", alpha=0.8)
    # 回归线（取最近 reg_window 天）
    reg_window = 60
    if len(closes) >= reg_window:
        t = np.arange(reg_window)
        recent_closes = closes[-reg_window:]
        recent_dates = dates[-reg_window:]
        preds = result.details["reg_a"] * t + result.details["reg_b"]
        ax1.plot(recent_dates, preds, "r--", linewidth=1.5, label="Regression (60d)")
        # 标记当日预测点
        ax1.scatter([recent_dates[-1]], [result.details["predicted"]],
                    color="red", s=60, zorder=5, marker="o")

    # 标记信号
    ylim = ax1.get_ylim()
    if result.signal == "buy" and result.confidence >= 3:
        ax1.scatter([dates[-1]], [closes[-1]], color="green", s=200,
                    zorder=5, marker="^", label=f"BUY (conf={result.confidence})")
    elif result.signal == "sell":
        ax1.scatter([dates[-1]], [closes[-1]], color="red", s=200,
                    zorder=5, marker="v", label=f"SELL (conf={result.confidence})")

    ax1.set_title(f"{code} — Mean Reversion Diagnostic  (signal={result.signal}, conf={result.confidence})")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("Price")

    # --- 子图2：残差偏离 ---
    if len(closes) >= reg_window:
        t_full = np.arange(reg_window)
        a = result.details["reg_a"]
        b = result.details["reg_b"]
        preds_all = a * t_full + b
        residuals = recent_closes - preds_all
        residual_std = result.details["residual_std"]
        z_res = residuals / residual_std if residual_std > 0 else residuals * 0

        colors_z = ["red" if z < -1.5 else "green" if z > 1.5 else "gray" for z in z_res]
        ax2.bar(recent_dates, z_res, width=1, color=colors_z, alpha=0.7)
        ax2.axhline(y=-1.5, color="green", linestyle="--", alpha=0.5, label="buy threshold (-1.5σ)")
        ax2.axhline(y=1.5, color="red", linestyle="--", alpha=0.5, label="sell threshold (+1.5σ)")
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax2.set_ylabel("Z-Residual")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax2.transAxes)

    # --- 子图3：成交量 ---
    volumes = df["volume"].values
    ax3.bar(dates, volumes, width=1, color="steelblue", alpha=0.6, label="Volume")
    if len(volumes) >= 21:
        ma20v = pd.Series(volumes).rolling(20).mean().values
        ax3.plot(dates, ma20v, "orange", linewidth=1, label="MA20 Volume")
    ax3.set_ylabel("Volume")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  图保存至: {save_path}")
    plt.show()


def scan_single(code: str, end_date: str, plot: bool = False):
    """测试单只股票。"""
    print(f"\n{'='*60}")
    print(f"  股票: {code}")
    print(f"{'='*60}")

    df = get_daily_kline_from_tdx(code, end_date)
    if df is None or len(df) < 60:
        print(f"  !! 数据不足（{len(df) if df is not None else 0}条），跳过")
        return None

    print(f"  数据量: {len(df)} 条, 日期范围: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")

    result = compute_signal(df, code=code)

    print(f"\n  —— 信号结果 ——")
    print(f"  信号: {result.signal:>8}  |  信心等级: {result.confidence}/5")
    print(f"  日期: {result.date}")
    print(f"\n  —— 信号A: 滚动回归残差 ——")
    print(f"  Z-Residual: {result.z_residual:>8.2f}")
    print(f"  回归斜率:   {result.reg_slope:>8.4f}")
    print(f"  回归截距:   {result.details['reg_b']:>8.2f}")
    print(f"  残差标准差: {result.details['residual_std']:>8.4f}")
    print(f"  预测价格:   {result.details['predicted']:>8.2f}")
    print(f"\n  —— 信号B: 能量衰竭 ——")
    print(f"  衰竭评分:   {result.energy_score}/5")
    print(f"  下行能量:   {result.drop_energy:.4f}")
    print(f"  是否减速:   {result.details['decelerating']}")
    print(f"  是否缩量:   {result.details['volume_shrink']}")
    print(f"  是否跌不动: {result.details['stalled']}")
    print(f"  收盘近低:   {result.details['near_low']}")
    print(f"\n  量比:       {result.volume_ratio:.2f}")

    if plot:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"diagnostic_{code}.png")
        plot_diagnostic(df, result, code, save_path=save_path)

    return result


def scan_block(block_name: str, end_date: str, max_stocks: int = 20):
    """扫描一个板块的所有成分股。"""
    from easy_tdx import TdxClient

    print(f"\n{'='*60}")
    print(f"  板块扫描: {block_name}")
    print(f"{'='*60}")

    # 获取成分股
    with TdxClient.from_best_host() as c:
        df_block = c.get_block_info("block_gn.dat")
        match = df_block[df_block["name"] == block_name]
        if len(match) == 0:
            print(f"  !! 未找到板块 '{block_name}'")
            return
        codes_raw = match.iloc[0]["codes"]

    # 补前缀
    codes = []
    for c in codes_raw:
        if c.startswith("60"):
            codes.append("sh" + c)
        elif c.startswith("00") or c.startswith("30"):
            codes.append("sz" + c)
        elif c.startswith("4") or c.startswith("8"):
            codes.append("bj" + c)

    print(f"  成分股: {len(codes)} 只")
    if max_stocks and len(codes) > max_stocks:
        print(f"  (限于测试，只扫描前 {max_stocks} 只)")
        codes = codes[:max_stocks]

    # 逐只扫描
    results = []
    for i, code in enumerate(codes):
        try:
            df = get_daily_kline_from_tdx(code, end_date)
            if df is None or len(df) < 60:
                continue
            r = compute_signal(df, code=code)
            if r.signal != "neutral":
                results.append(r)
                print(f"  [{i+1}/{len(codes)}] {code}: {r.signal:>8} (conf={r.confidence})")
        except Exception as e:
            pass

    # 汇总
    print(f"\n  —— 扫描汇总 ——")
    print(f"  扫描: {len(codes)} 只, 有效: {len(results)} 只有信号")
    buys = [r for r in results if r.signal == "buy"]
    sells = [r for r in results if r.signal == "sell"]
    print(f"  买入信号: {len(buys)}, 卖出信号: {len(sells)}")

    if buys:
        print(f"\n  —— 买入信号详情 (按信心排序) ——")
        buys.sort(key=lambda r: r.confidence, reverse=True)
        for r in buys:
            print(f"  {r.code:>10} | conf={r.confidence} | z={r.z_residual:>6.2f} | "
                  f"energy={r.energy_score} | drop_ene={r.drop_energy:.4f} | "
                  f"vol_ratio={r.volume_ratio:.2f}")

    return results


# ── 手动测试用例 ─────────────────────────────────────────────
TEST_CASES = [
    # (code, 说明)
    ("sz000032", "深桑达A - 前次探测 RSI=24.9，预期应检出回归信号"),
    ("sz000016", "深康佳A - 正常区间，预期中性"),
    ("sh600519", "贵州茅台 - 大市值趋势股，预期中性"),
    ("sz000002", "万科A - 长期下跌股"),
]


def main():
    parser = argparse.ArgumentParser(description="均值回归算法验证")
    parser.add_argument("--code", type=str, default=None,
                        help="单只股票代码，如 sz000032")
    parser.add_argument("--block", type=str, default=None,
                        help="板块名称，如 人工智能")
    parser.add_argument("--date", type=str, default="20260723",
                        help="截止日期 YYYYMMDD，默认 20260723")
    parser.add_argument("--plot", action="store_true", default=True,
                        help="是否绘制诊断图")
    parser.add_argument("--test", action="store_true", default=False,
                        help="运行预设测试用例")
    args = parser.parse_args()

    if args.code:
        scan_single(args.code, args.date, plot=args.plot)
    elif args.block:
        scan_block(args.block, args.date)
    elif args.test:
        print("=" * 60)
        print("  预设测试用例")
        print("=" * 60)
        for code, desc in TEST_CASES:
            print(f"\n  用例: {code} — {desc}")
            scan_single(code, args.date, plot=args.plot)
    else:
        # 默认：单股测试 + 板块扫描
        scan_single("sz000032", args.date, plot=True)
        scan_single("sz000016", args.date, plot=True)
        scan_single("sh600519", args.date, plot=True)


if __name__ == "__main__":
    main()
