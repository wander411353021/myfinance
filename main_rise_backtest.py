"""主升浪检测回测 / 验收框架（未来确认法标定真值）。

设计原则（对齐 mean_reversion.backtest 的「因果信号 + 事后标签 + 随机基准」范式）：
  - detect_main_rise 本身纯因果（只用当日及之前数据，无未来函数）；
  - 真值标签用「未来确认法」事后标定（段后表现验证），仅用于验收，不进信号；
  - 覆盖率 = 真主升浪天数中被检出覆盖的比例（目标 >80%）；
  - 误检率 = 检出天数中非真主升浪的比例（目标 <10%，即 precision>90%）；
  - 随机基准：乱选同等数量日子，命中真主升浪的天然比例（信号须显著优于它）。

用法：
    from main_rise_backtest import run_on_df, run_many, summarize
    stats = run_on_df(df, code="sz300437")
    print(summarize(stats))
"""

import numpy as np
import pandas as pd

from mean_reversion.signal_residual import compute_rolling_regression
from main_rise import detect_main_rise


# ────────────────────────────────────────────────────────────
# 真值标定：未来确认法（允许用未来数据，仅验收）
# ────────────────────────────────────────────────────────────
def label_main_rise_truth(closes, reg250=None, reg_win_long=250, reg_win_mid=120,
                          slope_annual_min=0.20, pos_gate=0.97, max_dd=0.25,
                          min_len=20, gain_min=0.30, confirm_win=20):
    """事后标定真主升浪段（未来确认法，O(n) 段扫描）。

    找所有「up 持续段」(中周期斜率够 + 价格在 reg250 上方)：对每段检查
    段内 running-peak 回撤≤max_dd、区间涨幅≥gain_min、段长≥min_len、
    且段后 confirm_win 日内不深跌 才确认为真主升浪。仅用于验收。
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    if reg250 is None:
        reg250, _ = compute_rolling_regression(closes, reg_win_long)
    _, slopes120 = compute_rolling_regression(closes, reg_win_mid)
    annual120 = np.exp(slopes120 * 250.0) - 1.0

    finite = np.isfinite(reg250) & np.isfinite(slopes120)
    up = finite & (annual120 > slope_annual_min) & (closes > reg250 * pos_gate)

    truth = np.zeros(n, dtype=bool)
    segs = []
    i = 0
    while i < n:
        if not up[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and up[j + 1]:
            j += 1
        s, e = i, j
        seg = closes[s:e + 1]
        run_peak = np.maximum.accumulate(seg)
        run_dd = np.max((run_peak - seg) / run_peak)
        invalid = bool(run_dd > max_dd)
        gain = closes[e] / closes[s] - 1.0
        if (e - s + 1 >= min_len) and (gain >= gain_min) and (not invalid):
            if e + 1 + confirm_win <= n:
                post_low = closes[e + 1:e + 1 + confirm_win].min()
                if post_low < closes[e] * (1 - max_dd):
                    i = j + 1
                    continue
            truth[s:e + 1] = True
            segs.append((s, e))
        i = j + 1
    return truth, segs


# ────────────────────────────────────────────────────────────
# 指标
# ────────────────────────────────────────────────────────────
def evaluate(detected, truth):
    d = detected.astype(bool)
    t = truth.astype(bool)
    n_det = int(d.sum())
    n_truth = int(t.sum())
    tp = int((d & t).sum())
    fp = int((d & ~t).sum())
    coverage = tp / n_truth if n_truth else 0.0
    precision = tp / n_det if n_det else 0.0
    false_alarm = fp / n_det if n_det else 0.0
    return {"coverage": coverage, "precision": precision, "false_alarm": false_alarm,
            "n_det": n_det, "n_truth": n_truth, "tp": tp, "fp": fp}


def random_baseline(truth, n_det, rng, n_trials=200):
    """随机选 n_det 个日子，命中真主升浪的天然比例均值。"""
    t = truth.astype(bool)
    n = len(t)
    n_truth = int(t.sum())
    if n_det <= 0 or n_truth == 0:
        return 0.0
    base = n_truth / n
    # 直接数学期望即可（无放回抽样的期望 = base）
    return base


# ────────────────────────────────────────────────────────────
# 单只 / 多只
# ────────────────────────────────────────────────────────────
def run_on_df(df, code="", reg_win_long=250, reg_win_mid=120,
              slope_annual_min=0.20, pos_gate=0.97, max_dd=0.25, min_len=20,
              gain_min=0.30, confirm_win=20, verbose=False):
    closes = df["close"].values.astype(float)
    volumes = df["volume"].values.astype(float) if "volume" in df.columns else None
    n = len(closes)
    reg250, _ = compute_rolling_regression(closes, reg_win_long)

    detected, score, segs = detect_main_rise(
        closes, reg250=reg250, reg_win_long=reg_win_long, reg_win_mid=reg_win_mid,
        slope_annual_min=slope_annual_min, pos_gate=pos_gate, max_dd=max_dd, min_len=min_len, volumes=volumes)
    truth, truth_segs = label_main_rise_truth(
        closes, reg250=reg250, reg_win_long=reg_win_long, reg_win_mid=reg_win_mid,
        slope_annual_min=slope_annual_min, pos_gate=pos_gate, max_dd=max_dd,
        min_len=min_len, gain_min=gain_min, confirm_win=confirm_win)

    ev = evaluate(detected, truth)
    rb = random_baseline(truth, ev["n_det"], None)
    stats = {"code": code, "n": n,
             "n_det_segs": len(segs), "n_truth_segs": len(truth_segs),
             "coverage": ev["coverage"], "precision": ev["precision"],
             "false_alarm": ev["false_alarm"], "n_det": ev["n_det"],
             "n_truth": ev["n_truth"], "tp": ev["tp"], "fp": ev["fp"],
             "rand_base": rb}
    if verbose:
        print(summarize(stats))
    return stats


def run_many(results):
    valid = [r for r in results if r and r.get("n_truth", 0) > 0]
    if not valid:
        return {"n_stocks": len(results), "n_valid": 0}
    # 日级聚合（按真值天数加权更公平；此处用简单平均 + 总量）
    tot_det = sum(r["n_det"] for r in valid)
    tot_truth = sum(r["n_truth"] for r in valid)
    tot_tp = sum(r["tp"] for r in valid)
    tot_fp = sum(r["fp"] for r in valid)
    coverage = tot_tp / tot_truth if tot_truth else 0.0
    precision = tot_tp / tot_det if tot_det else 0.0
    false_alarm = tot_fp / tot_det if tot_det else 0.0
    rb = np.mean([r["rand_base"] for r in valid])
    return {"n_stocks": len(results), "n_valid": len(valid),
            "n_det": tot_det, "n_truth": tot_truth, "tp": tot_tp, "fp": tot_fp,
            "coverage": coverage, "precision": precision, "false_alarm": false_alarm,
            "rand_base": rb}


def summarize(stats):
    if stats.get("n_valid", 1) == 0:
        return f"[{stats.get('code','?')}] 无有效样本"
    cov = stats.get("coverage", 0.0) * 100
    fa = stats.get("false_alarm", 0.0) * 100
    rb = stats.get("rand_base", 0.0) * 100
    return (f"[{stats.get('code','?')}] n={stats.get('n',0)}  检出段={stats.get('n_det_segs',0)} "
            f"真值段={stats.get('n_truth_segs',0)}\n"
            f"  覆盖率={cov:5.1f}%   误检率={fa:5.1f}%   (随机基准命中={rb:5.1f}%)")


if __name__ == "__main__":
    # 用法：
    #   真实池(84只):   python3 -m main_rise_backtest
    #   真实600只池:    python3 -m main_rise_backtest --universe result/universe_600.txt
    #   合成稳定性自检: python3 -m main_rise_backtest --sim 600
    import argparse, datetime
    from panic_reversal import CANDIDATE_POOL
    import tdx_quant

    def simulate_stock(rng, n=600):
        """合成一只股票（无未来函数检测的逻辑自检用，不代表真实分布）。"""
        close = np.empty(n)
        kind = rng.choice(["bull", "fake", "none", "bear"], p=[0.40, 0.25, 0.20, 0.15])
        base = 10.0
        if kind in ("bull", "fake"):
            close[:120] = base + rng.normal(0, 0.08, 120).cumsum() * 0.01
            if kind == "bull":
                tt = np.arange(400)
                close[120:520] = base * np.exp(0.0015 * tt) * (1 + rng.normal(0, 0.012, 400))
                close[520:600] = close[519] * (1 + rng.normal(0, 0.005, 80))
            else:  # fake: 一段真主升 + 紧接假主升(涨后深跌)
                tt = np.arange(250)
                close[120:370] = base * np.exp(0.0015 * tt) * (1 + rng.normal(0, 0.012, 250))
                fak = np.concatenate([close[369] * np.exp(0.006 * np.arange(20)),
                                      close[369] * 1.13 * np.exp(-0.009 * np.arange(20))])
                close[370:410] = fak[:40]
                close[410:600] = close[409] * (1 + rng.normal(0, 0.005, 190))
        elif kind == "none":
            close[:] = base * (1 + rng.normal(0, 0.01, n).cumsum() * 0.005)
        else:  # bear 阴跌
            close[:] = base * np.exp(-0.0008 * np.arange(n)) * (1 + rng.normal(0, 0.012, n))
        return pd.DataFrame({"close": close})

    def stability(results, n_boot=100, sample=200, seed=0):
        rng = np.random.default_rng(seed)
        valid = [r for r in results if r.get("n_truth", 0) > 0]
        if len(valid) < 2:
            return None
        if sample > len(valid):
            sample = len(valid)
        covs, fas = [], []
        for _ in range(n_boot):
            sub = list(rng.choice(valid, size=sample, replace=True))
            a = run_many(sub)
            covs.append(a["coverage"])
            fas.append(a["false_alarm"])
        covs = np.array(covs); fas = np.array(fas)
        return covs.mean(), covs.std(), fas.mean(), fas.std()

    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=int, default=0, help="合成池规模(如 600)")
    parser.add_argument("--pool", type=int, default=0, help="真实全市场取前N只(如 600)")
    parser.add_argument("--universe", type=str, default="", help="600只代码文件(每行一个)")
    args = parser.parse_args()

    if args.sim:
        rng = np.random.default_rng(20260825)
        results = [run_on_df(simulate_stock(rng, 600), code="SIM"+str(i))
                   for i in range(args.sim)]
    elif args.pool:
        end_date = datetime.date.today().strftime("%Y-%m-%d")
        from eltdx import TdxClient
        with TdxClient(timeout=5) as c:
            all_codes = c.get_a_share_codes_all()
        codes = [str(x) for x in all_codes]
        print(f"真实池: 取到 {len(codes)} 只, 开始回测...")
        results = []
        for k, code in enumerate(codes):
            if len(results) >= args.pool:
                break
            try:
                df = tdx_quant.get_daily_kline_from_tdx(code, end_date)
                if df is None or len(df) < 300:
                    continue
                results.append(run_on_df(df, code=code))
            except Exception as _e:  # noqa
                pass
            if (k + 1) % 50 == 0:
                print(f"  progress {k+1}/{len(codes)}  valid={len(results)}", flush=True)
    else:
        pool = CANDIDATE_POOL
        if args.universe:
            with open(args.universe) as fh:
                pool = [l.strip() for l in fh if l.strip()]
        end_date = datetime.date.today().strftime("%Y-%m-%d")
        results = []
        for code in pool:
            try:
                df = tdx_quant.get_daily_kline_from_tdx(code, end_date)
                if df is None or len(df) < 300:
                    continue
                results.append(run_on_df(df, code=code))
            except Exception as _e:  # noqa
                print(f"skip {code}: {_e}")

    agg = run_many(results)
    print("=== 主升浪检测 聚合（%d 有效 / %d 池） ===" % (agg.get("n_valid", 0), agg.get("n_stocks", 0)))
    print(f"  覆盖率   = {agg['coverage']*100:5.1f}%   (目标 >80%)")
    print(f"  误检率   = {agg['false_alarm']*100:5.1f}%   (目标 <10%)")
    print(f"  精度     = {agg['precision']*100:5.1f}%")
    print(f"  随机基准 = {agg['rand_base']*100:5.1f}%")
    if args.sim:
        st = stability(results)
        if st:
            cm, cs, fm, fs = st
            print(f"  稳定性(bootstrap 100次/每次200有效): 覆盖率={cm*100:.1f}±{cs*100:.1f}%  "
                  f"误检率={fm*100:.1f}±{fs*100:.1f}%")

