# -*- coding: utf-8 -*-
"""v3b early 假信号 4 因子验证 — 2026-10-18 polo4111
底座: early ∩ qfq_flag=0 ∩ fut60_days>=60 (n=67,276)
分层: 假=后60日max<5% / 一般=5~10% / 真=>=10%; 主升=后120日max>=50%(fut120_days>=120)
因子(全部只用<=信号日数据, 无未来函数; 紫线=gt2 复用 challenge_scan.compute):
  F1 均线结构: close>MA20 且 MA20>MA60 (移除不满足者=未摆脱下跌通道)
  F2 距紫线:   (gt2-close)/close>20% (遥不可及=假候选)
  F3 绝对量能: vol[i]/max(vol[i-250:i]) < thr (缩量反抽; thr=0.30/0.35/0.40 + 曲线)
  F4 前堆高压: dist_vol_high∈[0,5] / [0,10] (正压在前放量堆高点下方贴磨)
用法:
  python3 fake_filter_v3b_5y.py build   # 计算因子表 -> result/fake_filter_factors.csv
  python3 fake_filter_v3b_5y.py eval    # 单因子+组合评估 + 报告
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
import challenge_scan as cs

REPO = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(REPO, 'result', 'scan_cache_v3b')
SIG_CSV = os.path.join(REPO, 'result', 'backtest_v3b_5y_overcome.csv')
FACTOR_CSV = os.path.join(REPO, 'result', 'fake_filter_factors.csv')
LOG = os.path.join(REPO, 'result', 'fake_filter_v3b_5y.log')
NPROC = max(1, min(2, os.cpu_count() or 1))


def log(msg):
    with open(LOG, 'a') as f:
        f.write(time.strftime('%H:%M:%S ') + msg + '\n')
    print(msg, flush=True)


# ────────────────────────── 因子表构建 ──────────────────────────
def code_factors(args):
    code, rows = args  # rows: [{_ridx, sig_date}]
    fp = os.path.join(CACHE, f'{code}.csv')
    if not os.path.exists(fp):
        return None, code
    try:
        df = pd.read_csv(fp, parse_dates=['date'])
        if len(df) < 400:
            return None, code
        cl, darr, gt_p, gt2, c35s, _, _cA = cs.compute(df)
    except Exception as e:
        return None, code
    vol = df['volume'].values.astype(float)
    n = len(cl)
    darr = np.array(darr)
    out = []
    for r in rows:
        sd = str(r['sig_date'])
        i = int(np.searchsorted(darr, sd))
        if i >= n or darr[i] != sd:
            out.append(dict(_ridx=r['_ridx'], f1_ok=np.nan, f2_dist_purple=np.nan, f3_vr250=np.nan))
            continue
        # F1: 均线结构 (MA20/MA60 含信号日, 只用<=i数据)
        ma20 = cl[i - 19:i + 1].mean() if i >= 19 else np.nan
        ma60 = cl[i - 59:i + 1].mean() if i >= 59 else np.nan
        f1_ok = 1.0 if (np.isfinite(ma20) and np.isfinite(ma60) and cl[i] > ma20 and ma20 > ma60) else 0.0
        # F2: 距紫线距离 (紫线=gt2 第二阶梯)
        p2 = gt2[i]
        f2 = (p2 / cl[i] - 1.0) * 100.0 if (np.isfinite(p2) and p2 > 0 and cl[i] > 0) else np.nan
        # F3: 绝对量能 = 当日量 / 前250日(不含当日)最大单日量
        w0 = max(0, i - 250)
        vmax = vol[w0:i].max() if i - w0 > 0 else np.nan
        f3 = vol[i] / vmax if (np.isfinite(vmax) and vmax > 0) else np.nan
        out.append(dict(_ridx=r['_ridx'], f1_ok=f1_ok, f2_dist_purple=f2, f3_vr250=f3))
    return out, code


def build():
    t0 = time.time()
    df = pd.read_csv(SIG_CSV, dtype={'code': str})
    df['code'] = df.code.str.zfill(6)
    if 'qfq_flag' not in df.columns:
        raise SystemExit(
            '输入 %s 缺 qfq_flag 列。请先运行:\n'
            '  python add_qfq_flag.py\n'
            '（它会从 result/backtest_v3b_5y.csv 生成 *_qfq.csv；若尚未回溯，'
            '先跑 backtest_v3b_5y.py + merge_backtest.py）' % SIG_CSV)
    base = df[(df.sig_type == 'early') & (df.qfq_flag == 0) & (df.fut60_days >= 60)].copy()
    base['_ridx'] = np.arange(len(base))
    groups = [(c, g[['_ridx', 'sig_date']].to_dict('records')) for c, g in base.groupby('code')]
    log('build: codes=%d signals=%d nproc=%d' % (len(groups), len(base), NPROC))
    rows, misses = [], []
    with Pool(NPROC) as pool:
        for res, code in pool.imap_unordered(code_factors, groups, chunksize=4):
            if res is None:
                misses.append(code)
            else:
                rows.extend(res)
    fd = pd.DataFrame(rows).sort_values('_ridx').reset_index(drop=True)
    # 加入 F4 (来自 overcome CSV 的 dist_vol_high)
    out = pd.concat([base.drop(columns=['_ridx']).reset_index(drop=True), fd.drop(columns=['_ridx'])], axis=1)
    out.to_csv(FACTOR_CSV, index=False)
    log(f'build: done n={len(out)} 缺缓存={misses} 耗时{time.time() - t0:.0f}s -> {FACTOR_CSV}')


# ────────────────────────── 评估 ──────────────────────────
# 2026-10-16 参考口径（n=67,276）。⚠ 这些只是历史基线，重跑数据后以实际 df 为准：
#   main_eval() 会按当前 CSV 重算并覆盖 FAKE_N/TRUE_N/MZ_N，避免硬编码断言阻断复现。
BASE_N = 67276
FAKE_N = 19730
TRUE_N = 36408
W120_N = 63581
MZ_N = 12043  # 主升(底座内 fut120_days>=120 & fut120_max>=50)


def stats_of(sub):
    """给定子集(保留组), 返回统计字典。"""
    n = len(sub)
    s60 = sub.s60.mean() * 100 if n else np.nan
    w120 = sub[sub.fut120_days >= 120]
    s120 = w120.s120.mean() * 100 if len(w120) else np.nan
    fmed = sub.fut60_max.median() if n else np.nan
    return dict(n=n, s60=round(s60, 2), s120=round(s120, 2), fmed=round(float(fmed), 2), w120n=len(w120))


def evaluate_flag(flag, df, label, thr_desc):
    """flag: 长度为 len(df) 的 bool Series, True=剔除。返回统计dict"""
    rem = df[flag]
    keep = df[~flag]
    n_rem = int(len(rem))
    rem_fake = int(rem.fut60_max.lt(5).sum())
    rem_true = int(rem.fut60_max.ge(10).sum())
    rem_mz = int(((rem.fut120_days >= 120) & (rem.fut120_max >= 50)).sum())
    keep_fake = int(keep.fut60_max.lt(5).sum())
    keep_true = int(keep.fut60_max.ge(10).sum())
    ks = stats_of(keep)
    d = dict(
        label=label, thr=thr_desc,
        n_rem=n_rem,
        rem_fake=rem_fake, rem_true=rem_true, rem_mz=rem_mz,
        fake_rm_rate=round(rem_fake / FAKE_N * 100, 2),      # 剔假率: 剔除的假信号/总假
        fake_in_rem=round(rem_fake / n_rem * 100, 1) if n_rem else np.nan,  # 剔除集中假占比
        true_keep_rate=round(keep_true / TRUE_N * 100, 2),   # 真保留率
        keep_fake_n=keep_fake,
        keep_fake_rate=round(keep_fake / len(keep) * 100, 1) if len(keep) else np.nan,
        keep_n=ks['n'], keep_s60=ks['s60'], keep_s120=ks['s120'], keep_w120n=ks['w120n'], keep_fmed=ks['fmed'],
    )
    return d


def main_eval():
    global BASE_N, FAKE_N, TRUE_N, W120_N, MZ_N
    df = pd.read_csv(FACTOR_CSV, dtype={'code': str})
    df['code'] = df.code.str.zfill(6)
    n = len(df)
    # 按当前因子表重算口径常量（历史 67,276 仅作参考，不阻断复现）
    BASE_N = n
    FAKE_N = int(df.fut60_max.lt(5).sum())
    TRUE_N = int(df.fut60_max.ge(10).sum())
    W120_N = int((df.fut120_days >= 120).sum())
    MZ_N = int(((df.fut120_days >= 120) & (df.fut120_max >= 50)).sum())
    print('底座 n=%d fake=%d true=%d w120=%d mz=%d' % (BASE_N, FAKE_N, TRUE_N, W120_N, MZ_N))
    if n != 67276:
        print('ℹ 底座行数 %d != 历史参考 67276（数据重跑后属正常，所有比例按当前 n 计算）' % n)
    # F1
    f1_rem = df.f1_ok != 1.0
    # F2
    f2_rem = df.f2_dist_purple > 20.0
    # F3 阈值系列
    f3_thrs = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    f3_rem = {t: df.f3_vr250 < t for t in f3_thrs}
    # F4 区间系列
    f4_rem = {}
    for hi in (5.0, 10.0):
        f4_rem[hi] = df.dist_vol_high.between(0, hi, inclusive='both')
    rows = []
    rows.append(evaluate_flag(f1_rem, df, 'F1', 'close>MA20 且 MA20>MA60'))
    rows.append(evaluate_flag(f2_rem, df, 'F2', '距紫线 (gt2-close)/close>20%'))
    for t in f3_thrs:
        rows.append(evaluate_flag(f3_rem[t], df, 'F3', f'量/前250日最大量<{t:.2f}'))
    for hi in (5.0, 10.0):
        rows.append(evaluate_flag(f4_rem[hi], df, 'F4', f'dist_vol_high∈[0,{int(hi)}]'))
    single = pd.DataFrame(rows)
    single.to_csv(os.path.join(REPO, 'result', 'fake_filter_single.csv'), index=False)
    print(single[['label', 'thr', 'n_rem', 'rem_fake', 'rem_true', 'rem_mz', 'fake_rm_rate', 'true_keep_rate',
                  'keep_n', 'keep_s60', 'keep_s120', 'keep_fmed']].to_string(index=False))

    # ── 组合: 两两 / 三三 ──
    variants = {
        'F1': f1_rem,
        'F2': f2_rem,
        'F3@0.30': f3_rem[0.30], 'F3@0.35': f3_rem[0.35], 'F3@0.40': f3_rem[0.40],
        'F4@[0,5]': f4_rem[5.0], 'F4@[0,10]': f4_rem[10.0],
    }
    names = list(variants)
    combos = []
    import itertools
    for k in (2, 3):
        for combo in itertools.combinations(names, k):
            flag = pd.Series(False, index=df.index)
            for nm in combo:
                flag = flag | variants[nm]
            combos.append((combo, flag))
    # 也可计算保留>5万 的组合(剔除过多视为超约束)
    crows = []
    for combo, flag in combos:
        d = evaluate_flag(flag, df, '+'.join(combo), '+'.join(combo))
        crows.append(d)
    combo_df = pd.DataFrame(crows).sort_values(['true_keep_rate', 'fake_rm_rate'], ascending=[False, False])
    combo_df.to_csv(os.path.join(REPO, 'result', 'fake_filter_combo.csv'), index=False)
    print('\n== 组合(按真保留率降序) ==')
    print(combo_df[['label', 'n_rem', 'rem_fake', 'rem_true', 'rem_mz', 'fake_rm_rate', 'true_keep_rate',
                    'keep_n', 'keep_s60', 'keep_s120']].head(40).to_string(index=False))
    # 满足硬约束 (真保留率>=90%) 的组合里找 剔假最多 + S120最高
    ok = combo_df[combo_df.true_keep_rate >= 90.0].copy()
    print('\n== 满足真保留率>=90% 的组合 ==')
    print(ok[['label', 'n_rem', 'rem_fake', 'rem_true', 'rem_mz', 'fake_rm_rate', 'true_keep_rate',
              'keep_n', 'keep_s120', 'keep_s60']].sort_values('fake_rm_rate', ascending=False).to_string(index=False))
    return single, combo_df, df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['build', 'eval'])
    args = ap.parse_args()
    if args.mode == 'build':
        build()
    else:
        main_eval()
