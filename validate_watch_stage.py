# -*- coding: utf-8 -*-
"""观察池状态机广样本验证（诊断脚本，不参与生产信号）。

用法:
  python validate_watch_stage.py --limit 30 --datalen 2400 --end 20261023

包含:
  1. 多股票/长时间跨度阶段分布；
  2. 截断一致性（无未来函数）；
  3. 未来函数 oracle：仅用于检查因果版本是否漏掉后续推进，不进入生产。
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chip_panel import cost_series
from challenge_scan import ehlers_smooth
from algo_config import COST2_PCT, COST2_EHLERS, COST_PCT, COST_EHLERS
from watch_stage import compute_watch_stages, STAGE_NAMES
import tdx_quant as tq


def load_pool(path, limit):
    out = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            out.append(line.split(',')[0].strip())
            if limit and len(out) >= limit:
                break
    return out


def analyze(code, end, datalen):
    df = tq.get_daily_kline_from_tdx(code, end, datalen=datalen, with_turnover=True)
    if df is None or len(df) < 200:
        return None
    df = df.sort_values('date').reset_index(drop=True)
    df = df[df['date'].astype(str).str[:10].str.replace('-', '') <= end].reset_index(drop=True)
    cl = df['close'].values.astype(float)
    hi = df['high'].values.astype(float)
    lo = df['low'].values.astype(float)
    vo = df['volume'].values.astype(float)
    turn = np.clip(df['turnover'].values.astype(float), 0, 1)
    B = ehlers_smooth(cost_series(hi, lo, vo, turn, COST2_PCT), max(2, COST2_EHLERS))
    A = ehlers_smooth(cost_series(hi, lo, vo, turn, COST_PCT), max(2, COST_EHLERS))
    st = compute_watch_stages(cl, hi, lo, B, A, volume=vo, turnover=turn)
    n = len(st)

    # 截断一致性：生产状态机必须只依赖 <=cut 的数据
    trunc_bad = 0
    for cut in (n // 2, max(100, n - 30), n - 1):
        sp = compute_watch_stages(cl[:cut], hi[:cut], lo[:cut], B[:cut], A[:cut],
                                  volume=vo[:cut], turnover=turn[:cut])
        if not np.array_equal(sp, st[:cut]):
            trunc_bad += 1

    # 状态机结构异常
    advanced = True
    struct_bad = []
    for i in range(1, n):
        if st[i] > st[i - 1] + 1:
            struct_bad.append(('jump', i, int(st[i - 1]), int(st[i])))
        if st[i] == 4 and not np.any(st[:i] == 3):
            struct_bad.append(('s4_before_s3', i, 0, 4))
        if st[i] == 5 and not np.any(st[:i] == 4):
            struct_bad.append(('s5_before_s4', i, 0, 5))

    # 未来函数 oracle：只看未来 60 天是否继续推进（诊断，不进入生产）
    future = defaultdict(list)
    for i in range(1, n):
        if st[i] <= st[i - 1]:
            continue
        s = int(st[i])
        if s in (1, 2, 3, 4):
            fut = st[i + 1:min(n, i + 61)]
            future[s].append(bool(len(fut) and fut.max() > s))

    return {
        'code': code, 'n': n, 'st': st, 'trunc_bad': trunc_bad,
        'struct_bad': struct_bad,
        'counts': Counter(map(int, st)),
        'future': future,
        'dates': df['date'].astype(str).str[:10].values,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=30)
    ap.add_argument('--datalen', type=int, default=2400)
    ap.add_argument('--end', default='20261023')
    ap.add_argument('--pool', default='stock_pool_300.txt')
    args = ap.parse_args()

    codes = load_pool(args.pool, args.limit)
    counts = Counter(); future = defaultdict(list)
    trunc_bad = 0; struct_bad = []; ok = 0; errors = []
    for k, code in enumerate(codes, 1):
        try:
            r = analyze(code, args.end, args.datalen)
            if r is None:
                errors.append((code, 'short'))
                continue
            ok += 1
            counts.update(r['counts'])
            trunc_bad += 1 if r['trunc_bad'] else 0
            struct_bad.extend([(code,) + x for x in r['struct_bad']])
            for s, vals in r['future'].items():
                future[s].extend(vals)
            print('progress %d/%d %s bars=%d' % (k, len(codes), code, r['n']))
        except Exception as e:
            errors.append((code, str(e)))
            print('ERR %s %s' % (code, e))

    print('\n== summary ==')
    print('stocks ok/attempt', ok, '/', len(codes), 'errors', len(errors))
    print('stage days', {STAGE_NAMES[s]: counts.get(s, 0) for s in STAGE_NAMES})
    print('truncation_bad_stocks', trunc_bad)
    print('struct_bad', len(struct_bad))
    for x in struct_bad[:20]:
        print('  ', x)
    for s in (1, 2, 3, 4):
        v = future.get(s, [])
        if v:
            print('future_progress S%d: %d/%d %.1f%%' % (
                s, sum(v), len(v), 100.0 * sum(v) / len(v)))
    if errors:
        print('errors:', errors[:10])


if __name__ == '__main__':
    main()
