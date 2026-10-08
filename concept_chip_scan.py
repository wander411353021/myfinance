# -*- coding: utf-8 -*-
"""概念板块筹码扫描(2026-10-08 polo4111 定版)
用法:
    python3 concept_chip_scan.py [end=YYYYMMDD] [min_days=300] [out]
    python3 concept_chip_scan.py                       # 默认 end=今天, 全量418板块
    python3 concept_chip_scan.py 20261008 300 result/concept_chip_scan_20261008.csv
流程: 读 tdx_block_names.json(8805xx-8809xx 概念板块) -> eltdx 批量拉日线 -> 方案A换手率代理
      (turn=0.05*量/MA20量, 前20根固定0.05) -> chip_panel.cost_series 逐日递推 COST35/50/75
      -> 输出 csv + 控制台分类(强势:收>COST75 / 中间 / 受压:收<COST35)。
红线: 逐日递推 fengwo 只用当日及以前数据, 无未来函数; 指数无股本故用方案A代理(当日量+MA20当日值)。
"""
import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from eltdx import Client

from chip_panel import cost_series
from eltdx_compat import get_bars

_MAP = 'tdx_block_names.json'


def fetch_concept_indexes(cli, end, min_days=300, limit=None):
    """拉取全部概念板块日线(8805xx-8809xx), 返回 [{code,name,high,low,close,volume}]"""
    m = json.load(open(_MAP, encoding='utf-8'))
    concepts = sorted([c for c in m if 880500 <= int(c) <= 880999])
    if limit:
        concepts = concepts[:limit]
    out, bad = [], []
    for c in concepts:
        try:
            bars = get_bars(cli, 'sh%s' % c, 'day', 1600, end)
            if len(bars) < min_days:
                continue
            out.append(dict(code=c, name=m[c], days=len(bars),
                            high=np.array([b.high for b in bars], float),
                            low=np.array([b.low for b in bars], float),
                            close=np.array([b.close for b in bars], float),
                            volume=np.array([b.volume_lots or 0.0 for b in bars], float)))
        except Exception as e:
            bad.append((c, str(e)[:40]))
    return out, bad


def turnover_proxy(volume):
    """方案A换手率代理: turn=0.05*量/MA20量, clip(0.005,0.30), 前20根固定0.05"""
    V = np.asarray(volume, float)
    ma20 = np.convolve(V, np.ones(20) / 20, mode='valid')
    ma20 = np.concatenate([np.full(19, np.nan), ma20])
    ratio = np.where(ma20 > 0, V / np.maximum(ma20, 1e-9), 1.0)
    turn = (0.05 * ratio).clip(0.005, 0.30)
    turn[:20] = 0.05
    return turn


def scan(end=None, min_days=300, out=None, limit=None, quiet=False):
    end = end or datetime.now().strftime('%Y%m%d')
    out = out or 'concept_chip_scan_%s.csv' % end
    cli = Client(timeout=8.0)
    t0 = time.time()
    try:
        indexes, bad = fetch_concept_indexes(cli, end, min_days, limit)
        rows = []
        for it in indexes:
            turn = turnover_proxy(it['volume'])
            c35 = cost_series(it['high'], it['low'], it['volume'], turn, 0.35)
            c50 = cost_series(it['high'], it['low'], it['volume'], turn, 0.50)
            c75 = cost_series(it['high'], it['low'], it['volume'], turn, 0.75)
            v35, v50, v75 = float(c35[-1]), float(c50[-1]), float(c75[-1])
            if not all(np.isfinite((v35, v50, v75))):
                continue
            close = float(it['close'][-1])
            rows.append(dict(code=it['code'], name=it['name'], days=it['days'],
                             close=close, cost35=v35, cost50=v50, cost75=v75,
                             above75=close / v75 - 1, below35=close / v35 - 1))
    finally:
        cli.close()
    df = pd.DataFrame(rows).reset_index(drop=True).sort_values('above75', ascending=False)
    df.to_csv(out, index=False, encoding='utf-8-sig')
    if not quiet:
        strong = df[df.above75 > 0]
        pressed = df[df.below35 < 0]
        mid = df[(df.above75 <= 0) & (df.below35 >= 0)]
        print('完成 %d 板块(跳过/失败 %d), 耗时 %.0fs' % (len(df), len(bad), time.time() - t0))
        print('分类: 强势(收>COST75) %d | 中间 %d | 受压(收<COST35) %d' % (len(strong), len(mid), len(pressed)))
        print('输出:', out)
    return df


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    limit = None
    for a in sys.argv[1:]:
        if a.startswith('--limit='):
            limit = int(a.split('=')[1])
    end = args[0] if len(args) > 0 else None
    min_days = int(args[1]) if len(args) > 1 else 300
    out = args[2] if len(args) > 2 else None
    scan(end=end, min_days=min_days, out=out, limit=limit)
