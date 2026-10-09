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

# 2026-10-09 reasonix: 8805xx-8809xx 段混入 52 个"动态成分"板块(昨日涨停/跌停/连板/首板/
# 历史新高/重仓/次新/送转/解禁…), 成分股按日或短期重构 —— 指数是动态组合, "筹码成本"无实际
# 持有者对应(实测"昨日跌停"显示 -38% 虚假超跌)。默认排除, 只保留 373 个真实题材/概念板块。
DYNAMIC_KW = (
    # ① 按日/短期重构(行情统计型)
    '昨日', '昨曾', '昨ST', '昨收', '昨成交', '昨高', '连板', '首板', '断板', '涨停', '跌停',
    '异动', '换手', '成交', '新高', '新低', '指标股', '重仓', '次新', '送转', '解禁',
    '融资', '破净', '破发', '超跌', '活跃', '热股', '强势', '弱势', '突涨', '振荡', '上榜',
    # ② 定期(季/月)重构: 财务/属性/事件筛选型
    '预案', '转亏', '预增', '预盈', '预亏', '预减', '预告', '扭亏', '市净', '市盈', '百元',
    '低价', '高价', '贝塔', '户数', '每股', '股东', '评级', '季报', '年报', '中报', '绩优',
    '大基金', '北向', '陆股通', '社保', 'QFII', '信托', '保险', '券商', '基金', '微盘',
    '小盘', '大盘', '高分红', '安全分', '整体上市', '承诺不减', '股权激励', '可转债',
    'ST',
)


def is_dynamic_block(name):
    return any(k in name for k in DYNAMIC_KW)


def fetch_concept_indexes(cli, end, min_days=300, limit=None):
    """拉取全部概念板块日线(8805xx-8809xx), 返回 [{code,name,high,low,close,volume}]"""
    m = json.load(open(_MAP, encoding='utf-8'))
    concepts = sorted([c for c in m if 880500 <= int(c) <= 880999
                       and not is_dynamic_block(m[c])])
    if limit:
        concepts = concepts[:limit]
    out, bad = [], []
    for c in concepts:
        try:
            bars = get_bars(cli, 'sh%s' % c, 'day', 800, end)
            if len(bars) < min_days:
                continue
            # 2026-10-09 修复(reasonix): get_bars 锚定路径(all_pages)返回**时间倒序**,
            # 无锚定路径返回升序 —— 必须统一升序, 否则 cost_series 逐日递推方向反转、
            # close[-1] 取到最早价(880501 曾输出 2006 年价 970.53), 分类结论全错。
            bars = sorted(bars, key=lambda b: b.time)
            out.append(dict(code=c, name=m[c], days=len(bars),
                            high=np.array([b.high for b in bars], float),
                            low=np.array([b.low for b in bars], float),
                            close=np.array([b.close for b in bars], float),
                            volume=np.array([b.volume_lots or 0.0 for b in bars], float)))
        except Exception as e:
            bad.append((c, str(e)[:40]))
    return out, bad


def turnover_proxy(volume):
    """方案A换手率代理: turn=0.025*量/MA20量, clip(0.005,0.30), 前20根固定0.025
    2026-10-09 k标定: 0.05→0.025(3板块成分股真实换手中位0.019~0.031, reasonix的0.05偏高2倍;
    标定后深破档60日胜率更稳 存储芯片42.9%→56.3%, 样本更足)"""
    V = np.asarray(volume, float)
    ma20 = np.convolve(V, np.ones(20) / 20, mode='valid')
    ma20 = np.concatenate([np.full(19, np.nan), ma20])
    ratio = np.where(ma20 > 0, V / np.maximum(ma20, 1e-9), 1.0)
    turn = (0.025 * ratio).clip(0.005, 0.30)
    turn[:20] = 0.025
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
            _b35 = close / v35 - 1
            _a75 = close / v75 - 1
            # 2026-10-09 reasonix 验证: 概念板块"跌破 COST35 越深 → 后续反弹越强"
            #   (深度-8~-15%: 60日+34%/胜率67%; <-15%: +89%/72%), 收>COST75 反而平庸;
            #   故 zone 用中性/超跌语义, 排序按 below35 升序(超跌最深在前)。
            if _b35 < -0.15:
                _zone = '极深超跌'
            elif _b35 < -0.08:
                _zone = '深度跌破'
            elif _b35 < 0:
                _zone = '轻度跌破'
            elif _a75 > 0.08:
                _zone = '高位'
            elif _a75 > 0:
                _zone = '突破筹码峰'
            else:
                _zone = '中间'
            rows.append(dict(code=it['code'], name=it['name'], days=it['days'],
                             close=close, cost35=v35, cost50=v50, cost75=v75,
                             above75=_a75, below35=_b35, zone=_zone))
    finally:
        cli.close()
    df = pd.DataFrame(rows).reset_index(drop=True).sort_values('below35', ascending=True)  # 超跌最深在前
    df.to_csv(out, index=False, encoding='utf-8-sig')
    if not quiet:
        print('完成 %d 板块(跳过/失败 %d), 耗时 %.0fs' % (len(df), len(bad), time.time() - t0))
        print('分档(2026-10-09 验证: 跌破 COST35 越深, 统计上后续反弹越强):')
        for z in ('极深超跌', '深度跌破', '轻度跌破', '中间', '突破筹码峰', '高位'):
            print('  %-8s %d' % (z, int((df.zone == z).sum())))
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
