# -*- coding: utf-8 -*-
"""全市场黄金坑日报扫描(手动触发)。

用法:
    python scan_daily_golden_pit.py [YYYYMMDD]
    不传日期默认取最近交易日(数据末端)。

功能:
    1. 遍历全市场,检测所有黄金坑(定版参数:z_thr=-1.5, use_pre_std=True)
    2. 只输出"指定日期当天出坑"的坑(启动日 == 指定日期)
    3. 标注:快启动(滞后≤5天)/质量(strong/normal/weak)/高位坑(⚠️)
    4. 统计:当日出坑数、快启动/慢启动、质量分布
    5. 写入 result/daily/{date}.md

耗时:全市场 ~15-20 分钟(首次拉数据),热缓存后 ~10-15 分钟。
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panic_reversal as pr
import tdx_quant as tq
import numpy as np
from mean_reversion.signal_residual import compute_rolling_regression


def main():
    t0 = time.time()
    if len(sys.argv) > 1:
        END = sys.argv[1]
    else:
        END = time.strftime('%Y%m%d')

    all_df = tq.load_stock_list()
    codes_all = all_df['code'].astype(str).str.zfill(6).tolist()
    names = dict(zip(all_df['code'].astype(str).str.zfill(6), all_df['name']))

    hits = []
    n_ok = 0
    for k, code in enumerate(codes_all):
        try:
            if code[0] in '69':
                fcode = 'sh' + code
            elif code[0] in '03':
                fcode = 'sz' + code
            elif code[0] in '48':
                fcode = 'bj' + code
            else:
                fcode = 'sz' + code
            df = pr._load_df(fcode, END)
            if df is None or len(df) < 390:
                continue
            n_ok += 1
            c = df['close'].values.astype(float)
            v = df['volume'].values.astype(float)
            n = len(c)
            dates = df['date'].dt.strftime('%Y%m%d').values
            reg250, _ = compute_rolling_regression(c, window=250, use_log=True)
            pits = pr.detect_golden_pit(c, reg250)
            if not pits:
                continue
            q = pr.compute_pit_quality(pits, c, v)
            hp = pr.mark_high_pos(pits, c)
            for (s, b, lch), (sh, fl, ql), hpi in zip(pits, q, hp):
                if lch is None:
                    continue
                if dates[lch] != END:
                    continue  # 仅当日出坑
                hits.append(dict(code=code, name=names.get(code, ''), s=dates[s], b=dates[b],
                                 lch=dates[lch], bp=c[b], lp=c[lch], lag=lch - b, ql=ql,
                                 sh=sh, fl=fl, hp=hpi))
        except Exception:
            pass
        if k % 2000 == 0 and k > 0:
            print('  进度 %d/%d 当日出坑 %d (%.0fs)' % (k, len(codes_all), len(hits), time.time() - t0), flush=True)

    fast = [h for h in hits if h['lag'] <= 5]
    print('\n=== %s 当日出坑: %d 个 (有效股票 %d) ===' % (END, len(hits), n_ok), flush=True)
    print('快启动(≤5天): %d | 慢启动: %d' % (len(fast), len(hits) - len(fast)))
    for qq in ['strong', 'normal', 'weak']:
        print('  质量[%s]: %d' % (qq, sum(1 for h in hits if h['ql'] == qq)))
    print('  高位坑: %d' % sum(1 for h in hits if h['hp']))
    print()

    hits.sort(key=lambda x: (x['lag'] > 5, x['ql'] != 'strong', x['lag']))
    for h in hits:
        fs = '★' if h['lag'] <= 5 else ' '
        print("  %s %s %s~%s 坑底%s(%.2f) 出坑%s(%.2f) 滞后%d天 %s %s 缩%.2f/放%.2f %s" % (
            h['code'], h['name'], h['s'], h['b'], h['b'], h['bp'], h['lch'], h['lp'],
            h['lag'], fs, h['ql'], h['sh'], h['fl'], '⚠高位' if h['hp'] else ''))

    lines = [
        '# 黄金坑扫描日报 %s' % END, '',
        '- 全市场扫描,数据截止 %s' % END,
        '- 有效股票 %d | **当日出坑 %d 个**(快启动 %d,慢启动 %d)' % (n_ok, len(hits), len(fast), len(hits) - len(fast)),
        '', '## 当日出坑清单', '',
        '| 代码 | 名称 | 坑段 | 坑底 | 坑底价 | 出坑价 | 滞后(天) | 类型 | 质量 | 缩量比 | 放量比 | 高位坑 |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for h in hits:
        lines.append("| %s | %s | %s~%s | %s | %.2f | %.2f | %d | %s | %s | %.2f | %.2f | %s |" % (
            h['code'], h['name'], h['s'], h['b'], h['b'], h['bp'], h['lp'], h['lag'],
            '快启动' if h['lag'] <= 5 else '慢启动', h['ql'], h['sh'], h['fl'],
            '⚠️' if h['hp'] else ''))
    os.makedirs(os.path.join('result', 'daily'), exist_ok=True)
    out = os.path.join('result', 'daily', '%s.md' % END)
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n已写入 %s (耗时 %.0fs)' % (out, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
