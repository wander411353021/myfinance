# -*- coding: utf-8 -*-
"""派生统计: 两种模式 × H∈{40,60,80} × 样本切片。打印全部报告所需数字。"""
import numpy as np
import pandas as pd

df = pd.read_csv('result/confirm_signals_enriched.csv', dtype={'code': str})


def eval_mode(sub, mode, H):
    """返回该切片/模式/horizon 的交易统计 dict。"""
    out = dict()
    if mode == 'A':
        traded = sub[(sub.avail_days >= H) & (sub.buy_off_A <= H)].copy()
        out['no_trade'] = int(len(sub) - len(traded))  # 买入日超 horizon
        if len(traded) == 0:
            return out
        bo = traded.buy_off_A.values
        bp = traded.buy_px_A.values
        # 有效破位: 存在且在 horizon 内且晚于买入日
        brk = traded.brk_off_A.values
        brkpx = traded.brk_px_A.values
        use_brk = np.isfinite(brk) & (brk <= H) & (brk > bo)
        ex_off = np.where(use_brk, brk, H)
        ex_px = np.where(use_brk, brkpx, traded[f'px{H}'].values)
        reason = np.where(use_brk, 'cost_break', 'timeout')
        ret = ex_px / bp - 1.0
        out.update(n=len(traded), no_trade=out['no_trade'],
                   win=float((ret > 0).mean() * 100),
                   mean=float(ret.mean() * 100), med=float(np.median(ret) * 100),
                   p20=float((ret > 0.20).mean() * 100),
                   m10=float((ret < -0.10).mean() * 100),
                   hold=float(np.mean(ex_off - bo)),
                   brk_pct=float((reason == 'cost_break').mean() * 100))
    else:  # B: 信号日买入
        traded = sub[sub.avail_days >= H].copy()
        out['no_trade'] = 0
        if len(traded) == 0:
            return out
        bp = traded.sig_close.values
        brk = traded.brk_off_B.values
        brkpx = traded.brk_px_B.values
        use_brk = np.isfinite(brk) & (brk <= H)
        ex_off = np.where(use_brk, brk, H)
        ex_px = np.where(use_brk, brkpx, traded[f'px{H}'].values)
        reason = np.where(use_brk, 'cost_break', 'timeout')
        ret = ex_px / bp - 1.0
        out.update(n=len(traded),
                   win=float((ret > 0).mean() * 100),
                   mean=float(ret.mean() * 100), med=float(np.median(ret) * 100),
                   p20=float((ret > 0.20).mean() * 100),
                   m10=float((ret < -0.10).mean() * 100),
                   hold=float(np.mean(ex_off)),
                   brk_pct=float((reason == 'cost_break').mean() * 100))
    return out


def fmt(d):
    if 'n' not in d:
        return f"(无成交, no_trade={d.get('no_trade')})"
    return (f"n={d['n']} win={d['win']:.1f}% mean={d['mean']:+.2f}% "
            f"med={d['med']:+.2f}% >20%={d['p20']:.1f}% <-10%={d['m10']:.1f}% "
            f"hold={d['hold']:.1f}d brk={d['brk_pct']:.0f}% no_trade={d.get('no_trade',0)}")


print('=' * 90)
print('【0】样本基数')
v3b = df[df.sig_type == 'v3b']
early = df[df.sig_type == 'early']
print(f"v3b 总体={len(v3b)} (qfq=0: {(v3b.qfq_flag==0).sum()}, qfq=1: {(v3b.qfq_flag==1).sum()})")
print(f"early 总体={len(early)} (qfq=0: {(early.qfq_flag==0).sum()})")

print('=' * 90)
print('【1】核心: v3b + qfq=0, H=60, Mode A(确认后买) vs Mode B(信号日买)')
sub = v3b[v3b.qfq_flag == 0]
print(' Mode A:', fmt(eval_mode(sub, 'A', 60)))
print(' Mode B:', fmt(eval_mode(sub, 'B', 60)))

print('=' * 90)
print('【2】持有期敏感性 (v3b+qfq=0)')
for H in (40, 60, 80):
    print(f' H={H:>2}  A:', fmt(eval_mode(sub, 'A', H)))
for H in (40, 60, 80):
    print(f' H={H:>2}  B:', fmt(eval_mode(sub, 'B', H)))

print('=' * 90)
print('【3】分年度 (v3b+qfq=0, H=60)')
for y in sorted(sub.year.unique()):
    sy = sub[sub.year == y]
    print(f' {y}: n={len(sy)} | A: {fmt(eval_mode(sy,"A",60))}')
    print(f' {y}:          | B: {fmt(eval_mode(sy,"B",60))}')

print('=' * 90)
print('【4】qfq 对比 (v3b, H=60)')
sub_q1 = v3b[v3b.qfq_flag == 1]
print(' qfq=0 A:', fmt(eval_mode(sub, 'A', 60)))
print(' qfq=1 A:', fmt(eval_mode(sub_q1, 'A', 60)))
print(' 全样本A:', fmt(eval_mode(v3b, 'A', 60)))
print(' qfq=0 B:', fmt(eval_mode(sub, 'B', 60)))
print(' qfq=1 B:', fmt(eval_mode(sub_q1, 'B', 60)))
print(' 全样本B:', fmt(eval_mode(v3b, 'B', 60)))

print('=' * 90)
print('【5】early vs v3b (qfq=0, H=60)')
sub_e = early[early.qfq_flag == 0]
print(' early A:', fmt(eval_mode(sub_e, 'A', 60)))
print(' v3b   A:', fmt(eval_mode(sub, 'A', 60)))
print(' early B:', fmt(eval_mode(sub_e, 'B', 60)))
print(' v3b   B:', fmt(eval_mode(sub, 'B', 60)))

print('=' * 90)
print('【6】确认等待天数分布 (v3b+qfq=0)')
c_off = np.minimum(sub.big_off.values, sub.ov_off.values)
print(f" big_off: median={np.median(sub.big_off):.1f} mean={sub.big_off.mean():.2f} "
      f"p25={sub.big_off.quantile(.25):.0f} p75={sub.big_off.quantile(.75):.0f}")
print(f" ov_off : median={np.median(sub.ov_off):.1f} mean={sub.ov_off.mean():.2f} "
      f"p25={sub.ov_off.quantile(.25):.0f} p75={sub.ov_off.quantile(.75):.0f}")
print(f" confirm_off(最早确认): median={np.median(c_off):.1f} mean={c_off.mean():.2f} "
      f"p25={np.percentile(c_off,25):.0f} p75={np.percentile(c_off,75):.0f}")
print(f" buy_off_A=confirm+1: median={np.median(c_off)+1:.1f}; buy_off_A>60 占比="
      f"{(sub.buy_off_A>60).mean()*100:.1f}%")
