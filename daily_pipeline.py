# -*- coding: utf-8 -*-
"""每日管线: 全市场扫描 -> 股票池累加 -> 池内跟踪分析 -> 每日简报 (2026-10-14 polo4111 定版)。

流程(严格无未来函数, 全部判定只用<=当天数据):
  Step1 全市场扫描: 复用 scan_pressure_breakout.detect(压制期150天 上穿<=15% gap>=10% + 近20天置空 + 收盘>MA20)
  Step2 股票池累加: 新命中 -> result/watchlist.csv (去重, 保留 entry_date/grade/void_date)
  Step3 池内跟踪: 对池内每只更新最新行情 -> cur/ma20/post_ret/当日涨跌/距高点回撤/状态
  Step4 简报: result/daily_report_{date}.md (新加入Top + 池内状态变化 + 特殊事件)

用法: python3 daily_pipeline.py [YYYYMMDD] [--parts 8 --part k]  多进程分片同 scan
"""
import sys, os, time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tdx_quant as tq
import scan_pressure_breakout as scan
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression

WATCHLIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', 'watchlist.csv')
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
DONE_RET = 50  # 置空以来涨幅>=50% 视为"已走完"自动出池(不再跟踪) — 2026-10-14 polo4111 定


def full_scan(end_date, parts=1, part=0, limit=None, quiet=False):
    """Step1: 全市场扫描, 返回今日命中 DataFrame(已分级)。

    2026-10-15 reasonix 补: 支持 --parts/--part 分片(每片独立跑, 结果写
    result/scan_hits_{end}_part{k}.csv; 全量单进程 ~35 分钟 → 8 片并行 ~5 分钟)。
    ⚠️ 失败计数不再静默吞掉(旧实现 except: pass 无法区分"网络失败"与"无信号")。
    """
    all_df = tq.load_stock_list()
    codes = all_df['code'].astype(str).str.zfill(6).tolist()
    names = dict(zip(codes, all_df['name']))
    if parts > 1:
        codes = codes[part::parts]        # 分片: 交错切分, 各片负载均衡
    if limit:
        codes = codes[:limit]
    hits, n_ok, n_fail = [], 0, 0
    t0 = time.time()
    for i, code in enumerate(codes):
        try:
            r = scan.detect(code, end_date)
            if r is not None:
                r['name'] = names.get(code, '')
                hits.append(r)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            if n_fail <= 3:
                print(f'  ⚠ {code} 失败: {type(e).__name__} {str(e)[:60]}', flush=True)
        if not quiet and (i + 1) % 500 == 0:
            print(f'  [{part+1}/{parts}] 进度 {i+1}/{len(codes)} 命中{len(hits)} '
                  f'失败{n_fail} 耗时{time.time()-t0:.0f}s', flush=True)
    df = pd.DataFrame(hits)
    if len(df):
        df['grade'] = df.switch.map({0: 'A', 1: 'B'}).fillna('C')
        df['void_days'] = (pd.Timestamp(end_date) - pd.to_datetime(df.void_date)).dt.days
    print(f'  [分片{part+1}/{parts}] 完成: 扫描{len(codes)}只 成功{n_ok} 失败{n_fail} '
          f'命中{len(df)} 耗时{time.time()-t0:.0f}s', flush=True)
    if parts > 1:
        os.makedirs(RESULT_DIR, exist_ok=True)
        fp = os.path.join(RESULT_DIR, f'scan_hits_{end_date}_part{part}.csv')
        df.to_csv(fp, index=False, encoding='utf-8-sig')
        print(f'  分片结果 -> {fp}', flush=True)
    return df


def load_scan_hits(end_date):
    """合并当日分片结果(scan_hits_{end}_part*.csv); 无分片文件则返回 None。"""
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(RESULT_DIR, f'scan_hits_{end_date}_part*.csv')))
    if not files:
        return None
    dfs = [pd.read_csv(f, encoding='utf-8-sig') for f in files if os.path.getsize(f) > 0]
    dfs = [d for d in dfs if len(d)]
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df['code'] = df['code'].astype(str).str.zfill(6)
    df = df.drop_duplicates(subset=['code', 'void_date']).reset_index(drop=True)
    print(f'合并 {len(files)} 个分片 -> {len(df)} 只命中', flush=True)
    return df


def merge_watchlist(new_df, end_date):
    """Step2: 新命中合并进股票池(去重, 新票记 entry_date)。"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    wl = pd.read_csv(WATCHLIST, encoding='utf-8-sig') if os.path.exists(WATCHLIST) else pd.DataFrame(
        columns=['code', 'name', 'entry_date', 'grade', 'press_abr', 'press_gap', 'switch', 'void_date', 'void_days'])
    if len(wl) and 'base_close' not in wl.columns:
        wl['base_close'] = np.nan   # 旧池兼容: 缺列时补空(跟踪将回退按 void_date 定位)
    added = 0
    if len(new_df):
        new_df['code'] = new_df['code'].astype(str).str.zfill(6)
        # 已存在的不重复加; 用 (code, void_date) 判重(同一股票可能多次信号)
        if len(wl):
            exist_keys = set(zip(wl['code'].astype(str).str.zfill(6), wl['void_date']))
        else:
            exist_keys = set()
        for _, r in new_df.iterrows():
            key = (r['code'], r['void_date'])
            if key in exist_keys:
                continue
            wl = pd.concat([wl, pd.DataFrame([dict(
                code=r['code'], name=r['name'], entry_date=end_date, grade=r['grade'],
                press_abr=r['press_abr'], press_gap=r['press_gap'], switch=r['switch'],
                void_date=r['void_date'], void_days=r['void_days'],
                base_close=r.get('void_close', np.nan))])], ignore_index=True)   # base_close=入池基准价(防基准漂移)
            exist_keys.add(key)
            added += 1
    # 统一排序: grade A->B->C, 再按 entry_date
    order = {'A': 0, 'B': 1, 'C': 2}
    wl['grank'] = wl['grade'].map(order).fillna(9)
    wl = wl.sort_values(['grank', 'entry_date'], ascending=[True, False]).drop(columns=['grank']).reset_index(drop=True)
    wl.to_csv(WATCHLIST, index=False, encoding='utf-8-sig')
    return wl, added


def track_watchlist(wl, end_date):
    """Step3: 池内逐只更新最新行情与状态。"""
    rows = []
    for _, r in wl.iterrows():
        try:
            df = scan.cached_kline(str(r['code']).zfill(6), end_date)
            if df is None or len(df) < 40:
                continue
            c = df['close'].values.astype(float)
            dates = df['date'].values
            ma20 = pd.Series(c).rolling(20).mean().values[-1]
            cur = c[-1]
            # 置空日以来的路径
            # 2026-10-15 reasonix: 基准优先用**入池时记录的 base_close**(固定值), 避免"基准漂移"
            #   —— 数据更新后 base 变化可能使原 void_date 不再是置空日(688789: 入池时 09-22,
            #   次日数据更新后置空首日变 09-23), 用 void_date 重新定位会取到错误基准价
            #   (实测 post_ret +10.6% 虚高, 真值 -0.2%)。缺失 base_close 时回退按日期定位。
            vd = str(r['void_date'])[:10]
            _bc = r.get('base_close', np.nan) if hasattr(r, 'get') else np.nan
            base_idx = None
            if pd.notna(_bc) and float(_bc) > 0:
                _b = float(_bc)
                _mi = np.where(np.abs(c - _b) < 1e-6)[0]
                base_idx = _mi[0] if len(_mi) else None
            else:
                vmask = np.array([str(d)[:10] for d in dates]) == vd
                vi = np.where(vmask)[0]
                base_idx = vi[0] if len(vi) else None
            if base_idx is not None and base_idx < len(c) - 1:
                post_ret = (cur / c[base_idx] - 1) * 100
                high_ret = (np.nanmax(c[base_idx:]) / c[base_idx] - 1) * 100
                dd = (cur / np.nanmax(c[base_idx:]) - 1) * 100 if np.isfinite(np.nanmax(c[base_idx:])) else 0
                pct_1d = (cur / c[-2] - 1) * 100 if len(c) > 1 else 0
            else:
                post_ret = high_ret = dd = pct_1d = np.nan
            rows.append(dict(code=r['code'], name=r['name'], entry_date=str(r['entry_date'])[:10],
                             grade=r['grade'], void_date=str(r['void_date'])[:10],
                             cur=round(cur, 2), ma20=round(ma20, 2),
                             pct_1d=round(pct_1d, 1) if np.isfinite(pct_1d) else None,
                             post_ret=round(post_ret, 1) if np.isfinite(post_ret) else None,
                             high_ret=round(high_ret, 1) if np.isfinite(high_ret) else None,
                             dd=round(dd, 1) if np.isfinite(dd) else None,
                             above_ma20=cur > ma20))
        except Exception:
            pass
    tr = pd.DataFrame(rows)
    return tr


def build_report(new_df, wl, tr, end_date, added):
    """Step4: 生成每日简报。"""
    lines = []
    lines.append(f'# 压制→主升突破 每日简报 {end_date}')
    lines.append('')
    # 今日扫描
    lines.append(f'## 今日扫描: 全市场命中 {len(new_df) if len(new_df) else 0} 只, 新加入股票池 {added} 只')
    if len(new_df):
        g = new_df.grade.value_counts()
        lines.append(f'分级: A级(switch=0) {g.get("A",0)} / B级(switch=1) {g.get("B",0)} / C级(switch>=2) {g.get("C",0)}')
        top = new_df.sort_values('post_ret', ascending=False).head(10)
        lines.append('\n| 代码 | 名称 | 分级 | 上穿% | gap% | 置空日 | 突破以来% |')
        lines.append('|---|---|---|---|---|---|---|')
        for _, r in top.iterrows():
            lines.append(f'| {r.code} | {r["name"]} | {r.grade} | {r.press_abr} | {r.press_gap} | {r.void_date} | {r.post_ret:+.1f} |')
    lines.append('')
    # 池内状态 (已走完 post_ret>=DONE_RET 自动出池, 不再跟踪)
    if tr is not None and len(tr):
        tr = tr.copy()
        tr['code6'] = tr['code'].astype(str).str.zfill(6)
        wl = wl.copy()
        wl['code6'] = wl['code'].astype(str).str.zfill(6)
        done_map = dict(zip(tr['code6'], tr['post_ret'] >= DONE_RET))
        wl['status'] = wl['code6'].map(lambda c: 'done' if done_map.get(c, False) else 'active')
        wl = wl.drop(columns=['code6'])
        # 写回 watchlist(持久化 status)
        wl_order = wl['grade'].map({'A': 0, 'B': 1, 'C': 2}).fillna(9)
        wl = wl.assign(_o=wl_order).sort_values(['_o', 'entry_date'], ascending=[True, False]).drop(columns=['_o'])
        wl.to_csv(WATCHLIST, index=False, encoding='utf-8-sig')
        active = tr[~tr['code6'].isin(tr.loc[tr.post_ret >= DONE_RET, 'code6'])]
        done = tr[tr.post_ret >= DONE_RET]
        lines.append(f'## 股票池跟踪 ({len(active)} 只在池跟踪, 今日更新 {len(tr)} 只)')
        lines.append(f'### 已走完出池 {len(done)} 只 (置空以来涨幅≥{DONE_RET}%):')
        lines.append('| 代码 | 名称 | 分级 | 置空日 | 突破以来% |')
        lines.append('|---|---|---|---|---|')
        for _, r in done.sort_values('post_ret', ascending=False).iterrows():
            lines.append(f'| {r.code6} | {r["name"]} | {r.grade} | {r.void_date} | {r.post_ret:+.1f} |')
        lines.append('')
        tr = active
        lines.append(f'### 池内活跃 {len(tr)} 只今日动态')
        up = tr[tr.pct_1d > 0] if 'pct_1d' in tr else tr.iloc[0:0]
        down = tr[tr.pct_1d < 0] if 'pct_1d' in tr else tr.iloc[0:0]
        lines.append(f'今日上涨 {len(up)} / 下跌 {len(down)}')
        # 特殊事件: 今日跌破 MA20
        broke = tr[(~tr.above_ma20) & (tr.post_ret > 5)]
        if len(broke):
            lines.append('\n### ⚠️ 跌破 MA20(置空后涨幅>5%):')
            for _, r in broke.iterrows():
                lines.append(f'- {r.code6} {r["name"]}: 现价{r.cur} MA20={r.ma20} 置空{r.void_date} 以来{r.post_ret:+.1f}% 回撤{r.dd:.1f}%')
        # 创置空以来新高附近
        near_high = tr[(tr.dd > -3) & (tr.post_ret > 10)]
        if len(near_high):
            lines.append('\n### 🔥 贴近置空以来新高(涨幅>10%):')
            for _, r in near_high.head(15).iterrows():
                lines.append(f'- {r.code6} {r["name"]}: 现价{r.cur} 置空以来{r.post_ret:+.1f}% 距高点{r.dd:.1f}%')
    report = os.path.join(RESULT_DIR, f'daily_report_{end_date}.md')
    with open(report, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return report


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    END = args[0] if args else time.strftime('%Y%m%d')
    parts = int(sys.argv[sys.argv.index('--parts') + 1]) if '--parts' in sys.argv else 1
    part = int(sys.argv[sys.argv.index('--part') + 1]) if '--part' in sys.argv else 0
    limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
    merge_only = '--merge' in sys.argv          # 只合并分片结果(所有分片跑完后执行一次)
    print(f'== 每日管线 {END} == 分片{part+1}/{parts}' + (f' limit={limit}' if limit else ''), flush=True)
    if merge_only:
        merged = load_scan_hits(END)
        if merged is None:
            print(f'未找到 {END} 的分片结果(scan_hits_{END}_part*.csv)'); return
        new_df = merged
    else:
        new_df = full_scan(END, parts=parts, part=part, limit=limit)
        if parts > 1:
            # 分片模式: 每片只写命中文件, 由最后统一 --merge 合并后再入库(避免并发写 watchlist)
            print(f'分片{part+1}/{parts} 完成; 全部跑完后执行: '
                  f'python daily_pipeline.py {END} --merge', flush=True)
            return
    wl, added = merge_watchlist(new_df, END)
    print(f'股票池: {len(wl)} 只 (新增 {added})', flush=True)
    tr = track_watchlist(wl, END)
    report = build_report(new_df, wl, tr, END, added)
    print(f'简报: {report}', flush=True)
    if tr is not None and len(tr):
        tr.to_csv(os.path.join(RESULT_DIR, f'watch_snapshot_{END}.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
