# -*- coding: utf-8 -*-
"""每日股票池管线（v3 / v3b / early）— 2026-10-18 reasonix 重写

旧版 daily_pipeline 复用 scan_pressure_breakout（压制→置空突破），与当前信号体系脱节：
  · 入池口径错：等"置空突破"= 已经猛涨之后（skill 2026-10-15 已推翻）；
  · 没有接入 v3 / v3b / early / 大阳确认；
  · 出池只有 DONE_RET=50 一条，缺少 skill 定版的五条规则。

新版流程（严格无未来函数：所有信号判定只用 <= 当天数据）：
  Step1 全市场扫描  challenge_scan.scan_recent(code, end, fresh) → 最近 N 日的 v3/v3b/early
                    （P0-1 结构冷却；P0-2 置空备用线；P1-1 v3 已接入）
  Step2 入池        result/watchlist.csv（去重键 = code + signal_type + sig_date）
  Step3 池内跟踪    逐只更新行情/画像/大阳确认/量能/COST35，跑五条出池规则
  Step4 简报        result/daily_report_{date}.md（新增 / 状态迁移 / 归档复盘 / 活跃明细）

五条出池规则见 pool_rules.py；入池当天即参与（不等大阳确认）。

用法：
  python daily_pipeline.py 20261016                     # 单进程全流程
  python daily_pipeline.py 20261016 --parts 8 --part 0  # 分片扫描（只写 scan_hits 分片）
  python daily_pipeline.py 20261016 --merge             # 合并分片后入库/跟踪/出报告
  python daily_pipeline.py 20261016 --limit 50          # 小样本试跑
  python daily_pipeline.py 20261016 --fresh             # 忽略缓存强制重拉
  python daily_pipeline.py 20261016 --mark 600865       # 人工标记不关注（下次跟踪归档）
  python daily_pipeline.py 20261016 --unmark 600865
"""
import os
import sys
import time
import glob
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tdx_quant as tq
import challenge_scan as cs
import pool_rules as rules

REPO = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(REPO, 'result')
WATCHLIST = os.path.join(RESULT_DIR, 'watchlist.csv')
ARCHIVE = os.path.join(RESULT_DIR, 'watchlist_archive.csv')
DAILY_CACHE = os.path.join(RESULT_DIR, 'scan_cache_daily')
V3B_CACHE = os.path.join(RESULT_DIR, 'scan_cache_v3b')
FRESH_DAYS = 10          # 每日扫描窗口（覆盖周末/短假）
DATALEN = 1000           # 日更取数长度（reg250 + 摸线 250 + 缓冲）

POOL_COLS = [
    'code', 'name', 'signal_type', 'sig_date', 'sig_close', 'entry_date',
    'phase', 'status', 'grade', 'manual_status',
    'vol_ratio', 'rec_depth', 'rec_days', 'cool_days', 'hist_touch',
    'dist_high', 'dev35', 'band', 'profit', 'gap',
    'big_date', 'big_wait', 'big_pct', 'big_vol_ratio',
    'confirm_date', 'days_in_pool', 'days_since_big',
    'last_date', 'last_close', 'post_ret', 'high_ret', 'dd',
    'ma20', 'cost35', 'above_cost35', 'pct_1d', 'vol_ratio_now',
    'post_big_vol_ratio',
    'archive_date', 'archive_reason', 'archive_close', 'archive_ret',
]
SCAN_COLS = ['code', 'name'] + [c for c in cs.RECENT_FIELDS if c != 'code']


# ────────────────────────── 通用工具 ──────────────────────────
def _s(v):
    """安全转字符串：None/NaN -> ''。"""
    if v is None:
        return ''
    try:
        if pd.isna(v):
            return ''
    except (TypeError, ValueError):
        pass
    return str(v)


def _s8(v):
    """日期统一成 YYYYMMDD。"""
    s = _s(v).strip()
    if not s:
        return ''
    s = s[:10].replace('-', '').replace('/', '')
    return s[:8]


def _f(v, default=None):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


def _date_ok(df, end_date):
    try:
        last = str(df['date'].iloc[-1])[:10].replace('-', '')
        return last >= end_date
    except Exception:
        return False


def cached_kline(code, end_date, fresh=False):
    """日更行情缓存：DAILY_CACHE → 回溯缓存 V3B_CACHE → 联网。

    新鲜度口径与 scan_pressure_breakout 一致：末端日期够新，或缓存 12 小时内拉过
    （通达信末端常滞后于请求日）。
    """
    for cache_dir in (DAILY_CACHE, V3B_CACHE):
        fp = os.path.join(cache_dir, '%s.csv' % code)
        if fresh or not os.path.exists(fp):
            continue
        try:
            df = pd.read_csv(fp, parse_dates=['date'])
        except Exception:
            continue
        if len(df) < 400:
            continue
        if _date_ok(df, end_date) or (time.time() - os.path.getmtime(fp)) < 12 * 3600:
            return df
    try:
        df = tq.get_daily_kline_from_tdx(code, end_date, DATALEN, with_turnover=True)
    except Exception:
        return None
    if df is not None and len(df):
        os.makedirs(DAILY_CACHE, exist_ok=True)
        try:
            df.to_csv(os.path.join(DAILY_CACHE, '%s.csv' % code), index=False)
        except Exception:
            pass
    return df


# ────────────────────────── Step1: 扫描 ──────────────────────────
def full_scan(end_date, parts=1, part=0, limit=None, quiet=False, fresh=False):
    """全市场扫描最近 FRESH_DAYS 日的 v3b/early，返回命中 DataFrame。

    分片模式（parts>1）写 result/scan_hits_{end}_part{k}.csv，由 --merge 合并后入库。
    """
    all_df = tq.load_stock_list()
    all_df['code'] = all_df['code'].astype(str).str.zfill(6)
    codes = all_df['code'].tolist()
    names = dict(zip(codes, all_df['name'].fillna('')))
    if parts > 1:
        codes = codes[part::parts]
    if limit:
        codes = codes[:limit]

    rows, n_ok, n_fail = [], 0, 0
    fail_kind = {'no_data': 0, 'short': 0, 'error': 0}
    t0 = time.time()
    for i, code in enumerate(codes):
        try:
            df = cached_kline(code, end_date, fresh=fresh)
            if df is None:
                n_fail += 1
                fail_kind['no_data'] += 1
                continue
            if len(df) < cs.MIN_I + 50:
                n_fail += 1
                fail_kind['short'] += 1
                continue
            hits = cs.scan_recent(code, end_date, fresh=FRESH_DAYS, df=df)
            for r in hits:
                r['name'] = names.get(code, '')
            rows.extend(hits)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            fail_kind['error'] += 1
            if fail_kind['error'] <= 3:
                print('  [!] %s 失败: %s %s' % (code, type(e).__name__, str(e)[:60]), flush=True)
        if not quiet and (i + 1) % 500 == 0:
            print('  [%d/%d] 进度 %d/%d 命中%d 失败%d 耗时%.0fs'
                  % (part + 1, parts, i + 1, len(codes), len(rows), n_fail, time.time() - t0), flush=True)

    df = pd.DataFrame(rows, columns=SCAN_COLS) if rows else pd.DataFrame(columns=SCAN_COLS)
    if len(df):
        df['code'] = df['code'].astype(str).str.zfill(6)
        df = df.drop_duplicates(subset=['code', 'signal_type', 'sig_date']).reset_index(drop=True)
    print('  [分片%d/%d] 完成: 扫描%d只 成功%d 失败%d（无数据%d/数据不足%d/异常%d）命中%d 耗时%.0fs'
          % (part + 1, parts, len(codes), n_ok, n_fail,
             fail_kind['no_data'], fail_kind['short'], fail_kind['error'],
             len(df), time.time() - t0), flush=True)
    if parts > 1:
        os.makedirs(RESULT_DIR, exist_ok=True)
        fp = os.path.join(RESULT_DIR, 'scan_hits_%s_part%d.csv' % (end_date, part))
        df.to_csv(fp, index=False, encoding='utf-8-sig')
        print('  分片结果 -> %s' % fp, flush=True)
    return df


def load_scan_hits(end_date):
    """合并当日分片；无分片返回 None。"""
    files = sorted(glob.glob(os.path.join(RESULT_DIR, 'scan_hits_%s_part*.csv' % end_date)))
    if not files:
        return None
    dfs = []
    for fp in files:
        try:
            if os.path.getsize(fp) == 0:
                continue
            d = pd.read_csv(fp, dtype={'code': str}, encoding='utf-8-sig')
            if len(d):
                dfs.append(d)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame(columns=SCAN_COLS)
    df = pd.concat(dfs, ignore_index=True)
    df['code'] = df['code'].astype(str).str.zfill(6)
    df = df.drop_duplicates(subset=['code', 'signal_type', 'sig_date']).reset_index(drop=True)
    print('合并 %d 个分片 -> %d 个信号' % (len(files), len(df)), flush=True)
    return df


# ────────────────────────── Step2: 入池 ──────────────────────────
def _load_pool(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=POOL_COLS)
    df = pd.read_csv(path, dtype={'code': str}, encoding='utf-8-sig')
    df['code'] = df['code'].astype(str).str.zfill(6)
    for c in POOL_COLS:
        if c not in df.columns:
            df[c] = None
    return df


def _phase_rank(p):
    return {'watching': 0, 'tracking': 1, 'archived': 2}.get(_s(p), 9)


def _grade_rank(g):
    return {'A': 0, 'B': 1, 'C': 2}.get(_s(g), 9)


def _sort_pool(wl):
    if not len(wl):
        return wl
    wl = wl.copy()
    wl['_pr'] = wl['phase'].map(_phase_rank)
    wl['_gr'] = wl['grade'].map(_grade_rank)
    wl = wl.sort_values(['_pr', '_gr', 'sig_date'], ascending=[True, True, False])
    return wl.drop(columns=['_pr', '_gr']).reset_index(drop=True)


def _new_row(r, end_date):
    """扫描命中 -> 池行。入池即参与；若已出大阳则直接进 tracking。"""
    big_date = _s8(r.get('big_date'))
    has_big = bool(big_date)
    phase = rules.PHASE_TRACKING if has_big else rules.PHASE_WATCHING
    grade = rules.grade_of(
        vol_ratio=r.get('vol_ratio'), big_pct=r.get('big_pct'),
        cool_days=r.get('cool_days'), profit=r.get('profit'),
        dev35=r.get('dev35'), has_big=has_big)
    return dict(
        code=r['code'], name=r.get('name', ''), signal_type=r['signal_type'],
        sig_date=_s8(r['sig_date']), sig_close=r.get('sig_close'),
        entry_date=end_date, phase=phase, status='active', grade=grade, manual_status='',
        vol_ratio=r.get('vol_ratio'), rec_depth=r.get('rec_depth'), rec_days=r.get('rec_days'),
        cool_days=r.get('cool_days'), hist_touch=r.get('hist_touch'),
        dist_high=r.get('dist_high'), dev35=r.get('dev35'), band=r.get('band'),
        profit=r.get('profit'), gap=r.get('gap'),
        big_date=big_date if has_big else '',
        big_wait=r.get('big_wait') if has_big else None,
        big_pct=r.get('big_pct') if has_big else None,
        big_vol_ratio=r.get('big_vol_ratio') if has_big else None,
        confirm_date=big_date if has_big else '',
        days_in_pool=0,
        days_since_big=None,
        last_date=end_date, last_close=r.get('sig_close'),
        post_ret=0.0, high_ret=0.0, dd=0.0,
        ma20=None, cost35=None, above_cost35=None, pct_1d=None,
        vol_ratio_now=None, post_big_vol_ratio=None,
        archive_date='', archive_reason='', archive_close=None, archive_ret=None,
    )


def merge_pool(new_df, end_date):
    """新信号合并进池（去重键 code+signal_type+sig_date），返回 (池, 新增数)。"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    wl = _load_pool(WATCHLIST)
    # 旧 pressure 池迁移（2026-10-18）：无 signal_type/phase 的行不属于新管线，
    # 标记 archived 保留可查，不再静默忽略（否则用户会误以为池子丢了）。
    if len(wl):
        legacy_mask = wl['signal_type'].map(_s) == ''
        n_legacy = int(legacy_mask.sum())
        if n_legacy:
            wl.loc[legacy_mask, 'phase'] = rules.PHASE_ARCHIVED
            wl.loc[legacy_mask, 'status'] = 'archived'
            wl.loc[legacy_mask, 'archive_reason'] = 'legacy_pressure_pool'
            wl.loc[legacy_mask, 'archive_date'] = end_date
            print('[!] 检测到 %d 行旧 pressure 池（无 signal_type），已标记 archived，不参与新管线'
                  % n_legacy, flush=True)
    added = 0
    if new_df is not None and len(new_df):
        new_df = new_df.copy()
        new_df['code'] = new_df['code'].astype(str).str.zfill(6)
        exist = set(zip(wl['code'], wl['signal_type'].map(_s), wl['sig_date'].map(_s8)))
        rows = []
        for _, r in new_df.iterrows():
            key = (r['code'], _s(r.get('signal_type')), _s8(r.get('sig_date')))
            if key in exist:
                continue
            rows.append(_new_row(r, end_date))
            exist.add(key)
            added += 1
        if rows:
            wl = pd.concat([wl, pd.DataFrame(rows, columns=POOL_COLS)], ignore_index=True)
    wl = _sort_pool(wl)
    wl.to_csv(WATCHLIST, index=False, encoding='utf-8-sig')
    return wl, added


# ────────────────────────── Step3: 池内跟踪 ──────────────────────────
def _idx_of(darr, date_s):
    target = _s8(date_s)
    if not target:
        return None
    try:
        pos = np.where(np.asarray(darr) == target)[0]
    except Exception:
        return None
    return int(pos[0]) if len(pos) else None


def _metrics(cl, darr, vol, c35s, sig_i, sig_close, big_i=None):
    """计算池内指标（只用 <= 最后一天的数据）。"""
    n = len(cl)
    cur_i = n - 1
    cur = float(cl[cur_i])
    seg = cl[sig_i:]
    hi = float(np.nanmax(seg))
    m = dict(
        days_in_pool=cur_i - sig_i,
        last_date=darr[cur_i], last_close=round(cur, 2),
        post_ret=round((cur / sig_close - 1) * 100, 2) if sig_close else None,
        high_ret=round((hi / sig_close - 1) * 100, 2) if sig_close else None,
        dd=round((cur / hi - 1) * 100, 2) if hi else None,
        ma20=round(float(np.nanmean(cl[max(0, cur_i - 19):cur_i + 1])), 2),
        cost35=_f(c35s[cur_i]),
        above_cost35=(bool(cur >= c35s[cur_i]) if np.isfinite(c35s[cur_i]) else None),
        pct_1d=(round((cur / cl[cur_i - 1] - 1) * 100, 2)
                if cur_i >= 1 and cl[cur_i - 1] > 0 else None),
        vol_ratio_now=(round(float(vol[cur_i] / np.nanmean(vol[max(0, cur_i - 20):cur_i])), 2)
                       if cur_i > 0 and np.nanmean(vol[max(0, cur_i - 20):cur_i]) > 0 else None),
        prev_close=float(cl[cur_i - 1]) if cur_i >= 1 else None,
        prev_cost35=_f(c35s[cur_i - 1]) if cur_i >= 1 else None,
        big_i=big_i, big_found=big_i is not None,
        big_date=None, big_wait=None, big_pct=None, big_vol_ratio=None,
        days_since_big=None, post_big_vol_ratio=None,
    )
    if big_i is not None:
        m['big_date'] = darr[big_i]
        m['big_wait'] = big_i - sig_i
        m['days_since_big'] = cur_i - big_i
        if big_i >= 1 and cl[big_i - 1] > 0:
            m['big_pct'] = round((cl[big_i] / cl[big_i - 1] - 1) * 100, 1)
        base = np.nanmean(vol[max(0, big_i - 20):big_i])
        m['big_vol_ratio'] = round(float(vol[big_i] / base), 2) if base and base > 0 else None
        if m['days_since_big'] >= rules.VOL_CHECK_DAYS:
            post = np.nanmean(vol[big_i + 1:big_i + 1 + rules.VOL_CHECK_DAYS])
            m['post_big_vol_ratio'] = (round(float(post / base), 2)
                                       if base and base > 0 and np.isfinite(post) else None)
    return m


def track_pool(wl, end_date, quiet=False):
    """逐只更新池内行情与状态，执行五条出池规则。

    返回 (wl, snapshot, transitions, newly_archived)
    """
    transitions, newly_archived = [], []
    rows = []
    todo = wl[wl['phase'].map(_s).isin([rules.PHASE_WATCHING, rules.PHASE_TRACKING])]
    for k, (idx, r) in enumerate(todo.iterrows()):
        code = str(r['code']).zfill(6)
        old_phase = _s(r.get('phase')) or rules.PHASE_WATCHING
        out = dict(r)
        try:
            df = cached_kline(code, end_date)
            if df is None or len(df) < cs.MIN_I + 5:
                raise RuntimeError('no_data')
            cl, darr, gt_p, gt2, c35s, _ = cs.compute(df)
            vol = df['volume'].values.astype(float)
            sig_i = _idx_of(darr, r.get('sig_date'))
            if sig_i is None:
                raise RuntimeError('sig_date_missing')
            sig_close = _f(r.get('sig_close'), float(cl[sig_i]))
            # 大阳：优先用池里已记录的日期，否则扫描（事后质量标签，仅用于池状态）
            big_i = _idx_of(darr, r.get('big_date')) if _s8(r.get('big_date')) else None
            if big_i is None:
                bg = cs._first_bigline(cl, sig_i)
                big_i = bg[0] if bg else None
            m = _metrics(cl, darr, vol, c35s, sig_i, sig_close, big_i=big_i)
            manual = _s(r.get('manual_status')).lower() == 'skip'
            new_phase, reason, note = rules.next_phase(
                old_phase,
                days_in_pool=m['days_in_pool'],
                big_found=m['big_found'],
                days_since_big=m['days_since_big'],
                post_big_vol_ratio=m['post_big_vol_ratio'],
                close=m['last_close'], cost35=m['cost35'],
                prev_close=m['prev_close'], prev_cost35=m['prev_cost35'],
                manual_skip=manual)

            # 写回指标
            for key in ('days_in_pool', 'last_date', 'last_close', 'post_ret', 'high_ret', 'dd',
                        'ma20', 'cost35', 'above_cost35', 'pct_1d', 'vol_ratio_now',
                        'big_date', 'big_wait', 'big_pct', 'big_vol_ratio',
                        'days_since_big', 'post_big_vol_ratio'):
                out[key] = m.get(key)
            out['phase'] = new_phase
            out['status'] = 'archived' if new_phase == rules.PHASE_ARCHIVED else 'active'
            if new_phase == rules.PHASE_TRACKING and old_phase == rules.PHASE_WATCHING:
                out['confirm_date'] = m['big_date'] or end_date
            out['grade'] = rules.grade_of(
                vol_ratio=out.get('vol_ratio'), big_pct=out.get('big_pct'),
                cool_days=out.get('cool_days'), profit=out.get('profit'),
                dev35=out.get('dev35'), has_big=bool(_s8(out.get('big_date'))))
            if new_phase == rules.PHASE_ARCHIVED:
                out['archive_date'] = end_date
                out['archive_reason'] = reason or ''
                out['archive_close'] = m['last_close']
                out['archive_ret'] = round(m['post_ret'], 2) if m['post_ret'] is not None else None
                newly_archived.append(dict(out))
            if new_phase != old_phase or note:
                transitions.append(dict(
                    code=code, name=_s(r.get('name')), signal_type=_s(r.get('signal_type')),
                    sig_date=_s8(r.get('sig_date')), old_phase=old_phase, new_phase=new_phase,
                    reason=_s(reason), note=_s(note), post_ret=out.get('post_ret')))
        except Exception as e:
            out['last_date'] = _s(r.get('last_date'))
            transitions.append(dict(
                code=code, name=_s(r.get('name')), signal_type=_s(r.get('signal_type')),
                sig_date=_s8(r.get('sig_date')), old_phase=old_phase,
                new_phase=old_phase, reason='data_missing', note=str(e)[:80],
                post_ret=r.get('post_ret')))
        rows.append(out)
        if not quiet and (k + 1) % 100 == 0:
            print('  跟踪 %d/%d' % (k + 1, len(todo)), flush=True)

    # 合并回池（archived 行保持原样；本次处理的行用 out 覆盖）
    if rows:
        upd = pd.DataFrame(rows, columns=POOL_COLS).set_index(['code', 'signal_type', 'sig_date'])
        base = wl.copy()
        base['sig_date'] = base['sig_date'].map(_s8)
        base = base.set_index(['code', 'signal_type', 'sig_date'])
        base.update(upd)
        wl = base.reset_index()
    wl = _sort_pool(wl)
    snapshot = wl[wl['phase'].map(_s).isin([rules.PHASE_WATCHING, rules.PHASE_TRACKING])].copy()
    return wl, snapshot, transitions, newly_archived


def append_archive(archived_rows, end_date):
    """新归档行追加到 watchlist_archive.csv（去重）。"""
    if not archived_rows:
        return 0
    old = _load_pool(ARCHIVE)
    rows = pd.DataFrame(archived_rows, columns=POOL_COLS)
    rows['code'] = rows['code'].astype(str).str.zfill(6)
    if len(old):
        keys = set(zip(old['code'], old['signal_type'].map(_s), old['sig_date'].map(_s8),
                       old['archive_reason'].map(_s)))
    else:
        keys = set()
    add = []
    for _, r in rows.iterrows():
        key = (r['code'], _s(r.get('signal_type')), _s8(r.get('sig_date')),
               _s(r.get('archive_reason')))
        if key not in keys:
            add.append(dict(r))
            keys.add(key)
    if add:
        out = pd.concat([old, pd.DataFrame(add, columns=POOL_COLS)], ignore_index=True)
        out.to_csv(ARCHIVE, index=False, encoding='utf-8-sig')
    return len(add)


# ────────────────────────── Step4: 简报 ──────────────────────────
def _fmt(v, nd=1, pct=False):
    f = _f(v)
    if f is None:
        return '-'
    fmt = ('%+.' + str(nd) + 'f%%') if pct else ('%.' + str(nd) + 'f')
    return fmt % f


def build_report(new_df, wl, transitions, archived_rows, end_date, added):
    os.makedirs(RESULT_DIR, exist_ok=True)
    lines = ['# v3b / early 每日池 %s' % end_date, '']
    lines.append('## 一、今日新增入池 %d 只（全市场命中 %d，去重后新增 %d）'
                 % (added, len(new_df) if new_df is not None else 0, added))
    if new_df is not None and len(new_df):
        show = new_df.copy()
        show['_g'] = show.apply(lambda r: rules.grade_of(
            vol_ratio=r.get('vol_ratio'), big_pct=r.get('big_pct'),
            cool_days=r.get('cool_days'), profit=r.get('profit'),
            dev35=r.get('dev35'), has_big=bool(_s8(r.get('big_date')))), axis=1)
        show = show.sort_values(['_g', 'sig_date'], ascending=[True, False]).head(40)
        lines.append('')
        lines.append('| 代码 | 名称 | 类型 | 信号日 | 收盘 | 分级 | 量比 | 回踩% | 冷却 | dev35% | gap% | 大阳 |')
        lines.append('|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|')
        for _, r in show.iterrows():
            big = ('%s(+%.1f%%)' % (_s8(r.get('big_date')), _f(r.get('big_pct'), 0.0))
                   if _s8(r.get('big_date')) else '-')
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |'
                         % (r['code'], _s(r.get('name')), _s(r.get('signal_type')),
                            _s8(r.get('sig_date')), _fmt(r.get('sig_close'), 2), r['_g'],
                            _fmt(r.get('vol_ratio'), 2), _fmt(r.get('rec_depth')),
                            _fmt(r.get('cool_days'), 0), _fmt(r.get('dev35')),
                            _fmt(r.get('gap')), big))
    lines.append('')

    lines.append('## 二、状态迁移')
    if transitions:
        to_track = [t for t in transitions if t['new_phase'] == rules.PHASE_TRACKING]
        to_arch = [t for t in transitions if t['new_phase'] == rules.PHASE_ARCHIVED]
        if to_track:
            lines.append('### watching → tracking（大阳确认）%d 只' % len(to_track))
            lines.append('')
            lines.append('| 代码 | 名称 | 类型 | 信号日 | 确认后涨幅 |')
            lines.append('|---|---|---|---|---:|')
            for t in to_track:
                lines.append('| %s | %s | %s | %s | %s |'
                             % (t['code'], t['name'], t['signal_type'], t['sig_date'],
                                _fmt(t.get('post_ret'), 1, True)))
            lines.append('')
        if to_arch:
            lines.append('### 出池归档 %d 只' % len(to_arch))
            lines.append('')
            lines.append('| 代码 | 名称 | 类型 | 信号日 | 原因 | 信号以来% |')
            lines.append('|---|---|---|---|---|---:|')
            for t in to_arch[:60]:
                lines.append('| %s | %s | %s | %s | %s | %s |'
                             % (t['code'], t['name'], t['signal_type'], t['sig_date'],
                                rules.reason_text(t['reason']),
                                _fmt(t.get('post_ret'), 1, True)))
            if len(to_arch) > 60:
                lines.append('| ... | | | | 其余 %d 只见 watchlist_archive.csv | |' % (len(to_arch) - 60))
            lines.append('')
        if not to_track and not to_arch:
            lines.append('- 无状态迁移')
    else:
        lines.append('- 无状态迁移')
    lines.append('')

    lines.append('## 三、归档复盘')
    if archived_rows:
        df = pd.DataFrame(archived_rows)
        g = df.groupby('archive_reason').agg(
            n=('code', 'count'),
            avg_ret=('archive_ret', lambda s: pd.to_numeric(s, errors='coerce').mean()))
        lines.append('')
        lines.append('| 出池原因 | 数量 | 信号以来均值% |')
        lines.append('|---|---:|---:|')
        for reason, r in g.iterrows():
            lines.append('| %s | %d | %s |'
                         % (rules.reason_text(_s(reason)), int(r['n']),
                            _fmt(r['avg_ret'], 1, True) if pd.notna(r['avg_ret']) else '-'))
        lines.append('')
    else:
        lines.append('- 无新增归档')
    lines.append('')

    active = wl[wl['phase'].map(_s).isin([rules.PHASE_WATCHING, rules.PHASE_TRACKING])] if len(wl) else wl
    track = active[active['phase'].map(_s) == rules.PHASE_TRACKING] if len(active) else active
    watch = active[active['phase'].map(_s) == rules.PHASE_WATCHING] if len(active) else active
    lines.append('## 四、池内活跃（tracking %d / watching %d）' % (len(track), len(watch)))
    if len(track):
        lines.append('')
        lines.append('### 已确认（tracking）')
        lines.append('| 代码 | 名称 | 分级 | 信号日 | 现价 | 信号以来% | 距高点% | 跌破COST35 | 大阳后量比 |')
        lines.append('|---|---|---|---|---:|---:|---:|---|---:|')
        for _, r in track.sort_values('post_ret', ascending=False, na_position='last').head(60).iterrows():
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |'
                         % (r['code'], _s(r.get('name')), _s(r.get('grade')),
                            _s8(r.get('sig_date')), _fmt(r.get('last_close'), 2),
                            _fmt(r.get('post_ret'), 1, True), _fmt(r.get('dd'), 1, True),
                            '是' if _s(r.get('above_cost35')) == 'False' else '否',
                            _fmt(r.get('post_big_vol_ratio'), 2)))
    if len(watch):
        lines.append('')
        lines.append('### 等待确认（watching，入池 ≤%d 日）' % rules.WATCH_DAYS)
        lines.append('| 代码 | 名称 | 分级 | 信号日 | 现价 | 信号以来% | 已等待 | 回踩% | 冷却 |')
        lines.append('|---|---|---|---|---:|---:|---:|---:|---:|')
        for _, r in watch.sort_values('days_in_pool', ascending=False, na_position='last').head(60).iterrows():
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |'
                         % (r['code'], _s(r.get('name')), _s(r.get('grade')),
                            _s8(r.get('sig_date')), _fmt(r.get('last_close'), 2),
                            _fmt(r.get('post_ret'), 1, True), _fmt(r.get('days_in_pool'), 0),
                            _fmt(r.get('rec_depth')), _fmt(r.get('cool_days'), 0)))
    report = os.path.join(RESULT_DIR, 'daily_report_%s.md' % end_date)
    with open(report, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return report


# ────────────────────────── CLI ──────────────────────────
def main():
    global FRESH_DAYS
    ap = argparse.ArgumentParser()
    ap.add_argument('end_date', nargs='?', default=time.strftime('%Y%m%d'))
    ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--merge', action='store_true', help='合并分片后入库/跟踪/报告')
    ap.add_argument('--fresh', action='store_true', help='忽略缓存强制重拉')
    ap.add_argument('--fresh-days', type=int, default=FRESH_DAYS, help='扫描最近 N 个交易日')
    ap.add_argument('--mark', default=None, help='人工标记 CODE 不关注（下次跟踪归档）')
    ap.add_argument('--unmark', default=None)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    END = args.end_date
    FRESH_DAYS = args.fresh_days

    print('== 每日池 %s == 分片 %d/%d%s'
          % (END, args.part + 1, args.parts,
             (' limit=%d' % args.limit) if args.limit else ''), flush=True)

    # 人工标记
    if args.mark or args.unmark:
        wl = _load_pool(WATCHLIST)
        target = args.mark or args.unmark
        flag = 'skip' if args.mark else ''
        hit = wl['code'] == str(target).zfill(6)
        wl.loc[hit, 'manual_status'] = flag
        wl.to_csv(WATCHLIST, index=False, encoding='utf-8-sig')
        print('人工标记 %s -> manual_status=%r（%d 行）' % (target, flag, int(hit.sum())), flush=True)

    # Step1
    if args.merge:
        new_df = load_scan_hits(END)
        if new_df is None:
            print('未找到 %s 的分片结果（scan_hits_%s_part*.csv）' % (END, END))
            return
    else:
        new_df = full_scan(END, parts=args.parts, part=args.part,
                           limit=args.limit, quiet=args.quiet, fresh=args.fresh)
        if args.parts > 1:
            print('分片完成；全部跑完后执行: python daily_pipeline.py %s --merge' % END)
            return

    # Step2-4
    wl, added = merge_pool(new_df, END)
    print('股票池: %d 只（新增 %d）' % (len(wl), added), flush=True)
    wl, snapshot, transitions, archived = track_pool(wl, END, quiet=args.quiet)
    wl.to_csv(WATCHLIST, index=False, encoding='utf-8-sig')
    n_arch = append_archive(archived, END)
    report = build_report(new_df, wl, transitions, archived, END, added)
    print('归档新增: %d 行 -> %s' % (n_arch, ARCHIVE), flush=True)
    print('简报: %s' % report, flush=True)
    if len(snapshot):
        fp = os.path.join(RESULT_DIR, 'watch_snapshot_%s.csv' % END)
        snapshot.to_csv(fp, index=False, encoding='utf-8-sig')
        print('快照: %s（%d 只）' % (fp, len(snapshot)), flush=True)


if __name__ == '__main__':
    main()
