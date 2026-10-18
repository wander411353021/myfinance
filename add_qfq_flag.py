# -*- coding: utf-8 -*-
"""给 v3b/early 五年回溯结果附加 qfq_flag（前复权口径质量标记）— 2026-10-18

背景
----
2026-10-18 的假信号因子验证（fake_filter_v3b_5y.py）底座口径是
    early ∩ qfq_flag=0 ∩ fut60_days>=60
但 qfq_flag 的生成脚本随本地 reset 丢失（原临时脚本未入库），导致：
  · 只靠入库代码无法从 backtest_v3b_5y.csv 重建 67,276 行底座；
  · fake_filter / mainrise_deep_v2 依赖该列却无人生成。
本脚本把该字段补成**可复现、可缓存、可断点续跑**的独立后处理步骤。

qfq_flag 定义（本仓库定版）
--------------------------
qfq_flag = 1  信号日之后 QFQ_HORIZON(默认 60) 个交易日内，该股存在
              除权/股本变动记录 → 前复权因子可能重锚，fut60 收益口径受影响；
qfq_flag = 0  窗口内无此类记录；
qfq_flag = NaN 数据窗口不足（i+horizon 超出末端）或接口失败（无法判定）。

数据源（eltdx 通达信直连，非爬虫）
--------------------------------
1) 主路径 client.corporate.capital_changes(code)：取 date + category_raw；
   除权/股本事件类别 = 1(除权除息) / 2(送配股上市) / 3(非流通股上市) /
   5(股本变化) / 9(转配股上市) / 14(送认沽权证)。
   与 tdx_quant._historical_float_shares_records 的流通股本筛选口径分开：
   这里关心"是否发生复权事件"，故 cat1/14 也要计入。
2) 回退 client.get_xdxr(code)：结构随版本可能变化，用 _extract_dates 递归解析。

🚨 红线
------
qfq_flag 是**事后数据质量标记**（用到了信号日之后的除权记录），
只允许用于回溯统计的样本剔除，**严禁进入 signal()/scan 的买入判定**，
否则就是未来函数。signal 层仍只用 <= 当日数据。

用法
----
    python add_qfq_flag.py                 # 读 result/backtest_v3b_5y.csv
    python add_qfq_flag.py --horizon 60 --workers 4
    python add_qfq_flag.py --input result/backtest_v3b_5y.csv \
                           --output result/backtest_v3b_5y_qfq.csv
    python add_qfq_flag.py --limit 20      # 小样本试跑

产物
----
    result/backtest_v3b_5y_qfq.csv         原列 + qfq_flag + qfq_events + qfq_window_days
    result/xdxr_cache/{code}.json         除权日期缓存（可断点续跑）
    result/add_qfq_flag.log               运行日志
"""
import os
import sys
import json
import time
import argparse
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(REPO, 'result', 'backtest_v3b_5y.csv')
DEFAULT_OUTPUT = os.path.join(REPO, 'result', 'backtest_v3b_5y_qfq.csv')
CACHE_DIR = os.path.join(REPO, 'result', 'xdxr_cache')
LOG = os.path.join(REPO, 'result', 'add_qfq_flag.log')
KLINE_CACHE_DIRS = [
    os.path.join(REPO, 'result', 'scan_cache_v3b'),
    os.path.join(REPO, 'result', 'scan_cache'),
]
QFQ_HORIZON = 60
EVENT_CATEGORIES = (1, 2, 3, 5, 9, 14)


def log(msg):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, 'a') as f:
        f.write(time.strftime('%H:%M:%S ') + msg + '\n')
    print(msg, flush=True)


# ────────────────────────── 日期解析 ──────────────────────────
def _to_date(v):
    if v is None:
        return None
    try:
        t = pd.to_datetime(v, errors='coerce')
    except Exception:
        return None
    if pd.isna(t):
        return None
    return pd.Timestamp(t).normalize()


def _extract_dates(obj, depth=0):
    """从 eltdx 未知返回结构里递归提取日期（兼容 dict/list/dataclass/DataFrame）。"""
    out = []
    if obj is None or depth > 4:
        return out
    if isinstance(obj, pd.DataFrame):
        for col in obj.columns:
            if str(col).lower() in ('date', 'ex_date', 'record_date', 'datetime', 'time'):
                for v in obj[col].tolist():
                    d = _to_date(v)
                    if d is not None:
                        out.append(d)
                break
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in ('date', 'ex_date', 'record_date', 'datetime', 'time'):
                d = _to_date(v)
                if d is not None:
                    out.append(d)
            elif isinstance(v, (dict, list, tuple, pd.DataFrame)):
                out.extend(_extract_dates(v, depth + 1))
        return out
    if isinstance(obj, (list, tuple, set)):
        for x in obj:
            out.extend(_extract_dates(x, depth + 1))
        return out
    if isinstance(obj, str):
        d = _to_date(obj)
        return [d] if d is not None else []
    # dataclass / 普通对象
    for name in ('date', 'ex_date', 'record_date', 'datetime', 'time'):
        if hasattr(obj, name):
            d = _to_date(getattr(obj, name))
            if d is not None:
                out.append(d)
            break
    return out


def _fetch_event_dates(code):
    """返回 (sorted dates, source, error)。source ∈ capital_changes / xdxr / empty。"""
    from eltdx import TdxClient
    dates = []
    source = 'empty'
    err = None
    try:
        with TdxClient() as client:
            # 主路径：股本/除权记录（tdx_quant 已在用，结构稳定）
            try:
                cc = client.corporate.capital_changes(code)
                recs = getattr(cc, 'records', cc)
                for r in recs:
                    cat = getattr(r, 'category_raw', None)
                    if cat is None and isinstance(r, dict):
                        cat = r.get('category_raw')
                    if cat in EVENT_CATEGORIES:
                        d = _to_date(getattr(r, 'date', None) if not isinstance(r, dict) else r.get('date'))
                        if d is not None:
                            dates.append(d)
                if dates:
                    source = 'capital_changes'
            except Exception as e:
                err = 'capital_changes: %s' % e
            # 回退：除权除息整理接口
            if not dates:
                try:
                    xd_dates = _extract_dates(client.get_xdxr(code))
                    if xd_dates:
                        dates = xd_dates
                        source = 'xdxr'
                except Exception as e:
                    err = (err + '; ' if err else '') + 'xdxr: %s' % e
    except Exception as e:
        return [], 'error', str(e)
    uniq = sorted(set(dates))
    return uniq, source, err


def _cache_path(code):
    return os.path.join(CACHE_DIR, '%s.json' % code)


def get_event_dates(code, refresh=False):
    """带磁盘缓存的除权日期（失败不缓存，下次重试）。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    fp = _cache_path(code)
    if not refresh and os.path.exists(fp):
        try:
            with open(fp, 'r') as f:
                obj = json.load(f)
            return [pd.Timestamp(d) for d in obj.get('dates', [])], obj.get('source', 'cache'), None
        except Exception:
            pass
    dates, source, err = _fetch_event_dates(code)
    if source != 'error':
        with open(fp, 'w') as f:
            json.dump({'code': code, 'dates': [d.strftime('%Y-%m-%d') for d in dates],
                       'source': source, 'fetched_at': time.strftime('%Y-%m-%d %H:%M:%S')}, f)
    return dates, source, err


# ────────────────────────── 日线日期轴 ──────────────────────────
def _load_dates(code, end_date=None):
    """读本地缓存拿交易日数组；都没有则联网拉一次（只取日期）。"""
    for d in KLINE_CACHE_DIRS:
        fp = os.path.join(d, '%s.csv' % code)
        if os.path.exists(fp):
            try:
                df = pd.read_csv(fp, usecols=['date'])
                return pd.to_datetime(df['date']).dt.normalize().tolist(), 'cache:' + os.path.basename(d)
            except Exception:
                pass
    try:
        import tdx_quant as tq
        df = tq.get_daily_kline_from_tdx(code, end_date or '', 1800, with_turnover=False)
        if df is None or not len(df):
            return [], 'empty'
        return pd.to_datetime(df['date']).dt.normalize().tolist(), 'tdx'
    except Exception as e:
        return [], 'error:%s' % e


def flag_one(args):
    """worker: 给一只股票的所有信号打 qfq_flag。

    args = (code, [{'ridx': int, 'sig_date': 'YYYYMMDD'}, ...], horizon, refresh)
    """
    code, rows, horizon, refresh = args
    dates, dsrc = _load_dates(code)
    events, esrc, eerr = get_event_dates(code, refresh=refresh)
    dates_arr = np.array([np.datetime64(d) for d in dates]) if dates else np.array([])

    out = []
    for r in rows:
        sd = str(r['sig_date'])
        try:
            sig = pd.Timestamp('%s-%s-%s' % (sd[:4], sd[4:6], sd[6:8]))
        except Exception:
            out.append(dict(ridx=r['ridx'], qfq_flag=np.nan, qfq_events=np.nan,
                            qfq_window_days=np.nan, qfq_source='bad_date'))
            continue
        if len(dates_arr) == 0:
            out.append(dict(ridx=r['ridx'], qfq_flag=np.nan, qfq_events=np.nan,
                            qfq_window_days=np.nan, qfq_source=dsrc))
            continue
        i = int(np.searchsorted(dates_arr, np.datetime64(sig)))
        if i >= len(dates_arr) or dates_arr[i] != np.datetime64(sig):
            out.append(dict(ridx=r['ridx'], qfq_flag=np.nan, qfq_events=np.nan,
                            qfq_window_days=np.nan, qfq_source='sig_date_missing'))
            continue
        j = min(i + horizon, len(dates_arr) - 1)
        win_days = j - i
        if win_days < horizon:
            out.append(dict(ridx=r['ridx'], qfq_flag=np.nan, qfq_events=np.nan,
                            qfq_window_days=win_days, qfq_source=esrc))
            continue
        lo, hi = sig, pd.Timestamp(dates_arr[j])
        n_ev = sum(1 for d in events if lo < d <= hi)
        out.append(dict(ridx=r['ridx'], qfq_flag=1 if n_ev > 0 else 0,
                        qfq_events=n_ev, qfq_window_days=win_days, qfq_source=esrc))
    return code, dsrc, esrc, eerr, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=DEFAULT_INPUT)
    ap.add_argument('--output', default=DEFAULT_OUTPUT)
    ap.add_argument('--horizon', type=int, default=QFQ_HORIZON,
                    help='信号日后检查窗口(交易日), 默认 60 与 fut60 对齐')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=None, help='只处理前 N 只股票(试跑)')
    ap.add_argument('--refresh', action='store_true', help='忽略除权缓存重新拉取')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print('输入不存在: %s\n请先运行 backtest_v3b_5y.py 并合并分片' % args.input)
        sys.exit(1)
    df = pd.read_csv(args.input, dtype={'code': str, 'sig_date': str})
    df['code'] = df.code.str.zfill(6)
    df['sig_date'] = df.sig_date.str.zfill(8)
    if 'qfq_flag' in df.columns:
        log('输入已含 qfq_flag, 将覆盖重算')
        df = df.drop(columns=[c for c in ('qfq_flag', 'qfq_events', 'qfq_window_days', 'qfq_source') if c in df.columns])
    df['_ridx'] = np.arange(len(df))

    groups = []
    for code, g in df.groupby('code'):
        groups.append((code, g[['_ridx', 'sig_date']].rename(columns={'_ridx': 'ridx'}).to_dict('records'),
                       args.horizon, args.refresh))
    if args.limit:
        groups = groups[:args.limit]
    log('start: input=%s rows=%d codes=%d horizon=%d workers=%d'
        % (args.input, len(df), len(groups), args.horizon, args.workers))

    results = []
    n_err = 0
    t0 = time.time()
    with Pool(max(1, args.workers)) as pool:
        for k, (code, dsrc, esrc, eerr, rows) in enumerate(
                pool.imap_unordered(flag_one, groups, chunksize=4), 1):
            results.extend(rows)
            if esrc in ('error',) or dsrc.startswith('error'):
                n_err += 1
                if n_err <= 5:
                    log('WARN %s dsrc=%s esrc=%s err=%s' % (code, dsrc, esrc, eerr))
            if k % 200 == 0:
                log('progress %d/%d rows=%d err=%d %.0fs'
                    % (k, len(groups), len(results), n_err, time.time() - t0))

    res = pd.DataFrame(results)
    if len(res) == 0:
        print('没有可写出的结果（输入为空或全部失败）')
        sys.exit(1)
    res = res.drop_duplicates(subset=['ridx']).set_index('ridx')
    # 只保留本次处理过的行（--limit 时其余行保持原样）
    df = df.set_index('_ridx')
    for col in ('qfq_flag', 'qfq_events', 'qfq_window_days', 'qfq_source'):
        df[col] = res[col] if col in res.columns else np.nan
    df = df.reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    n1 = int((df.qfq_flag == 1).sum())
    n0 = int((df.qfq_flag == 0).sum())
    nn = int(df.qfq_flag.isna().sum())
    log('done rows=%d flag1=%d (%.1f%%) flag0=%d nan=%d err_codes=%d -> %s'
        % (len(df), n1, n1 * 100.0 / max(1, n1 + n0), n0, nn, n_err, args.output))
    print('qfq_flag 分布: 1=%d / 0=%d / NaN=%d；NaN 含窗口不足与接口失败' % (n1, n0, nn))


if __name__ == '__main__':
    main()
