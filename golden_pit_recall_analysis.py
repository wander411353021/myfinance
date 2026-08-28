# -*- coding: utf-8 -*-
"""黄金坑漏报分析 v2（优化版）：单例连接 + 数据缓存 + 逐步保存结果防环境重置丢失。

现有定版规则：z<-1.5 + 快启动(lch-b<=5) + 坑长(b-s+1>=8) + 出坑确认
分析：用 z_thr=-1.0（更宽）检测所有坑，按上述规则命中与否分组，
     统计各组20日收益，并对"未命中但收益好"的坑做特征归因。
"""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_pit_v2_backtest import compute_rolling_regression, detect_golden_pit_v1, rolling_std

WORKDIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(WORKDIR, '.cache_kline')
RESULT = os.path.join(WORKDIR, 'gp_recall_result.jsonl')

_client = None
def get_client():
    global _client
    if _client is None:
        from eltdx import TdxClient
        _client = TdxClient()
        _client.connect()
    return _client

def fetch_kline_tdx(symbol, datalen=1023, end_date='20260828'):
    """带缓存取数：缓存到 .cache_kline/{symbol}.npy"""
    cache_f = os.path.join(CACHE, f'{symbol}.npy')
    if os.path.exists(cache_f):
        arr = np.load(cache_f, allow_pickle=True)
        return arr.item()
    try:
        client = get_client()
        data = client.bars.get(symbol, period='day', count=datalen,
                               adjust='qfq', anchor_date=end_date, all_pages=True)
        bars = getattr(data, 'bars', None)
        if not bars:
            return None
        rows = []
        for b in bars:
            if float(b.close) > 0:
                rows.append([float(b.close), float(b.high), float(b.low),
                             float(b.volume_lots), b.time])
        # all_pages 返回顺序可能混乱,必须按时间升序排序
        rows.sort(key=lambda r: r[4])
        # 只保留最近 1100 根（tdx 全量太长，拖慢回归）
        rows = rows[-1100:]
        if len(rows) < 310:
            return None
        d = {'close': np.array([r[0] for r in rows], dtype=float),
             'high': np.array([r[1] for r in rows], dtype=float),
             'low': np.array([r[2] for r in rows], dtype=float),
             'vol': np.array([r[3] for r in rows], dtype=float),
             'ts': np.array([int(r[4].timestamp()) for r in rows], dtype=np.int64)}
        np.save(cache_f, d)
        return d
    except Exception:
        return None


def pit_features(c, v, reg250, s, b, lch, symbol, ts=None):
    n = len(c)
    if lch is None or lch + 20 >= n:
        return None
    resid = c - reg250
    rstd = rolling_std(resid, 60)
    if s >= 60:
        pre_std = np.std(resid[s-60:s]) if np.std(resid[s-60:s]) > 0 else rstd[b]
    else:
        pre_std = rstd[b] if rstd[b] > 0 else 1.0
    z_b = resid[b] / pre_std if pre_std > 0 else 0.0
    launch = lch - b
    pit_len = b - s + 1
    w = min(20, b)
    if w >= 10:
        x = np.arange(w); y = reg250[b-w+1:b+1]
        slope = np.polyfit(x, y, 1)[0] / (abs(reg250[b]) + 1e-9)
    else:
        slope = 0.0
    gain20 = c[lch+20] / c[lch] - 1.0
    pre_vol = np.mean(v[max(0, b-20):b]) if b >= 20 else np.mean(v[:b])
    post_peak = np.max(v[lch+1:lch+8]) / (pre_vol + 1e-9) if lch+8 < n else 0.0
    return {
        'z_b': round(z_b, 3), 'launch': int(launch), 'pit_len': int(pit_len),
        'reg_slope': round(slope, 5), 'gain20': round(gain20, 4),
        'post_peak': round(post_peak, 2), 'symbol': symbol,
        'year': int(__import__('datetime').datetime.fromtimestamp(ts[lch]).year) if ts is not None else 0,
    }


def scan(symbol, z_thr=-1.0):
    d = fetch_kline_tdx(symbol)
    if d is None:
        return []
    c = d['close']; v = d['vol']
    reg250, _ = compute_rolling_regression(c, window=250, use_log=True)
    pits = detect_golden_pit_v1(c, reg250, z_thr=z_thr)
    out = []
    for s, b, lch in pits:
        f = pit_features(c, v, reg250, s, b, lch, symbol, d.get('ts'))
        if f:
            out.append(f)
    return out


def main():
    pool_file = sys.argv[1] if len(sys.argv) > 1 else 'stock_pool_300.txt'
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    start_from = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    stocks = []
    for line in open(os.path.join(WORKDIR, pool_file), encoding='utf-8'):
        line = line.strip()
        if line:
            stocks.append(line.split(',')[0].strip())
    stocks = stocks[start_from:start_from + top]
    print(f"扫描 {len(stocks)} 只 (start={start_from}) ...", flush=True)
    t0 = time.time()
    done = 0
    with open(RESULT, 'a') as f:
        for i, sym in enumerate(stocks):
            try:
                zt = float(os.environ.get('GP_ZTHR', '-1.5'))
                for feat in scan(sym, zt):
                    f.write(json.dumps(feat) + '\n')
            except Exception as e:
                pass
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(stocks)} 耗时{time.time()-t0:.0f}s", flush=True)
    print(f"完成 {done} 只, 结果已追加 {RESULT}", flush=True)
    try:
        if _client is not None:
            _client.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()
