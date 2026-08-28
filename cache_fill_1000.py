# -*- coding: utf-8 -*-
"""补齐 1000 池日K缓存到 .cache_kline（复用 golden_pit_recall_analysis.fetch_kline_tdx）
用法: python3 cache_fill_1000.py stock_pool_1000.txt 1000 [start]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden_pit_recall_analysis as m

def main():
    pool_file = sys.argv[1] if len(sys.argv) > 1 else 'stock_pool_1000.txt'
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    stocks = []
    for line in open(os.path.join(m.WORKDIR, pool_file)):
        line = line.strip()
        if line:
            stocks.append(line.split(',')[0].strip())
    stocks = stocks[start:start + top]
    t0 = time.time()
    done = ok = skip = 0
    for i, sym in enumerate(stocks):
        if os.path.exists(os.path.join(m.CACHE, f'{sym}.npy')):
            skip += 1
            done += 1
            continue
        d = m.fetch_kline_tdx(sym)
        if d is not None:
            ok += 1
        done += 1
        if done % 100 == 0:
            print(f'  {done}/{len(stocks)} 新增{ok} 耗时{time.time()-t0:.0f}s', flush=True)
    print(f'完成: 处理{done} 新增缓存{ok} 已跳过{skip} 总耗时{time.time()-t0:.0f}s')
    try:
        m.get_client().close()
    except Exception:
        pass

if __name__ == '__main__':
    main()
