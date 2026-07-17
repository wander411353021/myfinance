"""
Daily multi-threaded stock scanner (V10 fast_mode).
Scans all A-share stocks for buy signals on a given date.

Usage:
    D:/ProgramData/miniconda3/envs/chip_analyzer/python.exe daily_scan.py 20260625
    D:/ProgramData/miniconda3/envs/chip_analyzer/python.exe daily_scan.py 20260625 --threads 16
"""
import sys, os, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, r'e:\chip_analyzer_ui\new_algo')

# Must set before any matplotlib import
import matplotlib; matplotlib.use('Agg')

from price_segmenter_v10 import run_segmentation
from juejing import get_stock_klines_from_juejing, get_all_stock_info
import pandas as pd

# ── Thread-safe print ──────────────────────────────────────
_print_lock = Lock()

def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


# ── Worker ─────────────────────────────────────────────────
def scan_stock(code, name, end_date):
    """
    Scan a single stock for V10 buy signals.
    Returns (code, name) if buy signal found, None otherwise.
    """
    try:
        df = get_stock_klines_from_juejing(code, end_date, fqt=1)
        if df is None or len(df) == 0:
            return None
        df = df[df['close'] > 0].reset_index(drop=True)
        if len(df) < 50:  # too few bars
            return None
        df['date'] = pd.to_datetime(df['date'])
        has_buy = run_segmentation(df, name=code, fast_mode=True)
        if has_buy:
            return (code, name)
    except Exception as e:
        tprint(f"  [{code}] error: {e}")
    return None


def scan_batch(codes, end_date, tid):
    """Scan a batch of stocks (called by each thread)."""
    results = []
    total = len(codes)
    for i, (code, name) in enumerate(codes):
        res = scan_stock(code, name, end_date)
        if res:
            results.append(res)
        if (i + 1) % 100 == 0 or i == total - 1:
            tprint(f"  Thread {tid}: {i+1}/{total} done, found {len(results)} so far")
    return results


# ── Main ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Daily V10 stock scanner')
    parser.add_argument('end_date', help='End date in YYYYMMDD format')
    parser.add_argument('--threads', '-t', type=int, default=8,
                        help='Max threads (default: 8)')
    args = parser.parse_args()

    end_date = args.end_date
    max_threads = args.threads

    print(f"=== Daily Scanner V10 ===")
    print(f"Date: {end_date}, Threads: {max_threads}")
    print()

    # Load stock list
    print("Loading stock list...")
    stocks_df = get_all_stock_info()
    codes = list(zip(stocks_df['code'].values, stocks_df['name'].values))
    total = len(codes)
    print(f"Total stocks: {total}")
    print()

    # Split into batches for each thread
    batch_size = (total + max_threads - 1) // max_threads
    batches = []
    for t in range(max_threads):
        start = t * batch_size
        end = min(start + batch_size, total)
        if start < total:
            batches.append((codes[start:end], t))
    
    print(f"Batches: {len(batches)}, ~{batch_size} stocks each")
    print()

    # Run scan
    t0 = time.time()
    all_results = []

    with ThreadPoolExecutor(max_workers=len(batches)) as executor:
        futures = {executor.submit(scan_batch, batch, end_date, tid): tid 
                   for batch, tid in batches}
        for future in as_completed(futures):
            tid = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                tprint(f"  Thread {tid} finished: {len(results)} signals")
            except Exception as e:
                tprint(f"  Thread {tid} failed: {e}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Sort and save
    all_results.sort(key=lambda x: x[0])
    out_path = f"e:/chip_analyzer_ui/new_algo/result/{end_date}_signals.csv"
    df_out = pd.DataFrame(all_results, columns=['code', 'name'])
    df_out.to_csv(out_path, index=False)

    print(f"Signals found: {len(all_results)}")
    print(f"Saved to: {out_path}")


if __name__ == '__main__':
    main()
