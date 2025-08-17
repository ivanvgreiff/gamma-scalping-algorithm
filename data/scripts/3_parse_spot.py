#!/usr/bin/env python3
"""Parse raw Binance spot kline files into a unified parsed dataset.

Reads daily feather files produced by the downloader in
`data/raw/spot/feather/binance/<symbol>/<date>.feather` and concatenates
them into a single feather file under `data/parsed/spot/`.

The output preserves the original 1h bar granularity (or whatever interval
was downloaded) and adds a UTC datetime column.

Usage (defaults to BTCUSDT):
    python data/scripts/3_parse_spot.py --symbol BTCUSDT --interval 1h

Re-running is idempotent; it will overwrite the unified file unless
`--append` is specified (append performs a simple concat + drop_duplicates
on timestamp).
"""

from pathlib import Path
import pandas as pd
import argparse
import sys


def load_daily_files(raw_dir: Path) -> list[pd.DataFrame]:
    files = sorted(raw_dir.glob('*.feather'))
    if not files:
        print(f"No feather files found in {raw_dir}")
        return []

    dfs = []
    total = len(files)
    for i, f in enumerate(files, 1):
        try:
            df = pd.read_feather(f)
            if 'timestamp' not in df.columns:
                print(f"  Skipping {f.name}: no 'timestamp' column")
                continue
            dfs.append(df)
        except Exception as e:
            print(f"  Error reading {f.name}: {e}")

        # progress feedback
        if i % 50 == 0 or i == total:
            print(f"  Loaded {i:,}/{total:,} files")

    return dfs



def unify(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize mixed timestamp units and add datetime column.

    Raw spot files appear to switch units over the history:
    - Early data: microseconds (~1.55e15)
    - Some potential millisecond values (<1e14) in other datasets
    - Later data: nanoseconds (~1.73e18)

    Strategy: vectorized detection per row:
      ns  : timestamp > 1e17 -> divide by 1000
      us  : 1e14 <= ts <= 1e17 -> keep
      ms  : 1e11 <= ts < 1e14  -> *1000
      s   : ts  < 1e11 -> *1_000_000
    Result standardized to microseconds, then converted.
    """
    ts = df['timestamp'].astype('int64').copy()
    mask_ns = ts > 10**17
    mask_us = (ts >= 10**14) & (ts <= 10**17)
    mask_ms = (ts >= 10**11) & (ts < 10**14)
    mask_s  = ts < 10**11

    if mask_ns.any():
        ts.loc[mask_ns] = ts.loc[mask_ns] // 1000  # ns -> us
    if mask_ms.any():
        ts.loc[mask_ms] = ts.loc[mask_ms] * 1000   # ms -> us
    if mask_s.any():
        ts.loc[mask_s] = ts.loc[mask_s] * 1_000_000  # s -> us
    # microseconds left untouched
    df['timestamp'] = ts
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='us', utc=True)

    # Optional: report summary once (small cost) for debugging
    counts = {
        'ns_rows': int(mask_ns.sum()),
        'us_rows': int(mask_us.sum()),
        'ms_rows': int(mask_ms.sum()),
        's_rows': int(mask_s.sum()),
    }
    print(f"Timestamp unit normalization: {counts}")

    cols_pref = [c for c in ['datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
                             'quote_volume', 'trades_count', 'taker_buy_volume', 'taker_buy_quote_volume'] if c in df.columns]
    return df[cols_pref].sort_values('timestamp').reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description='Unify raw spot kline feather files into parsed dataset.')
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--interval', default='1h', help='Downloaded interval (informational only for filename).')
    parser.add_argument('--append', action='store_true', help='Append to existing parsed file instead of overwrite.')
    args = parser.parse_args()

    raw_dir = Path(f'./data/raw/spot/feather/binance/{args.symbol}')
    out_dir = Path('./data/parsed/spot')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'{args.symbol}_{args.interval}.feather'

    if not raw_dir.exists():
        print(f'Raw spot directory not found: {raw_dir}')
        sys.exit(1)

    print(f'=== PARSING SPOT KLINES ({args.symbol}, interval {args.interval}) ===')
    dfs = load_daily_files(raw_dir)
    if not dfs:
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    combined = unify(combined)

    if args.append and out_file.exists():
        try:
            existing = pd.read_feather(out_file)
            merged = pd.concat([existing, combined], ignore_index=True)
            merged = merged.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            combined = merged
        except Exception as e:
            print(f"Warning: could not read existing file for append ({e}); proceeding with overwrite.")

    combined.to_feather(out_file)

    print('Output written:')
    print(f'  File: {out_file}')
    print(f'  Rows: {len(combined):,}')
    print(f"  Date range: {combined['datetime'].min()} to {combined['datetime'].max()}")
    print('=== DONE ===')


if __name__ == '__main__':
    main()
