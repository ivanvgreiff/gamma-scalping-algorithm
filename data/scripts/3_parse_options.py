#!/usr/bin/env python3
"""
Options Data Parser for Deribit BTC Data (FAST, NO DEDUP)

Parses daily trade files and separates data by individual options.
Creates a feather file for each unique option plus ONE aggregated metadata file
(`options_metadata.feather`) covering all instruments (no per-option metadata.json files).
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def parse_instrument_name(instrument):
    """Parse Deribit instrument name into components.
    
    Format: BTC-19JUL19-10000-P
    Returns: dict with asset, expiry_date, strike_price, option_type
    """
    try:
        parts = instrument.split('-')
        if len(parts) != 4:
            return None
        
        asset = parts[0]
        expiry_str = parts[1]
        strike_price = int(parts[2])
        option_type = 'CALL' if parts[3] == 'C' else 'PUT'
        
        # Parse expiry date (format: 19JUL19)
        expiry_date = datetime.strptime(expiry_str, '%d%b%y').date()
        
        return {
            'asset': asset,
            'expiry_date': expiry_date.isoformat(),
            'strike_price': strike_price,
            'option_type': option_type
        }
    except Exception:
        return None

def generate_metadata(instrument, trades_df, files_count):
    """Generate flat metadata dict for a single option (no nested objects)."""
    parsed = parse_instrument_name(instrument)
    if not parsed:
        return None
    dts = pd.to_datetime(trades_df['timestamp'], unit='us', utc=True)
    return {
        'instrument': instrument,
        'asset': parsed['asset'],
        'expiry_date': parsed['expiry_date'],
        'strike_price': parsed['strike_price'],
        'option_type': parsed['option_type'],
        'total_trades': int(len(trades_df)),
        'total_volume': float(trades_df['quantity'].sum()),
        'first_trade': dts.min().isoformat(),
        'last_trade': dts.max().isoformat(),
        'price_min': float(trades_df['price'].min()),
        'price_max': float(trades_df['price'].max()),
        'iv_min': float(trades_df['iv'].min()),
        'iv_max': float(trades_df['iv'].max()),
        'data_files_processed': int(files_count)
    }

def save_option_data(instrument, trades_df, output_dir):
    """Save trades (metadata handled separately)."""
    option_dir = output_dir / instrument
    option_dir.mkdir(parents=True, exist_ok=True)
    trades_df.sort_values('timestamp').reset_index(drop=True).to_feather(option_dir / 'trades.feather')
    return len(trades_df)

def _read_and_group(file_path: Path, columns):
    """Read a feather file and group by instrument (no dedup)."""
    try:
        df = pd.read_feather(file_path, columns=columns) if columns else pd.read_feather(file_path)
        if df.empty:
            return {}, 0
        groups = {inst: [g] for inst, g in df.groupby('instrument')}
        return groups, len(df)
    except Exception as e:
        print(f"   Error {file_path.name}: {e}")
        return {}, 0

def process_all_files(input_dir, output_dir, parallel=1, columns=None):
    print("=== DERIBIT OPTIONS PARSER (FAST, NO DEDUP) ===\n")
    files = sorted(list(input_dir.glob('*.feather')))
    if not files:
        print(f"  No feather files found in {input_dir}")
        return 0, 0
    print(f"  Found {len(files)} files")
    print(f"  Date range: {files[0].stem} -> {files[-1].stem}")
    start_time = time.time()

    instruments_data: dict[str, list[pd.DataFrame]] = defaultdict(list)
    files_per_instrument = defaultdict(int)
    total_trades = 0

    def handle_result(res_groups, trade_count):
        nonlocal total_trades
        total_trades += trade_count
        for inst, lst in res_groups.items():
            instruments_data[inst].extend(lst)
            files_per_instrument[inst] += 1

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futures = {ex.submit(_read_and_group, fp, columns): fp for fp in files}
            for i, fut in enumerate(as_completed(futures), 1):
                groups, cnt = fut.result()
                handle_result(groups, cnt)
                if i % 50 == 0 or i == len(files):
                    elapsed = time.time() - start_time
                    print(f"  Processed {i}/{len(files)} files ({total_trades:,} trades) in {elapsed:.1f}s")
    else:
        for i, fp in enumerate(files, 1):
            groups, cnt = _read_and_group(fp, columns)
            handle_result(groups, cnt)
            if i % 50 == 0 or i == len(files):
                elapsed = time.time() - start_time
                print(f"  Processed {i}/{len(files)} files ({total_trades:,} trades) in {elapsed:.1f}s")

    print(f"\nAggregating & saving (unique instruments: {len(instruments_data)})")
    saved_options = 0
    saved_trades = 0
    metadata_rows = []
    for idx, (instrument, dfs) in enumerate(instruments_data.items(), 1):
        try:
            combined_df = pd.concat(dfs, ignore_index=True)
            trades_count = save_option_data(instrument, combined_df, output_dir)
            meta = generate_metadata(instrument, combined_df, files_per_instrument[instrument])
            if meta:
                metadata_rows.append(meta)
            saved_options += 1
            saved_trades += trades_count
            if idx % 200 == 0 or idx == len(instruments_data):
                print(f"  Saved {idx}/{len(instruments_data)} instruments")
        except Exception as e:
            print(f"  Error saving {instrument}: {e}")
            continue

    # Write aggregated metadata
    if metadata_rows:
        meta_df = pd.DataFrame(metadata_rows)
        meta_path = output_dir / 'options_metadata.feather'
        meta_df.to_feather(meta_path)
        # Also lightweight CSV for quick human inspection (optional, small index omitted)
        try:
            meta_df.to_csv(output_dir / 'options_metadata.csv', index=False)
        except Exception as e:
            print(f"  (Warning) Could not write CSV metadata: {e}")
        print(f"\n  Aggregated metadata written: {meta_path} ({len(meta_df)} rows)")

    elapsed = time.time() - start_time
    print(f"\nPARSING COMPLETE in {elapsed:.1f}s")
    print(f"  Options saved: {saved_options}")
    print(f"  Total trades saved: {saved_trades:,}")
    print(f"  Output location: {output_dir}")
    return saved_options, saved_trades

def main():
    parser = argparse.ArgumentParser(description='Fast parser for Deribit BTC options into per-instrument datasets + aggregated metadata file.')
    parser.add_argument('--input-dir', default='./data/raw/options/feather/deribit/BTC', help='Directory with daily feather files.')
    parser.add_argument('--output-dir', default='./data/parsed/options', help='Destination directory.')
    parser.add_argument('--parallel', type=int, default=1, help='Number of threads for reading/grouping.')
    parser.add_argument('--columns', nargs='*', help='Subset of columns to load (default loads all).')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        saved_options, saved_trades = process_all_files(
            input_dir=input_dir,
            output_dir=output_dir,
            parallel=args.parallel,
            columns=args.columns,
        )
        print("\nSUMMARY:")
        print(f"  Individual option directories created: {saved_options}")
        print("  Each directory contains: trades.feather")
        print("  Aggregated metadata: options_metadata.feather (+ CSV copy)")
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()