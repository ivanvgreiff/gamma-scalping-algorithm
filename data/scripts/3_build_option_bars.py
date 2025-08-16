#!/usr/bin/env python3
"""Build OHLCV / volume / VWAP bar data from per-option trade files.

Reads each option's `trades.feather` (produced by `3_parse_options.py`) and
produces resampled bars for a SINGLE user-specified interval.

Accepted interval syntax ONLY:
    <number>min  (e.g. 1min, 5min, 15min)
    <number>h    (e.g. 1h, 4h)
    <number>d    (e.g. 1d)

Example:
    python data/scripts/4_build_option_bars.py --interval 5min

Output file: bars_<interval>.feather (interval exactly as provided, lower‑cased).
"""

from pathlib import Path
import pandas as pd
import argparse
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed


def _prepare_trades(trades_path: Path) -> pd.DataFrame | None:
    if not trades_path.exists():
        return None
    df = pd.read_feather(trades_path)
    if df.empty:
        return None
    required = {'timestamp', 'price', 'quantity'}
    if not required.issubset(df.columns):
        return None
    # Normalize timestamp units (expect microseconds; convert ms if needed)
    ts_sample = int(df['timestamp'].iloc[0])
    if ts_sample < 10**14:  # likely ms
        df['timestamp'] = df['timestamp'] * 1000
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='us', utc=True)
    df['notional'] = df['price'] * df['quantity']
    df = df.sort_values('datetime').set_index('datetime')
    return df


def build_interval_bars(df: pd.DataFrame, pandas_interval: str) -> pd.DataFrame:
    o = df['price'].resample(pandas_interval).first()
    h = df['price'].resample(pandas_interval).max()
    l = df['price'].resample(pandas_interval).min()
    c = df['price'].resample(pandas_interval).last()
    vol = df['quantity'].resample(pandas_interval).sum()
    trades_ct = df['price'].resample(pandas_interval).count()
    notional = df['notional'].resample(pandas_interval).sum()
    vwap = notional / vol.replace({0: pd.NA})
    bars = pd.DataFrame({
        'open': o,
        'high': h,
        'low': l,
        'close': c,
        'volume': vol,
        'trades': trades_ct,
        'vwap': vwap,
    })
    bars = bars.dropna(subset=['open', 'high', 'low', 'close'])
    bars.reset_index(inplace=True)
    bars['timestamp'] = (bars['datetime'].astype('int64') // 1000) * 1000
    return bars


def process_option_dir(option_dir: Path, interval_display: str, pandas_interval: str, skip_existing: bool) -> tuple[str, Path | None]:
    trades_path = option_dir / 'trades.feather'
    df = _prepare_trades(trades_path)
    if df is None:
        return option_dir.name, None
    out_file = option_dir / f'bars_{interval_display}.feather'
    if skip_existing and out_file.exists():
        return option_dir.name, out_file
    try:
        bars = build_interval_bars(df, pandas_interval)
        if not bars.empty:
            bars.to_feather(out_file)
            return option_dir.name, out_file
    except Exception as e:
        print(f"  Failed {interval_display} {option_dir.name}: {e}")
    return option_dir.name, None


def filter_option_dirs(base_dir: Path, expiry: str | None, opt_type: str | None) -> list[Path]:
    dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if expiry:
        dirs = [d for d in dirs if f'-{expiry}-' in d.name]
    if opt_type:
        # opt_type expected 'C' or 'P'
        dirs = [d for d in dirs if d.name.endswith(f'-{opt_type}')]
    return dirs


def main():
    parser = argparse.ArgumentParser(description='Build OHLCV/VWAP bars from option trade data for a single interval.')
    parser.add_argument('--base-dir', default='./data/parsed/options', help='Directory containing per-option folders.')
    parser.add_argument('--interval', default='1h', help='Interval like 5min, 1h, 1d (also accepts pandas aliases 5T,1H,1D)')
    parser.add_argument('--max-workers', type=int, default=8)
    parser.add_argument('--expiry', help='Filter option directories by expiry fragment (e.g. 10APR21)')
    parser.add_argument('--type', choices=['C', 'P'], help='Filter by option type (C or P)')
    parser.add_argument('--progress-every', type=int, default=200, help='Print progress every N instruments (default 200)')
    parser.add_argument('--skip-existing', action='store_true', help='Skip intervals whose bar files already exist')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f'Base directory not found: {base_dir}')
        sys.exit(1)

    option_dirs = filter_option_dirs(base_dir, args.expiry, args.type)

    # (quiet warnings option removed by user request)

    print(f'=== BUILDING OPTION BARS ===')
    print(f'Options found: {len(option_dirs)} (after filtering)')

    # Validate but do not transform interval (pandas supports e.g. 5min, 1h, 1d directly)
    interval_display = args.interval.strip().lower()
    import re
    if not re.fullmatch(r'\d+(min|h|d)', interval_display):
        print(f"Invalid interval '{args.interval}'. Use forms like 5min, 1h, 1d")
        sys.exit(1)
    pandas_interval = interval_display  # direct use
    print(f'Interval: {interval_display}')

    completed = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(process_option_dir, d, interval_display, pandas_interval, args.skip_existing): d for d in option_dirs}
        for fut in as_completed(futures):
            name, out_file = fut.result()
            completed += 1
            if completed % args.progress_every == 0 or completed == len(option_dirs):
                status = out_file.name if out_file else 'no bars'
                print(f'  [{completed}/{len(option_dirs)}] {name} -> {status}')

    print('=== DONE ===')


if __name__ == '__main__':
    main()
