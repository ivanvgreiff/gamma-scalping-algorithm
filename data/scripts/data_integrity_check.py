#!/usr/bin/env python3
"""
Data Integrity Checker for Deribit BTC Data

Checks for:
1. Duplicate trade IDs within and across files
2. Missing trade IDs (gaps in sequence)
3. Date spillover (trades with timestamps from wrong dates)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def check_file_integrity(file_path):
    """Check integrity of a single file."""
    issues = {
        'duplicates': [],
        'date_spillover': [],
        'monotonic_issues': []
    }
    
    try:
        df = pd.read_feather(file_path)
        expected_date = datetime.strptime(file_path.stem, '%Y-%m-%d').date()
        
        # 1. Check for duplicate trade IDs
        if 'id' in df.columns:
            duplicate_ids = df[df.duplicated(subset=['id'], keep=False)]
            if not duplicate_ids.empty:
                issues['duplicates'] = duplicate_ids['id'].tolist()
        
        # 2. Check for date spillover
        if 'timestamp' in df.columns:
            # Convert timestamp to UTC datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='us', utc=True)
            df['date'] = df['datetime'].dt.date
            
            wrong_date_trades = df[df['date'] != expected_date]
            if not wrong_date_trades.empty:
                for _, trade in wrong_date_trades.iterrows():
                    issues['date_spillover'].append({
                        'trade_id': trade['id'],
                        'expected_date': expected_date,
                        'actual_date': trade['date'],
                        'timestamp': trade['datetime'].isoformat()
                    })
        
        # 3. Check monotonic order
        monotonic_issues = check_monotonic_order(df)
        if monotonic_issues:
            issues['monotonic_issues'] = monotonic_issues
        
        return issues, len(df)
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return issues, 0

def check_monotonic_order(df):
    """Check if timestamps and trade IDs are in monotonic increasing order."""
    issues = []
    
    if 'timestamp' in df.columns and len(df) > 1:
        if not df['timestamp'].is_monotonic_increasing:
            issues.append('timestamps_not_monotonic')
    
    if 'id' in df.columns and len(df) > 1:
        # Convert trade IDs to numeric for comparison
        try:
            numeric_ids = pd.to_numeric(df['id'], errors='coerce')
            if not numeric_ids.dropna().is_monotonic_increasing:
                issues.append('trade_ids_not_monotonic')
        except:
            issues.append('trade_ids_not_numeric')
    
    return issues

def get_largest_time_gap(df):
    """Find the largest time gap between consecutive trades."""
    if 'timestamp' in df.columns and len(df) > 1:
        # Convert to datetime for gap calculation
        df_sorted = df.sort_values('timestamp')
        timestamps = pd.to_datetime(df_sorted['timestamp'], unit='us', utc=True)
        
        # Calculate gaps between consecutive trades
        gaps = timestamps.diff().dropna()
        if len(gaps) > 0:
            max_gap = gaps.max()
            max_gap_idx = gaps.idxmax()
            
            return {
                'max_gap_seconds': max_gap.total_seconds(),
                'max_gap_str': str(max_gap),
                'gap_start_time': timestamps.iloc[timestamps.index.get_loc(max_gap_idx) - 1].isoformat(),
                'gap_end_time': timestamps.iloc[timestamps.index.get_loc(max_gap_idx)].isoformat()
            }
    
    return None

def plot_time_gaps(time_gaps_data, data_type):
    """Plot the largest time gaps per file over time."""
    df_gaps = pd.DataFrame(time_gaps_data)
    df_gaps = df_gaps.sort_values('date')
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Main plot - time gaps over time
    plt.subplot(2, 1, 1)
    plt.plot(df_gaps['date'], df_gaps['max_gap_hours'], 'o-', alpha=0.7, markersize=3)
    plt.title(f'Largest Time Gap per File - {data_type.capitalize()} Data')
    plt.ylabel('Max Gap (hours)')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    # Highlight files with large gaps (> 1 hour)
    large_gaps = df_gaps[df_gaps['max_gap_hours'] > 1]
    if len(large_gaps) > 0:
        plt.scatter(large_gaps['date'], large_gaps['max_gap_hours'], 
                   color='red', s=50, alpha=0.8, zorder=5, label='> 1 hour gap')
        plt.legend()
    
    # Histogram of gap sizes
    plt.subplot(2, 1, 2)
    gap_hours_nonzero = df_gaps[df_gaps['max_gap_hours'] > 0]['max_gap_hours']
    if len(gap_hours_nonzero) > 0:
        plt.hist(gap_hours_nonzero, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Time Gaps (excluding zero gaps)')
        plt.xlabel('Gap Size (hours)')
        plt.ylabel('Number of Files')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_gap = gap_hours_nonzero.mean()
        median_gap = gap_hours_nonzero.median()
        plt.axvline(mean_gap, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_gap:.2f}h')
        plt.axvline(median_gap, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_gap:.2f}h')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No time gaps found', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Distribution of Time Gaps')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n📈 TIME GAP STATISTICS for {data_type.upper()}:")
    print(f"  Files with time gaps: {len(gap_hours_nonzero)}/{len(df_gaps)}")
    if len(gap_hours_nonzero) > 0:
        print(f"  Average gap (non-zero): {gap_hours_nonzero.mean():.2f} hours")
        print(f"  Median gap (non-zero): {gap_hours_nonzero.median():.2f} hours")
        print(f"  Files with gaps > 1 hour: {len(large_gaps)}")
        print(f"  Files with gaps > 6 hours: {len(df_gaps[df_gaps['max_gap_hours'] > 6])}")
        print(f"  Files with gaps > 24 hours: {len(df_gaps[df_gaps['max_gap_hours'] > 24])}")

def main():
    print("=== DERIBIT BTC DATA INTEGRITY CHECKER ===\n")
    
    # Define data paths
    options_path = Path('./data/raw/options/feather/deribit/BTC')
    futures_path = Path('./data/raw/futures/feather/deribit/BTC')
    
    total_issues = {
        'duplicates': 0,
        'date_spillover': 0,
        'monotonic_issues': 0
    }
    
    for data_type, data_path in [("OPTIONS", options_path), ("FUTURES", futures_path)]:
        print(f"Checking {data_type} data in: {data_path}")
        
        if not data_path.exists():
            print(f"❌ Path does not exist: {data_path}\n")
            continue
        
        files = sorted(list(data_path.glob('*.feather')))
        if not files:
            print(f"❌ No feather files found in: {data_path}\n")
            continue
        
        print(f"Found {len(files)} files")
        print(f"Date range: {files[0].stem} to {files[-1].stem}")
        
        duplicate_files = []
        spillover_files = []
        monotonic_files = []
        total_trades = 0
        max_time_gap = None
        time_gaps_per_file = []
        
        # Check each file
        for i, file_path in enumerate(files):
            issues, trade_count = check_file_integrity(file_path)
            total_trades += trade_count
            
            # Check time gaps in this file
            try:
                df = pd.read_feather(file_path)
                file_time_gap = get_largest_time_gap(df)
                if file_time_gap:
                    # Track for overall maximum
                    if max_time_gap is None or file_time_gap['max_gap_seconds'] > max_time_gap['max_gap_seconds']:
                        max_time_gap = file_time_gap
                        max_time_gap['file'] = file_path.name
                    
                    # Store for plotting
                    file_date = datetime.strptime(file_path.stem, '%Y-%m-%d')
                    time_gaps_per_file.append({
                        'date': file_date,
                        'file': file_path.name,
                        'max_gap_hours': file_time_gap['max_gap_seconds'] / 3600,
                        'max_gap_seconds': file_time_gap['max_gap_seconds']
                    })
                else:
                    # No gaps found in this file
                    file_date = datetime.strptime(file_path.stem, '%Y-%m-%d')
                    time_gaps_per_file.append({
                        'date': file_date,
                        'file': file_path.name,
                        'max_gap_hours': 0,
                        'max_gap_seconds': 0
                    })
            except:
                pass
            
            # Track files with issues
            if issues['duplicates']:
                duplicate_files.append({
                    'file': file_path.name,
                    'duplicates': issues['duplicates']
                })
                total_issues['duplicates'] += len(issues['duplicates'])
            
            if issues['date_spillover']:
                spillover_files.append({
                    'file': file_path.name,
                    'spillovers': issues['date_spillover']
                })
                total_issues['date_spillover'] += len(issues['date_spillover'])
            
            if issues['monotonic_issues']:
                monotonic_files.append({
                    'file': file_path.name,
                    'issues': issues['monotonic_issues']
                })
                total_issues['monotonic_issues'] += len(issues['monotonic_issues'])
            
            # Progress indicator
            if (i + 1) % 50 == 0 or i == len(files) - 1:
                print(f"  Processed {i + 1}/{len(files)} files...")
        
        # Report results
        print(f"\n--- {data_type} RESULTS ---")
        print(f"Total trades processed: {total_trades:,}")
        
        if duplicate_files:
            print(f"\n❌ DUPLICATE TRADE IDs: {total_issues['duplicates']} duplicates in {len(duplicate_files)} files")
            for file_info in duplicate_files[:5]:  # Show first 5 files
                print(f"  {file_info['file']}: {len(file_info['duplicates'])} duplicates")
                print(f"    IDs: {file_info['duplicates'][:10]}")  # Show first 10 IDs
            if len(duplicate_files) > 5:
                print(f"  ... and {len(duplicate_files) - 5} more files with duplicates")
        else:
            print("✅ NO DUPLICATE TRADE IDs")
        
        if spillover_files:
            print(f"\n❌ DATE SPILLOVER: {total_issues['date_spillover']} trades in {len(spillover_files)} files")
            for file_info in spillover_files[:5]:  # Show first 5 files
                print(f"  {file_info['file']}: {len(file_info['spillovers'])} spillover trades")
                for spillover in file_info['spillovers'][:3]:  # Show first 3 trades
                    print(f"    ID {spillover['trade_id']}: expected {spillover['expected_date']}, actual {spillover['actual_date']}")
            if len(spillover_files) > 5:
                print(f"  ... and {len(spillover_files) - 5} more files with spillover")
        else:
            print("✅ NO DATE SPILLOVER")
        
        if monotonic_files:
            print(f"\n❌ NON-MONOTONIC ORDER: {total_issues['monotonic_issues']} issues in {len(monotonic_files)} files")
            for file_info in monotonic_files[:5]:  # Show first 5 files
                print(f"  {file_info['file']}: {', '.join(file_info['issues'])}")
            if len(monotonic_files) > 5:
                print(f"  ... and {len(monotonic_files) - 5} more files with ordering issues")
        else:
            print("✅ TIMESTAMPS AND TRADE IDs IN MONOTONIC ORDER")
        
        if max_time_gap:
            gap_hours = max_time_gap['max_gap_seconds'] / 3600
            print(f"\n📊 LARGEST TIME GAP: {max_time_gap['max_gap_str']} ({gap_hours:.2f} hours)")
            print(f"  File: {max_time_gap['file']}")
            print(f"  From: {max_time_gap['gap_start_time']}")
            print(f"  To:   {max_time_gap['gap_end_time']}")
        else:
            print(f"\n📊 NO TIME GAP DATA AVAILABLE")
        
        # Plot time gaps over time
        if time_gaps_per_file and len(time_gaps_per_file) > 1:
            plot_time_gaps(time_gaps_per_file, data_type)
        
        print()
    
    # Final summary
    print("=== OVERALL SUMMARY ===")
    total_all_issues = sum(total_issues.values())
    if total_all_issues == 0:
        print("🎉 ALL CHECKS PASSED - Data integrity is perfect!")
    else:
        print(f"⚠️  TOTAL ISSUES FOUND: {total_all_issues}")
        for issue_type, count in total_issues.items():
            if count > 0:
                print(f"  - {issue_type.replace('_', ' ').title()}: {count}")
    
    return total_all_issues

if __name__ == "__main__":
    exit_code = main()
    sys.exit(min(exit_code, 1))  # Return 1 if any issues found, 0 if perfect