"""
Gamma scalping strategy implementation.
Uses per-option simulation with all available data for clear P&L attribution.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from typing import List, Dict, Any
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from simulation.gamma_scalping_simulator import GammaScalpingSimulator, GammaScalpingResult
from backtest.backtest_engine import BacktestConfig, StrategyConfig
from concurrent.futures import ThreadPoolExecutor, as_completed


def analyze_option_chain_from_dict(
    options_dict: Dict[str, pd.DataFrame],
    spot_data: pd.DataFrame,
    target_dte: int = 30
) -> pd.DataFrame:
    """Analyze available option chain from per-option dict and select tradeable options.

    Computes liquidity and moneyness using spot_data.
    """
    if not options_dict or spot_data.empty:
        return pd.DataFrame()
    
    # Build a compact metadata frame per option
    rows = []
    for symbol, df in options_dict.items():
        if df.empty:
            continue
        # Ensure required cols exist
        required = {'timestamp','strike','expiry','option_type','close'}
        if not required.issubset(df.columns):
            continue
        df_sorted = df.sort_values('timestamp')
        start_time = df_sorted['timestamp'].min()
        
        # Get spot price when this option first appears for moneyness calculation
        spot_row = spot_data[spot_data['timestamp'] >= start_time].head(1)
        if spot_row.empty:
            continue
        start_spot = spot_row['close'].iloc[0]
        if pd.isna(start_spot) or start_spot <= 0:
            continue
            
        # Calculate DTE from when option first appears (its natural lifecycle)
        expiry = df_sorted['expiry'].iloc[0]
        dte = (expiry - start_time).days
        
        # Skip if option is already expired when it appears
        if dte <= 0:
            continue
            
        avg_volume = df['volume'].mean() if 'volume' in df.columns else 0.0
        avg_oi = df['open_interest'].mean() if 'open_interest' in df.columns else 0.0
        moneyness = df_sorted['strike'].iloc[0] / start_spot
        
        rows.append({
            'symbol': symbol,
            'strike': df_sorted['strike'].iloc[0],
            'expiry': expiry,
            'option_type': df_sorted['option_type'].iloc[0],
            'dte': dte,  # Natural DTE when option appears
            'moneyness': moneyness,
            'avg_volume': avg_volume,
            'avg_oi': avg_oi,
            'num_observations': len(df_sorted),
            'first_timestamp': start_time  # Track when option enters our dataset
        })
    unique_options = pd.DataFrame(rows)
    
    # We already built all metrics in rows above
    df = unique_options.copy()
    
    # Filter for liquid options near target DTE
    if not df.empty:
        df = df[
            (df['dte'] >= target_dte - 10) & 
            (df['dte'] <= target_dte + 10) &
            (df['num_observations'] >= 20)  # Ensure enough data
        ]
        
        # Classify moneyness
        df['moneyness_class'] = pd.cut(
            df['moneyness'],
            bins=[0, 0.95, 1.05, float('inf')],
            labels=['ITM', 'ATM', 'OTM']
        )
    
    return df


def run_single_option_backtest(
    spot_data: pd.DataFrame,
    option_data: pd.DataFrame,
    strategy_config: StrategyConfig,
    simulator: GammaScalpingSimulator
) -> GammaScalpingResult:
    """Run backtest for a single option"""
    
    # Get option data for this specific contract
    if option_data.empty:
        return None
        
    # Align by timestamp intersection
    common_ts = np.intersect1d(spot_data['timestamp'].values, option_data['timestamp'].values)
    if len(common_ts) < 10:
        return None
    spot_aligned = spot_data[spot_data['timestamp'].isin(common_ts)].sort_values('timestamp')
    option_aligned = option_data[option_data['timestamp'].isin(common_ts)].sort_values('timestamp')
    
    if len(spot_aligned) < 10 or len(option_aligned) < 10:
        return None
        
    # Run simulation
    result = simulator.simulate(
        spot_prices=spot_aligned,
        option_data=option_aligned,
        hedge_threshold=strategy_config.hedge_threshold,
        position_size=strategy_config.position_size
    )
    
    return result


def run_portfolio_backtest(
    strategy_configs: List[StrategyConfig],
    spot_data: pd.DataFrame,
    options_dict: Dict[str, pd.DataFrame],
    max_options: int = 50,
    target_dte: int = 30,
    moneyness_filter: str = 'ATM',  # 'ATM', 'OTM', 'ITM', or 'all'
    select: str = 'filtered',       # 'filtered' uses DTE/moneyness/liquidity; 'all' runs every option
    workers: int = 1,               # >1 uses threads to run options in parallel per strategy
    aggregate_timeseries: bool = False,
    return_raw: bool = False
) -> pd.DataFrame | Dict[str, Any]:
    """
    Run backtests across multiple options and strategies.
    
    Args:
        strategy_configs: List of strategy configurations to test
        options_data: Options DataFrame with required columns (and spot_price if no spot_data provided)
        spot_data: Optional spot price DataFrame with required columns
        max_options: Maximum number of options to test
        target_dte: Target days to expiry
        moneyness_filter: Filter for option moneyness
    """
    
    print("=" * 60)
    print("GAMMA SCALPING BACKTEST")
    print("=" * 60)
    # Derive period from spot data
    if not spot_data.empty:
        data_start = pd.to_datetime(spot_data['timestamp']).min()
        data_end = pd.to_datetime(spot_data['timestamp']).max()
        print(f"Period: {data_start.date()} to {data_end.date()}")
    print(f"Target DTE: {target_dte} days")
    print(f"Moneyness filter: {moneyness_filter}")
    if select == 'all':
        print("Selection: all options (no chain filtering)")
    else:
        print("Selection: filtered by DTE/moneyness/liquidity")
    if workers and workers > 1:
        print(f"Parallel: threads x{workers}")
    print()
    
    # Validate inputs
    # Validate inputs
    spot_required = {"timestamp", "close"}
    if not spot_required.issubset(spot_data.columns):
        raise ValueError(f"spot_data missing columns: {sorted(spot_required - set(spot_data.columns))}")
    # At least one option
    if not options_dict:
        print("No options provided")
        return pd.DataFrame()
    
    # Ensure datetime types
    # Ensure datetime types for spot and options
    if not np.issubdtype(spot_data["timestamp"].dtype, np.datetime64):
        spot_data = spot_data.copy()
        spot_data["timestamp"] = pd.to_datetime(spot_data["timestamp"]) 
    # Normalize option frames
    norm_options: Dict[str, pd.DataFrame] = {}
    total_rows = 0
    for sym, df in options_dict.items():
        if df.empty:
            continue
        if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"]) 
        if not np.issubdtype(df["expiry"].dtype, np.datetime64):
            df = df.copy()
            df["expiry"] = pd.to_datetime(df["expiry"]).dt.tz_localize(None)
        norm_options[sym] = df
        total_rows += len(df)
    options_dict = norm_options
    print(f"Loaded spot rows: {len(spot_data):,} | options rows total: {total_rows:,}")
    
    # Analyze option chain
    # Build selection (no heavy concatenation of full data)
    if select == 'all':
        selected_symbols = [sym for sym, df in options_dict.items() if not df.empty]
        option_chain = None
        print(f"Selected {len(selected_symbols)} options for testing (all)")
    else:
        print("\nAnalyzing option chain...")
        option_chain = analyze_option_chain_from_dict(options_dict, spot_data, target_dte)
        if option_chain.empty:
            print("No suitable options found")
            return pd.DataFrame()

        # Apply moneyness filter
        if moneyness_filter != 'all':
            option_chain = option_chain[option_chain['moneyness_class'] == moneyness_filter]
        # Sort by liquidity (volume * OI) and take top options
        option_chain['liquidity_score'] = option_chain['avg_volume'] * np.sqrt(option_chain['avg_oi'] + 1)
        option_chain = option_chain.nlargest(min(max_options, len(option_chain)), 'liquidity_score')
        selected_symbols = option_chain['symbol'].tolist()

        print(f"Selected {len(selected_symbols)} options for testing")
        if not option_chain.empty:
            print(f"Strike range: {option_chain['strike'].min():.0f} - {option_chain['strike'].max():.0f}")
            print(f"DTE range: {option_chain['dte'].min()} - {option_chain['dte'].max()} days")
    
    # Initialize simulator with consistent parameters
    simulator = GammaScalpingSimulator(
        commission_rate=0.0005,
        slippage_bps=10,
        risk_free_rate=0.01
    )
    
    # Run backtests
    all_results = []
    ts_frames: List[pd.DataFrame] = []  # collect per-strategy time series if requested
    raw_results_map: Dict[str, List[GammaScalpingResult]] = {} if return_raw else None
    
    for strategy_config in strategy_configs:
        print(f"\n--- Testing strategy: {strategy_config.name} ---")
        strategy_results = []

        # Collect full GammaScalpingResult objects if requested (for downstream notebook wrangling)
        collect_raw = aggregate_timeseries or return_raw
        raw_results: List[GammaScalpingResult] = [] if collect_raw else None

        if workers and workers > 1:
            # Parallel execution per option (thread-based)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = []
                for sym in selected_symbols:
                    opt_data = options_dict.get(sym, pd.DataFrame())
                    if opt_data.empty:
                        continue
                    futures.append(ex.submit(
                        run_single_option_backtest,
                        spot_data,
                        opt_data,
                        strategy_config,
                        simulator  # Use the same simulator instance
                    ))

                pbar = tqdm(total=len(futures), desc="Options", leave=False)
                for fut in as_completed(futures):
                    result = fut.result()
                    pbar.update(1)
                    if result:
                        if collect_raw:
                            raw_results.append(result)
                        strategy_results.append({
                            'strategy': strategy_config.name,
                            'hedge_threshold': strategy_config.hedge_threshold,
                            'symbol': result.option_symbol,
                            'strike': result.strike,
                            'expiry': result.expiry,
                            'option_type': result.option_type,
                            'option_pnl': result.option_pnl,
                            'hedge_pnl': result.hedge_pnl,
                            'total_pnl': result.total_pnl,
                            'commission_cost': result.commission_cost,
                            'slippage_cost': result.slippage_cost,
                            'num_hedges': result.num_hedges,
                            'avg_delta': result.avg_delta,
                            'avg_gamma': result.avg_gamma,
                            'gamma_pnl': result.total_gamma_pnl,
                            'theta_cost': result.total_theta_cost
                        })
                pbar.close()
        else:
            # Sequential execution
            pbar = tqdm(total=len(selected_symbols), desc="Options")
            for sym in selected_symbols:
                opt_data = options_dict.get(sym, pd.DataFrame())
                if opt_data.empty:
                    pbar.update(1)
                    continue
                result = run_single_option_backtest(
                    spot_data=spot_data,
                    option_data=opt_data,
                    strategy_config=strategy_config,
                    simulator=simulator
                )
                if result:
                    if collect_raw:
                        raw_results.append(result)
                    strategy_results.append({
                        'strategy': strategy_config.name,
                        'hedge_threshold': strategy_config.hedge_threshold,
                        'symbol': result.option_symbol,
                        'strike': result.strike,
                        'expiry': result.expiry,
                        'option_type': result.option_type,
                        'option_pnl': result.option_pnl,
                        'hedge_pnl': result.hedge_pnl,
                        'total_pnl': result.total_pnl,
                        'commission_cost': result.commission_cost,
                        'slippage_cost': result.slippage_cost,
                        'num_hedges': result.num_hedges,
                        'avg_delta': result.avg_delta,
                        'avg_gamma': result.avg_gamma,
                        'gamma_pnl': result.total_gamma_pnl,
                        'theta_cost': result.total_theta_cost
                    })
                pbar.update(1)
            pbar.close()

        # Per-strategy summary and optional collections
        if strategy_results:
            strategy_df = pd.DataFrame(strategy_results)

            # Calculate strategy-level statistics
            total_pnl = strategy_df['total_pnl'].sum()
            avg_pnl = strategy_df['total_pnl'].mean()
            win_rate = (strategy_df['total_pnl'] > 0).mean()

            print(f"\nResults for {strategy_config.name}:")
            print(f"  Total P&L: ${total_pnl:,.2f}")
            print(f"  Average P&L per option: ${avg_pnl:,.2f}")
            print(f"  Win rate: {win_rate:.1%}")
            print(f"  Options traded: {len(strategy_df)}")

            all_results.extend(strategy_results)

            if return_raw and raw_results is not None:
                # Store raw results keyed by strategy
                raw_results_map[strategy_config.name] = raw_results

            if aggregate_timeseries and raw_results:
                # Merge per-option P&L histories and average across options at each timestamp
                merged_ts: pd.DataFrame | None = None
                for res in raw_results:
                    if res.pnl_history is None or res.pnl_history.empty:
                        continue
                    ts = res.pnl_history[['timestamp', 'total_pnl']].copy()
                    ts = ts.rename(columns={'total_pnl': f'pnl_{res.option_symbol}'})
                    if merged_ts is None:
                        merged_ts = ts
                    else:
                        merged_ts = merged_ts.merge(ts, on='timestamp', how='outer')
                if merged_ts is not None and not merged_ts.empty:
                    merged_ts = merged_ts.sort_values('timestamp')
                    value_cols = [c for c in merged_ts.columns if c.startswith('pnl_')]
                    merged_ts['avg_total_pnl'] = merged_ts[value_cols].mean(axis=1, skipna=True)
                    merged_ts['sum_total_pnl'] = merged_ts[value_cols].sum(axis=1, skipna=True)
                    merged_ts['strategy'] = strategy_config.name
                    merged_ts['active'] = merged_ts[value_cols].notna().sum(axis=1)
                    ts_frames.append(merged_ts[['timestamp', 'strategy', 'avg_total_pnl', 'sum_total_pnl', 'active']])
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    if not results_df.empty:
        # Add summary statistics by strategy
        summary = results_df.groupby(['strategy', 'hedge_threshold']).agg({
            'total_pnl': ['sum', 'mean', 'std'],
            'option_pnl': 'sum',
            'hedge_pnl': 'sum',
            'commission_cost': 'sum',
            'slippage_cost': 'sum',
            'num_hedges': 'mean',
            'gamma_pnl': 'sum',
            'theta_cost': 'sum'
        }).round(2)
        
        print("\n" + "=" * 60)
        print("SUMMARY BY STRATEGY")
        print("=" * 60)
        print(summary)
    
    # Build return payloads
    if aggregate_timeseries and ts_frames:
        ts_df = pd.concat(ts_frames, ignore_index=True)
    else:
        ts_df = None

    if return_raw or aggregate_timeseries:
        return {
            'summary_df': results_df,
            'timeseries_df': ts_df,
            'raw_results': raw_results_map if return_raw else None,
        }

    # Backward-compatible return
    return results_df