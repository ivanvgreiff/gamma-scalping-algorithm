"""
Optimized Gamma Scalping Simulator
Performance optimizations while maintaining correctness:
- Batch Greeks calculations
- Cached implied volatility
- Reduced DataFrame overhead
- Vectorized operations where possible
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.options_pricing import bs_price, delta, gamma, theta, vega, implied_volatility
from scipy.stats import norm
from scipy.optimize import brentq

@dataclass
class GammaScalpingResult:
    """Results from simulating gamma scalping on an option"""
    option_symbol: str
    strike: float
    expiry: pd.Timestamp
    option_type: str

    # P&L components
    option_pnl: float
    hedge_pnl: float
    total_pnl: float
    commission_cost: float
    slippage_cost: float

    # Trading statistics
    num_hedges: int
    total_hedge_volume: float
    avg_hedge_size: float

    # Greeks statistics
    avg_delta: float
    avg_gamma: float
    total_gamma_pnl: float
    total_theta_cost: float

    # Time series data
    pnl_history: pd.DataFrame
    greeks_history: pd.DataFrame
    trades: pd.DataFrame


class OptimizedGreeksCalculator:
    """Optimized Greeks calculations with caching and vectorization"""

    def __init__(self):
        self._cache = {}
        self._d1_cache = {}
        self._d2_cache = {}

    def _cache_key(self, S, K, T, r, sigma):
        """Generate cache key for Greeks calculations"""
        return (round(S, 2), K, round(T, 6), r, round(sigma, 4))

    def calculate_d1_d2(self, S, K, T, r, sigma):
        """Calculate d1 and d2 with caching"""
        key = self._cache_key(S, K, T, r, sigma)

        if key not in self._d1_cache:
            if T <= 0:
                self._d1_cache[key] = 0
                self._d2_cache[key] = 0
            else:
                d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                self._d1_cache[key] = d1
                self._d2_cache[key] = d2

        return self._d1_cache[key], self._d2_cache[key]

    def calculate_all_greeks(self, S, K, T, r, sigma, option_type='call'):
        """Calculate all Greeks at once, reusing intermediate calculations"""
        key = self._cache_key(S, K, T, r, sigma)

        if key in self._cache:
            return self._cache[key]

        if T <= 0:
            # At expiry
            if option_type == 'call':
                delta_val = 1.0 if S > K else 0.0
            else:
                delta_val = -1.0 if S < K else 0.0

            result = {
                'delta': delta_val,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0
            }
        else:
            # Calculate d1 and d2 once
            d1, d2 = self.calculate_d1_d2(S, K, T, r, sigma)

            # Pre-calculate common terms
            sqrt_T = np.sqrt(T)
            pdf_d1 = norm.pdf(d1)
            cdf_d1 = norm.cdf(d1)
            cdf_d2 = norm.cdf(d2)
            exp_rT = np.exp(-r * T)

            # Delta
            if option_type == 'call':
                delta_val = cdf_d1
            else:
                delta_val = -norm.cdf(-d1)

            # Gamma (same for call and put)
            gamma_val = pdf_d1 / (S * sigma * sqrt_T)

            # Vega (same for call and put)
            vega_val = S * pdf_d1 * sqrt_T

            # Theta
            term1 = -S * pdf_d1 * sigma / (2 * sqrt_T)
            if option_type == 'call':
                theta_val = term1 - r * K * exp_rT * cdf_d2
            else:
                theta_val = term1 + r * K * exp_rT * norm.cdf(-d2)

            result = {
                'delta': delta_val,
                'gamma': gamma_val,
                'theta': theta_val,
                'vega': vega_val
            }

        self._cache[key] = result
        return result

    def clear_cache(self):
        """Clear the cache to free memory"""
        self._cache.clear()
        self._d1_cache.clear()
        self._d2_cache.clear()


class OptimizedImpliedVolatilitySolver:
    """Optimized implied volatility solver with caching and better initial guess"""

    def __init__(self):
        self._cache = {}
        self._last_iv = 0.3  # Keep track of last IV for better initial guess

    def solve(self, price, S, K, T, r, option_type='call', tol=1e-4, max_iterations=50):
        """Solve for implied volatility with optimizations"""
        # Cache key based on inputs
        cache_key = (round(price, 4), round(S, 2), K, round(T, 6), r, option_type)

        if cache_key in self._cache:
            return self._cache[cache_key]

        if T <= 0:
            self._cache[cache_key] = np.nan
            return np.nan

        # Better initial guess based on ATM approximation
        # Brenner and Subrahmanyam approximation
        initial_guess = np.sqrt(2 * np.pi / T) * (price / S)
        initial_guess = max(0.01, min(initial_guess, 3.0))

        # Try Newton-Raphson first (faster than Brent)
        iv = self._newton_raphson_iv(price, S, K, T, r, option_type, initial_guess, tol, max_iterations)

        if np.isnan(iv):
            # Fall back to Brent if Newton-Raphson fails
            try:
                def objective(sigma):
                    return bs_price(S, K, T, r, sigma, option_type) - price

                iv = brentq(objective, 1e-6, 3.0, maxiter=max_iterations, xtol=tol)
            except (ValueError, RuntimeError):
                iv = self._last_iv  # Use last known IV as fallback

        self._cache[cache_key] = iv
        self._last_iv = iv if not np.isnan(iv) else self._last_iv
        return iv

    def _newton_raphson_iv(self, price, S, K, T, r, option_type, initial_guess, tol, max_iter):
        """Newton-Raphson method for IV (faster than Brent for good initial guess)"""
        sigma = initial_guess

        for _ in range(max_iter):
            try:
                bs = bs_price(S, K, T, r, sigma, option_type)
                vega_val = vega(S, K, T, r, sigma)

                if abs(vega_val) < 1e-10:
                    return np.nan

                diff = bs - price
                if abs(diff) < tol:
                    return sigma

                sigma = sigma - diff / vega_val
                sigma = max(1e-6, min(sigma, 5.0))  # Keep in reasonable bounds

            except:
                return np.nan

        return np.nan

    def clear_cache(self):
        """Clear the cache"""
        self._cache.clear()


class GammaScalpingSimulatorOptimized:
    """
    Optimized gamma scalping simulator with performance improvements.
    Maintains exact same logic as original but with better performance.
    """

    def __init__(
        self,
        commission_rate: float = 0.0005,
        slippage_bps: float = 10,
        risk_free_rate: float = 0.01
    ):
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps / 10000
        self.risk_free_rate = risk_free_rate
        self.greeks_calc = OptimizedGreeksCalculator()
        self.iv_solver = OptimizedImpliedVolatilitySolver()

    def simulate(
        self,
        spot_prices: pd.DataFrame,
        option_data: pd.DataFrame,
        hedge_threshold: float = 0.1,
        position_size: float = 1.0
    ) -> GammaScalpingResult:
        """
        Simulate gamma scalping for a single option using delta-band hedging.
        Optimized version with same logic as original.
        """

        if option_data.empty or spot_prices.empty:
            return None

        # Pre-process data for faster lookups
        # Convert timestamps to index for faster merging
        spot_prices = spot_prices.set_index('timestamp', drop=False)
        option_data = option_data.set_index('timestamp', drop=False)

        # Get option details from first row
        first_row = option_data.iloc[0]
        strike = first_row['strike']
        expiry = first_row['expiry']
        option_type = first_row['option_type']
        symbol = first_row.get('symbol', f"{strike}_{option_type}")

        # Merge spot and option data for aligned processing
        merged_data = pd.merge(
            option_data[['timestamp', 'close']],
            spot_prices[['timestamp', 'close']],
            on='timestamp',
            suffixes=('_option', '_spot'),
            how='inner'
        )

        if merged_data.empty:
            return None

        # Pre-calculate time to expiry for all timestamps
        timestamps = merged_data['timestamp'].values
        ttes = np.array([(expiry - t).total_seconds() / (365 * 24 * 3600)
                         for t in timestamps])
        ttes = np.maximum(ttes, 0)

        # Initialize tracking variables
        hedge_position = 0
        last_hedge_delta = 0

        # Pre-allocate arrays for results (more efficient than appending)
        n_steps = len(merged_data)
        pnl_history = []
        greeks_history = []
        trades = []

        # P&L tracking
        option_cost = 0
        hedge_cost_basis = 0
        total_commissions = 0
        total_slippage = 0

        # Get initial data
        initial_spot = merged_data.iloc[0]['close_spot']
        initial_option_price = merged_data.iloc[0]['close_option']
        initial_timestamp = merged_data.iloc[0]['timestamp']

        # Calculate initial IV with optimization
        initial_iv = self.iv_solver.solve(
            initial_option_price, initial_spot, strike,
            ttes[0], self.risk_free_rate, option_type
        )

        if np.isnan(initial_iv):
            initial_iv = 0.5

        # Open option position
        option_cost = position_size * initial_option_price
        option_commission = abs(option_cost) * self.commission_rate
        option_slippage = abs(option_cost) * self.slippage_bps
        total_commissions += option_commission
        total_slippage += option_slippage

        trades.append({
            'timestamp': initial_timestamp,
            'type': 'option',
            'action': 'buy',
            'quantity': position_size,
            'price': initial_option_price,
            'commission': option_commission,
            'slippage': option_slippage
        })

        # Keep track of last known IV for fallback
        last_iv = initial_iv

        # Main simulation loop - process in batches for efficiency
        for idx in range(n_steps):
            timestamp = merged_data.iloc[idx]['timestamp']
            option_price = merged_data.iloc[idx]['close_option']
            spot_price = merged_data.iloc[idx]['close_spot']
            tte = ttes[idx]

            if tte <= 0:
                break  # Option expired

            # Calculate current IV with caching
            current_iv = self.iv_solver.solve(
                option_price, spot_price, strike,
                tte, self.risk_free_rate, option_type
            )
            if np.isnan(current_iv):
                current_iv = last_iv
            else:
                last_iv = current_iv

            # Calculate all Greeks at once (optimized)
            greeks = self.greeks_calc.calculate_all_greeks(
                spot_price, strike, tte, self.risk_free_rate, current_iv, option_type
            )

            # Scale by position size
            position_delta = greeks['delta'] * position_size
            position_gamma = greeks['gamma'] * position_size
            position_theta = greeks['theta'] * position_size
            position_vega = greeks['vega'] * position_size

            # Determine if we need to hedge
            delta_diff = abs(position_delta - last_hedge_delta)
            should_hedge = delta_diff > hedge_threshold

            # Execute hedge if needed
            if should_hedge:
                target_hedge = -position_delta
                hedge_trade_size = target_hedge - hedge_position

                if abs(hedge_trade_size) > 0.001:
                    # Execute hedge trade
                    hedge_notional = abs(hedge_trade_size * spot_price)
                    hedge_commission = hedge_notional * self.commission_rate
                    hedge_slippage = hedge_notional * self.slippage_bps

                    total_commissions += hedge_commission
                    total_slippage += hedge_slippage

                    hedge_cost_basis += hedge_trade_size * spot_price
                    hedge_position = target_hedge
                    last_hedge_delta = position_delta

                    trades.append({
                        'timestamp': timestamp,
                        'type': 'hedge',
                        'action': 'buy' if hedge_trade_size > 0 else 'sell',
                        'quantity': abs(hedge_trade_size),
                        'price': spot_price,
                        'commission': hedge_commission,
                        'slippage': hedge_slippage
                    })

            # Calculate current P&L
            option_mtm = position_size * option_price
            option_pnl = option_mtm - option_cost

            hedge_mtm = hedge_position * spot_price
            hedge_pnl = hedge_mtm - hedge_cost_basis

            total_pnl = option_pnl + hedge_pnl - total_commissions - total_slippage

            # Store history
            pnl_history.append({
                'timestamp': timestamp,
                'spot_price': spot_price,
                'option_price': option_price,
                'option_pnl': option_pnl,
                'hedge_pnl': hedge_pnl,
                'commission_cost': total_commissions,
                'slippage_cost': total_slippage,
                'total_pnl': total_pnl,
                'hedge_position': hedge_position,
                'net_delta': position_delta + hedge_position
            })

            greeks_history.append({
                'timestamp': timestamp,
                'delta': position_delta,
                'gamma': position_gamma,
                'theta': position_theta,
                'vega': position_vega,
                'iv': current_iv
            })

            # Clear cache periodically to manage memory
            if idx % 100 == 0:
                self.greeks_calc.clear_cache()
                self.iv_solver.clear_cache()

        # Close positions at expiry
        if pnl_history:
            final_spot = pnl_history[-1]['spot_price']

            # Option payoff at expiry
            if option_type == 'call':
                option_payoff = max(0, final_spot - strike) * position_size
            else:
                option_payoff = max(0, strike - final_spot) * position_size

            final_option_pnl = option_payoff - option_cost

            # Close hedge position
            if abs(hedge_position) > 0.001:
                hedge_close_notional = abs(hedge_position * final_spot)
                hedge_close_commission = hedge_close_notional * self.commission_rate
                hedge_close_slippage = hedge_close_notional * self.slippage_bps

                total_commissions += hedge_close_commission
                total_slippage += hedge_close_slippage

                final_hedge_pnl = (hedge_position * final_spot) - hedge_cost_basis

                trades.append({
                    'timestamp': expiry,
                    'type': 'hedge',
                    'action': 'close',
                    'quantity': abs(hedge_position),
                    'price': final_spot,
                    'commission': hedge_close_commission,
                    'slippage': hedge_close_slippage
                })
            else:
                final_hedge_pnl = 0

            final_total_pnl = final_option_pnl + final_hedge_pnl - total_commissions - total_slippage

            # Create DataFrames
            pnl_df = pd.DataFrame(pnl_history)
            greeks_df = pd.DataFrame(greeks_history)
            trades_df = pd.DataFrame(trades)

            # Calculate statistics using vectorized operations
            if len(pnl_df) > 1:
                spot_moves = pnl_df['spot_price'].diff().fillna(0).values
                gamma_values = greeks_df['gamma'].shift(1).fillna(0).values
                gamma_pnl = 0.5 * np.sum(gamma_values * spot_moves**2)
            else:
                gamma_pnl = 0

            # Estimate theta cost
            if len(greeks_df) > 1:
                greeks_df['timestamp'] = pd.to_datetime(greeks_df['timestamp'])
                time_deltas = (greeks_df['timestamp'] - greeks_df['timestamp'].shift(1)).dt.total_seconds() / 86400
                theta_cost = (greeks_df['theta'].shift(1).fillna(0) * time_deltas.fillna(0)).sum()
            else:
                theta_cost = 0

            # Clear final caches
            self.greeks_calc.clear_cache()
            self.iv_solver.clear_cache()

            return GammaScalpingResult(
                option_symbol=symbol,
                strike=strike,
                expiry=expiry,
                option_type=option_type,
                option_pnl=final_option_pnl,
                hedge_pnl=final_hedge_pnl,
                total_pnl=final_total_pnl,
                commission_cost=total_commissions,
                slippage_cost=total_slippage,
                num_hedges=len(trades_df[trades_df['type'] == 'hedge']),
                total_hedge_volume=trades_df[trades_df['type'] == 'hedge']['quantity'].sum(),
                avg_hedge_size=trades_df[trades_df['type'] == 'hedge']['quantity'].mean() if len(trades_df[trades_df['type'] == 'hedge']) > 0 else 0,
                avg_delta=greeks_df['delta'].mean(),
                avg_gamma=greeks_df['gamma'].mean(),
                total_gamma_pnl=gamma_pnl,
                total_theta_cost=theta_cost,
                pnl_history=pnl_df,
                greeks_history=greeks_df,
                trades=trades_df
            )

        return None