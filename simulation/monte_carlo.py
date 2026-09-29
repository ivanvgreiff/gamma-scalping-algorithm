"""Vectorized Monte Carlo engine for discrete delta hedging of a long European call.

The underlying follows GBM with the *realized* volatility, while the option is priced
and hedged with the *implied* volatility. The hedger holds a self-financing cash
account, so the final P&L is exactly ``payoff + cash`` and can be attributed to
payoff-minus-premium, hedge trading, financing and transaction costs.

This is the engine used in ``notebooks/gamma_scalping_analysis.ipynb``.
"""

import numpy as np
from scipy.stats import norm

def _bs_d1_d2(S, K, r, sigma, tau):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    eps = 1e-12
    tau_safe = np.maximum(tau, eps)
    sigma_safe = np.maximum(sigma, eps)
    d1 = (np.log(S / K) + (r + 0.5 * sigma_safe**2) * tau_safe) / (sigma_safe * np.sqrt(tau_safe))
    d2 = d1 - sigma_safe * np.sqrt(tau_safe)
    return d1, d2

def call_price_bs(S, K, r, sigma, tau):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)
    payoff_at_maturity = np.maximum(S - K, 0.0)
    price = np.where(tau <= 0, payoff_at_maturity, None)
    mask = tau > 0
    if np.any(mask):
        d1, d2 = _bs_d1_d2(S[mask], K[mask], r, sigma if np.isscalar(sigma) else sigma[mask], tau[mask])
        part = S[mask] * norm.cdf(d1) - K[mask] * np.exp(-r * tau[mask]) * norm.cdf(d2)
        if price is None:
            price = np.zeros_like(S)
        price[mask] = part
    return price

def delta_bs(S, K, r, sigma, tau):
    tau = np.asarray(tau, dtype=float)
    if np.all(tau <= 0):
        return (S > K).astype(float)
    d1, _ = _bs_d1_d2(S, K, r, sigma, tau)
    return norm.cdf(d1)

def gamma_bs(S, K, r, sigma, tau):
    tau = np.asarray(tau, dtype=float)
    eps = 1e-12
    sqrt_tau = np.sqrt(np.maximum(tau, eps))
    sigma_safe = np.maximum(sigma, eps)
    d1, _ = _bs_d1_d2(S, K, r, sigma_safe, tau)
    return norm.pdf(d1) / (S * sigma_safe * sqrt_tau)

def theta_bs(S, K, r, sigma, tau):
    # Per year
    tau = np.asarray(tau, dtype=float)
    d1, d2 = _bs_d1_d2(S, K, r, sigma, tau)
    term1 = -0.5 * sigma * S * norm.pdf(d1) / np.sqrt(np.maximum(tau, 1e-12))
    term2 = r * K * np.exp(-r * tau) * norm.cdf(d2)
    return term1 - term2

def vega_bs(S, K, r, sigma, tau):
    d1, _ = _bs_d1_d2(S, K, r, sigma, tau)
    return S * norm.pdf(d1) * np.sqrt(np.maximum(tau, 1e-12))

def simulate_delta_hedge_paths_mc(
    S0=100.0,
    K=100.0,
    r=0.0,
    sigma_real=0.20,
    sigma_imp=None,
    T=1.0,
    n_steps=390,
    n_paths=10000,
    seed=None,
    transaction_cost_bps=0.0,
    record_timeseries=False
):
    sigma_imp = sigma_real if sigma_imp is None else sigma_imp
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # Initial option premium and delta under implied vol
    sqrt_T = np.sqrt(T)
    d1_0 = (np.log(S0 / K) + (r + 0.5 * sigma_imp**2) * T) / (sigma_imp * sqrt_T)
    d2_0 = d1_0 - sigma_imp * sqrt_T
    call_premium = S0 * norm.cdf(d1_0) - K * np.exp(-r * T) * norm.cdf(d2_0)
    delta0 = norm.cdf(d1_0)

    # State
    S = np.full(n_paths, S0)
    cash = -np.full(n_paths, call_premium)
    current_delta = np.zeros(n_paths)

    financing = np.zeros(n_paths)
    tx_costs = np.zeros(n_paths)
    hedge_trading = np.zeros(n_paths)

    # Initial hedge: short delta shares
    notional0 = delta0 * S0
    current_delta[:] = delta0
    cash += notional0
    tc0 = transaction_cost_bps * 1e-4 * abs(notional0)
    cash -= tc0
    tx_costs -= tc0

    # Random shocks for realized vol simulation
    Z = rng.standard_normal((n_steps, n_paths))
    drift = (r - 0.5 * sigma_real**2) * dt
    vol_step = sigma_real * np.sqrt(dt)

    # Optional recording for a single path (index 0)
    rec = None
    if record_timeseries and n_paths > 0:
        option_val0 = float(np.atleast_1d(call_price_bs(S0, K, r, sigma_imp, T))[0])
        stock_val0 = -delta0 * S0
        rec = {
            'S': [S0],
            'delta': [delta0],
            'cash': [cash[0]],
            'stock': [stock_val0],
            'option_value': [option_val0],
            'trade': [-delta0],  # from 0 to -delta0 short
            'pnl_cum': [option_val0 + cash[0] + stock_val0],
            'time': [0.0],
        }

    for step in range(1, n_steps + 1):
        S_prev = S.copy()
        S *= np.exp(drift + vol_step * Z[step - 1])

        # Accrue financing
        old_cash = cash.copy()
        cash *= np.exp(r * dt)
        financing += (cash - old_cash)

        # P/L from hedge trading over step (stock move) for existing position
        hedge_trading += (-current_delta) * (S - S_prev)

        if step == n_steps:
            break

        # Re-hedge
        t_curr = step * dt
        tau = max(T - t_curr, 0.0)
        sqrt_tau = np.sqrt(tau)
        d1 = (np.log(S / K) + (r + 0.5 * sigma_imp**2) * tau) / (sigma_imp * sqrt_tau)
        new_delta = norm.cdf(d1)
        trade = new_delta - current_delta
        trade_notional = -trade * S
        cash -= trade_notional
        tc = transaction_cost_bps * 1e-4 * np.abs(trade_notional)
        cash -= tc
        tx_costs -= tc
        current_delta = new_delta

        if rec is not None:
            cash0 = cash[0]
            option_val = float(np.atleast_1d(call_price_bs(S[0], K, r, sigma_imp, tau))[0])
            stock_val = -current_delta[0] * S[0]
            portfolio_val = option_val + cash0 + stock_val
            rec['S'].append(S[0])
            rec['delta'].append(current_delta[0])
            rec['cash'].append(cash0)
            rec['stock'].append(stock_val)
            rec['option_value'].append(option_val)
            rec['trade'].append(trade[0])
            rec['pnl_cum'].append(portfolio_val)
            rec['time'].append(t_curr)

    # Maturity payoff & liquidation
    payoff = np.maximum(S - K, 0.0)
    last_delta_path0 = current_delta[0] if rec is not None else None
    liquid_notional = -current_delta * S
    cash += liquid_notional
    tc = transaction_cost_bps * 1e-4 * np.abs(liquid_notional)
    cash -= tc
    tx_costs -= tc
    current_delta = np.zeros_like(current_delta)

    pnl = payoff + cash
    payoff_component = payoff - call_premium

    result = {
        'pnl': pnl,
        'payoff_component': payoff_component,
        'financing_component': financing,
        'transaction_cost_component': tx_costs,
        'hedge_trading_component': hedge_trading,
        'call_premium': call_premium,
        'initial_delta': delta0,
        'final_delta': current_delta.copy(),
    }

    if rec is not None:
        option_val_final = float(np.maximum(S[0] - K, 0.0))
        stock_val_final = -current_delta[0] * S[0]
        portfolio_val_final = option_val_final + cash[0] + stock_val_final
        trade_final = -last_delta_path0 if last_delta_path0 is not None else 0.0
        rec['S'].append(S[0])
        rec['delta'].append(current_delta[0])
        rec['cash'].append(cash[0])
        rec['stock'].append(stock_val_final)
        rec['option_value'].append(option_val_final)
        rec['trade'].append(trade_final)
        rec['pnl_cum'].append(portfolio_val_final)
        rec['time'].append(T)
        result['timeseries'] = rec

    return result