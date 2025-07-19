 # models/options_pricing.py
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# --- Helper Functions ---
def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

# --- Black-Scholes Pricing ---
def bs_price(S, K, T, r, sigma, option_type='call'):
    """Price a European option using the Black-Scholes formula."""
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)

    if option_type == 'call':
        return S * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-D2) - S * norm.cdf(-D1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

# --- Greeks ---
def delta(S, K, T, r, sigma, option_type='call'):
    D1 = d1(S, K, T, r, sigma)
    return norm.cdf(D1) if option_type == 'call' else -norm.cdf(-D1)

def gamma(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    return norm.pdf(D1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(D1) * np.sqrt(T)

def theta(S, K, T, r, sigma, option_type='call'):
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    term1 = -S * norm.pdf(D1) * sigma / (2 * np.sqrt(T))
    if option_type == 'call':
        return term1 - r * K * np.exp(-r * T) * norm.cdf(D2)
    else:
        return term1 + r * K * np.exp(-r * T) * norm.cdf(-D2)

def rho(S, K, T, r, sigma, option_type='call'):
    D2 = d2(S, K, T, r, sigma)
    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(D2)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-D2)

# --- Implied Volatility Solver ---
def implied_volatility(price, S, K, T, r, option_type='call', tol=1e-6, max_iterations=100):
    """Find implied volatility using Brent's method."""
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - price

    try:
        return brentq(objective, 1e-6, 5.0, maxiter=max_iterations, xtol=tol)
    except (ValueError, RuntimeError):
        return np.nan  # Or raise a warning/log if needed
