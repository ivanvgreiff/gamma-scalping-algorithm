# models/volatility/estimator.py

import pandas as pd
import numpy as np

from .target import log_returns

ann_days  = 365           
ann_hours  = 24 * ann_days

# close-to-close realized-vol estimator
def close_to_close_vol(df_spot: pd.DataFrame, window: int, *, freq: str = "daily") -> pd.Series:
    """
    Close-to-close realized volatility estimator (rolling std of log returns),
    annualized in percent. Returns a Series named 'rv_est_pct' indexed at the
    chosen frequency (daily or hourly), same style as 'rv_parkinson_pct'.
    """
    freq = freq.lower()
    if freq not in {"daily", "hourly"}:
        raise ValueError("freq must be 'daily' or 'hourly'")

    log_ret = log_returns(df_spot, out_freq=freq)

    ann = 365 if freq == "daily" else 24 * 365

    # rolling realized-vol estimate (annualized, %)
    rv = log_ret.rolling(window).std(ddof=0)
    vol = rv * np.sqrt(ann) * 100.0
    vol.name = "rv_est_pct"

    return vol

_LOG2_4 = 4.0 * np.log(2.0)

# Parkinson volatility estimator
def parkinson_vol(df_spot: pd.DataFrame, window: int, *, freq: str = "daily") -> pd.Series:
    """
    NEW METHOD: Rolling Parkinson volatility (annualized, %) from OHLC data.

    Parameters
    ----------
    df_spot : DataFrame
        Hourly OHLC with DatetimeIndex and columns: 'open','high','low','close'.
    window : int
        Rolling window length in periods of `freq` (days if daily, hours if hourly).
    freq : {'daily','hourly'}
        Frequency at which to compute the rolling estimator.

    Returns
    -------
    pd.Series
        'rv_parkinson_pct' aligned to the chosen frequency index, annualized in %.
    """
    freq = freq.lower()
    if freq not in {"daily", "hourly"}:
        raise ValueError("freq must be 'daily' or 'hourly'")

    df_spot = df_spot.sort_index()

    if freq == "hourly":
        ann  = ann_hours
        ohlc = df_spot[['high','low']].asfreq('h')
        idx  = ohlc.index
    else:
        ann  = ann_days
        ohlc = (df_spot[['high','low']]
                .resample('1D')
                .agg({'high':'max','low':'min'}))
        idx  = ohlc.index

    # Parkinson per-period variance proxy: (ln(H/L))^2 / (4 ln 2)
    range_sq = np.log(ohlc['high'] / ohlc['low'])**2
    park_var = (range_sq / _LOG2_4).rolling(window).mean()

    # per-period vol -> annualized % 
    park_vol = np.sqrt(park_var) * np.sqrt(ann) * 100.0
    park_vol.name = 'rv_parkinson_pct'

    return park_vol.reindex(idx)

# Rogers–Satchell volatility estimator
def rs_vol(df_spot: pd.DataFrame, window: int, *, freq: str = "daily") -> pd.Series:
    """
    NEW METHOD: Rolling Rogers–Satchell volatility (annualized, %) from OHLC data.

    Parameters
    ----------
    df_spot : DataFrame
        Hourly OHLC with DatetimeIndex and columns: 'open','high','low','close'.
    window : int
        Rolling window length in periods of `freq` (days if daily, hours if hourly).
    freq : {'daily','hourly'}
        Frequency at which to compute the rolling estimator.

    Returns
    -------
    pd.Series
        'rv_rs_pct' aligned to the chosen frequency index, annualized in %.
    """
    freq = freq.lower()
    if freq not in {"daily", "hourly"}:
        raise ValueError("freq must be 'daily' or 'hourly'")

    df_spot = df_spot.sort_index()

    if freq == "hourly":
        ann  = ann_hours
        ohlc = df_spot[['open','high','low','close']].asfreq('h')
        idx  = ohlc.index
    else:
        ann  = ann_days
        ohlc = (df_spot[['open','high','low','close']]
                .resample('1D')
                .agg({'open':'first','high':'max','low':'min','close':'last'}))
        idx  = ohlc.index

    # Rogers–Satchell per-period variance:
    # u = ln(H/C), d = ln(L/C), o = ln(O/C)
    # RS_var = u*(u - o) + d*(d - o)
    c = ohlc['close']
    o = ohlc['open']
    h = ohlc['high']
    l = ohlc['low']

    u = np.log(h / c)
    d = np.log(l / c)
    o_rel = np.log(o / c)

    rs_var_per_period = u * (u - o_rel) + d * (d - o_rel)

    # Rolling mean of per-period variance, then sqrt and annualize
    rs_var_roll = rs_var_per_period.rolling(window).mean().clip(lower=0)
    rs_vol_per_period = np.sqrt(rs_var_roll)

    rs_vol_ann_pct = rs_vol_per_period * np.sqrt(ann) * 100.0
    rs_vol_ann_pct.name = 'rv_rs_pct'

    return rs_vol_ann_pct.reindex(idx)
