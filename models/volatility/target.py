# models/volatility/target.py

import numpy as np
import pandas as pd

ann_days  = 365           
ann_hours  = 24 * ann_days

def log_returns(df_hourly: pd.DataFrame, out_freq: str = "hourly") -> pd.Series:
    """
    Compute log-returns from HOURLY OHLC data, returned at 'hourly' or 'daily'.

    Parameters
    ----------
    df_hourly : DataFrame with at least column 'close' and a DatetimeIndex at hourly freq
    out_freq  : 'hourly' or 'daily'
        - 'hourly': log(close_t / close_{t-1}) at hourly steps
        - 'daily' : log(C_t^daily / C_{t-1}^daily) using the last hourly close of each UTC day

    Returns
    -------
    pd.Series of log-returns at the requested frequency.
    """
    df = df_hourly.sort_index()

    if out_freq.lower() == "hourly":
        r_h = np.log(df["close"]).diff()
        r_h.name = "logret_hourly"
        return r_h

    if out_freq.lower() == "daily":
        # take last close of each day, then diff
        daily_close = df["close"].resample("1D").last()
        r_d = np.log(daily_close).diff()
        r_d.name = "logret_daily"
        return r_d

    raise ValueError("out_freq must be 'hourly' or 'daily'")

def realized_future_vol(
    df_hourly: pd.DataFrame,
    h: int,
    *,
    freq: str = "hourly",
    percent: bool = True,
    return_variance: bool = False,
    ann: float | None = None,
) -> pd.Series:
    
    """
    Forward-looking realized volatility (or variance) from hourly OHLC data.

    For each timestamp t, computes realized vol/var over the next h periods (hours or days),
    annualizes it, and aligns the result back to t.
    """
    
    if h < 1:
        raise ValueError("h must be >= 1")
    freq = freq.lower()
    if freq not in {"hourly", "daily"}:
        raise ValueError("freq must be 'hourly' or 'daily'")

    r = log_returns(df_hourly, out_freq=freq)

    # annualize
    if ann is None:
        ann = ann_hours if freq == "hourly" else ann_days

    # forward-looking sum of squared returns
    r2_fwd = (r.astype(float)**2).rolling(window=h, min_periods=h).sum().shift(-h)

    out = (ann / h) * r2_fwd
    if not return_variance:
        out = np.sqrt(out)

    if percent:
        out *= 100

    unit = "h" if freq == "hourly" else "d"
    out.name = f"rv_{'var' if return_variance else 'vol'}_future_{h}{unit}" + ("_pct" if percent else "")
    
    return out
