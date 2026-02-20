# models/volatility/forecast.py

import pandas as pd
import numpy as np
from arch import arch_model

from .target import log_returns

# ann_days  = 365           
# ann_hours  = 24 * ann_days

# # Naive forecast from any annualized % estimator series
# def naive_forecast(est_pct: pd.Series, horizon: int = 1) -> pd.Series:
#     """
#     Naive volatility forecast (annualized, %).
#     Uses today's estimator as the forecast for the next h days.
#     Output remains annualized, %.

#     Parameters
#     ----------
#     est_pct : pd.Series
#         Annualized % volatility estimator (e.g. rv_est_pct / rv_parkinson_pct / rv_rs_pct).
#     horizon : int, default=1
#         Forecast horizon in calendar days (or hours if your estimator is hourly).

#     Returns
#     -------
#     pd.Series
#         Series of forecasts, annualized %, named f'fcst_naive_h{h}_pct'.
#     """
#     if horizon < 1:
#         raise ValueError("horizon must be >= 1")

#     # Forecast = current estimate shifted forward to align with t+h target
#     f = est_pct.shift(1) * np.sqrt(horizon)
#     f.name = f"fcst_naive_h{h}_pct"
#     return f

class GARCHVolatilityForecaster:
    """
    GARCH(1,1) forecaster for financial time series volatility.
    
    This class fits a GARCH(1,1) model to historical log returns and
    produces volatility forecasts for different horizons (1d, 5d, 21d).
    
    Output is annualized volatility, so it's directly comparable to implied vol.
    """
    
    def __init__(self, returns: pd.Series, annualization_factor: int = 365):
        """
        Initialize the GARCH forecaster.

        Parameters
        ----------
        returns : pd.Series
            Series of log returns (should be clean, no NaNs).
        annualization_factor : int
            Factor to annualize daily volatility (252 trading days by default).
        """
        self.returns = returns.dropna()
        self.annualization_factor = annualization_factor
        self.model = None
        self.res = None

    def fit(self):
        """
        Fit a GARCH(1,1) model to the returns.
        """
        self.model = arch_model(
            self.returns * 100,   # scale returns to percentage to improve stability
            vol="GARCH",
            p=1,
            q=1,
            mean="Zero",         # assume zero mean for returns
            dist="normal"
        )
        self.res = self.model.fit(disp="off")

    def forecast(self, horizon: int = 1) -> float:
        """
        Forecast volatility for a given horizon (in days).

        Parameters
        ----------
        horizon : int
            Number of days ahead to forecast (1 = next day, 5 = next week, 21 = next month).

        Returns
        -------
        float
            Annualized forecasted volatility (in decimal, e.g., 0.20 = 20%).
        """
        if self.res is None:
            raise ValueError("Model not fitted. Call .fit() first.")

        # Get variance forecast
        forecast = self.res.forecast(horizon=horizon, reindex=False)
        variance_forecast = forecast.variance.values[-1, :]

        # Convert to volatility (std dev)
        daily_vols = np.sqrt(variance_forecast) / 100  # back to decimals

        # For multi-day horizons: average variance across days, then annualize
        horizon_vol = np.sqrt(np.mean(daily_vols**2))

        # Annualize volatility
        annualized_vol = horizon_vol * np.sqrt(self.annualization_factor)

        return annualized_vol

    def summary(self):
        """
        Return a summary of the fitted GARCH model.
        """
        if self.res is None:
            raise ValueError("Model not fitted. Call .fit() first.")
        return self.res.summary()