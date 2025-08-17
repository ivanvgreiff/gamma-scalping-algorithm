import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict
from scipy import stats, optimize
from arch import arch_model


class VolatilityEstimator:
    def __init__(self, annualization_factor: int = 365):
        self.annualization_factor = annualization_factor

    def close_to_close(self, prices: pd.Series, window: int = 30) -> pd.Series:
        log_returns = np.log(prices / prices.shift(1))
        return log_returns.rolling(window=window).std() * np.sqrt(self.annualization_factor)

    def parkinson(self, high: pd.Series, low: pd.Series, window: int = 30) -> pd.Series:
        hl_ratio = np.log(high / low)
        factor = 1 / (4 * np.log(2))
        return np.sqrt(factor * (hl_ratio ** 2).rolling(window=window).mean() * self.annualization_factor)

    def garman_klass(self, open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 30) -> pd.Series:
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)
        rs = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
        return np.sqrt(rs.rolling(window=window).mean() * self.annualization_factor)

    def rogers_satchell(self, open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 30) -> pd.Series:
        log_ho = np.log(high / open_)
        log_hc = np.log(high / close)
        log_lo = np.log(low / open_)
        log_lc = np.log(low / close)
        rs = log_ho * log_hc + log_lo * log_lc
        return np.sqrt(rs.rolling(window=window).mean() * self.annualization_factor)

    def yang_zhang(self, open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 30) -> pd.Series:
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        log_oc = np.log(open_ / close.shift(1))
        overnight_var = (log_oc - log_oc.rolling(window=window).mean())**2
        overnight_vol = overnight_var.rolling(window=window).mean()
        log_co = np.log(close / open_)
        open_close_var = (log_co - log_co.rolling(window=window).mean())**2
        open_close_vol = open_close_var.rolling(window=window).mean()
        rs_vol = self.rogers_satchell(open_, high, low, close, window)**2 / self.annualization_factor
        yz_variance = overnight_vol + k * open_close_vol + (1 - k) * rs_vol
        return np.sqrt(yz_variance * self.annualization_factor)

    def calculate_all(self, df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        results = pd.DataFrame(index=df.index)
        if 'close' in df.columns:
            results['vol_close'] = self.close_to_close(df['close'], window)
        if all(col in df.columns for col in ['high', 'low']):
            results['vol_parkinson'] = self.parkinson(df['high'], df['low'], window)
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            results['vol_garman_klass'] = self.garman_klass(df['open'], df['high'], df['low'], df['close'], window)
            results['vol_rogers_satchell'] = self.rogers_satchell(df['open'], df['high'], df['low'], df['close'], window)
            results['vol_yang_zhang'] = self.yang_zhang(df['open'], df['high'], df['low'], df['close'], window)
        return results