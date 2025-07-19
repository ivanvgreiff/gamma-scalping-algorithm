 # backtests/backtest_runner.py
import pandas as pd
from strategies.gamma_scalping import GammaScalpingStrategy
from models.options_pricing import implied_volatility

def load_data():
    """
    Load preprocessed options and spot price data.
    Must return a DataFrame with columns:
    ['timestamp', 'spot', 'strike', 'expiry', 'option_price', 'option_type']
    """
    return pd.read_csv("processed/backtest_data.csv", parse_dates=['timestamp'])

def run_backtest():
    df = load_data()

    # Set up strategy
    r = 0.01
    hedge_threshold = 0.05  # e.g., rebalance if delta > 5%
    qty = 1  # Number of option contracts held

    # Pick one option contract (e.g., ATM call)
    option_row = df.iloc[0]
    K = option_row['strike']
    expiry = pd.to_datetime(option_row['expiry'])
    option_type = option_row['option_type']

    strategy = GammaScalpingStrategy(
        hedge_threshold=hedge_threshold,
        option_position={
            'K': K,
            'T': 0,        # Will be updated dynamically
            'S': 0,        # Will be updated dynamically
            'sigma': 0,    # Will be updated dynamically
            'qty': qty,
            'option_type': option_type
        },
        r=r
    )

    timestamps = []
    hedge_actions = []

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        S = row['spot']
        t = row['timestamp']
        expiry = pd.to_datetime(row['expiry'])
        T = (expiry - t).total_seconds() / (365 * 24 * 3600)
        option_price = row['option_price']

        sigma = implied_volatility(
            price=option_price,
            S=S,
            K=K,
            T=T,
            r=r,
            option_type=option_type
        )

        strategy.option_position.update({'S': S, 'T': T, 'sigma': sigma})
        strategy.step(S_new=S, T_new=T, sigma_new=sigma)

        timestamps.append(t)
        hedge_actions.append(strategy.hedge_position)

    return timestamps, hedge_actions, strategy.pnls
