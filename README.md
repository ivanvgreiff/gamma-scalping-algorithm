# Gamma Scalping Algorithm

This repository implements a gamma scalping strategy for cryptocurrency options (e.g., BTC options on Deribit). The project is organized for research, backtesting, and live trading, with modular components for data, modeling, strategy logic, and risk management.

## Project Structure

```
gamma_scalping/
├── data/
│   ├── raw/               # Historical BTC & options data (CSV, JSON)
│   ├── processed/         # Cleaned, feature-engineered datasets
│   └── loaders/           # Scripts to download or update data
├── models/
│   ├── options_pricing.py # Black-Scholes, Greeks, IV calculations
│   └── volatility.py      # Realized vol, GARCH, etc.
├── strategies/
│   └── gamma_scalping.py  # Core logic: buy, hedge, track PnL
├── backtests/
│   ├── backtest_runner.py # Run and analyze strategy
│   └── results/           # Charts, logs, PnL summaries
├── live_trading/
│   ├── deribit_api.py     # Order placement, live IV, greeks
│   ├── hedger.py          # Real-time delta adjustment engine
│   └── risk_manager.py    # Position sizing, margin tracking
├── notebooks/
│   ├── exploratory.ipynb  # IV vs RV analysis, visualizations
│   └── sanity_checks.ipynb
├── config/
│   └── params.yaml        # Parameters: hedging threshold, expiries, etc.
├── utils/
│   └── logger.py          # Logging, alerts, etc.
├── tests/
│   └── test_models.py     # Unit tests for greeks, pricing, etc.
├── requirements.txt
└── README.md
```

### Directory & File Descriptions

- **data/**: Data management and storage.
  - `raw/`: Raw historical BTC and options data (CSV, JSON).
  - `processed/`: Cleaned and feature-engineered datasets for modeling and backtesting.
  - `loaders/`: Scripts to download, update, or preprocess data.

- **models/**: Financial models and calculations.
  - `options_pricing.py`: Implements Black-Scholes, Greeks, and implied volatility calculations.
  - `volatility.py`: Realized volatility, GARCH models, and related analytics.

- **strategies/**: Core trading logic.
  - `gamma_scalping.py`: Main gamma scalping strategy logic (buying options, hedging, tracking PnL).

- **backtests/**: Backtesting framework and results.
  - `backtest_runner.py`: Script to run and analyze backtests.
  - `results/`: Stores backtest charts, logs, and PnL summaries.

- **live_trading/**: Live trading and execution modules.
  - `deribit_api.py`: Handles API connections, order placement, and live data.
  - `hedger.py`: Real-time delta hedging engine.
  - `risk_manager.py`: Position sizing, margin, and risk tracking.

- **notebooks/**: Jupyter notebooks for research and analysis.
  - `exploratory.ipynb`: IV vs RV analysis, data exploration, and visualizations.
  - `sanity_checks.ipynb`: Data and model sanity checks.

- **config/**: Configuration files.
  - `params.yaml`: Strategy parameters (hedging thresholds, expiries, etc.).

- **utils/**: Utility scripts.
  - `logger.py`: Logging, alerts, and helper functions.

- **tests/**: Unit and integration tests.
  - `test_models.py`: Tests for pricing, Greeks, and volatility models.

- **requirements.txt**: Python dependencies.
- **README.md**: Project overview and documentation.

---

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Explore notebooks:**
   - See `notebooks/exploratory.ipynb` for data analysis and visualizations.
3. **Run backtests:**
   - Use `backtests/backtest_runner.py` to simulate the strategy.
4. **Live trading:**
   - Integrate with Deribit using modules in `live_trading/`.

---

## Notes
- This structure is modular and extensible for research, backtesting, and live trading.
- Fill in each module as you develop the strategy.
