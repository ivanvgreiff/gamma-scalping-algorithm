import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
import pyarrow.feather as feather
import re
from datetime import datetime, timedelta

class DataLoader:
    """
    Unified data loader for spot and options data.
    Handles feather files and provides synchronized data for backtesting.
    """
    
    def __init__(self, data_path: str = 'data/parsed'):
        self.data_path = Path(data_path)
        self.spot_path = self.data_path / 'spot'
        self.options_path = self.data_path / 'options'
        
    def load_spot_data(
        self,
        symbol: str = 'BTCUSDT',
        timeframe: str = '1h',
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """Load spot price data"""
        
        file_path = self.spot_path / f"{symbol}_{timeframe}.feather"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Spot data file not found: {file_path}")
        
        # Load data
        df = feather.read_feather(file_path)
        
        # Ensure timestamp column
        if 'timestamp' not in df.columns and 'time' in df.columns:
            df['timestamp'] = df['time']
        elif 'timestamp' not in df.columns and df.index.name in ['timestamp', 'time']:
            df['timestamp'] = df.index
        
        # Convert to datetime if needed
        if df['timestamp'].dtype == 'int64':
            # Assume microseconds if large integer
            if df['timestamp'].iloc[0] > 1e12:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by date range
        if start_date:
            df = df[df['timestamp'] >= start_date]
        if end_date:
            df = df[df['timestamp'] <= end_date]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def parse_option_symbol(self, symbol: str) -> Dict:
        """
        Parse option symbol to extract components.
        Example: BTC-10APR20-6000-C -> {underlying: BTC, expiry: 10APR20, strike: 6000, type: C}
        """
        
        pattern = r'([A-Z]+)-(\d+[A-Z]+\d+)-(\d+)-([CP])'
        match = re.match(pattern, symbol)
        
        if match:
            underlying, expiry_str, strike, option_type = match.groups()
            
            # Parse expiry date
            expiry = self._parse_expiry_date(expiry_str)
            
            return {
                'underlying': underlying,
                'expiry': expiry,
                'strike': float(strike),
                'option_type': 'call' if option_type == 'C' else 'put',
                'symbol': symbol
            }
        
        return {}
    
    def _parse_expiry_date(self, expiry_str: str) -> pd.Timestamp:
        """Parse expiry date from string like '10APR20'"""
        
        # Extract components
        match = re.match(r'(\d+)([A-Z]+)(\d+)', expiry_str)
        if match:
            day, month_str, year_str = match.groups()
            
            # Month mapping
            months = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
                'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
                'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }
            
            month = months.get(month_str, 1)
            year = 2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str)
            
            # Set expiry to 8:00 UTC (standard for crypto options)
            return pd.Timestamp(year=year, month=month, day=int(day), hour=8)
        
        return pd.Timestamp.now()
    
    def load_option_data(
        self,
        option_symbol: str,
        data_type: str = 'bars_1h',
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """Load data for a specific option"""
        
        file_path = self.options_path / option_symbol / f"{data_type}.feather"
        
        if not file_path.exists():
            return pd.DataFrame()
        
        # Load data
        df = feather.read_feather(file_path)
        
        # Ensure timestamp column
        if 'timestamp' not in df.columns and 'time' in df.columns:
            df['timestamp'] = df['time']
        elif 'timestamp' not in df.columns and df.index.name in ['timestamp', 'time']:
            df['timestamp'] = df.index
        
        # Convert to datetime
        if df['timestamp'].dtype == 'int64':
            # Assume microseconds if large integer
            if df['timestamp'].iloc[0] > 1e12:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add option metadata
        option_info = self.parse_option_symbol(option_symbol)
        for key, value in option_info.items():
            df[key] = value
        
        # Filter by date range
        if start_date:
            df = df[df['timestamp'] >= start_date]
        if end_date:
            df = df[df['timestamp'] <= end_date]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def load_all_options(
        self,
        underlying: str = 'BTC',
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        min_volume: float = 0,
        min_oi: float = 0
    ) -> pd.DataFrame:
        """Load all available options data"""
        
        all_options = []
        
        # Iterate through all option directories
        for option_dir in self.options_path.iterdir():
            if option_dir.is_dir() and option_dir.name.startswith(underlying):
                option_data = self.load_option_data(
                    option_dir.name,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not option_data.empty:
                    # Filter by volume and OI if specified
                    if min_volume > 0 and 'volume' in option_data.columns:
                        option_data = option_data[option_data['volume'] >= min_volume]
                    if min_oi > 0 and 'open_interest' in option_data.columns:
                        option_data = option_data[option_data['open_interest'] >= min_oi]
                    
                    if not option_data.empty:
                        all_options.append(option_data)
        
        if all_options:
            return pd.concat(all_options, ignore_index=True).sort_values('timestamp')
        
        return pd.DataFrame()

    def list_option_symbols(self, underlying: str = 'BTC') -> List[str]:
        """List available option symbols (directory names) for an underlying."""
        symbols: List[str] = []
        if not self.options_path.exists():
            return symbols
        for option_dir in self.options_path.iterdir():
            if option_dir.is_dir() and option_dir.name.startswith(underlying):
                symbols.append(option_dir.name)
        return sorted(symbols)

    def iter_options(
        self,
        underlying: str = 'BTC',
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        min_volume: float = 0,
        min_oi: float = 0,
        data_type: str = 'bars_1h'
    ) -> Iterator[Tuple[str, pd.DataFrame]]:
        """Yield per-option DataFrames without merging with spot.

        Yields tuples of (symbol, option_df) filtered by date and basic liquidity constraints.
        """
        for symbol in self.list_option_symbols(underlying=underlying):
            df = self.load_option_data(symbol, data_type=data_type, start_date=start_date, end_date=end_date)
            if df.empty:
                continue
            if min_volume > 0 and 'volume' in df.columns:
                df = df[df['volume'] >= min_volume]
            if min_oi > 0 and 'open_interest' in df.columns:
                df = df[df['open_interest'] >= min_oi]
            if df.empty:
                continue
            yield symbol, df
    
    def create_synchronized_dataset(
        self,
        spot_symbol: str = 'BTCUSDT',
        underlying: str = 'BTC',
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        option_filters: Optional[Dict] = None,
        convert_to_usd: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create synchronized spot and options datasets for backtesting.
        
        Args:
            convert_to_usd: If True, converts BTC option prices to USD
        
        Returns:
            Tuple of (spot_data, options_data) DataFrames with aligned timestamps
        """
        
        # Load spot data
        spot_data = self.load_spot_data(spot_symbol, start_date=start_date, end_date=end_date)
        
        # Load options data
        options_data = self.load_all_options(
            underlying=underlying,
            start_date=start_date,
            end_date=end_date,
            min_volume=option_filters.get('min_volume', 0) if option_filters else 0,
            min_oi=option_filters.get('min_oi', 0) if option_filters else 0
        )
        
        if spot_data.empty or options_data.empty:
            return spot_data, options_data
        
        # Align timestamps
        common_timestamps = set(spot_data['timestamp']) & set(options_data['timestamp'])
        
        spot_data = spot_data[spot_data['timestamp'].isin(common_timestamps)]
        options_data = options_data[options_data['timestamp'].isin(common_timestamps)]
        
        # Sort both datasets
        spot_data = spot_data.sort_values('timestamp').reset_index(drop=True)
        options_data = options_data.sort_values('timestamp').reset_index(drop=True)
        
        # Convert option prices from BTC to USD if requested
        if convert_to_usd and not options_data.empty:
            # Merge spot prices with options data to get USD conversion rate
            options_data = options_data.merge(
                spot_data[['timestamp', 'close']].rename(columns={'close': 'spot_price'}),
                on='timestamp',
                how='left'
            )
            
            # Convert price columns from BTC to USD
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in options_data.columns:
                    # Store original BTC price
                    options_data[f'{col}_btc'] = options_data[col]
                    # Convert to USD
                    options_data[col] = options_data[col] * options_data['spot_price']
        
        return spot_data, options_data

    def create_dataset_dict(
        self,
        spot_symbol: str = 'BTCUSDT',
        underlying: str = 'BTC',
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        option_filters: Optional[Dict] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Return separate spot DataFrame and a dict of per-option DataFrames.

        The options dict is keyed by option symbol directory name and values are
        raw option DataFrames (no spot merge). Columns include at least
        ['timestamp','open','high','low','close','strike','expiry','option_type','symbol']
        if available.
        """
        # Load spot
        spot_df = self.load_spot_data(spot_symbol, start_date=start_date, end_date=end_date)

        # Load options per symbol
        min_volume = option_filters.get('min_volume', 0) if option_filters else 0
        min_oi = option_filters.get('min_oi', 0) if option_filters else 0

        options_dict: Dict[str, pd.DataFrame] = {}
        for symbol, df in self.iter_options(
            underlying=underlying,
            start_date=start_date,
            end_date=end_date,
            min_volume=min_volume,
            min_oi=min_oi,
            data_type='bars_1h',
        ):
            options_dict[symbol] = df.sort_values('timestamp').reset_index(drop=True)

        return spot_df, options_dict
    
    def get_option_chain(
        self,
        timestamp: pd.Timestamp,
        underlying: str = 'BTC',
        min_dte: int = 7,
        max_dte: int = 30
    ) -> pd.DataFrame:
        """Get available option chain at a specific timestamp"""
        
        all_options = []
        
        for option_dir in self.options_path.iterdir():
            if option_dir.is_dir() and option_dir.name.startswith(underlying):
                option_info = self.parse_option_symbol(option_dir.name)
                
                if option_info:
                    # Calculate days to expiry
                    dte = (option_info['expiry'] - timestamp).days
                    
                    if min_dte <= dte <= max_dte:
                        # Load data for this timestamp
                        option_data = self.load_option_data(option_dir.name)
                        
                        if not option_data.empty:
                            # Get data at specific timestamp
                            mask = option_data['timestamp'] == timestamp
                            if mask.any():
                                all_options.append(option_data[mask])
        
        if all_options:
            chain = pd.concat(all_options, ignore_index=True)
            chain['dte'] = (chain['expiry'] - timestamp).dt.days
            chain['moneyness'] = chain['strike'] / chain['close']  # Assuming spot close available
            return chain.sort_values(['expiry', 'strike', 'option_type'])
        
        return pd.DataFrame()
    
    def calculate_implied_forward(
        self,
        option_chain: pd.DataFrame,
        risk_free_rate: float = 0.01
    ) -> float:
        """Calculate implied forward price from put-call parity"""
        
        if option_chain.empty:
            return np.nan
        
        # Find matching calls and puts
        strikes = option_chain['strike'].unique()
        
        implied_forwards = []
        
        for strike in strikes:
            calls = option_chain[(option_chain['strike'] == strike) & 
                                (option_chain['option_type'] == 'call')]
            puts = option_chain[(option_chain['strike'] == strike) & 
                               (option_chain['option_type'] == 'put')]
            
            if not calls.empty and not puts.empty:
                call_price = calls['close'].iloc[0]
                put_price = puts['close'].iloc[0]
                tte = calls['dte'].iloc[0] / 365
                
                # Put-call parity: C - P = S - K * exp(-r*T)
                # Rearranged: S = C - P + K * exp(-r*T)
                implied_forward = call_price - put_price + strike * np.exp(-risk_free_rate * tte)
                implied_forwards.append(implied_forward)
        
        if implied_forwards:
            return np.median(implied_forwards)
        
        return np.nan
    
    def get_data_summary(self) -> pd.DataFrame:
        """Get summary of available data"""
        
        summary = []
        
        # Spot data summary
        for spot_file in self.spot_path.glob('*.feather'):
            df = feather.read_feather(spot_file)
            summary.append({
                'type': 'spot',
                'symbol': spot_file.stem,
                'start_date': df.index.min() if df.index.name == 'timestamp' else df['timestamp'].min() if 'timestamp' in df else None,
                'end_date': df.index.max() if df.index.name == 'timestamp' else df['timestamp'].max() if 'timestamp' in df else None,
                'rows': len(df)
            })
        
        # Options data summary
        for option_dir in self.options_path.iterdir():
            if option_dir.is_dir():
                bars_file = option_dir / 'bars_1h.feather'
                if bars_file.exists():
                    df = feather.read_feather(bars_file)
                    option_info = self.parse_option_symbol(option_dir.name)
                    
                    summary.append({
                        'type': 'option',
                        'symbol': option_dir.name,
                        'strike': option_info.get('strike'),
                        'expiry': option_info.get('expiry'),
                        'option_type': option_info.get('option_type'),
                        'start_date': df.index.min() if df.index.name == 'timestamp' else df['timestamp'].min() if 'timestamp' in df else None,
                        'end_date': df.index.max() if df.index.name == 'timestamp' else df['timestamp'].max() if 'timestamp' in df else None,
                        'rows': len(df)
                    })
        
        return pd.DataFrame(summary)