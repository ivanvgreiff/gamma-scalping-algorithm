import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.gamma_scalping_simulator import GammaScalpingSimulator, GammaScalpingResult
from models.options_pricing import implied_volatility

@dataclass
class BacktestConfig:
    """Configuration for backtest run"""
    initial_capital: float = 100000
    commission_rate: float = 0.0005
    slippage_bps: float = 10
    risk_free_rate: float = 0.01
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    
@dataclass
class StrategyConfig:
    """Configuration for strategy variant"""
    name: str
    hedge_threshold: float = 0.1  # Delta threshold for rebalancing
    option_selection: str = 'atm'  # 'atm', 'otm_call', 'otm_put', 'straddle'
    otm_percent: float = 0.05  # For OTM selection (5% OTM)
    position_size: float = 1.0  # Number of option contracts
    vol_adjustment: bool = False  # Adjust thresholds based on volatility

class BacktestEngine:
    """
    Backtesting engine for gamma scalping strategies.
    Uses per-option simulation for clear P&L attribution.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.simulator = GammaScalpingSimulator(
            commission_rate=config.commission_rate,
            slippage_bps=config.slippage_bps,
            risk_free_rate=config.risk_free_rate
        )
        
    def select_options(
        self,
        spot_price: float,
        available_options: pd.DataFrame,
        config: StrategyConfig,
        current_timestamp: pd.Timestamp
    ) -> pd.DataFrame:
        """Select options based on strategy configuration"""
        
        if available_options.empty:
            return pd.DataFrame()
        
        # Filter by days to expiry (prefer 7-30 DTE)
        available_options['dte'] = (available_options['expiry'] - current_timestamp).dt.days
        options = available_options[(available_options['dte'] >= 7) & (available_options['dte'] <= 30)]
        
        if options.empty:
            options = available_options
        
        selected = pd.DataFrame()
        
        if config.option_selection == 'atm':
            # Select ATM option
            options['moneyness'] = abs(options['strike'] - spot_price) / spot_price
            selected = options.nsmallest(1, 'moneyness')
            
        elif config.option_selection == 'otm_call':
            # Select OTM call
            target_strike = spot_price * (1 + config.otm_percent)
            calls = options[options['option_type'] == 'call']
            calls['distance'] = abs(calls['strike'] - target_strike)
            selected = calls.nsmallest(1, 'distance')
            
        elif config.option_selection == 'otm_put':
            # Select OTM put
            target_strike = spot_price * (1 - config.otm_percent)
            puts = options[options['option_type'] == 'put']
            puts['distance'] = abs(puts['strike'] - target_strike)
            selected = puts.nsmallest(1, 'distance')
            
        elif config.option_selection == 'straddle':
            # Select ATM straddle (both call and put)
            options['moneyness'] = abs(options['strike'] - spot_price) / spot_price
            atm_strike = options.nsmallest(1, 'moneyness')['strike'].iloc[0]
            selected = options[options['strike'] == atm_strike]
        
        return selected
    
    def run_backtest(
        self,
        spot_data: pd.DataFrame,
        options_data: pd.DataFrame,
        strategy_config: StrategyConfig
    ) -> Dict[str, Any]:
        """
        Run backtest for a specific strategy configuration.
        Simulates each selected option independently.
        
        Args:
            spot_data: DataFrame with columns ['timestamp', 'close']
            options_data: DataFrame with columns ['timestamp', 'strike', 'expiry', 
                         'option_type', 'close', 'volume', 'open_interest']
            strategy_config: Strategy configuration
        
        Returns:
            Dictionary with aggregated backtest results
        """
        
        # Filter data by date range
        if self.config.start_date:
            spot_data = spot_data[spot_data['timestamp'] >= self.config.start_date]
            options_data = options_data[options_data['timestamp'] >= self.config.start_date]
        if self.config.end_date:
            spot_data = spot_data[spot_data['timestamp'] <= self.config.end_date]
            options_data = options_data[options_data['timestamp'] <= self.config.end_date]
        
        # Get initial timestamp and spot price
        if spot_data.empty or options_data.empty:
            return {'error': 'No data available for backtest period'}
            
        initial_timestamp = spot_data['timestamp'].min()
        initial_spot = spot_data[spot_data['timestamp'] == initial_timestamp]['close'].iloc[0]
        
        # Select options to trade
        initial_options = options_data[options_data['timestamp'] == initial_timestamp]
        selected_options = self.select_options(initial_spot, initial_options, strategy_config, initial_timestamp)
        
        if selected_options.empty:
            return {'error': 'No suitable options found'}
        
        # Run simulation for each selected option
        results = []
        for _, option in selected_options.iterrows():
            # Get data for this specific option
            option_data = options_data[
                (options_data['strike'] == option['strike']) &
                (options_data['expiry'] == option['expiry']) &
                (options_data['option_type'] == option['option_type'])
            ]
            
            if option_data.empty:
                continue
                
            # Run simulation
            result = self.simulator.simulate(
                spot_prices=spot_data,
                option_data=option_data,
                hedge_threshold=strategy_config.hedge_threshold,
                position_size=strategy_config.position_size
            )
            
            if result:
                results.append(result)
        
        # Aggregate results
        if not results:
            return {'error': 'No successful simulations'}
            
        total_option_pnl = sum(r.option_pnl for r in results)
        total_hedge_pnl = sum(r.hedge_pnl for r in results)
        total_pnl = sum(r.total_pnl for r in results)
        total_commissions = sum(r.commission_cost for r in results)
        total_slippage = sum(r.slippage_cost for r in results)
        
        # Calculate metrics
        total_return = total_pnl / self.config.initial_capital
        avg_return = total_pnl / len(results) / self.config.initial_capital
        
        # Simple Sharpe calculation
        if len(results) > 1:
            returns = [r.total_pnl / self.config.initial_capital for r in results]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
            
        # Calculate max drawdown from aggregated P&L history
        if results[0].pnl_history is not None and not results[0].pnl_history.empty:
            cumulative_pnl = sum(r.pnl_history['total_pnl'].values for r in results if r.pnl_history is not None)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = (cumulative_pnl - running_max) / self.config.initial_capital
            max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        else:
            max_drawdown = 0
            
        # Return results dictionary
        return {
            'strategy_name': strategy_config.name,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'option_pnl': total_option_pnl,
            'hedge_pnl': total_hedge_pnl,
            'total_pnl': total_pnl,
            'commission_cost': total_commissions,
            'slippage_cost': total_slippage,
            'num_options': len(results),
            'avg_hedges': np.mean([r.num_hedges for r in results]),
            'results': results  # Keep individual option results
        }
    
    def run_multiple_strategies(
        self,
        spot_data: pd.DataFrame,
        options_data: pd.DataFrame,
        strategy_configs: List[StrategyConfig]
    ) -> pd.DataFrame:
        """Run backtest for multiple strategy configurations"""
        
        results = []
        
        for config in strategy_configs:
            print(f"Running backtest for strategy: {config.name}")
            result = self.run_backtest(spot_data, options_data, config)
            
            # Extract key metrics
            if 'error' not in result:
                summary = {
                    'strategy': config.name,
                    'hedge_threshold': config.hedge_threshold,
                    'option_selection': config.option_selection,
                    'total_return': result.get('total_return', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'option_pnl': result.get('option_pnl', 0),
                    'hedge_pnl': result.get('hedge_pnl', 0),
                    'total_pnl': result.get('total_pnl', 0),
                    'num_options': result.get('num_options', 0)
                }
                results.append(summary)
        
        return pd.DataFrame(results)
    
    def parameter_optimization(
        self,
        spot_data: pd.DataFrame,
        options_data: pd.DataFrame,
        base_config: StrategyConfig,
        param_grid: Dict[str, List[Any]]
    ) -> pd.DataFrame:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            param_grid: Dictionary of parameters to optimize
                       e.g., {'hedge_threshold': [0.05, 0.1, 0.15]}
        """
        
        results = []
        
        # Generate all parameter combinations
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for values in itertools.product(*param_values):
            # Create config with current parameters
            config = StrategyConfig(
                name=f"opt_{'-'.join(map(str, values))}",
                hedge_threshold=base_config.hedge_threshold,
                option_selection=base_config.option_selection,
                position_size=base_config.position_size
            )
            
            # Update with optimization parameters
            for name, value in zip(param_names, values):
                setattr(config, name, value)
            
            # Run backtest
            result = self.run_backtest(spot_data, options_data, config)
            
            # Store results
            param_result = {param: value for param, value in zip(param_names, values)}
            param_result.update({
                'total_return': result['total_return'],
                'sharpe_ratio': result['sharpe_ratio'],
                'max_drawdown': result['max_drawdown']
            })
            results.append(param_result)
        
        return pd.DataFrame(results)