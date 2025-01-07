from typing import Dict, List, Tuple, Union, Any, Optional, Callable
from enum import Enum
import numpy as np
from itertools import product
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

class OptimizeTarget(Enum):
    """Defines the possible optimization targets"""
    NET_PROFIT = "Net Profit"
    MAX_DRAWDOWN = "Max Drawdown"
    WIN_RATE = "Win Rate"
    PROFIT_FACTOR = "Profit Factor"
    RETURN_OVER_MAX_DRAWDOWN = "Return/Drawdown"
    SHARPE_RATIO = "Sharpe Ratio"  # Added Sharpe Ratio as optimization target

class ConstraintType(Enum):
    """Defines the possible constraint types"""
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="

@dataclass
class OptimizationResult:
    """Data class to store optimization results"""
    best_parameters: Dict[str, int]
    best_performance: Dict[str, Any]
    target: str
    score: float
    metrics: Dict[str, float]
    max_drawdown: float
    
class Optimizer:
    def __init__(self, backtester: object, logger: Optional[logging.Logger] = None):
        """
        Initialize the optimizer with a backtester instance
        
        Args:
            backtester: Object that implements run_backtest_with_data method
            logger: Optional logger instance for debugging
        """
        if not hasattr(backtester, 'run_backtest_with_data'):
            raise ValueError("Backtester must have 'run_backtest_with_data' method")
        
        self.backtester = backtester
        self.historical_data = None
        self.interval = None
        self.period = None
        self.logger = logger or logging.getLogger(__name__)
        self._constraint_ops = {
            ConstraintType.GREATER_THAN: lambda x, y: x > y,
            ConstraintType.GREATER_EQUAL: lambda x, y: x >= y,
            ConstraintType.LESS_THAN: lambda x, y: x < y,
            ConstraintType.LESS_EQUAL: lambda x, y: x <= y,
            ConstraintType.EQUAL: lambda x, y: x == y,
            ConstraintType.NOT_EQUAL: lambda x, y: x != y
        }

    def _validate_constraints(self, result: Dict, constraints: List[Tuple]) -> bool:
        """
        Validate if the result meets all specified constraints
        
        Args:
            result: Backtest result dictionary
            constraints: List of (metric, operator, value) tuples
            
        Returns:
            bool: True if all constraints are satisfied
        """
        if not constraints:
            return True
            
        for metric, op, value in constraints:
            try:
                metric_value = float(result['data']['metrics'].get(metric, float('-inf')))
                if not self._constraint_ops[ConstraintType(op)](metric_value, float(value)):
                    return False
            except (KeyError, ValueError) as e:
                self.logger.warning(f"Error checking constraint {metric}: {str(e)}")
                return False
        return True

    def _calculate_objective(self, result: Dict, target: OptimizeTarget) -> float:
        """
        Calculate the objective value based on the selected optimization target
        
        Args:
            result: Backtest result dictionary
            target: OptimizeTarget enum value
            
        Returns:
            float: Calculated objective value (negative for maximization)
        """
        try:
            if result.get('status') != 'success':
                return float('inf')
                
            data = result.get('data', {})
            trades = data.get('trades', [])
            metrics = data.get('metrics', {})
            max_drawdown = float(data.get('max_drawdown', float('inf')))
            current_capital = float(data.get('current_capital', 0))
            initial_capital = float(data.get('initial_capital', 1))
            
            objective_functions = {
                OptimizeTarget.NET_PROFIT: 
                    lambda: -(current_capital - initial_capital),
                OptimizeTarget.MAX_DRAWDOWN:
                    lambda: max_drawdown,
                OptimizeTarget.WIN_RATE:
                    lambda: -float(metrics.get('Win Rate', 0)),
                OptimizeTarget.PROFIT_FACTOR:
                    lambda: -float(metrics.get('Profit Factor', 0)),
                OptimizeTarget.RETURN_OVER_MAX_DRAWDOWN:
                    lambda: -((current_capital - initial_capital) / max_drawdown) if max_drawdown > 0 else float('inf'),
                OptimizeTarget.SHARPE_RATIO:
                    lambda: -float(metrics.get('Sharpe Ratio', 0))
            }
            
            return objective_functions.get(target, lambda: float('inf'))()
                
        except Exception as e:
            self.logger.error(f"Error calculating objective: {str(e)}")
            return float('inf')

    def _evaluate_params(
        self,
        params: Dict[str, int],
        strategy_type: str,
        symbol: str,
        optimize_target: OptimizeTarget,
        constraints: List[Tuple]
    ) -> Tuple[Dict[str, int], float, Dict]:
        """
        Evaluate a single parameter combination
        
        Args:
            params: Dictionary of parameter values
            strategy_type: Type of trading strategy
            symbol: Trading symbol
            optimize_target: Optimization target
            constraints: List of constraints
            
        Returns:
            Tuple containing parameters, score, and backtest result
        """
        try:
            if not hasattr(self.backtester.trading_engine, 'strategy') or \
               self.backtester.trading_engine.strategy is None:
                raise ValueError("Strategy not properly initialized")
                
            self.backtester.trading_engine.strategy.params = params
            result = self.backtester.run_backtest_with_data(
                interval=self.interval,
                period=self.period
            )
            
            if not self._validate_constraints(result, constraints):
                return params, float('inf'), result
                
            score = self._calculate_objective(result, optimize_target)
            return params, score, result
            
        except Exception as e:
            self.logger.error(f"Error evaluating parameters {params}: {str(e)}")
            return params, float('inf'), None

    def optimize(
        self,
        strategy_type: str,
        param_ranges: Dict[str, Union[Tuple[int, int], Tuple[int, int, int]]],
        initial_guess: Union[Dict[str, int], List[int]],
        symbol: str,
        interval: str,
        period: str,
        optimize_target: OptimizeTarget = OptimizeTarget.NET_PROFIT,
        constraints: List[Tuple[str, str, Union[str, ConstraintType]]] = None,
        max_workers: int = 4
    ) -> OptimizationResult:
        """
        Optimize strategy parameters using parallel grid search
        
        Args:
            strategy_type: Type of trading strategy
            param_ranges: Dictionary of parameter ranges (min, max) or (min, max, step)
            initial_guess: Initial parameter values as dict or list
            symbol: Trading symbol
            optimize_target: Optimization target
            constraints: List of (metric, operator, value) constraints
            max_workers: Maximum number of parallel workers
            
        Returns:
            OptimizationResult object containing optimal parameters and performance
        """
        if period is not None:
            self.period = period
        if interval is not None:
            self.interval = interval
        if self.historical_data is None:
            self.backtester.get_stock_data(symbol, interval=interval, period=period)
        
        # Convert initial_guess list to dict if necessary
        param_names = list(param_ranges.keys())
        if isinstance(initial_guess, list):
            initial_guess = dict(zip(param_names, initial_guess))
        
        # Register strategy with initial parameters
        try:
            self.backtester.register_strategy(
                strategy_type=strategy_type,
                symbol=symbol,
                params=initial_guess
            )
        except Exception as e:
            self.logger.error(f"Error registering strategy: {str(e)}")
            raise ValueError(f"Failed to register strategy: {str(e)}")
            
        if not hasattr(self.backtester.trading_engine, 'strategy') or \
           self.backtester.trading_engine.strategy is None:
            raise ValueError("Strategy registration failed")
        
        best_params = None
        best_score = float('inf')
        best_result = None
        
        # Generate parameter combinations
        param_names = list(param_ranges.keys())
        param_values = []
        for range_tuple in param_ranges.values():
            if len(range_tuple) == 2:
                start, end = range_tuple
                step = 1
            else:
                start, end, step = range_tuple
            param_values.append(list(range(start, end + 1, step)))
        combinations = [dict(zip(param_names, values)) for values in product(*param_values)]

        # Evaluate combinations in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_params = {
                executor.submit(
                    self._evaluate_params,
                    params,
                    strategy_type,
                    symbol,
                    optimize_target,
                    constraints
                ): params for params in combinations
            }
            
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    params, score, result = future.result()
                    if score < best_score:
                        best_score = score
                        best_params = params
                        best_result = result
                except Exception as e:
                    self.logger.error(f"Error processing parameters {params}: {str(e)}")
        
        # Calculate final metrics
        metrics = best_result.get('data', {}).get('metrics', {}) if best_result else {}
        max_drawdown = best_result.get('data', {}).get('max_drawdown', {})
    
        return OptimizationResult(
            best_parameters=best_params,
            best_performance=best_result,
            target=optimize_target.value,
            score=best_score,
            metrics=metrics,
            max_drawdown=max_drawdown
        )
        