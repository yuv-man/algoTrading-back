from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

class BaseStrategy(ABC):
    def __init__(self, symbol: str, params: dict = None):
        """
        Initialize base strategy
        
        Args:
            symbol (str): Trading symbol
            params (dict): Strategy parameters
        """
        self.symbol = symbol
        self.params = params or {}
        self.position = 0
        self.portfolio_value = 100000  # Default starting capital
        self.trades = []
        self.current_indicators = {}

    def update_position_backtest(self, trade_type: str, size: int, price: float, 
                               timestamp: pd.Timestamp, indicators: Dict[str, float]) -> None:
        """
        Update position and record trade during backtesting
        
        Args:
            trade_type (str): 'BUY' or 'SELL'
            size (int): Number of shares/contracts
            price (float): Trade execution price
            timestamp (pd.Timestamp): Timestamp of the trade
            indicators (Dict[str, float]): Dictionary of indicator values at trade time
        """
        # Update position and portfolio
        trade_value = size * price
        if trade_type == "BUY":
            self.position += size
            self.portfolio_value -= trade_value
        else:  # SELL
            self.position -= size
            self.portfolio_value += trade_value

        # Calculate trade metrics
        trade_info = self._create_trade_record(
            timestamp=timestamp,
            trade_type=trade_type,
            size=size,
            price=price,
            indicators=indicators
        )
        
        # Record the trade
        self.trades.append(trade_info)

    def update_position_live(self, trade_type: str, size: int, price: float) -> None:
        """
        Update position and record trade during live trading
        
        Args:
            trade_type (str): 'BUY' or 'SELL'
            size (int): Number of shares/contracts
            price (float): Trade execution price
        """
        # Update position and portfolio
        trade_value = size * price
        if trade_type == "BUY":
            self.position += size
            self.portfolio_value -= trade_value
        else:  # SELL
            self.position -= size
            self.portfolio_value += trade_value

        # Calculate trade metrics
        trade_info = self._create_trade_record(
            timestamp=pd.Timestamp.now(),
            trade_type=trade_type,
            size=size,
            price=price,
            indicators=self.current_indicators
        )
        
        # Record the trade
        self.trades.append(trade_info)

    def create_trade_record(self, entry_time: pd.Timestamp, exit_time: pd.Timestamp, 
                       trade_type: str, size: int, entry_price: float, 
                       exit_price: float) -> Dict:
        """
        Create a standardized trade record with profit calculations
        
        Args:
            entry_time (pd.Timestamp): Time of trade entry
            exit_time (pd.Timestamp): Time of trade exit
            trade_type (str): 'LONG' or 'SHORT'
            size (int): Trade size
            entry_price (float): Entry price
            exit_price (float): Exit price
            indicators (Dict[str, float]): Indicator values at entry
            
        Returns:
            Dict: Complete trade record with profit calculations
        """
        # Calculate trade value and profits
        
        if trade_type == 'LONG':
            profit_pct = ((exit_price - entry_price) / entry_price) * 100
        else:  # SHORT
            profit_pct = ((entry_price - exit_price) / entry_price) * 100

        profit = (profit_pct/100) * size
        self.portfolio_value += profit
         
        return {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'duration': exit_time - entry_time,
            'type': trade_type,
            'size': size,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit': round(profit, 2),
            'profit_pct': round(profit_pct, 2),
            'position': self.position,
            'portfolio_value': self.portfolio_value,
        }

    def update_indicators(self, indicators: Dict[str, float]) -> None:
        """
        Update current indicator values
        
        Args:
            indicators (Dict[str, float]): Current indicator values
        """
        self.current_indicators = indicators

    @abstractmethod
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading signals
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            pd.DataFrame: Data with signals
        """
        pass

    @abstractmethod
    def generate_trade_decision(self, current_data: Dict) -> tuple:
        """
        Generate trading decision based on current market data
        
        Args:
            current_data (Dict): Current market data
            
        Returns:
            tuple: (trade_type, size, price) or (None, None, None)
        """
        pass

    def get_trade_history(self) -> pd.DataFrame:
        """
        Get trade history as DataFrame
        
        Returns:
            pd.DataFrame: Trade history
        """
        return pd.DataFrame(self.trades)

    def get_current_position(self) -> Dict[str, Any]:
        """
        Get current position information
        
        Returns:
            Dict[str, Any]: Current position details
        """
        return {
            'symbol': self.symbol,
            'position': self.position,
            'portfolio_value': self.portfolio_value,
            'current_indicators': self.current_indicators
        }

    def calculate_stop_loss(self, action: str, entry_price: float) -> float:
        """Calculate stop loss price - should be implemented by each strategy"""
        raise NotImplementedError("Subclass must implement calculate_stop_loss()")
        
    def calculate_trailing_stop(self, action: str, current_price: float) -> float:
        """Calculate trailing stop loss price - should be implemented by each strategy"""
        raise NotImplementedError("Subclass must implement calculate_trailing_stop()")

    def reset(self) -> None:
        """Reset strategy state"""
        self.position = 0
        self.portfolio_value = 100000
        self.trades = []
        self.current_indicators = {}