from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.order import Order
from IBKRWrapper import IBKRWrapper
from typing import Dict, Tuple, List
import pandas as pd
from datetime import datetime
import threading
import time
import numpy as np
import queue
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingEngine:
    def __init__(self):
        self.app = IBKRWrapper()
        self.strategy = None
        self.contract = None
        self.is_live_trading = False
        self.order_id = 0
        self.positions = {}
        self.historical_data = None
        self.risk_limits = {
            'max_position': 1000,
            'max_loss_percent': 2.0,
            'max_trade_size': 100
        }
        
        # Connect to IBKR
    def _connect(self):
        """Establish connection to IBKR"""
        self.app.connect('127.0.0.1', 7497, clientId=1)
        
        # Start the connection thread
        con_thread = threading.Thread(target=self.app.run, daemon=True)
        con_thread.start()
        time.sleep(1)  # Give time for connection to establish
    
    def create_contract(self, symbol):
        """Create a stock contract"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = 'STK'
        contract.exchange = 'SMART'
        contract.currency = 'USD'
        return contract
    
    def create_order(self, action, quantity, order_type='MKT', limit_price=None):
        """
        Create an order object with proper quantity handling
        
        Args:
            action (str): 'BUY' or 'SELL'
            quantity (float): Number of shares/contracts
            order_type (str): 'MKT' for Market or 'LMT' for Limit
            limit_price (float): Limit price (required for 'LMT' orders)
        """
        self.order_id += 1
        order = Order()
        order.orderId = self.order_id
        order.action = action
        # Convert quantity to integer to avoid fractional share issues
        order.totalQuantity = int(quantity)
        order.orderType = order_type
        
        if order_type == 'LMT':
            if limit_price is None:
                raise ValueError("Limit price required for LMT orders")
            order.lmtPrice = limit_price
            
        return order
    
    def set_strategy(self, strategy):
        """Set the trading strategy"""
        self.strategy = strategy
        self.contract = self.create_contract(strategy.symbol)
    
    def check_risk_limits(self, action, quantity, price):
        """Check if trade complies with risk management rules"""
        if action == "BUY":
            new_position = self.strategy.position + quantity
        else:
            new_position = self.strategy.position - quantity
            
        # Check position limits
        if abs(new_position) > self.risk_limits['max_position']:
            return False, "Position limit exceeded"
            
        # Check trade size
        if quantity > self.risk_limits['max_trade_size']:
            return False, "Trade size too large"
            
        # Check max loss
        if self.strategy.portfolio_value > 0:
            potential_loss = (self.strategy.portfolio_value - new_position * price) / self.strategy.portfolio_value * 100
            if potential_loss > self.risk_limits['max_loss_percent']:
                return False, "Maximum loss limit exceeded"
                
        return True, "Trade allowed"
    
    def execute_trade(self, action, quantity, price=None):
        """Execute trade with risk checks and proper quantity handling"""
        # Round quantity to nearest whole number
        quantity = int(round(quantity))
        
        if quantity == 0:
            print("Trade rejected: Quantity too small (rounded to 0)")
            return False
            
        is_allowed, message = self.check_risk_limits(action, quantity, price or self.strategy.current_price)
        
        if not is_allowed:
            print(f"Trade rejected: {message}")
            return False
            
        order = self.create_order(action, quantity, 'MKT' if price is None else 'LMT', price)
        self.app.placeOrder(order.orderId, self.contract, order)
        self.strategy.update_position(action, quantity, price or self.strategy.current_price)
        
        print(f"Order placed: {action} {quantity} shares at {'market price' if price is None else price}")
        return True

    
    
    def backtest(self, data, initial_capital, position_size=10000):
        """Run backtest with proper column name handling"""
        try:
            if data is None or len(data) == 0:
                raise ValueError("No historical data available for backtesting")

            # Ensure column names are lowercase
            data.columns = data.columns.str.lower()
            
            # Verify required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = set(required_columns) - set(data.columns)
            
            if missing_columns:
                raise ValueError(f"Missing required columns for backtesting: {missing_columns}")
    
            self.historical_data = data
            signals = self.strategy.calculate_signals(self.historical_data)
            return self.run_backtest(signals, initial_capital, position_size)
        
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            raise

    
    def run_backtest(self, df: pd.DataFrame, initial_capital: float, position_size: int) -> Tuple[List[Dict], float, float]:
        """
        Run backtest on signals, track positions, and generate trade records with portfolio metrics
        
        Args:
            df (pd.DataFrame): DataFrame with signals
            initial_capital (float): Starting capital for the backtest
            position_size (int): Number of units per trade
            
        Returns:
            Tuple[List[Dict], float, float]: List of trades, maximum drawdown, final portfolio value
        """
        trades = []
        current_trade = None
        df = df.copy()
        
        # Portfolio tracking
        current_portfolio = initial_capital
        peak_portfolio = initial_capital
        max_drawdown = 0
        
        # Initialize position column
        df['position'] = 0
        
        # Iterate through data to track positions and record trades
        for i in range(1, len(df)):
            current_idx = df.index[i]
            prev_idx = df.index[i-1]
            
            # Get current signal and previous position
            current_signal = df.loc[current_idx, 'signal']
            prev_position = df.loc[prev_idx, 'position']
            
            # Default to previous position
            df.loc[current_idx, 'position'] = prev_position
            
            # Update portfolio value for open position and calculate drawdown
            if prev_position == 1: 
                price_change = (df.loc[current_idx, 'close'] - df.loc[prev_idx, 'close'])/df.loc[prev_idx, 'close']
                portfolio_change = price_change * position_size
                current_portfolio += portfolio_change
                
                # Update maximum drawdown in monetary terms
                peak_portfolio = max(peak_portfolio, current_portfolio)
                current_drawdown = peak_portfolio - current_portfolio  # Absolute dollar amount
                max_drawdown = max(max_drawdown, current_drawdown)

            # Entry signal (only if not in position)
            if current_signal == 1 and prev_position == 0:
                df.loc[current_idx, 'position'] = 1
                current_trade = {
                    'entry_time': current_idx,
                    'entry_price': df.loc[current_idx, 'close'],
                    'type': 'LONG',
                    'size': position_size
                }
                    
            # Exit signal (only if in position)
            elif current_signal == -1 and prev_position == 1:
                df.loc[current_idx, 'position'] = 0
                if current_trade is not None:
                    
                    trade_record = self.strategy.create_trade_record(
                        entry_time=current_trade['entry_time'],
                        exit_time=current_idx,
                        trade_type=current_trade['type'],
                        size=current_trade['size'],
                        entry_price=current_trade['entry_price'],
                        exit_price=df.loc[current_idx, 'close'],
                    )
                    
                    trades.append(trade_record)
                    current_trade = None
        
        # Close any open position on the final day
        last_idx = df.index[-1]
        if current_trade is not None:
            df.loc[last_idx, 'position'] = 0
            
            # Calculate final portfolio value including last trade
            #final_price_change = (df.loc[last_idx, 'close'] - current_trade['entry_price'])/current_trade['entry_price']
            #final_portfolio_change = final_price_change * position_size
            #current_portfolio += final_portfolio_change
            
            trade_record = self.strategy.create_trade_record(
                entry_time=current_trade['entry_time'],
                exit_time=last_idx,
                trade_type=current_trade['type'],
                size=current_trade['size'],
                entry_price=current_trade['entry_price'],
                exit_price=df.loc[last_idx, 'close'],
            )
            
            trades.append(trade_record)
        
        return {'trades': trades, 'max_drawdown': max_drawdown , 'current_portfolio': current_portfolio, 'data': df}

    def start_trading(self, interval):
        """Initialize and start live trading"""
        if not self.app.isConnected():
            self._connect()

        self.interval = interval
        # Request historical data first
        self.historical_data = self.app.request_historical_data(self.contract, self.interval)
        time.sleep(5)  # Wait for historical data
        
        if len(self.historical_data) > 0:
            # Initialize strategy with historical data
            df = pd.DataFrame(self.historical_data)
            self.strategy.initialize(df)
            
            # Start live trading
            self.start_live_trading()
        else:
            raise ValueError("Failed to get historical data")
    
    def start_live_trading(self):
        """Start live trading with the strategy"""
        if not self.strategy:
            raise ValueError("No strategy set. Call set_strategy() first.")
            
        self.is_live_trading = True
        interval = self.interval * 60
        self.app.reqRealtimeData(0, self.contract, interval)
        
        while self.is_live_trading:
            try:
                current_data = self.app.data_queue.get(timeout=1)
                action, quantity, price = self.strategy.generate_trade_decision(current_data)
                
                if action:
                    if self.position == 0:  # No position
                        if self.execute_trade(action, quantity, price):
                            stop_price = self.calculate_stop_loss(action, price)
                            self.place_stop_loss_order(action, quantity, stop_price)
                    else:  # Position exists
                        if self.should_update_position(action):
                            new_stop_price = self.calculate_stop_loss(action, price)
                            self.update_stop_loss_order(new_stop_price)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in live trading: {str(e)}")
                self.stop_live_trading()
    
    def stop_live_trading(self):
        """Stop live trading"""
        self.is_live_trading = False
        self.app.cancelRealTimeBars(0)
        self.app.disconnect_wrapper()

    def should_update_position(self, new_action: str) -> bool:
        """Check if position should be updated"""
        return (self.position > 0 and new_action == "BUY") or \
               (self.position < 0 and new_action == "SELL")
               
    def update_stop_loss_order(self, new_stop_price: float):
        """Update existing stop loss order"""
        if self.stop_order_id is not None:
            self.cancelOrder(self.stop_order_id)
            action = "SELL" if self.position > 0 else "BUY"
            self.place_stop_loss_order(action, abs(self.position), new_stop_price)
    
    def get_historical_data(self, duration="1 D", interval="1 min", symbol=None):
        """Get historical data with error handling"""

        def validate_dataframe(df, timeframe):
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {timeframe} data: {missing_cols}")
        try:
            if not self.contract and symbol is None:
                raise ValueError("No contract set. Call set_strategy() first.")
            elif not self.contract and symbol is not None:
                self.contract = self.create_contract(symbol)
            
            
            if not self.app.isConnected():
                self._connect()

            intraday_intervals = ['1 secs', '5 secs', '10 secs', '15 secs', '30 secs',
                            '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins',
                            '20 mins', '30 mins', '1 hour', '2 hours', '3 hours', '4 hours']
        
            is_intraday = interval in intraday_intervals
    
            # Get intraday data
            intraday_df = self.app.histData(1, self.contract, duration, interval)
            
            validate_dataframe(intraday_df, "intraday")
            
            # For intraday intervals, also get daily data for the last year
            if is_intraday:
                # Wait a bit between requests to avoid rate limiting
                time.sleep(1)
                daily_df = self.app.histData(2, self.contract, "1 Y", "1 day")
                validate_dataframe(daily_df, "daily")

                return intraday_df, daily_df
            else:
                # For non-intraday intervals, just return the single dataframe
                if 'Timestamp' in intraday_df.columns:
                    intraday_df['Timestamp'] = pd.to_datetime(intraday_df['Timestamp'])
                    intraday_df.set_index('Timestamp', inplace=True)
                intraday_df.sort_index(inplace=True)
                return None, intraday_df
    
        except Exception as e:
            logging.error(f"Error fetching historical data: {str(e)}")
            raise
