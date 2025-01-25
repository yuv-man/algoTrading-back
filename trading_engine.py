from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.order import Order
from IBKRWrapper import IBKRWrapper
from MongoDBWrapper import MongoDBWrapper
from typing import Dict, Tuple, List
import pandas as pd
from datetime import datetime
import threading
import time
import numpy as np
import queue
import logging
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingEngine:
    def __init__(self, socketio=None):
        self.app = IBKRWrapper()
        self.strategy = None
        self.contract = None
        self.is_live_trading = False
        self.order_id = 0
        self.positions = {}
        self.stop_loss = {}
        self.stop_order_ids = {}
        self.initial_portfolio_value = None
        self.last_profit_update = 0
        self.historical_data = None
        self.socketio = socketio
        self.profit_update_interval = 15 * 60
        self.active_trades = {}  # Format: {symbol: {'is_trading': bool, 'strategy': str}}
        self.trading_threads = {}  # Format: {symbol: thread}
        self.account_code = None
        self.isConnected = False
        self.mongo = MongoDBWrapper()
        self.risk_limits = {
            'max_position': 1000,
            'max_loss_percent': 2.0,
            'max_trade_size': 100
        }

    def _connect(self):
        """Establish connection to IBKR via socket"""
        try:
            if hasattr(self.app, 'con_thread') and getattr(self.app, 'con_thread', None) is not None:
                if self.app.is_connected:
                    self.isConnected = True
                    return True
                else:
                    self.app.disconnect()
                    time.sleep(1)
    
            HOST = '127.0.0.1'
            PORT = 7497
            CLIENT_ID = random.randint(1000, 9999)
    
            self.app.connect(HOST, PORT, clientId=CLIENT_ID)
            logger.info(f"Attempting socket connection to {HOST}:{PORT}")
    
            self.app.con_thread = threading.Thread(
                target=self.app.run,
                daemon=True,
                name=f'IBConnection_{CLIENT_ID}'
            )
            self.app.con_thread.start()
    
            # Wait for connection acknowledgment
            timeout = 10
            if not self.app.connect_event.wait(timeout):
                logger.error("Connection timeout")
                self.isConnected = False
                return False

            # Add delay after connection before requesting account info
            time.sleep(1)
    
            # After connection is established, get the account code
            timeout_acc = 5
            if not self.app.accounts_received.wait(timeout_acc):
                logger.error("Account retrieval timeout")
                return False
    
            if self.app.managed_accounts:
                self.account_code = self.app.managed_accounts[0]
                logger.info(f"Using account: {self.account_code}")
            else:
                logger.error("No accounts found")
                return False
    
            logger.info("Successfully connected to IBKR")
            self.isConnected = True
            return True
    
        except Exception as e:
            logger.error(f"Socket connection error: {str(e)}")
            self.isConnected = False
            return False

    def get_account_info(self, timeout=30):
        """Get detailed account information"""
        try:

            if not self.isConnected:
                self._connect()
            if not self.account_code:
                logger.error("No account code available")
                return None

            # Reset events and data
            self.app.account_value = None
            self.app.account_event.clear()
            
            # Add delay before requesting account updates
            time.sleep(0.5)
            
            # Request account updates
            self.app.reqAccountUpdates(True, self.account_code)
            
            if not self.app.account_event.wait(timeout):
                raise TimeoutError("Account data request timed out")
            
            account_info = {
                'totalValue': self.app.account_details['NetLiquidation'],
                'availableFunds': self.app.account_details['AvailableFunds'],
                'buyingPower': self.app.account_details['BuyingPower'],
                'portfolioValue': self.app.account_details['GrossPositionValue'],
                'initialMargin': self.app.account_details['FullInitMarginReq'],
                'maintenanceMargin': self.app.account_details['FullMaintMarginReq']
            }
            
            return account_info
            
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return None
        finally:
            if self.app.is_connected:
                self.app.reqAccountUpdates(False, self.account_code)

    def create_contract(self, symbol):
        """Create a stock contract"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = 'STK'
        contract.exchange = 'SMART'
        contract.currency = 'USD'
        return contract

    def set_strategy(self, strategy):
        """Set the trading strategy"""
        self.strategy = strategy
        self.contract = self.create_contract(strategy.symbol)

    def get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        try:
            positions_df = self.app.request_positions()
            if positions_df.empty:
                return 0.0
            return positions_df['marketValue'].sum()
        except Exception as e:
            logger.error(f"Error getting portfolio value: {str(e)}")
            return 0.0

    def get_positions_summary(self) -> List[Dict]:
        """Get summary of current positions"""
        try:
            if not self.isConnected:
                self._connect()
            positions_df = self.app.request_positions()
            print(positions_df)
            if positions_df.empty:
                return []
            
            positions_list = []
            for _, pos in positions_df.iterrows():
                positions_list.append({
                    'symbol': pos['symbol'],
                    'position': pos['position'],
                    'avgCost': round(pos['avgCost'], 2),
                    'marketValue': round(pos['marketValue'], 2)
                })
            return positions_list
        except Exception as e:
            logger.error(f"Error getting positions summary: {str(e)}")
            return []

    def calculate_current_profit(self) -> Dict:
        """Calculate current profit/loss"""
        try:
            current_value = self.get_portfolio_value()
            positions = self.get_positions_summary()
            
            if self.initial_portfolio_value is None:
                self.initial_portfolio_value = current_value
                return {
                    'profit': 0,
                    'profit_percentage': 0,
                    'current_value': round(current_value, 2),
                    'positions': positions,
                    'timestamp': datetime.now().isoformat()
                }
            
            profit = current_value - self.initial_portfolio_value
            profit_percentage = (profit / self.initial_portfolio_value) * 100
            
            return {
                'profit': round(profit, 2),
                'profit_percentage': round(profit_percentage, 2),
                'current_value': round(current_value, 2),
                'positions': positions,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error calculating profit: {str(e)}")
            return None

    def emit_profit_update(self):
        """Emit profit update through WebSocket"""
        current_time = time.time()
        if current_time - self.last_profit_update >= self.profit_update_interval:
            profit_data = self.calculate_current_profit()
            if profit_data:
                self.socketio.emit('profit_update', profit_data)
                self.last_profit_update = current_time

    def check_risk_limits(self, action: str, quantity: float, price: float) -> Tuple[bool, str]:
        """Check if trade complies with risk management rules"""
        # Get current position from actual portfolio
        positions = self.get_positions_summary()
        current_position = next(
            (pos['position'] for pos in positions if pos['symbol'] == self.strategy.symbol), 
            0
        )

        if action == "BUY":
            new_position = current_position + quantity
        else:
            new_position = current_position - quantity
            
        # Check position limits
        if abs(new_position) > self.risk_limits['max_position']:
            return False, "Position limit exceeded"
            
        # Check trade size
        if quantity > self.risk_limits['max_trade_size']:
            return False, "Trade size too large"
            
        # Check max loss
        current_value = self.get_portfolio_value()
        if current_value > 0:
            potential_loss = (current_value - new_position * price) / current_value * 100
            if potential_loss > self.risk_limits['max_loss_percent']:
                return False, "Maximum loss limit exceeded"
                
        return True, "Trade allowed"

    def execute_trade(self, action: str, quantity: float, price: float = None) -> bool:
        """Execute trade with risk management"""
        try:
            quantity = int(round(quantity))
            if quantity == 0:
                logger.warning("Trade rejected: Quantity too small (rounded to 0)")
                return False
            
            is_allowed, message = self.check_risk_limits(action, quantity, price)
            if not is_allowed:
                logger.warning(f"Trade rejected: {message}")
                return False
            
            order_id = self.app.place_order(
                direction=action,
                quantity=quantity,
                contract=self.contract,
                order_type='LMT' if price else 'MKT',
                limit_price=price
            )
            
            logger.info(f"Order placed: {action} {quantity} shares at {'market price' if price is None else price}")
            return bool(order_id)
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return False

    def place_stop_loss_order(self, action: str, quantity: float, stop_price: float) -> int:
        """Place stop loss order"""
        try:
            exit_action = 'SELL' if action == 'BUY' else 'BUY'
            stop_id = self.app.place_order(
                direction=exit_action,
                quantity=quantity,
                contract=self.contract,
                order_type='STP',
                stop_price=stop_price
            )
            logger.info(f"Stop loss placed: {exit_action} {quantity} @ {stop_price}")
            return stop_id
        except Exception as e:
            logger.error(f"Error placing stop loss: {str(e)}")
            return None

    def update_stop_loss_order(self, symbol: str, quantity: float, new_stop_price: float) -> bool:
        """Update existing stop loss order"""
        try:
            if symbol in self.stop_order_ids:
                self.app.modify_order(
                    order_id=self.stop_order_ids[symbol],
                    new_quantity=quantity,
                    new_stop_price=new_stop_price
                )
                self.stop_loss[symbol] = new_stop_price
                logger.info(f"Stop loss updated for {symbol}: {quantity} @ {new_stop_price}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating stop loss: {str(e)}")
            return False

    def should_update_position(self, action: str) -> bool:
        """Check if position should be updated"""
        positions = self.get_positions_summary()
        current_position = next(
            (pos['position'] for pos in positions if pos['symbol'] == self.strategy.symbol), 
            0
        )
        return (current_position > 0 and action == "BUY") or (current_position < 0 and action == "SELL")

    def get_historical_data(self, duration="1 D", interval="1 min", symbol=None):
        """Get historical data with error handling"""
        def validate_dataframe(df, timeframe):
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {timeframe} data: {missing_cols}")

        try:
            if not self.contract and symbol is None:
                raise ValueError("No contract set. Call set_strategy() first.")
            elif not self.contract and symbol is not None:
                self.contract = self.create_contract(symbol)

            if not self.isConnected:
                self._connect()

            intraday_intervals = ['1 secs', '5 secs', '10 secs', '15 secs', '30 secs',
                            '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins',
                            '20 mins', '30 mins', '1 hour', '2 hours', '3 hours', '4 hours']
        
            is_intraday = interval in intraday_intervals
    
            # Get intraday data
            intraday_df = self.app.histData(1, self.contract, duration, interval)
            validate_dataframe(intraday_df, "intraday")
            
            # For intraday intervals, also get daily data
            if is_intraday:
                time.sleep(1)
                daily_df = self.app.histData(2, self.contract, "1 Y", "1 day")
                validate_dataframe(daily_df, "daily")
                return intraday_df, daily_df
            else:
                if 'timestamp' in intraday_df.columns:
                    intraday_df['timestamp'] = pd.to_datetime(intraday_df['timestamp'])
                    intraday_df.set_index('timestamp', inplace=True)
                    intraday_df.sort_index(inplace=True)
                return None, intraday_df
    
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise

    def is_symbol_trading(self, symbol: str) -> bool:
        """Check if a specific symbol is currently being traded"""
        return symbol in self.active_trades and self.active_trades[symbol]['is_trading']

    def start_trading(self, symbol: str, interval: str, strategy_type: str):
       """Initialize and start live trading for a specific symbol"""
       try:
           if not self.isConnected:
               self._connect()
               
           self.interval = interval
           
           if not self.contract and symbol is None:
               raise ValueError("No contract set. Call set_strategy() first.")
           elif not self.contract and symbol is not None:
               self.contract = self.create_contract(symbol)
    
           position_data = {
               'symbol': symbol,
               'interval': interval,
               'entry_time': datetime.now(),
               'status': 'Open', 
               'profit': 0,
               'profitPct': 0,
               'strategy': strategy_type
           }
           self.mongo.insert_position(position_data)
    
           self.historical_data = self.app.request_historical_data(self.contract, self.interval)
           time.sleep(5)
    
           if len(self.historical_data) > 0:
               df = pd.DataFrame(self.historical_data)
               self.strategy.initialize(df)
               
               interval_seconds = self.interval * 60
               self.app.reqRealtimeData(0, self.contract, interval_seconds)
               
               self.active_trades[symbol] = {
                   'is_trading': True,
                   'strategy': self.strategy.__class__.__name__,
                   'start_time': pd.Timestamp.now()
               }
               
               # Start live trading thread
               trading_thread = threading.Thread(
                   target=self.start_live_trading,
                   args=(symbol, interval)
               )
               trading_thread.start()
               
               logger.info(f"Trading initialized successfully for {symbol}")
           else:
               raise ValueError("Failed to get historical data")
               
       except Exception as e:
           logger.error(f"Error initializing trading for {symbol}: {str(e)}")
           self.stop_trading(symbol)
           raise

    def start_live_trading(self, symbol: str, interval: str):
        try:
            while self.is_symbol_trading(symbol):
                try:
                    current_data = self.app.data_queue.get(timeout=1)
                    action, quantity, price = self.strategy.generate_trade_decision(current_data)
                    
                    if action:
                        positions = self.get_positions_summary()
                        position = self.validate_position(positions, symbol)
                        
                        if not position:
                            success = self.execute_trade(action, quantity, price)
                            if success:
                                stop_price = self.strategy.calculate_stop_loss(action, price)
                                stop_id = self.place_stop_loss_order(action, quantity, stop_price)
                                if stop_id:
                                    self.stop_order_ids[symbol] = stop_id
                                    self.stop_loss[symbol] = stop_price
                        elif self.should_update_position(action):
                            self.update_stop_loss(symbol, action, quantity, price)
                    
                    self.emit_profit_update()
                    
                except queue.Empty:
                    self.emit_profit_update()
                    continue
        except Exception as e:
            logger.error(f"Live trading error for {symbol}: {str(e)}")
            self.socketio.emit('trading_error', {'message': str(e), 'symbol': symbol})
            raise
        finally:
            self.stop_trading(symbol)
                    
        except Exception as e:
            logger.error(f"Error in live trading loop for {symbol}: {str(e)}")
            self.socketio.emit('trading_error', {'message': str(e), 'symbol': symbol})
            raise
        finally:
            self.stop_trading(symbol)

    def stop_trading(self, symbol: str):
        """Stop trading for a specific symbol"""
        if symbol in self.active_trades:
            self.active_trades[symbol]['is_trading'] = False
            if symbol in self.stop_order_ids:
                self.app.cancelOrder(self.stop_order_ids[symbol])
                del self.stop_order_ids[symbol]
            
            # Clean up thread if exists
            if symbol in self.trading_threads:
                del self.trading_threads[symbol]
            
            logger.info(f"Trading stopped for {symbol}")

    def stop_all_trading(self):
        """Stop all active trading sessions"""
        for symbol in list(self.active_trades.keys()):
            self.stop_trading(symbol)
        self.app.disconnect_wrapper()
        
    def get_active_trades(self):
        """Get information about all active trading sessions"""
        return {
            symbol: {
                'strategy': info['strategy'],
                'start_time': info['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'running_time': (pd.Timestamp.now() - info['start_time']).total_seconds() / 60
            }
            for symbol, info in self.active_trades.items()
            if info['is_trading']
        }

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

    def run_backtest(self, df: pd.DataFrame, initial_capital: float, position_size: int) -> Dict:
        """Run backtest on signals and track positions"""
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
                
                # Update maximum drawdown
                peak_portfolio = max(peak_portfolio, current_portfolio)
                current_drawdown = peak_portfolio - current_portfolio
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
            trade_record = self.strategy.create_trade_record(
                entry_time=current_trade['entry_time'],
                exit_time=last_idx,
                trade_type=current_trade['type'],
                size=current_trade['size'],
                entry_price=current_trade['entry_price'],
                exit_price=df.loc[last_idx, 'close'],
            )
            trades.append(trade_record)
        
        return {
            'trades': trades, 
            'max_drawdown': max_drawdown, 
            'current_portfolio': current_portfolio, 
            'data': df
        }

    def get_orders(self) -> pd.DataFrame:
        """Get current open orders"""
        try:
            if not self.isConnected:
                self._connect()
            
            orders = self.app.request_open_orders()
            return orders
        except Exception as e:
            logger.error(f"Error fetching open orders: {str(e)}")
            raise