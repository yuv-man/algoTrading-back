from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import BarData
from typing import Dict, Optional, Type
import pandas as pd
import threading
import time
import queue
from datetime import datetime, timedelta
import pytz
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveTrading(EWrapper, EClient):
    def __init__(self, symbol):
        EWrapper.__init__(self)
        EClient.__init__(self, self)
        self.symbol = symbol
        self.strategy = None
        self.interval = None
        self.next_valid_order_id = None
        self.data_queue = queue.Queue()
        self.historical_data = []
        self.current_position = 0
        self.active_orders = {}
        self.contract = self._create_contract()
        self.connected = False
        self.connection_event = threading.Event()
        self.error_messages = []
        self.is_trading = False
        self.liquidation_complete = False
        self.trading_stopped = threading.Event()
        self.pending_orders_complete = threading.Event()
        self.pending_orders_complete.set()
        self.historical_data_event = threading.Event()
        
    def _create_contract(self) -> Contract:
        """Create an IBKR contract object for the symbol"""
        contract = Contract()
        contract.symbol = self.symbol
        contract.secType = 'STK'
        contract.exchange = 'SMART'
        contract.currency = 'USD'
        return contract

    def connect_client(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1, max_retries: int = 3) -> bool:
        """Connect to IBKR TWS/Gateway with retries"""
        retry_count = 0
        while retry_count < max_retries:
            try:
                if self.connected:
                    return True
                    
                logger.info(f"Attempting to connect (attempt {retry_count + 1}/{max_retries})...")
                self.connection_event.clear()
                
                # Clean up any existing connection
                try:
                    self.disconnect()
                except:
                    pass
                    
                time.sleep(1)  # Wait before connecting
                
                # Connect to IB
                self.connect(host, port, client_id)
                
                # Wait for connection confirmation
                connection_timeout = 15
                if self.connection_event.wait(timeout=connection_timeout):
                    logger.info("Successfully connected to IBKR")
                    time.sleep(1)  # Allow connection to stabilize
                    return True
                else:
                    logger.warning(f"Connection attempt {retry_count + 1} timed out")
                    self.disconnect()
                    
            except Exception as e:
                logger.error(f"Connection attempt {retry_count + 1} failed: {str(e)}")
                
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(2)  # Wait before retry
                
        logger.error("Failed to connect after all retry attempts")
        return False

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str):
        """Handle error messages from IBKR"""
        error_msg = f"Error {errorCode}: {errorString}"
        self.error_messages.append(error_msg)
        logger.error(error_msg)
        
        # Handle specific error codes
        if errorCode in [1100, 1101, 1102]:  # Connection-related errors
            self.connected = False
            self.connection_event.clear()

    def nextValidId(self, orderId: int):
        """Callback when the next valid order ID is received"""
        self.next_valid_order_id = orderId
        self.connected = True
        self.connection_event.set()
        
    def historicalData(self, reqId: int, bar: BarData):
        """Process historical data bars"""
        bar_dict = {
            'date': pd.to_datetime(bar.date),
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }
        self.historical_data.append(bar_dict)
        
    def historicalDataEnd(self, reqId: int, start: str, end: str):
        """Called when all historical data has been received"""
        self.historical_data_event.set()

    def realtimeBar(self, reqId: int, time: int, open_: float, high: float, 
                    low: float, close: float, volume: int, wap: float, 
                    count: int):
        """Process real-time bar updates"""
        bar_dict = {
            'date': pd.to_datetime(time, unit='s'),
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }
        self.data_queue.put(bar_dict)
        
    def _get_historical_data(self, period: str = '1 D', interval: str = '1 min') -> pd.DataFrame:
        """Request and get historical data from IBKR with improved error handling"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if not self.connected:
                    if not self.connect_client():
                        raise ConnectionError("Failed to connect to IBKR")
                
                # Clear previous data and event
                self.historical_data = []
                self.historical_data_event.clear()
                
                # Request historical data
                end_datetime = datetime.now(pytz.UTC).strftime('%Y%m%d %H:%M:%S')
                self.reqHistoricalData(
                    reqId=1,
                    contract=self.contract,
                    endDateTime=end_datetime,
                    durationStr=period,
                    barSizeSetting=interval,
                    whatToShow='TRADES',
                    useRTH=1,
                    formatDate=1,
                    keepUpToDate=False,
                    chartOptions=[]
                )
                
                # Wait for data with timeout
                if self.historical_data_event.wait(timeout=30):
                    if self.historical_data:
                        df = pd.DataFrame(self.historical_data)
                        if not df.empty and self.strategy:
                            self.strategy.price_history = df['close'].tolist()
                        return df
                        
                logger.warning(f"Historical data attempt {retry_count + 1} failed")
                
            except Exception as e:
                logger.error(f"Error fetching historical data (attempt {retry_count + 1}): {str(e)}")
                
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(2)
                
        raise ConnectionError("Failed to fetch historical data after all retry attempts")

    def _place_order(self, action: str, quantity: int, price: float) -> int:
        """Place a new order with stop loss and trailing stop from strategy"""
        if self.next_valid_order_id is None:
            raise ValueError("No valid order ID available")
            
        try:
            # Initial stop loss from strategy
            stop_loss = self.strategy.calculate_stop_loss(action, price)
                
            # Main order
            order = Order()
            order.action = action
            order.totalQuantity = quantity
            order.orderType = 'MKT'
            order.transmit = False
            
            # Stop loss order
            stop_order = Order()
            stop_order.action = "SELL" if action == "BUY" else "BUY"
            stop_order.totalQuantity = quantity
            stop_order.orderType = 'STP'
            stop_order.auxPrice = stop_loss
            stop_order.parentId = self.next_valid_order_id
            stop_order.transmit = True
            
            # Place orders
            self.placeOrder(self.next_valid_order_id, self.contract, order)
            self.placeOrder(self.next_valid_order_id + 1, self.contract, stop_order)
            
            # Track orders
            self.active_orders[self.next_valid_order_id] = {
                'action': action,
                'quantity': quantity,
                'price': price,
                'stop_loss': stop_loss,
                'status': 'PENDING'
            }
            
            order_id = self.next_valid_order_id
            self.next_valid_order_id += 2
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            raise

    def start_trading(self):
        """Start the live trading system with improved error handling"""
        if not self.strategy:
            raise ValueError("Strategy not set")
            
        try:
            # Ensure connection
            if not self.connected:
                if not self.connect_client():
                    raise ConnectionError("Failed to connect to IBKR")
                    
            # Get historical data
            hist_df = self._get_historical_data(interval=self.interval)
            if hist_df.empty:
                raise ValueError("No historical data received")
                
            # Start real-time data stream
            self.reqRealTimeBars(2, self.contract, 5, 'TRADES', False, [])
            
            self.is_trading = True
            self.trading_stopped.clear()
            
            logger.info("Starting trading loop...")
            while self.is_trading and not self.trading_stopped.is_set():
                try:
                    bar_data = self.data_queue.get(timeout=1)
                    
                    if not self.connected:
                        logger.warning("Connection lost, attempting to reconnect...")
                        if not self.connect_client():
                            raise ConnectionError("Failed to reconnect")
                    
                    if self.is_trading:
                        action, quantity, price = self.strategy.generate_trade_decision(bar_data)
                        
                        if action and quantity and price:
                            order_id = self._place_order(action, quantity, price)
                            
                            # Check for trailing stop updates
                            if order_id in self.active_orders:
                                new_stop = self.strategy.calculate_trailing_stop(
                                    self.active_orders[order_id]['action'],
                                    bar_data['close']
                                )
                                if new_stop:
                                    current_stop = self.active_orders[order_id]['stop_loss']
                                    if ((action == "BUY" and new_stop > current_stop) or 
                                        (action == "SELL" and new_stop < current_stop)):
                                        self.update_stop_loss(order_id, new_stop)
                            
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in trading loop: {str(e)}")
                    if not self.connected:
                        continue
                    else:
                        raise
                    
        except Exception as e:
            logger.error(f"Fatal error in trading: {str(e)}")
            self.stop_trading(liquidate=True)
            raise

    def run(self, symbol, interval):
        """Run the trading system in a separate thread with symbol and interval"""
        if not self.strategy:
            raise ValueError("Strategy not set")
            
        self.symbol = symbol
        self.interval = interval
        
        # Start trading in a separate thread
        trading_thread = threading.Thread(target=self.start_trading)
        trading_thread.daemon = True
        trading_thread.start()
        return trading_thread

    def set_strategy(self, strategy_class: Type, strategy_params: dict):
        """Set or update the trading strategy"""
        self.strategy = strategy_class(self.symbol, strategy_params)
        if self.historical_data:
            df = pd.DataFrame(self.historical_data)
            self.strategy.price_history = df['close'].tolist()

# Usage example:
if __name__ == "__main__":
    try:
        # Create trading instance
        trading_system = LiveTrading(symbol="AAPL")

        # Set up strategy
        strategy_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'stoch_k_period': 14,
            'stoch_d_period': 3,
            'atr_period': 60,
            'stoch_oversold': 30,
            'position_size': 100
        }
        
        # Set strategy
        trading_system.set_strategy(MACDStochStrategy, strategy_params)
        
        # Start trading
        trading_thread = trading_system.run("AAPL", "1 min")
        
        print("Trading system running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nInitiating graceful shutdown...")
        trading_system.stop_trading(liquidate=True)
        trading_thread.join(timeout=60)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if 'trading_system' in locals():
            trading_system.stop_trading(liquidate=True)