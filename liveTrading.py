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

class LiveTrading(EWrapper, EClient):
    def __init__(self, symbol: str, strategy_class: Type, strategy_params: dict, interval: str):
        EClient.__init__(self, self)
        self.symbol = symbol
        self.strategy = strategy_class(symbol, strategy_params)
        self.interval = interval
        self.next_valid_order_id = None
        self.data_queue = queue.Queue()
        self.historical_data = []
        self.current_position = 0
        self.active_orders = {}
        self.contract = self._create_contract()
        self.connected = False
        self.error_messages = []
        self.is_trading = False
        self.liquidation_complete = False
        self.trading_stopped = threading.Event()
        self.pending_orders_complete = threading.Event()
        self.pending_orders_complete.set()  # Initially set to True
        
    def _create_contract(self) -> Contract:
        """Create an IBKR contract object for the symbol"""
        contract = Contract()
        contract.symbol = self.symbol
        contract.secType = 'STK'
        contract.exchange = 'SMART'
        contract.currency = 'USD'
        return contract
        
    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str):
        """Handle error messages from IBKR"""
        error_msg = f"Error {errorCode}: {errorString}"
        self.error_messages.append(error_msg)
        print(error_msg)

    def nextValidId(self, orderId: int):
        """Callback when the next valid order ID is received"""
        self.next_valid_order_id = orderId
        self.connected = True
        
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
        
    def _place_order(self, action: str, quantity: int, price: float) -> int:
        """Place a new order with a stop loss"""
        if self.next_valid_order_id is None:
            raise ValueError("No valid order ID available")
            
        # Calculate stop loss (2% below entry for buys, 2% above for sells)
        stop_loss = price * 0.98 if action == "BUY" else price * 1.02
            
        # Main order
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = 'MKT'
        order.transmit = False  # Don't transmit until we attach stop loss
        
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
        
    def orderStatus(self, orderId: int, status: str, filled: float,
                   remaining: float, avgFillPrice: float, permId: int,
                   parentId: int, lastFillPrice: float, clientId: int,
                   whyHeld: str, mktCapPrice: float):
        """Handle order status updates"""
        if orderId in self.active_orders:
            self.active_orders[orderId]['status'] = status
            if status == 'Filled':
                if self.active_orders[orderId]['action'] == 'BUY':
                    self.current_position += filled
                else:
                    self.current_position -= filled
                    
                # Check if this was a liquidation order
                if self.active_orders[orderId].get('liquidation', False) and self.current_position == 0:
                    self.liquidation_complete = True
                    
            # Check if all pending orders are complete
            active_pending = any(order['status'] == 'Pending' for order in self.active_orders.values())
            if not active_pending:
                self.pending_orders_complete.set()
            else:
                self.pending_orders_complete.clear()
                    
    def _get_historical_data(self):
        """Request historical data from IBKR"""
        # Convert interval to IBKR format
        duration_str = '1 D'  # Default to 1 day of historical data
        if self.interval == '1min':
            bar_size = '1 min'
        elif self.interval == '5min':
            bar_size = '5 mins'
        else:
            bar_size = '1 min'  # Default to 1 minute
            
        end_datetime = datetime.now(pytz.UTC).strftime('%Y%m%d %H:%M:%S')
        self.reqHistoricalData(1, self.contract, end_datetime,
                             duration_str, bar_size, 'TRADES',
                             1, 1, False, [])
                             
    def liquidate_positions(self):
        """Liquidate all current positions"""
        if self.current_position == 0:
            self.liquidation_complete = True
            return

        # Determine action based on current position
        action = "SELL" if self.current_position > 0 else "BUY"
        quantity = abs(self.current_position)
        
        # Create market order for liquidation
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = 'MKT'
        order.transmit = True
        
        # Place liquidation order
        if self.next_valid_order_id is not None:
            self.placeOrder(self.next_valid_order_id, self.contract, order)
            self.active_orders[self.next_valid_order_id] = {
                'action': action,
                'quantity': quantity,
                'status': 'PENDING',
                'liquidation': True
            }
            self.next_valid_order_id += 1
            self.pending_orders_complete.clear()
        
    def stop_trading(self, liquidate: bool = True):
        """Stop the trading system and optionally liquidate positions"""
        print("Stopping trading system...")
        self.is_trading = False
        self.trading_stopped.set()
        
        if liquidate:
            print("Liquidating positions...")
            self.liquidate_positions()
            
            # Wait for liquidation to complete (with timeout)
            timeout = 30  # 30 seconds timeout for liquidation
            start_time = time.time()
            while not self.liquidation_complete and time.time() - start_time < timeout:
                time.sleep(0.1)
                
            if not self.liquidation_complete:
                print("Warning: Liquidation did not complete within timeout period")
        
        # Wait for any pending orders to complete
        self.pending_orders_complete.wait(timeout=30)
        
        # Cancel any remaining orders
        for order_id in self.active_orders:
            if self.active_orders[order_id]['status'] == 'PENDING':
                self.cancelOrder(order_id)
        
        # Disconnect from IBKR
        self.disconnect()
        print("Trading system stopped")
            
    def start_trading(self):
        """Start the live trading system"""
        # Connect to IBKR
        self.connect('127.0.0.1', 7497, 0)
        
        # Wait for connection and valid order ID
        timeout = 10
        start_time = time.time()
        while not self.connected and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        if not self.connected:
            raise ConnectionError("Failed to connect to IBKR")
            
        # Get historical data
        self._get_historical_data()
        time.sleep(5)  # Wait for historical data
        
        # Convert historical data to DataFrame
        hist_df = pd.DataFrame(self.historical_data)
        if not hist_df.empty:
            self.strategy.price_history = hist_df['close'].tolist()
        
        # Start real-time data stream
        self.reqRealTimeBars(2, self.contract, 5, 'TRADES', False, [])
        
        self.is_trading = True
        self.trading_stopped.clear()
        
        # Main trading loop
        try:
            while self.is_trading and not self.trading_stopped.is_set():
                try:
                    bar_data = self.data_queue.get(timeout=1)
                    
                    if self.is_trading:  # Check again to prevent new orders during shutdown
                        action, quantity, price = self.strategy.generate_trade_decision(bar_data)
                        
                        if action and quantity and price:
                            self._place_order(action, quantity, price)
                            
                except queue.Empty:
                    continue
                    
        except Exception as e:
            print(f"Error in trading loop: {str(e)}")
            self.stop_trading(liquidate=True)
            
    def run(self):
        """Run the trading system in a separate thread"""
        trading_thread = threading.Thread(target=self.start_trading)
        trading_thread.daemon = True  # Make thread daemon so it stops when main program stops
        trading_thread.start()
        return trading_thread

# Usage example:
if __name__ == "__main__":
    # Example parameters
    symbol = "AAPL"
    strategy_params = {
        'window': 20,
        'num_std': 2,
        'position_size': 100
    }
    interval = "1min"
    
    # Create and start trading system
    trading_system = LiveTrading(
        symbol=symbol,
        strategy_class=BollingerBandsStrategy,
        strategy_params=strategy_params,
        interval=interval
    )
    
    try:
        # Start trading in a separate thread
        trading_thread = trading_system.run()
        
        # Let it run for a while
        print("Trading system running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        # Stop trading and liquidate positions
        print("\nInitiating graceful shutdown...")
        trading_system.stop_trading(liquidate=True)
        
        # Wait for the trading thread to finish
        trading_thread.join(timeout=60)
        
        if trading_thread.is_alive():
            print("Warning: Trading thread did not terminate within timeout")
        else:
            print("Trading system shutdown complete")