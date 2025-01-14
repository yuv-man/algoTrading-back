from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order_state import OrderState
from ibapi.order import Order
import queue
import time
import logging
from threading import Thread, Lock, Event
from datetime import datetime
import pandas as pd
from typing import Union, List


class IBKRWrapper(EWrapper, EClient):
    def __init__(self):
        EWrapper.__init__(self)
        EClient.__init__(self, self)
        self.client_id=1
        
        # Data storage
        self.data_queue = queue.Queue()
        self.current_data = {}
        self.historical_data = []
        self.historical_data_end = False
        
        # Connection management
        self.is_connected = False
        self.is_market_data_connected = False
        self.is_historical_data_connected = False
        self.connection_lock = Lock()
        self.reconnect_count = 0
        self.max_reconnect_attempts = 5
        self.reconnect_wait_time = 30  # seconds

        # New data storage for positions and orders
        self.positions = {}
        self.open_orders = {}
        self.completed_orders = {}
        self.positions_end = False
        self.orders_end = False
        
        # Position and order events
        self.position_event = Event()
        self.order_event = Event()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
    def error(self, reqId, errorCode, errorString):
        """Handle various IB API error codes"""
        self.logger.error(f"Error {errorCode}: {errorString}")
        
        # Connection-related errors
        if errorCode in [2110]:  # Connectivity between TWS and server is broken
            self.handle_disconnection()
            
        elif errorCode in [2103, 2105, 2157]:  # Market data, HMDS, and Sec-def farm connection errors
            self.handle_market_data_disconnection(errorString)
            
        # Other specific error handling can be added here
        
    def handle_disconnection(self):
        """Handle main connection loss"""
        with self.connection_lock:
            if self.is_connected:
                self.logger.warning("Main connection lost. Attempting to reconnect...")
                self.is_connected = False
                self.start_reconnection()

    def handle_market_data_disconnection(self, error_string):
        """Handle market data farm disconnections"""
        with self.connection_lock:
            if self.is_market_data_connected:
                self.logger.warning(f"Market data connection lost: {error_string}")
                self.is_market_data_connected = False
                self.wait_for_market_data_reconnection()

    def connectionClosed(self):
        """Called when the connection is closed"""
        self.logger.warning("Connection closed by server")
        self.handle_disconnection()

    def start_reconnection(self):
        """Start the reconnection process"""
        def reconnect_process():
            while not self.is_connected and self.reconnect_count < self.max_reconnect_attempts:
                try:
                    self.reconnect_count += 1
                    self.logger.info(f"Reconnection attempt {self.reconnect_count}/{self.max_reconnect_attempts}")
                    
                    # Disconnect if still connected
                    if self.is_connected:
                        self.disconnect()
                    
                    # Wait before reconnecting
                    time.sleep(self.reconnect_wait_time)
                    
                    # Try to reconnect
                    self.connect('127.0.0.1', 7497, self.client_id)
                    
                    # Wait for connection confirmation
                    timeout = time.time() + 30
                    while not self.is_connected and time.time() < timeout:
                        time.sleep(1)
                    
                    if self.is_connected:
                        self.logger.info("Successfully reconnected")
                        self.reconnect_count = 0
                        break
                        
                except Exception as e:
                    self.logger.error(f"Reconnection attempt failed: {str(e)}")
                    
            if not self.is_connected:
                self.logger.error("Failed to reconnect after maximum attempts")
                
        Thread(target=reconnect_process, daemon=True).start()

    def wait_for_market_data_reconnection(self):
        """Wait for market data farms to reconnect"""
        def market_data_reconnect_watch():
            timeout = time.time() + (self.reconnect_wait_time * 2)
            while not self.is_market_data_connected and time.time() < timeout:
                time.sleep(5)
            
            if not self.is_market_data_connected:
                self.logger.error("Market data reconnection timeout")
                
        Thread(target=market_data_reconnect_watch, daemon=True).start()

    def connectAck(self):
        """Called when connection is established"""
        self.is_connected = True
        self.logger.info("Connected to IBKR")

    def marketDataConnected(self):
        """Called when market data connection is established"""
        self.is_market_data_connected = True
        self.logger.info("Market data connection established")

    def historicalDataEnd(self, reqId, start, end):
        """Called when all historical data has been received"""
        self.historical_data_end = True
        self.logger.info(f"Historical data complete - {start} to {end}")

    def safe_request_data(self, request_func, *args, **kwargs):
        """Safely make data requests with connection checking"""
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            try:
                if not self.is_connected:
                    self.logger.warning("Not connected. Waiting for connection...")
                    time.sleep(5)
                    attempt += 1
                    continue
                    
                request_func(*args, **kwargs)
                return True
                
            except Exception as e:
                self.logger.error(f"Error making request: {str(e)}")
                attempt += 1
                time.sleep(2)
                
        return False

    def disconnect_wrapper(self):
        """Safely disconnect from IBKR"""
        self.is_connected = False
        self.is_market_data_connected = False
        self.disconnect()
        self.logger.info("Disconnected from IBKR")

    def nextValidId(self, orderId):
        """Called when the next valid order ID is received"""
        self.next_valid_order_id = orderId
        print(f"Next Valid Order ID: {orderId}")

    def historicalData(self, reqId, bar):
        """Process incoming historical data"""
        data = {
            'Timestamp': bar.date,
            'Open': bar.open,
            'High': bar.high,
            'Low': bar.low,
            'Close': bar.close,
            'Volume': bar.volume,
            'BarCount': bar.barCount
        }
        self.historical_data.append(data)
        
    def realtimeBar(self, reqId, time, open_, high, low, close, volume, wap, count):
        """Process incoming real-time bar data"""
        data = {
            'Timestamp': datetime.fromtimestamp(time),
            'Open': open_,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume,
            'Average': wap,
            'BarCount': count
        }
        self.current_data = data
        self.data_queue.put(data)

    def contractDetails(self, reqId, contractDetails):
        """Process contract details"""
        self.contract_details = contractDetails
        print(f"Contract details received for reqId: {reqId}")

    @staticmethod
    def usTechStk(symbol, sec_type="STK", currency="USD", exchange="ISLAND"):
        """Create a US stock contract"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.currency = currency
        contract.exchange = exchange
        return contract

    def histData(self, req_num, contract, duration, candle_size, end_datetime=''):
        """
        Request historical data with better error handling
        """
        try:
            if not self.is_connected:
                raise ConnectionError("Not connected to IBKR")
    
            # Reset data structures
            self.historical_data = []  
            self.historical_data_end = False
            
            print(f"Requesting historical data for {contract.symbol}")
            
            # Make the request
            self.reqHistoricalData(
                reqId=req_num,
                contract=contract,
                endDateTime=end_datetime,
                durationStr=duration,
                barSizeSetting=candle_size,
                whatToShow='ADJUSTED_LAST',
                useRTH=1,
                formatDate=1,
                keepUpToDate=0,
                chartOptions=[]
            )
            
            # Wait for data with timeout
            timeout = 60  # 60 seconds timeout
            start_time = time.time()
            
            while not self.historical_data_end:
                time.sleep(0.1)
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Historical data request timed out after {timeout} seconds")
                
            if not self.historical_data:
                raise ValueError(f"No historical data received for {contract.symbol}")
                
            # Convert to DataFrame
            df = pd.DataFrame(self.historical_data)
            df.columns = df.columns.str.lower()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error requesting historical data: {str(e)}")
            raise

    def request_historical_data(self, contract, interval):
        """Request historical data from IB"""
        self.historical_data = []
        endDateTime = ''
        duration = '2 D'  # Request 2 days of data
        
        self.reqHistoricalData(
            reqId=1,
            contract=contract,
            endDateTime=endDateTime,
            durationStr=duration,
            barSizeSetting=interval,
            whatToShow='TRADES',
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
        return self.historical_data
        
    def reqRealtimeData(self, req_num, contract, interval):
        """Request real-time data for a contract"""
        self.reqRealTimeBars(
            reqId=req_num,
            contract=contract,
            barSize=interval,
            whatToShow='TRADES',
            useRTH=True,
            realTimeBarsOptions=[]
        )

    def placeOrder(self, contract, order):
        """Place a new order"""
        if self.next_valid_order_id is None:
            raise ValueError("No valid order ID received from IBKR")
            
        order_id = self.next_valid_order_id
        self.next_valid_order_id += 1
        super().placeOrder(order_id, contract, order)
        return order_id

    def place_order(self, direction: str, quantity: float, contract: Contract = None,
                   order_type: str = 'MKT', limit_price: float = None, 
                   stop_price: float = None, tif: str = 'GTC',
                   profit_target: float = None, stop_loss: float = None,
                   transmit: bool = True, oca_group: str = None) -> Union[int, List[int]]:
        """
        Main order placement function that handles all order types
        
        Args:
            direction (str): 'BUY' or 'SELL'
            quantity (float): Number of shares/contracts
            contract (Contract): IB contract object (if None, uses current contract)
            order_type (str): 'MKT', 'LMT', 'STP', 'STP_LMT', 'BRACKET'
            limit_price (float): Limit price for limit orders
            stop_price (float): Stop price for stop orders
            tif (str): Time in force - 'GTC', 'DAY', 'IOC'
            profit_target (float): Optional profit target price for bracket orders
            stop_loss (float): Optional stop loss price for bracket/stop orders
            transmit (bool): Whether to transmit the order immediately
            oca_group (str): OCA group name for related orders
            
        Returns:
            Union[int, List[int]]: Order ID(s) of placed order(s)
        """
        try:
            if self.next_valid_order_id is None:
                raise ValueError("No valid order ID received from IBKR")
                
            contract = contract or self.contract
            if contract is None:
                raise ValueError("No contract specified")

            # Validate inputs
            direction = direction.upper()
            order_type = order_type.upper()
            
            if direction not in ['BUY', 'SELL']:
                raise ValueError("Direction must be 'BUY' or 'SELL'")
            
            if order_type not in ['MKT', 'LMT', 'STP', 'STP_LMT', 'BRACKET']:
                raise ValueError("Invalid order type")

            # Handle bracket orders separately
            if order_type == 'BRACKET':
                return self._place_bracket_order(
                    direction=direction,
                    quantity=quantity,
                    contract=contract,
                    entry_price=limit_price,
                    profit_target=profit_target,
                    stop_loss=stop_loss,
                    tif=tif
                )

            # Create the appropriate order object
            order = self._create_base_order(
                direction=direction,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                tif=tif,
                transmit=transmit,
                oca_group=oca_group
            )

            # Get order ID and increment
            order_id = self.next_valid_order_id
            self.next_valid_order_id += 1

            # Place the order
            self.placeOrder(order_id, contract, order)
            
            self.logger.info(
                f"Order placed - ID: {order_id}, Type: {order_type}, "
                f"Direction: {direction}, Quantity: {quantity}"
            )
            
            return order_id

        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            raise

    def _create_base_order(self, direction: str, quantity: float, 
                          order_type: str = 'MKT', limit_price: float = None,
                          stop_price: float = None, transmit: bool = True, 
                          tif: str = 'GTC', oca_group: str = None) -> Order:
        """Internal helper to create base order object"""
        order = Order()
        order.action = direction
        order.totalQuantity = abs(float(quantity))
        order.orderType = order_type
        order.transmit = transmit
        order.tif = tif
        
        # Add prices based on order type
        if order_type in ['LMT', 'STP_LMT']:
            if limit_price is None or limit_price <= 0:
                raise ValueError(f"Limit price required for {order_type} orders")
            order.lmtPrice = limit_price
            
        if order_type in ['STP', 'STP_LMT']:
            if stop_price is None or stop_price <= 0:
                raise ValueError(f"Stop price required for {order_type} orders")
            order.auxPrice = stop_price
            order.outsideRth = True
            
        if oca_group:
            order.ocaGroup = oca_group
            order.ocaType = 1  # Cancel all other orders on fill
            
        return order

    def _place_bracket_order(self, direction: str, quantity: float, contract: Contract,
                           entry_price: float = None, profit_target: float = None,
                           stop_loss: float = None, tif: str = 'GTC') -> List[int]:
        """Internal helper to place bracket orders"""
        order_ids = []
        exit_direction = 'SELL' if direction == 'BUY' else 'BUY'
        
        # Create entry order
        entry_type = 'LMT' if entry_price is not None else 'MKT'
        entry_order = self._create_base_order(
            direction=direction,
            quantity=quantity,
            order_type=entry_type,
            limit_price=entry_price,
            transmit=False,
            tif=tif
        )
        
        # Get entry order ID
        entry_id = self.next_valid_order_id
        self.next_valid_order_id += 1
        order_ids.append(entry_id)
        
        # Create profit target if specified
        if profit_target is not None:
            profit_order = self._create_base_order(
                direction=exit_direction,
                quantity=quantity,
                order_type='LMT',
                limit_price=profit_target,
                transmit=False,
                tif=tif
            )
            profit_order.parentId = entry_id
            
            profit_id = self.next_valid_order_id
            self.next_valid_order_id += 1
            order_ids.append(profit_id)
            
            # Place profit target order
            self.placeOrder(profit_id, contract, profit_order)
        
        # Create stop loss if specified
        if stop_loss is not None:
            stop_order = self._create_base_order(
                direction=exit_direction,
                quantity=quantity,
                order_type='STP',
                stop_price=stop_loss,
                transmit=True,  # Last order can be transmitted
                tif=tif
            )
            stop_order.parentId = entry_id
            
            stop_id = self.next_valid_order_id
            self.next_valid_order_id += 1
            order_ids.append(stop_id)
            
            # Place stop loss order
            self.placeOrder(stop_id, contract, stop_order)
        
        # Place entry order last
        self.placeOrder(entry_id, contract, entry_order)
        
        self.logger.info(
            f"Bracket order placed - Entry ID: {entry_id}, "
            f"Direction: {direction}, Quantity: {quantity}"
        )
        
        return order_ids

    def modify_order(self, order_id: int, 
                    new_quantity: float = None,
                    new_limit_price: float = None,
                    new_stop_price: float = None) -> bool:
        """
        Modify an existing order
        
        Args:
            order_id (int): ID of the order to modify
            new_quantity (float): New quantity
            new_limit_price (float): New limit price
            new_stop_price (float): New stop price
            
        Returns:
            bool: Success/failure
        """
        try:
            if order_id not in self.open_orders:
                raise ValueError(f"Order ID {order_id} not found in open orders")
                
            existing_order = self.open_orders[order_id]
            
            # Create modification order based on existing order
            modify_order = self._create_base_order(
                direction=existing_order['action'],
                quantity=new_quantity or existing_order['totalQuantity'],
                order_type=existing_order['orderType'],
                limit_price=new_limit_price or existing_order.get('lmtPrice'),
                stop_price=new_stop_price or existing_order.get('auxPrice'),
                tif=existing_order.get('tif', 'GTC')
            )
            
            # Place modification
            self.placeOrder(order_id, self.contract, modify_order)
            
            self.logger.info(f"Order modified - ID: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error modifying order: {str(e)}")
            return False

    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        """Called when a position is received"""
        position_data = {
            'account': account,
            'symbol': contract.symbol,
            'secType': contract.secType,
            'exchange': contract.exchange,
            'currency': contract.currency,
            'position': position,
            'avgCost': avgCost,
            'marketValue': 0  # Will be updated when we receive market data
        }
        
        key = f"{contract.symbol}_{contract.secType}_{contract.exchange}"
        self.positions[key] = position_data
        self.logger.info(f"Position received: {position_data}")

    def positionEnd(self):
        """Called when all positions have been received"""
        self.positions_end = True
        self.position_event.set()
        self.logger.info("Positions download completed")

    # Order-related callbacks
    def openOrder(self, orderId: int, contract: Contract, order: Order, orderState: OrderState):
        """Called when an open order is received"""
        order_data = {
            'orderId': orderId,
            'symbol': contract.symbol,
            'secType': contract.secType,
            'exchange': contract.exchange,
            'action': order.action,
            'orderType': order.orderType,
            'totalQuantity': order.totalQuantity,
            'lmtPrice': order.lmtPrice if hasattr(order, 'lmtPrice') else None,
            'auxPrice': order.auxPrice if hasattr(order, 'auxPrice') else None,
            'status': orderState.status,
            'filled': order.filledQuantity if hasattr(order, 'filledQuantity') else 0,
            'remaining': order.remainingQuantity if hasattr(order, 'remainingQuantity') else order.totalQuantity,
            'avgFillPrice': orderState.avgFillPrice,
            'lastFillPrice': orderState.lastFillPrice,
            'whyHeld': orderState.whyHeld,
            'timestamp': datetime.now()
        }
        
        self.open_orders[orderId] = order_data
        self.logger.info(f"Open order received: {order_data}")

    def openOrderEnd(self):
        """Called when all open orders have been received"""
        self.orders_end = True
        self.order_event.set()
        self.logger.info("Open orders download completed")

    def orderStatus(self, orderId: int, status: str, filled: float, remaining: float,
                   avgFillPrice: float, permId: int, parentId: int, lastFillPrice: float,
                   clientId: int, whyHeld: str, mktCapPrice: float):
        """Called when the status of an order changes"""
        if orderId in self.open_orders:
            self.open_orders[orderId].update({
                'status': status,
                'filled': filled,
                'remaining': remaining,
                'avgFillPrice': avgFillPrice,
                'lastFillPrice': lastFillPrice,
                'whyHeld': whyHeld
            })
            
            # If order is completed, move it to completed_orders
            if status in ['Filled', 'Cancelled', 'Inactive']:
                self.completed_orders[orderId] = self.open_orders.pop(orderId)
                
        self.logger.info(f"Order status update - ID: {orderId}, Status: {status}")

    # Methods to request position and order data
    def request_positions(self, timeout=30):
        """
        Request all current positions
        Returns a DataFrame of positions after receiving all data
        """
        try:
            # Reset position data and event
            self.positions = {}
            self.positions_end = False
            self.position_event.clear()
            
            # Request positions
            self.reqPositions()
            
            # Wait for positions with timeout
            if not self.position_event.wait(timeout):
                raise TimeoutError("Position request timed out")
                
            if not self.positions:
                self.logger.warning("No positions found")
                return pd.DataFrame()
                
            # Convert positions to DataFrame
            df = pd.DataFrame(self.positions.values())
            
            # Add market value (you might want to request market data separately)
            return df
            
        except Exception as e:
            self.logger.error(f"Error requesting positions: {str(e)}")
            raise

    def request_open_orders(self, timeout=30):
        """
        Request all open orders
        Returns a DataFrame of open orders after receiving all data
        """
        try:
            # Reset order data and event
            self.open_orders = {}
            self.orders_end = False
            self.order_event.clear()
            
            # Request open orders
            self.reqOpenOrders()
            
            # Wait for orders with timeout
            if not self.order_event.wait(timeout):
                raise TimeoutError("Open orders request timed out")
                
            if not self.open_orders:
                self.logger.warning("No open orders found")
                return pd.DataFrame()
                
            # Convert orders to DataFrame
            df = pd.DataFrame(self.open_orders.values())
            return df
            
        except Exception as e:
            self.logger.error(f"Error requesting open orders: {str(e)}")
            raise

    def get_completed_orders(self):
        """
        Get all completed orders
        Returns a DataFrame of completed orders
        """
        if not self.completed_orders:
            return pd.DataFrame()
            
        return pd.DataFrame(self.completed_orders.values())

    def get_all_orders(self):
        """
        Get all orders (both open and completed)
        Returns a DataFrame of all orders
        """
        all_orders = {**self.open_orders, **self.completed_orders}
        if not all_orders:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_orders.values())
        return df

    def get_position_value(self, symbol):
        """
        Get the current position and value for a specific symbol
        """
        for key, pos in self.positions.items():
            if pos['symbol'] == symbol:
                return pos
        return None

    def data_to_dataframe(self):
        """Convert historical data to pandas DataFrame"""
        if not self.historical_data:
            return None
            
        df = pd.DataFrame(self.historical_data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        return df
