from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import queue
import time
import logging
from threading import Thread, Lock
from datetime import datetime
import pandas as pd

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
                    if self.isConnected():
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
        Request historical data
        
        Args:
            req_num (int): Request identifier
            contract (Contract): The contract to request data for
            duration (str): Duration of data (e.g., "1 D", "1 M", "1 Y")
            candle_size (str): Bar size setting (e.g., "1 min", "1 hour")
            end_datetime (str): End date and time for the request (format: "YYYYMMDD HH:mm:ss")
            
        Returns:
            pandas.DataFrame: DataFrame containing the historical data
        """
        # Reset data structures for this request
        self.historical_data = []  
        self.historical_data_end = False
        
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
        
        # Wait for data to arrive
        timeout = 60  # 60 seconds timeout
        start_time = time.time()
        
        while not self.historical_data_end:
            time.sleep(0.1)  # Small delay to prevent CPU spinning
            if time.time() - start_time > timeout:
                print(f"Historical data request timed out after {timeout} seconds")
                break
            
        # Convert to DataFrame if we have data
        if self.historical_data:
            df = pd.DataFrame(self.historical_data)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df.set_index('Timestamp', inplace=True)
            return df
        
        return None

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
            barSizeSetting=self.interval,
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

    
    def stopOrder(direction,quantity,st_price):
        order = Order()
        order.action = direction
        order.orderType = "STP"
        order.totalQuantity = quantity
        order.auxPrice = st_price
        return order

    def marketOrder(direction,quantity):
        order = Order()
        order.action = direction
        order.orderType = "MKT"
        order.totalQuantity = quantity
        return order

    def create_order(self, action, quantity, order_type='MKT', lmt_price=None):
        """
        Create an order object
        
        Args:
            action (str): 'BUY' or 'SELL'
            quantity (float): Number of shares/contracts
            order_type (str): 'MKT' for Market or 'LMT' for Limit
            lmt_price (float): Limit price (required for 'LMT' orders)
        """
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = order_type
        
        if order_type == 'LMT':
            if lmt_price is None:
                raise ValueError("Limit price required for LMT orders")
            order.lmtPrice = lmt_price
            
        return order

    def data_to_dataframe(self):
        """Convert historical data to pandas DataFrame"""
        if not self.historical_data:
            return None
            
        df = pd.DataFrame(self.historical_data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        return df
