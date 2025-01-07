import threading
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import pandas as pd
from ibapi.contract import Contract
from strategies import *
from trading_engine import TradingEngine
from trendAnalyzer import StockTrendAnalyzer
from strategyValidator import StrategyValidator, StrategyHandler
from optimizer import Optimizer, OptimizeTarget
from inspect import getmembers, isclass
import os
import sys


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingApplication:
    def __init__(self):
        """
        Initialize the trading application
        """
        self.trading_engine = TradingEngine()
        self.symbol = None
        self.sec_type = None
        self.currency = None
        self.exchange = None
        self.interval = None
        self.period = None
        self.start_date = None
        self.end_date = None
        self.strategy_name = None
        self.initial_capital = 100000
        self.current_capital = self.initial_capital
        self.intraday_data = None
        self.daily_data = None
        self.position_size = 100000
        self.max_drawdown = 0
        self.trades = None
        self.saveHistoricalDataToCSV = False
        self.loadHistoricalDataFromCSV = True

    def register_strategy(self, symbol, strategy_type: str, params: Dict[str, Any], sec_type: str = "STK", currency: str = "USD", 
                 exchange: str = "SMART") -> Dict:
        """Register a new trading strategy"""
        self.symbol = symbol
        self.sec_type = sec_type
        self.currency = currency
        self.exchange = exchange
        try:
            if strategy_type == 'MACDStoch':
                strategy = MACDStochStrategy(self.symbol, params)
            elif strategy_type == 'RSI':
                strategy = RSIStrategy(self.symbol, params)
            elif strategy_type == 'BollingerBands':
                strategy = BollingerBandsStrategy(self.symbol, params)
            elif strategy_type == 'MACD':
                strategy = MACDStrategy(self.symbol, params)
            elif strategy_type == 'VWAP':
                strategy = VWAPStrategy(self.symbol, params)
            else:
                return {
                    'status': 'error',
                    'message': f'Unknown strategy type: {strategy_type}'
                }
            
            self.strategy_name = strategy_type
            self.trading_engine.set_strategy(strategy)
            logger.info(f"Strategy registered: {strategy_type} for {self.symbol}")
            
            return {
                'status': 'success',
                'message': 'Strategy registered',
                'details': {
                    'strategy_type': strategy_type,
                    'symbol': self.symbol,
                    'params': params
                }
            }
            
        except Exception as e:
            logger.error(f"Error registering strategy: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_stock_data(self, symbol: str, interval: str = "5 mins", period: Optional[str] = None,start_date: Optional[str] = None, 
                     end_date: Optional[str] = None ) -> Dict:
        
        self.period = self.format_for_reqhistoricaldata(start_time=start_date, end_time=end_date, period=period)
        self.interval = interval

        if period is None and (start_date is None or end_date is None):
            raise ValueError("Must provide either period or both start_date and end_date")
        if period is not None and (start_date is not None or end_date is not None):
            raise ValueError("Cannot provide both period and date range")
            
        try:
            if self.loadHistoricalDataFromCSV:
                csv_path_intraday = 'stock_data_MSFT/MSFT_None_to_20241222 12:53:03_intraday.csv'
                csv_path_daily = 'stock_data_MSFT/MSFT_None_to_20241222 12:53:03_daily.csv'
                self.intraday_data = self.load_historical_data(csv_path=csv_path_intraday)
                self.daily_data = self.load_historical_data(csv_path=csv_path_daily)

                
            else:    
                self.intraday_data, self.daily_data = self.trading_engine.get_historical_data(
                        duration=self.period,
                        interval=self.interval,
                        symbol=symbol
                    )
            
            intra_day_data_with_trends = self.daily_trends_analyze()
            return {'intraday_data': intra_day_data_with_trends, 'daily_data': self.daily_data}         

        except Exception as e:
            logger.error(f"Error during data collecting: {str(e)}")
            return {'status': 'error', 'message': str(e)}


    def run_backtest(self, interval: str = "5 min", period: Optional[str] = None,start_date: Optional[str] = None, 
                     end_date: Optional[str] = None) -> Dict:

        
        """Run strategy backtest"""
        self.period = self.format_for_reqhistoricaldata(start_time=start_date, end_time=end_date, period=period)
        self.interval = interval

        # Validate inputs
        if period is None and (start_date is None or end_date is None):
            raise ValueError("Must provide either period or both start_date and end_date")
        if period is not None and (start_date is not None or end_date is not None):
            raise ValueError("Cannot provide both period and date range")
        try:
            if not self.trading_engine.strategy:
                return {'status': 'error', 'message': 'No strategy registered'}

            if self.loadHistoricalDataFromCSV:
                csv_path_intraday = 'stock_data_MSFT/MSFT_None_to_20241222 12:53:03_intraday.csv'
                csv_path_daily = 'stock_data_MSFT/MSFT_None_to_20241222 12:53:03_daily.csv'
                self.intraday_data = self.load_historical_data(csv_path=csv_path_intraday)
                self.daily_data = self.load_historical_data(csv_path=csv_path_daily)
            # Get historical data using trading engine
            else: 
                self.intraday_data, self.daily_data = self.trading_engine.get_historical_data(
                    duration=self.period,
                    interval=self.interval
                )
            if self.saveHistoricalDataToCSV:
                self.download_stock_data_to_file(data=self.intraday_data, after='intraday')
                self.download_stock_data_to_file(data=self.daily_data, after='daily')
            
            print('historical data successfully downloaded')
            
            # Run backtest
            result = self.trading_engine.backtest(self.intraday_data, initial_capital = self.initial_capital, position_size=self.position_size)
            self.current_capital = result['current_portfolio']
            self.max_drawdown = result['max_drawdown']
            data = result['data']
            self.download_stock_data_to_file(data=data, after='1')
            # Calculate metrics
            metrics = self.calculate_metrics(result['trades'])
            #self.print_results(metrics)

            #when in backend
            buy_and_hold_result = self.calculate_buy_and_hold()

            
            trades_list = []
            for trade in result['trades']:
                serialized_trade = {
                    key: str(value) if isinstance(value, (pd.Timestamp, pd.Timedelta)) 
                    else float(value) if isinstance(value, (np.floating, np.integer))
                    else value
                    for key, value in trade.items()
                }
                trades_list.append(serialized_trade)
            
            return {
                'status': 'success',
                'data': {
                    'trades': trades_list,  # Use the explicitly serialized trades
                    'metrics': metrics,
                    'current_capital': float(self.current_capital),
                    'max_drawdown': float(self.max_drawdown),
                    'start_date': self.start_date,
                    'end_date':self.end_date,
                    'buy_and_hold_profit': round(buy_and_hold_result['total_profit'],2),
                    'buy_and_hold_profit_pct': round(buy_and_hold_result['profit_percentage'],2),
                    'initial_capital': self.initial_capital,
                    'current_capital': round(self.current_capital,2)    
                }}            
            
        except Exception as e:
            logger.error(f"Error during backtest: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def calculate_buy_and_hold(self) -> Dict[str, float]:
        """Calculate buy and hold performance for comparison"""
        df = self.intraday_data
        if df is None or len(df) == 0:
            return {'total_profit': 0, 'profit_percentage': 0}
        
        first_price = df['close'].iloc[0]
        last_price = df['close'].iloc[-1]
        
        total_profit = (last_price - first_price) * (self.initial_capital / first_price)
        profit_percentage = (last_price - first_price) / first_price * 100
        
        return {
            'total_profit': total_profit,
            'profit_percentage': profit_percentage
        }

    def calculate_metrics(self, trades: List[Dict]) -> Dict[str, Dict]:
        """Calculate comprehensive metrics for all trades"""
        # Initialize results structure
        results = {
            'Total': {},
            'Long': {},
            'Short': {}
        }
        
        if not trades:
            for category in results:
                results[category] = self._get_empty_metrics_back()
            return results
        
        trades_df = pd.DataFrame(trades)
        self.trades = trades_df

        categories = {
            'Total': trades_df,
            'Long': trades_df[trades_df['type'] == 'LONG'],
            'Short': trades_df[trades_df['type'] == 'SHORT']
        }
        
        for category_name, category_df in categories.items():
            if len(category_df) == 0:
                results[category_name] = self._get_empty_metrics_back()
                continue
                
            results[category_name] = self._calculate_category_metrics_back(category_df)
        
        return results

    def _get_empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            "Profits": 0,
            "Losses": 0,
            "Net Profit": 0,
            "% Profit": 0,
            "Winning Trades": 0,
            "Max Loss": 0,
            "Number of Trades": 0,
            "Number of Winning Trades": 0,
            "Number of Losing Trades": 0,
            "Number of Even Trades": 0,
            "Number of Trends": 0,
            "Number of Trends Intra Day": 0,
            "Avg Trade": 0,
            "Avg Winning Trade": 0,
            "Avg Losing Trade": 0,
            "Ratio Avg Win/Avg Loss": 0
        }

    def _calculate_category_metrics(self, category_df: pd.DataFrame) -> Dict:
        """Calculate metrics for a specific category of trades"""
        category_df['profit'] = category_df.apply(
            lambda x: (x['exit_price'] - x['entry_price'])/x['entry_price'] * x['size'] if x['type'] == 'LONG'
            else (x['entry_price'] - x['exit_price'])/x['entry_price'] * x['size'],
            axis=1
        )
        
        winning_trades = category_df[category_df['profit'] > 0]
        losing_trades = category_df[category_df['profit'] < 0]
        even_trades = category_df[category_df['profit'] == 0]
        
        category_df['entry_date'] = pd.to_datetime(category_df['entry_time']).dt.date
        category_df['exit_date'] = pd.to_datetime(category_df['exit_time']).dt.date
        
        return {
            "Profits": round(float(winning_trades['profit'].sum()) if len(winning_trades) > 0 else 0,2),
            "Losses": round(float(abs(losing_trades['profit'].sum())) if len(losing_trades) > 0 else 0,2),
            "Net Profit": round(float(category_df['profit'].sum()),2),
            "% Profit": round(float(category_df['profit_pct'].sum()) if len(category_df) > 0 else 0,2),
            "Winning Trades": round(len(winning_trades) / len(category_df) * 100,2),
            "Max Loss": round(float(category_df['profit'].min()),2),
            "Number of Trades": len(category_df),
            "Number of Winning Trades": len(winning_trades),
            "Number of Losing Trades": len(losing_trades),
            "Number of Even Trades": len(even_trades),
            "Number of Trends": len(category_df[category_df['entry_date'] != category_df['exit_date']]),
            "Number of Trends Intra Day": len(category_df[category_df['entry_date'] == category_df['exit_date']]),
            "Avg Trade": round(float(category_df['profit'].mean()),2),
            "Avg Winning Trade": round(float(winning_trades['profit'].mean()) if len(winning_trades) > 0 else 0,2),
            "Avg Losing Trade": round(float(losing_trades['profit'].mean()) if len(losing_trades) > 0 else 0,2),
            "Ratio Avg Win/Avg Loss": round(float(abs(winning_trades['profit'].mean() / losing_trades['profit'].mean()))
            if len(winning_trades) > 0 and len(losing_trades) > 0 else 0 ,2)
        }

    def _get_empty_metrics_back(self) -> Dict:
        """Return empty metrics structure"""
        return {
            "profits": 0,
            "losses": 0,
            "net_profit": 0,
            "profit_pct": 0,
            "winning_rate": 0,
            "max_loss": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "even_trades": 0,
            "num_trends": 0,
            "num_trends_intraday": 0,
            "avg_trade": 0,
            "avg_winning_trade": 0,
            "avg_losing_trade": 0,
            "win_loss_ratio": 0
        }

    def _calculate_category_metrics_back(self, category_df: pd.DataFrame) -> Dict:
        """Calculate metrics for a specific category of trades"""
        category_df['profit'] = category_df.apply(
            lambda x: (x['exit_price'] - x['entry_price'])/x['entry_price'] * x['size'] if x['type'] == 'LONG'
            else (x['entry_price'] - x['exit_price'])/x['entry_price'] * x['size'],
            axis=1
        )
        
        winning_trades = category_df[category_df['profit'] > 0]
        losing_trades = category_df[category_df['profit'] < 0]
        even_trades = category_df[category_df['profit'] == 0]
        
        category_df['entry_date'] = pd.to_datetime(category_df['entry_time']).dt.date
        category_df['exit_date'] = pd.to_datetime(category_df['exit_time']).dt.date
        
        return {
            "profits": round(float(winning_trades['profit'].sum()) if len(winning_trades) > 0 else 0,2),
            "losses": round(float(abs(losing_trades['profit'].sum())) if len(losing_trades) > 0 else 0,2),
            "net_profit": round(float(category_df['profit'].sum()),2),
            "profit_pct": round(float(category_df['profit_pct'].sum()) if len(category_df) > 0 else 0,2),
            "winning_rate": round(len(winning_trades) / len(category_df) * 100,2),
            "max_loss": round(float(category_df['profit'].min()),2),
            "total_trades": len(category_df),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "even_trades": len(even_trades),
            "num_trends": len(category_df[category_df['entry_date'] != category_df['exit_date']]),
            "num_trends_intraday": len(category_df[category_df['entry_date'] == category_df['exit_date']]),
            "avg_trade": round(float(category_df['profit'].mean()),2),
            "avg_winning_trade": round(float(winning_trades['profit'].mean()) if len(winning_trades) > 0 else 0,2),
            "avg_losing_trade": round(float(losing_trades['profit'].mean()) if len(losing_trades) > 0 else 0,2),
            "win_loss_ratio": round(float(abs(winning_trades['profit'].mean() / losing_trades['profit'].mean()))
            if len(winning_trades) > 0 and len(losing_trades) > 0 else 0 ,2)
        }

    def print_results(self, results: Dict[str, Dict[str, Any]]):
        """Print formatted results"""
        buy_and_hold_result = self.calculate_buy_and_hold()
        
        print("\nBACKTEST RESULTS")
        print("=" * 120)
        self._print_header(buy_and_hold_result)
        print("-" * 120)
        self._print_metrics(results)

    def _print_header(self, buy_and_hold_result: Dict[str, float]):
        """Print the header section of results"""
        print(f"Symbol:                  {self.symbol}")
        print(f"Start Date:              {self.start_date}")
        print(f"End Date:                {self.end_date}")
        print(f"Interval:                {self.interval}")
        print(f"Start Date Capital:      {self.initial_capital}")
        print(f"End Date Capital:        {round(self.current_capital,2)}")
        print(f"Strategy Name:           {self.strategy_name}")
        print(f"Buy & Hold Net Profit:   $ {round(buy_and_hold_result['total_profit'],2)}")
        print(f"Buy & Hold Profit:       {round(buy_and_hold_result['profit_percentage'],2)}%")
        print(f"Strategy Profit:         $ {round(self.current_capital-self.initial_capital,2)}")
        print(f"Strategy Yield:          {round((self.current_capital-self.initial_capital)/self.initial_capital*100,2)}%")
        print(f"Max Draw Down:           $ {round(self.max_drawdown,2)}")


    def _print_metrics(self, results: Dict[str, Dict[str, Any]]):
        """Print results in a three-column format with negative numbers in red and brackets,
        adding $ for monetary values and % for percentages"""
        metrics_order = [
            "Profits", "Losses",
            "Net Profit", "% Profit", "Winning Trades", "Max Loss", 
            "Number of Trades", "Number of Winning Trades", "Number of Losing Trades",
            "Number of Even Trades", "Number of Trends", "Number of Trends Intra Day",
            "Avg Trade", "Avg Winning Trade", "Avg Losing Trade", "Ratio Avg Win/Avg Loss"
        ]
        
        # Define which metrics should have which symbols
        monetary_metrics = {
            "Net Profit", "Profits", "Losses", "Max Loss",
            "Avg Trade", "Avg Winning Trade", "Avg Losing Trade"
        }
        percentage_metrics = {"% Profit", "Ratio Avg Win/Avg Loss", "Winning Trades"}
        
        # ANSI escape codes for colors
        RED = '\033[91m'
        RESET = '\033[0m'
        
        def format_value(value, metric, width=25):
            """Format a value with special handling for negative numbers, maintaining alignment"""
            if isinstance(value, (int, float)):
                # Determine symbol
                prefix = "$" if metric in monetary_metrics else ""
                suffix = "%" if metric in percentage_metrics else ""
                
                if value < 0:
                    # Format negative numbers with consistent width
                    num_str = f"{prefix}{(value)}{suffix}"
                    formatted = f"{RED}({num_str}){RESET}"
                    # Pad with spaces to maintain alignment
                    padding = width - len(num_str) - 2  # -2 for brackets
                    return " " * max(0, padding) + formatted
                
                num_str = f"{prefix}{value:}{suffix}"
                padding = width - len(num_str)
                return " " * max(0, padding) + num_str
            return f"{str(value):>{width}}"

        # Print header
        print(f"{'Metric':<25} {'Total':>25} {'Long':>25} {'Short':>25}")
        print("-" * 120)
        
        # Print each metric
        for metric in metrics_order:
            total_val = results['Total'][metric]
            long_val = results['Long'][metric]
            short_val = results['Short'][metric]
            
            # Format each column with proper alignment
            total_str = format_value(total_val, metric)
            long_str = format_value(long_val, metric)
            short_str = format_value(short_val, metric)
            
            print(f"{metric:<25}{total_str}{long_str}{short_str}")

    def start_trading(self, interval) -> Dict:
        """Start live trading"""
        self.interval = interval
        try:
            result = self.trading_engine.start_trading(self.interval)
            logger.info("Live trading started")
            return {
                'status': 'success',
                'message': 'Live trading started',
                'strategy': self.strategy_name
            }
        except Exception as e:
            logger.error(f"Error starting trading: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def stop_trading(self) -> Dict:
        """Stop live trading"""
        try:
            self.trading_engine.stop_live_trading()
            logger.info("Live trading stopped")
            
            metrics = self.calculate_metrics(self.trading_engine.strategy.trades)
            
            return {
                'status': 'success',
                'message': 'Live trading stopped',
                'final_stats': {
                    'metrics': metrics,
                    'total_trades': len(self.trading_engine.strategy.trades),
                    'final_portfolio_value': self.trading_engine.strategy.portfolio_value
                }
            }
        except Exception as e:
            logger.error(f"Error stopping trading: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    
    def parse_date_string(date_str: str) -> datetime:
        """
        Parse date string in format 'YYYY-MM-DD' to datetime
        """
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Invalid date format. Expected 'YYYY-MM-DD', got {date_str}") from e
    
    def convert_times_to_ibkr_period(start_time: datetime, end_time: datetime) -> Tuple[str, str]:
        """
        Convert start and end times to IBKR duration string and bar size
        Returns tuple of (duration_str, bar_size)
        """
        time_diff = end_time - start_time
        total_seconds = time_diff.total_seconds()
        
        if total_seconds <= 86400:  # 1 day
            duration = f"{int(total_seconds)} S"
        elif total_seconds <= 604800:  # 1 week
            duration = f"{int(total_seconds / 86400)} D"
        elif total_seconds <= 2592000:  # 30 days
            duration = f"{int(total_seconds / 604800)} W"
        elif total_seconds <= 31536000:  # 1 year
            duration = f"{int(total_seconds / 2592000)} M"
        else:
            duration = f"{int(total_seconds / 31536000)} Y"
        
        bar_size = get_bar_size(total_seconds)
        return duration
    
    def format_for_reqhistoricaldata(self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        period: Optional[str] = None
    ) -> str:
        """
        Format time parameters for TWS reqHistoricalData
        Accepts either datetime objects or date strings in 'YYYY-MM-DD' format
        Returns tuple of (duration_str, bar_size)
        """
        if period is not None:
            end = datetime.now()
            if period.endswith('D') or period.endswith('d'):
                days = int(period[:-1])
                start = end - timedelta(days=days)
            elif period.endswith('W') or period.endswith('w'):
                weeks = int(period[:-1])
                start = end - timedelta(weeks=weeks)
            elif period.endswith('M') or period.endswith('m'):
                months = int(period[:-1])
                start = end - relativedelta(months=months)
            elif period.endswith('Y') or period.endswith('y'):
                years = int(period[:-1])
                start = end - relativedelta(years=years)
            else:
                raise ValueError(f"Invalid period format: {period}")
            
            self.start_date = start.strftime('%Y%m%d %H:%M:%S')
            self.start_date = None
            self.end_date = end.strftime('%Y%m%d %H:%M:%S') 
            return period
        
        if start_time is None or end_time is None:
            raise ValueError("Must provide either period or both start_time and end_time")
        
        # Convert string dates to datetime if necessary
        if isinstance(start_time, str):
            start_time = parse_date_string(start_time)
        if isinstance(end_time, str):
            end_time = parse_date_string(end_time)
            
        if end_time < start_time:
            raise ValueError("end_time must be after start_time")
            
        return convert_times_to_ibkr_period(start_time, end_time)
        
    def download_stock_data_to_file(self,output_dir=None, data=None, after=None):
        """
        Downloads stock data for a given symbol and date range, then saves it to a file.
        
        Args:
            symbol (str): Stock ticker symbol.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            output_dir (str, optional): Directory to save the file. Defaults to "stock_data_<symbol>".
            
        Returns:
            str: Path of the saved file.
        """
        # Set the default output directory if none is provided
        if output_dir is None:
            output_dir = f"stock_data_{self.symbol}"
        
        # Create the output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Fetch stock data using yfinance
        
        
        if data is None:
            raise ValueError(f"No data found for symbol '{symbol}' in the specified date range.")
        
        # Save the data to a CSV file
        file_path = os.path.join(output_dir, f"{self.symbol}_{self.start_date}_to_{self.end_date}_{after}.csv")
        data.to_csv(file_path)
        print(f"Data saved to {file_path}")
        
        return file_path
        
    def load_historical_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load and format historical data from a CSV file.
        Expected CSV columns: Date, Open, High, Low, Close, Volume
        Returns formatted DataFrame with datetime index
        """
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Check required columns
            required_columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Convert date to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Set date as index
            df.set_index('timestamp', inplace=True)
            
            # Sort by date
            df.sort_index(inplace=True)
            
            # Convert numeric columns to float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows with missing values
            df.dropna(inplace=True)
            # Store the data period
            if len(df) > 0:
                self.start_date = df.index[0].strftime('%Y%m%d %H:%M:%S')
                self.end_date = df.index[-1].strftime('%Y%m%d %H:%M:%S')
            
            logger.info(f"Loaded {len(df)} rows of historical data from {csv_path}")
            logger.info(f"Period: {self.start_date} to {self.end_date}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            raise
            
    def daily_trends_analyze(self): 
        analyzer = StockTrendAnalyzer(self.symbol, self.daily_data)
        self.peaks, self.troughs = analyzer.find_peaks_and_troughs()
        self.uptrends, self.downtrends, self.daily_data = analyzer.identify_trends()
        #plt = analyzer.visualize_trends()
        #plt.show()
        if self.intraday_data is not None:
            intraday_data = analyzer.apply_daily_trends_to_intraday(self.intraday_data)
            return intraday_data
        return None

    
    def get_strategies(self) -> List[Dict[str, Any]]:
        """
        Dynamically extracts metadata for each trading strategy from the strategy classes.
        
        Returns:
            List[Dict[str, Any]]: List of strategy metadata dictionaries
        """
        strategies = []
        
        # Get all classes from the strategies module
        strategy_classes = [
            cls for _, cls in getmembers(sys.modules['strategies'], isclass)
            if issubclass(cls, BaseStrategy) and cls != BaseStrategy
        ]
        
        for strategy_class in strategy_classes:
            # Create temporary instance to access default params
            # Using empty string as symbol since we just need the params
            temp_instance = strategy_class('', None)
            
            # Extract strategy name (remove 'Strategy' suffix if present)
            strategy_name = strategy_class.__name__
            if strategy_name.endswith('Strategy'):
                strategy_name = strategy_name[:-8]
                
            # Get default parameters from the class
            default_params = {}
            if hasattr(temp_instance, 'params'):
                default_params = temp_instance.params
                
            strategies.append({
                'id': strategy_name,
                'type': strategy_name,
                'params': default_params
            })
        
        return strategies
    
    def get_strategy_instance(strategy_name: str, symbol: str, custom_params: Dict = None) -> Any:
        """
        Creates and returns an instance of the specified strategy with optional custom parameters.
        
        Args:
            strategy_name (str): Name of the strategy to instantiate
            symbol (str): Trading symbol for the strategy
            custom_params (Dict, optional): Custom parameters to override defaults
            
        Returns:
            BaseStrategy: Instance of the requested strategy
        
        Raises:
            ValueError: If strategy_name is not recognized
        """
        # Get all strategy classes
        strategy_classes = {
            cls.__name__.replace('Strategy', ''): cls
            for _, cls in getmembers(sys.modules['strategies'], isclass)
            if issubclass(cls, BaseStrategy) and cls != BaseStrategy
        }
        
        if strategy_name not in strategy_classes:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available strategies: {list(strategy_classes.keys())}")
        
        strategy_class = strategy_classes[strategy_name]
        return strategy_class(symbol=symbol, params=custom_params)

    def add_strategy(self, strategy_json) -> Dict[str, Any]:
        handler = StrategyHandler()
        creation_result = handler.create_strategy_from_json(strategy_json)
        
        if creation_result['success']:
            # If creation successful, validate and save the strategy
            validator = StrategyValidator()
            validation_result = validator.validate_and_save_strategy(
                creation_result['strategy_class'],
                creation_result['class_source']
            )
            print(validation_result)
        else:
            print(creation_result)

    def run_backtest_with_data(self, interval: str = "5 min", period: Optional[str] = None,start_date: Optional[str] = None, 
                     end_date: Optional[str] = None) -> Dict:

        
        """Run strategy backtest"""
        self.period = self.format_for_reqhistoricaldata(start_time=start_date, end_time=end_date, period=period)
        self.interval = interval

        # Validate inputs
        if period is None and (start_date is None or end_date is None):
            raise ValueError("Must provide either period or both start_date and end_date")
        if period is not None and (start_date is not None or end_date is not None):
            raise ValueError("Cannot provide both period and date range")
        try:
            if not self.trading_engine.strategy:
                return {'status': 'error', 'message': 'No strategy registered'}

            result = self.trading_engine.backtest(self.intraday_data, initial_capital = self.initial_capital, position_size=self.position_size)
            self.current_capital = result['current_portfolio']
            self.max_drawdown = result['max_drawdown']
            data = result['data']
            #self.download_stock_data_to_file(data=data, after='1')
            # Calculate metrics
            metrics = self.calculate_metrics(result['trades'])
            
            return {
                'status': 'success',
                'data': {
                    'metrics': metrics,
                    'current_capital': float(self.current_capital),
                    'max_drawdown': float(self.max_drawdown),
                    'start_date': self.start_date,
                    'end_date':self.end_date,
                    'initial_capital': self.initial_capital,
                    'current_capital': round(self.current_capital,2)    
                }}            
            
        except Exception as e:
            logger.error(f"Error during backtest: {str(e)}")
            return {'status': 'error', 'message': str(e)}


    def optimize_strategy(self, strategy_type: str, param_ranges: List, initial_guess: Dict, optimize_target: str ,symbol: str, interval: str, period: str, start_date: str, end_date: str) -> Dict:

        optimizer = Optimizer(self)

        try:
            target = self.find_optimize_target(optimize_target)
            if target is None:
                return {'status': 'error', 'message': 'Optimize target is invalid'}
            param_ranges = params = {
                name: tuple(values["range"]) 
                for name, values in param_ranges.items()
                }
    
            result = optimizer.optimize(
                strategy_type=strategy_type,
                param_ranges=param_ranges,
                initial_guess=initial_guess,
                symbol=symbol,
                optimize_target=target,
                interval=interval,
                period=period
                )
    
            return result
                
        except Exception as e:
            logger.error(f"Error during optimizition: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def find_optimize_target(self, value: str) -> OptimizeTarget:
        """
        Find the corresponding OptimizeTarget enum member for a given string.
        """
        try:
            # Access the enum member using its name
            return OptimizeTarget[value]
        except KeyError:
            raise ValueError(f"{value} is not a valid optimization target. Allowed values: {', '.join(OptimizeTarget.__members__.keys())}")
        
            