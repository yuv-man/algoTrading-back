import pandas as pd
import numpy as np
import xgboost as xgb
from ib_insync import *
import datetime as dt
import time

class AITradingBot:
    def __init__(self, symbol='AAPL', timeframe='1 min', lookback_period=20):
        """
        Initialize the trading bot with IBKR connection and parameters
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_period = lookback_period
        
        # Connect to Interactive Brokers
        self.ib = IB()
        try:
            self.ib.connect('127.0.0.1', 7497, clientId=1)
            print("Successfully connected to IBKR")
        except Exception as e:
            print(f"Failed to connect to IBKR: {e}")
            
        # Create contract for the specified symbol
        self.contract = Stock(self.symbol, 'SMART', 'USD')
        
        # Initialize XGBoost model
        self.model = None
        
    def prepare_features(self, data):
        """
        Calculate technical indicators as features
        """
        df = data.copy()
        
        # Calculate basic technical indicators
        df['returns'] = df['close'].pct_change()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_upper'] = df['bb_middle'] + 2 * df['close'].rolling(window=20).std()
        df['bb_lower'] = df['bb_middle'] - 2 * df['close'].rolling(window=20).std()
        
        # Create target variable (1 for price increase, 0 for decrease)
        df['target'] = (df['returns'].shift(-1) > 0).astype(int)
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """
        Calculate Relative Strength Index
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def train_model(self, historical_data):
        """
        Train XGBoost model on historical data
        """
        # Prepare features
        df = self.prepare_features(historical_data)
        df = df.dropna()
        
        # Define features for training
        feature_columns = ['returns', 'sma_20', 'rsi', 'macd', 'signal']
        X = df[feature_columns]
        y = df['target']
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1
        )
        self.model.fit(X, y)

    def bars_to_dataframe(self, bars):
        """
        Convert IBKR bar data to pandas DataFrame
        """
        data = {
            'date': [],
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': [],
            'barCount': []
        }
        
        for bar in bars:
            data['date'].append(bar.date)
            data['open'].append(bar.open)
            data['high'].append(bar.high)
            data['low'].append(bar.low)
            data['close'].append(bar.close)
            data['volume'].append(bar.volume)
            data['barCount'].append(bar.barCount)
            
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        return df
        
    def get_historical_data(self):
        """
        Fetch historical data from IBKR
        """
        bars = self.ib.reqHistoricalData(
            self.contract,
            endDateTime='',
            durationStr=f'{self.lookback_period} D',
            barSizeSetting=self.timeframe,
            whatToShow='TRADES',
            useRTH=True
        )
        
        df = self.bars_to_dataframe(bars)
        return df
    
    def get_position_size(self, confidence, account_value):
        """
        Calculate position size based on model confidence and risk management
        """
        max_position = account_value * 0.02  # 2% risk per trade
        return max_position * confidence

    def handle_exit(self, signum, frame):
        """
        Handle graceful shutdown of the bot
        """
        print("\nReceived exit signal. Shutting down gracefully...")
        self.stop()
        
    def stop(self):
        """
        Stop the trading bot and cleanup
        """
        self.is_running = False
        # Close all positions
        self.close_all_positions()
        # Disconnect from IBKR
        if self.ib.isConnected():
            self.ib.disconnect()
        # Save trading results
        self.save_trading_results()
        print("Trading bot stopped successfully")

    def close_all_positions(self):
        """
        Close all open positions
        """
        positions = self.ib.positions()
        for position in positions:
            if position.contract.symbol == self.symbol:
                quantity = position.position
                if quantity != 0:
                    order = MarketOrder('SELL' if quantity > 0 else 'BUY', 
                                      abs(quantity))
                    trade = self.ib.placeOrder(position.contract, order)
                    print(f"Closed position of {quantity} shares")
                    self.ib.sleep(1)  # Give time for order to process

    def record_trade(self, order_type: str, shares: float, price: float, 
                    confidence: float):
        """
        Record trade details for later analysis
        """
        trade_info = {
            'timestamp': dt.datetime.now(),
            'type': order_type,
            'shares': shares,
            'price': price,
            'confidence': confidence,
            'value': shares * price
        }
        self.trades_history.append(trade_info)

    def save_trading_results(self):
        """
        Save trading results to CSV file
        """
        if self.trades_history:
            df = pd.DataFrame(self.trades_history)
            filename = f"trading_results_{self.symbol}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename)
            print(f"Trading results saved to {filename}")

    def analyze_performance(self) -> Dict:
        """
        Analyze trading performance and generate statistics
        """
        if not self.trades_history:
            return {"error": "No trading history available"}

        df = pd.DataFrame(self.trades_history)
        
        # Calculate basic metrics
        total_trades = len(df)
        buy_trades = len(df[df['type'] == 'BUY'])
        sell_trades = len(df[df['type'] == 'SELL'])
        
        # Calculate P&L
        df['pnl'] = 0.0
        for i in range(1, len(df)):
            if df.iloc[i]['type'] == 'SELL' and df.iloc[i-1]['type'] == 'BUY':
                df.iloc[i, df.columns.get_loc('pnl')] = (
                    df.iloc[i]['price'] - df.iloc[i-1]['price']
                ) * df.iloc[i]['shares']

        total_pnl = df['pnl'].sum()
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        
        # Calculate win rate and average returns
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 2%)
        returns = df[df['pnl'] != 0]['pnl'].pct_change()
        sharpe_ratio = (returns.mean() - 0.02/252) / returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        
        return {
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "total_pnl": total_pnl,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "sharpe_ratio": sharpe_ratio
        }

    def plot_performance(self):
        """
        Generate performance visualization plots
        """
        if not self.trades_history:
            print("No trading history available for plotting")
            return

        df = pd.DataFrame(self.trades_history)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot cumulative P&L
        cumulative_pnl = df['pnl'].cumsum()
        ax1.plot(cumulative_pnl.index, cumulative_pnl.values)
        ax1.set_title('Cumulative P&L Over Time')
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('Cumulative P&L ($)')
        
        # Plot trade values with color coding
        colors = ['g' if x > 0 else 'r' for x in df['pnl']]
        ax2.bar(df.index, df['pnl'], color=colors)
        ax2.set_title('Individual Trade P&L')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('P&L ($)')
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f"performance_plot_{self.symbol}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()

    def execute_trade(self, prediction, confidence):
        """
        Execute trade based on model prediction and record it
        """
        account = self.ib.accountSummary()
        account_value = float([x.value for x in account if x.tag == 'NetLiquidation'][0])
        
        position_size = self.get_position_size(confidence, account_value)
        current_position = self.ib.positions()
        
        if prediction == 1 and not current_position:  # Buy signal
            order = MarketOrder('BUY', position_size)
            trade = self.ib.placeOrder(self.contract, order)
            self.ib.sleep(1)  # Wait for fill
            if trade.orderStatus.status == 'Filled':
                self.record_trade('BUY', position_size, trade.orderStatus.avgFillPrice, confidence)
                print(f"Placed BUY order for {position_size} shares at {trade.orderStatus.avgFillPrice}")
            
        elif prediction == 0 and current_position:  # Sell signal
            order = MarketOrder('SELL', position_size)
            trade = self.ib.placeOrder(self.contract, order)
            self.ib.sleep(1)  # Wait for fill
            if trade.orderStatus.status == 'Filled':
                self.record_trade('SELL', position_size, trade.orderStatus.avgFillPrice, confidence)
                print(f"Placed SELL order for {position_size} shares at {trade.orderStatus.avgFillPrice}")

    
    def run(self):
        """
        Main trading loop
        """
        print(f"Starting trading bot for {self.symbol}")
        self.is_running = True
        
        # Get historical data and train model
        historical_data = self.get_historical_data()
        self.train_model(historical_data)
        
        while self.is_running:
            try:
                # Get latest data
                current_data = self.get_historical_data().tail(1)
                features = self.prepare_features(current_data)[['returns', 'sma_20', 'rsi', 'macd', 'signal']]
                
                # Make prediction
                prediction = self.model.predict(features)
                confidence = self.model.predict_proba(features)[0][prediction[0]]
                
                # Execute trade based on prediction
                self.execute_trade(prediction[0], confidence)
                
                # Wait for next timeframe
                time.sleep(60)  # Adjust based on your timeframe
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(10)
                continue

    def backtest(self, start_date: str, end_date: str, initial_capital: float = 100000.0):
        """
        Backtest the trading strategy
        
        Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        initial_capital (float): Initial capital for backtesting
        """
        print(f"Starting backtest from {start_date} to {end_date}")
        
        # Get historical data for backtesting period
        backtest_data = self.get_historical_data_for_backtest(start_date, end_date)
        
        # Prepare features
        df = self.prepare_features(backtest_data)
        
        # Split data for training and testing
        train_data = df[df.index < pd.to_datetime(end_date) - pd.Timedelta(days=30)]
        test_data = df[df.index >= pd.to_datetime(end_date) - pd.Timedelta(days=30)]
        
        # Train model on training data
        X_train = train_data[['returns', 'sma_20', 'rsi', 'macd', 'signal']]
        y_train = train_data['target']
        
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1
        )
        self.model.fit(X_train, y_train)
        
        # Initialize backtest variables
        capital = initial_capital
        position = 0
        trades = []
        daily_returns = []
        
        # Run backtest
        for i in range(len(test_data)):
            current_data = test_data.iloc[i:i+1]
            features = current_data[['returns', 'sma_20', 'rsi', 'macd', 'signal']]
            
            # Get prediction and confidence
            prediction = self.model.predict(features)[0]
            confidence = self.model.predict_proba(features)[0][prediction]
            
            # Current price and next price (for calculating returns)
            current_price = current_data['close'].iloc[0]
            
            # Trading logic
            if prediction == 1 and position == 0:  # Buy signal
                position_size = self.calculate_position_size(capital, confidence)
                shares = position_size / current_price
                position = shares
                trade_cost = shares * current_price
                capital -= trade_cost
                
                trades.append({
                    'date': current_data.index[0],
                    'type': 'BUY',
                    'shares': shares,
                    'price': current_price,
                    'confidence': confidence,
                    'value': trade_cost
                })
                
            elif prediction == 0 and position > 0:  # Sell signal
                trade_value = position * current_price
                capital += trade_value
                
                trades.append({
                    'date': current_data.index[0],
                    'type': 'SELL',
                    'shares': position,
                    'price': current_price,
                    'confidence': confidence,
                    'value': trade_value
                })
                
                position = 0
            
            # Calculate daily return
            portfolio_value = capital + (position * current_price)
            daily_returns.append({
                'date': current_data.index[0],
                'portfolio_value': portfolio_value,
                'return': (portfolio_value - initial_capital) / initial_capital
            })
        
        # Calculate backtest metrics
        backtest_results = self.calculate_backtest_metrics(trades, daily_returns, initial_capital)
        
        # Plot backtest results
        self.plot_backtest_results(trades, daily_returns)
        
        return backtest_results

    def get_historical_data_for_backtest(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical data for backtesting period
        """
        bars = self.ib.reqHistoricalData(
            self.contract,
            end=end_date,
            durationStr='1 Y',
            barSizeSetting=self.timeframe,
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        df = self.bars_to_dataframe(bars)
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        return df

    def calculate_position_size(self, capital: float, confidence: float) -> float:
        """
        Calculate position size based on capital and confidence
        """
        max_position = capital * 0.02  # 2% risk per trade
        return max_position * confidence

    def calculate_backtest_metrics(self, trades: List[Dict], daily_returns: List[Dict], 
                                 initial_capital: float) -> Dict:
        """
        Calculate comprehensive backtest metrics
        """
        trades_df = pd.DataFrame(trades)
        returns_df = pd.DataFrame(daily_returns)
        
        # Calculate basic metrics
        total_trades = len(trades_df)
        buy_trades = len(trades_df[trades_df['type'] == 'BUY'])
        sell_trades = len(trades_df[trades_df['type'] == 'SELL'])
        
        # Calculate P&L for each round trip
        pnl = []
        for i in range(0, len(trades_df) - 1, 2):
            if i + 1 < len(trades_df):
                buy_trade = trades_df.iloc[i]
                sell_trade = trades_df.iloc[i + 1]
                trade_pnl = (sell_trade['price'] - buy_trade['price']) * buy_trade['shares']
                pnl.append(trade_pnl)
        
        # Performance metrics
        total_pnl = sum(pnl)
        winning_trades = len([p for p in pnl if p > 0])
        losing_trades = len([p for p in pnl if p < 0])
        
        # Return metrics
        returns = returns_df['return'].values
        daily_returns_pct = np.diff(returns_df['portfolio_value']) / returns_df['portfolio_value'][:-1]
        
        # Calculate advanced metrics
        sharpe_ratio = np.mean(daily_returns_pct) / np.std(daily_returns_pct) * np.sqrt(252)
        max_drawdown = self.calculate_max_drawdown(returns_df['portfolio_value'])
        
        return {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'total_pnl': total_pnl,
            'final_portfolio_value': returns_df['portfolio_value'].iloc[-1],
            'total_return': (returns_df['portfolio_value'].iloc[-1] - initial_capital) / initial_capital,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'average_win': np.mean([p for p in pnl if p > 0]) if winning_trades > 0 else 0,
            'average_loss': np.mean([p for p in pnl if p < 0]) if losing_trades > 0 else 0,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': abs(sum([p for p in pnl if p > 0])) / abs(sum([p for p in pnl if p < 0])) if losing_trades > 0 else float('inf')
        }

    def calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """
        Calculate maximum drawdown from peak
        """
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown

    def plot_backtest_results(self, trades: List[Dict], daily_returns: List[Dict]):
        """
        Create comprehensive visualization of backtest results
        """
        returns_df = pd.DataFrame(daily_returns)
        trades_df = pd.DataFrame(trades)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot portfolio value over time
        ax1.plot(returns_df['date'], returns_df['portfolio_value'])
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        
        # Plot trades on the portfolio value chart
        for idx, trade in trades_df.iterrows():
            color = 'g' if trade['type'] == 'BUY' else 'r'
            ax1.scatter(trade['date'], trade['price'], color=color, marker='^' if trade['type'] == 'BUY' else 'v')
        
        # Plot daily returns
        ax2.plot(returns_df['date'], returns_df['return'])
        ax2.set_title('Daily Returns')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Return (%)')
        
        # Plot trade P&L
        trade_pnl = []
        for i in range(0, len(trades_df) - 1, 2):
            if i + 1 < len(trades_df):
                buy_trade = trades_df.iloc[i]
                sell_trade = trades_df.iloc[i + 1]
                pnl = (sell_trade['price'] - buy_trade['price']) * buy_trade['shares']
                trade_pnl.append(pnl)
        
        ax3.bar(range(len(trade_pnl)), trade_pnl)
        ax3.set_title('Trade P&L')
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('P&L ($)')
        
        plt.tight_layout()
        plt.savefig(f"backtest_results_{self.symbol}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()



##run bot
if __name__ == "__main__":
    # Initialize and run the trading bot
    bot = AITradingBot(symbol='AAPL', timeframe='15 mins', lookback_period=20)
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        bot.stop()
        
        # Analyze and display results
        results = bot.analyze_performance()
        print("\nTrading Results:")
        for key, value in results.items():
            print(f"{key}: {value}")
        
        # Generate performance plots
        bot.plot_performance()


###run backtest

if __name__ == "__main__":
    # Initialize the trading bot
    bot = AITradingBot(symbol='AAPL', timeframe='1 min', lookback_period=20)
    
    # Run backtest first
    backtest_results = bot.backtest(
        start_date='2023-01-01',
        end_date='2023-12-31',
        initial_capital=100000.0
    )
    
    print("\nBacktest Results:")
    for key, value in backtest_results.items():
        print(f"{key}: {value}")
    
    # If backtest results are satisfactory, run live trading
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        bot.stop()
        
        # Analyze and display results
        results = bot.analyze_performance()
        print("\nLive Trading Results:")
        for key, value in results.items():
            print(f"{key}: {value}")
        
        # Generate performance plots
        bot.plot_performance()

