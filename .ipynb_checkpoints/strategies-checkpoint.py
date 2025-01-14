from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from copy import deepcopy
from baseStrategy import BaseStrategy
import xgboost as xgb
from datetime import datetime
import logging


class MACDStochStrategy(BaseStrategy):
    def __init__(self, symbol: str, params: dict = None):
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'stoch_k_period': 14,
            'stoch_d_period': 3,
            'atr_period': 60,
            'stoch_oversold': 30,
            'position_size': 10000,
            'lookback': 100
        }
        params = {**default_params, **(params or {})}
        super().__init__(symbol, params)
        
        # Initialize strategy-specific attributes
        self.current_macd = None
        self.current_signal = None
        self.current_stoch = None
        self.current_atr = None
        self.prev_stoch = None
        self.price_history = []
        
    def calculate_macd(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line"""
        exp1 = data['close'].ewm(span=self.params['fast_period'], adjust=False).mean()
        exp2 = data['close'].ewm(span=self.params['slow_period'], adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=self.params['signal_period'], adjust=False).mean()
        return macd, macd_signal

    
    def calculate_stoch(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Stochastic Oscillator"""
        low_min = data['low'].rolling(window=self.params['stoch_k_period']).min()
        high_max = data['high'].rolling(window=self.params['stoch_k_period']).max()
        
        k_line = 100 * ((data['close'] - low_min) / (high_max - low_min))
        d_line = k_line.rolling(window=self.params['stoch_d_period']).mean()
        return d_line
    
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(window=self.params['atr_period']).mean()

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on indicators
        
        Args:
            data (pd.DataFrame): Price data with required columns
            
        Returns:
            pd.DataFrame: DataFrame with added signal column
        """
        df = data.copy()
        
        # Calculate indicators
        df['macd'], df['macd_signal'] = self.calculate_macd(df)
        df['stoch'] = self.calculate_stoch(df)
        df['atr'] = self.calculate_atr(df)
        df.dropna(inplace=True)
        
        # Initialize signal column
        df['signal'] = 0
        
        # Define conditions
        buy_condition = (
            (df['macd'] > df['macd_signal']) &
            (df['stoch'] > self.params['stoch_oversold']) &
            (df['stoch'] > df['stoch'].shift(1))
        )
        
        sell_condition = (
            df['low'] < (df['close'].shift(1) - df['atr'].shift(1))
        )
        
        # Generate signals
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df
    
    def generate_trade_decision(self, current_data: dict) -> Tuple[Optional[str], Optional[int], Optional[float]]:
        """
        Generate trading decision based on current market data
        
        Args:
            current_data (dict): Dictionary containing current market data
            
        Returns:
            Tuple[Optional[str], Optional[int], Optional[float]]: 
                Trading action (BUY/SELL), position size, and price
        """
        # Update indicators with current data
        df = pd.DataFrame([current_data])
        
        # Calculate current indicators
        macd_values = self.calculate_macd(df)
        self.current_macd = pd.concat([self.current_macd, pd.Series(macd_values[0])]).iloc[-self.params['lookback']:]
        self.current_signal = pd.concat([self.current_signal, pd.Series(macd_values[1])]).iloc[-self.params['lookback']:]
        
        stoch_value = self.calculate_stoch(df)
        self.current_stoch = pd.concat([self.current_stoch, pd.Series(stoch_value)]).iloc[-self.params['lookback']:]
        
        atr_value = self.calculate_atr(df)
        self.current_atr = pd.concat([self.current_atr, pd.Series(atr_value)]).iloc[-self.params['lookback']:]
        
        # Get latest values for decision making
        current_macd = self.current_macd.iloc[-1]
        current_macd_signal = self.current_signal.iloc[-1]
        current_stoch = self.current_stoch.iloc[-1]
        prev_stoch = self.current_stoch.iloc[-2] if len(self.current_stoch) > 1 else None
        current_atr = self.current_atr.iloc[-1]
        prev_close = df['close'].iloc[-2]
        
        # Buy Signal
        if (self.position <= 0 and
            current_macd > current_macd_signal and
            current_stoch > self.params['stoch_oversold'] and
            prev_stoch is not None and
            current_stoch > prev_stoch):
            
            return "BUY", self.params['position_size'], current_data['close']
        
        # Sell Signal
        elif (self.position > 0 and
              current_data['low'] < (prev_close - current_atr)):
            
            return "SELL", self.params['position_size'], current_data['close']
        
        return None, None, None

    def initialize(self, data: pd.DataFrame):
        """Initialize strategy with historical data"""
        self.atr = self.calculate_atr(data).iloc[-1]
        
    def calculate_stop_loss(self, action: str, entry_price: float) -> float:
        """Calculate initial stop loss based on ATR"""
        stop_distance = self.atr * self.params['stop_loss_atr_multiple']
        
        if action == "BUY":
            return entry_price - stop_distance
        return entry_price + stop_distance
        
    def calculate_trailing_stop(self, action: str, current_price: float) -> float:
        """Calculate trailing stop loss based on ATR"""
        stop_distance = self.atr * self.params['trailing_stop_atr_multiple']
        
        if action == "BUY":
            return current_price - stop_distance
        return current_price + stop_distance


class RSIStrategy(BaseStrategy):
    def __init__(self, symbol: str, params: dict = None):
        default_params = {
            'rsi_period': 14,
            'overbought': 70,
            'oversold': 30,
            'position_size': 100
        }
        params = {**default_params, **(params or {})}
        super().__init__(symbol, params)
        self.current_rsi = None
        
    def calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI indicator"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.params['rsi_period']).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['RSI'] = self.calculate_rsi(df)
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['RSI'] < self.params['oversold'], 'signal'] = 1  # Buy signal
        df.loc[df['RSI'] > self.params['overbought'], 'signal'] = -1  # Sell signal
        
        return df
    
    def generate_trade_decision(self, current_data: dict) -> Tuple[Optional[str], Optional[int], Optional[float]]:
        # Update price history for RSI calculation
        if not hasattr(self, 'price_history'):
            self.price_history = []
        self.price_history.append(current_data['close'])
        
        # Keep only necessary price history
        if len(self.price_history) > self.params['rsi_period'] + 1:
            self.price_history.pop(0)
        
        # Need enough data to calculate RSI
        if len(self.price_history) <= self.params['rsi_period']:
            return None, None, None
            
        # Calculate current RSI
        price_series = pd.Series(self.price_history)
        self.current_rsi = self.calculate_rsi(pd.DataFrame({'close': price_series})).iloc[-1]
        
        # Generate trading decisions
        if self.current_rsi < self.params['oversold'] and self.position <= 0:
            return "BUY", self.params['position_size'], current_data['close']
        elif self.current_rsi > self.params['overbought'] and self.position > 0:
            return "SELL", self.params['position_size'], current_data['close']
            
        return None, None, None

class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, symbol: str, params: dict = None):
        default_params = {
            'window': 20,
            'num_std': 2,
            'position_size': 100
        }
        params = {**default_params, **(params or {})}
        super().__init__(symbol, params)
        
    def calculate_bollinger_bands(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle_band = data['close'].rolling(window=self.params['window']).mean()
        std = data['close'].rolling(window=self.params['window']).std()
        upper_band = middle_band + (std * self.params['num_std'])
        lower_band = middle_band - (std * self.params['num_std'])
        return upper_band, middle_band, lower_band
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['upper_band'], df['middle_band'], df['lower_band'] = self.calculate_bollinger_bands(df)
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['close'] < df['lower_band'], 'signal'] = 1  # Buy when price below lower band
        df.loc[df['close'] > df['upper_band'], 'signal'] = -1  # Sell when price above upper band
        
        return df
    
    def generate_trade_decision(self, current_data: dict) -> Tuple[Optional[str], Optional[int], Optional[float]]:
        if not hasattr(self, 'price_history'):
            self.price_history = []
        self.price_history.append(current_data['close'])
        
        if len(self.price_history) > self.params['window']:
            self.price_history.pop(0)
            
        if len(self.price_history) < self.params['window']:
            return None, None, None
            
        # Calculate current Bollinger Bands
        df = pd.DataFrame({'close': self.price_history})
        upper, middle, lower = self.calculate_bollinger_bands(df)
        current_price = current_data['close']
        
        if current_price < lower.iloc[-1] and self.position <= 0:
            return "BUY", self.params['position_size'], current_price
        elif current_price > upper.iloc[-1] and self.position > 0:
            return "SELL", self.params['position_size'], current_price
            
        return None, None, None

class MACDStrategy(BaseStrategy):
    def __init__(self, symbol: str, params: dict = None):
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'position_size': 100
        }
        params = {**default_params, **(params or {})}
        super().__init__(symbol, params)
        
    def calculate_macd(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicators"""
        exp1 = data['close'].ewm(span=self.params['fast_period'], adjust=False).mean()
        exp2 = data['close'].ewm(span=self.params['slow_period'], adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.params['signal_period'], adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['macd'], df['signal_macd'], df['histogram'] = self.calculate_macd(df)
        
        # Generate signals based on MACD crossovers
        df['signal'] = 0
        df.loc[df['macd'] > df['signal_macd'], 'signal'] = 1
        df.loc[df['macd'] < df['signal_macd'], 'signal'] = -1
        
        # Generate trading signals on crossovers
        df['signal'] = df['signal'].diff()
        
        return df
    
    def generate_trade_decision(self, current_data: dict) -> Tuple[Optional[str], Optional[int], Optional[float]]:
        if not hasattr(self, 'price_history'):
            self.price_history = []
        self.price_history.append(current_data['close'])
        
        min_period = max(self.params['fast_period'], self.params['slow_period']) + self.params['signal_period']
        if len(self.price_history) > min_period:
            self.price_history.pop(0)
            
        if len(self.price_history) < min_period:
            return None, None, None
            
        # Calculate current MACD
        df = pd.DataFrame({'close': self.price_history})
        macd, signal, _ = self.calculate_macd(df)
        
        # Check for crossovers
        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2] and self.position <= 0:
            return "BUY", self.params['position_size'], current_data['close']
        elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2] and self.position > 0:
            return "SELL", self.params['position_size'], current_data['close']
            
        return None, None, None

class VWAPStrategy(BaseStrategy):
    def __init__(self, symbol: str, params: dict = None):
        default_params = {
            'window': 20,
            'std_dev_multiplier': 2,
            'position_size': 100
        }
        params = {**default_params, **(params or {})}
        super().__init__(symbol, params)
        
    def calculate_vwap(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate VWAP and bands"""
        df = data.copy()
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['price_volume'] = df['typical_price'] * df['volume']
        
        cumulative_pv = df['price_volume'].rolling(window=self.params['window']).sum()
        cumulative_volume = df['volume'].rolling(window=self.params['window']).sum()
        
        vwap = cumulative_pv / cumulative_volume
        
        # Calculate VWAP bands
        std = df['typical_price'].rolling(window=self.params['window']).std()
        upper_band = vwap + (std * self.params['std_dev_multiplier'])
        lower_band = vwap - (std * self.params['std_dev_multiplier'])
        
        return vwap, upper_band, lower_band
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['vwap'], df['upper_band'], df['lower_band'] = self.calculate_vwap(df)
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['close'] < df['lower_band'], 'signal'] = 1  # Buy below lower band
        df.loc[df['close'] > df['upper_band'], 'signal'] = -1  # Sell above upper band
        
        return df
    
    def generate_trade_decision(self, current_data: dict) -> Tuple[Optional[str], Optional[int], Optional[float]]:
        if not hasattr(self, 'price_volume_history'):
            self.price_volume_history = []
            
        typical_price = (current_data['high'] + current_data['low'] + current_data['close']) / 3
        self.price_volume_history.append({
            'typical_price': typical_price,
            'volume': current_data['volume'],
            'high': current_data['high'],
            'low': current_data['low'],
            'close': current_data['close']
        })
        
        if len(self.price_volume_history) > self.params['window']:
            self.price_volume_history.pop(0)
            
        if len(self.price_volume_history) < self.params['window']:
            return None, None, None
            
        # Calculate current VWAP and bands
        df = pd.DataFrame(self.price_volume_history)
        vwap, upper_band, lower_band = self.calculate_vwap(df)
        
        current_price = current_data['close']
        
        if current_price < lower_band.iloc[-1] and self.position <= 0:
            return "BUY", self.params['position_size'], current_price
        elif current_price > upper_band.iloc[-1] and self.position > 0:
            return "SELL", self.params['position_size'], current_price
            
        return None, None, None

# xgboost_strategy.py


class XGBoostStrategy(BaseStrategy):
    def __init__(self, symbol: str, lookback_period: int = 20):
        """
        Initialize XGBoost trading strategy
        
        Args:
            symbol (str): Trading symbol
            lookback_period (int): Number of periods for technical indicators
        """
        self.symbol = symbol
        self.lookback_period = lookback_period
        self.model = None
        self.position = 0
        self.current_price = 0
        self.portfolio_value = 0
        self.logger = logging.getLogger(__name__)
        
        # Stop loss settings
        self.trailing_stop = False
        self.stop_loss_pct = 0.02  # 2% fixed stop loss
        self.trailing_stop_pct = 0.03  # 3% trailing stop
        self.current_stop_loss = None
        self.highest_price_since_entry = None
        self.lowest_price_since_entry = None

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators as features
        
        Args:
            data (pd.DataFrame): Price data with OHLCV columns
        
        Returns:
            pd.DataFrame: DataFrame with technical indicators
        """
        df = data.copy()
        
        # Calculate basic technical indicators
        df['returns'] = df['close'].pct_change()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Additional features from Bollinger Bands
        df['price_above_upper'] = (df['close'] > df['bb_upper']).astype(int)
        df['price_below_lower'] = (df['close'] < df['bb_lower']).astype(int)
        
        # Create target variable (1 for price increase, 0 for decrease)
        df['target'] = (df['returns'].shift(-1) > 0).astype(int)
        
        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def train_model(self, data: pd.DataFrame) -> None:
        """
        Train XGBoost model on historical data
        
        Args:
            data (pd.DataFrame): Processed data with features
        """
        try:
            df = data.dropna()
            
            # Define features for training
            feature_columns = [
                'returns', 'sma_20', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_width', 'bb_position', 'price_above_upper', 'price_below_lower'
            ]
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
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from model predictions
        
        Args:
            data (pd.DataFrame): Current market data
        
        Returns:
            pd.DataFrame: Data with trading signals (-1 sell, 0 hold, 1 buy)
        """
        try:
            df = self.prepare_features(data)
            features = df[[
                'returns', 'sma_20', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_width', 'bb_position', 'price_above_upper', 'price_below_lower'
            ]].dropna()
            
            if len(features) == 0:
                return pd.DataFrame()
                
            predictions = self.model.predict(features)
            probabilities = self.model.predict_proba(features)
            
            # Generate trade signals (1 for buy, -1 for sell, 0 for hold)
            df['confidence'] = [prob[1] for prob in probabilities]
            df['prediction'] = predictions
            
            # Add hold signals based on confidence threshold and current position
            confidence_threshold = 0.6
            df['trade_signal'] = 0  # Default to hold
            
            # Buy signals (high confidence and price near lower band)
            df.loc[(df['prediction'] == 1) & 
                  (df['confidence'] > confidence_threshold), 'trade_signal'] = 1
            
            # Sell signals (low confidence or price near upper band)
            df.loc[(df['prediction'] == 0) & 
                  (df['confidence'] > confidence_threshold), 'trade_signal'] = -1
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating signals: {str(e)}")
            raise

    def generate_trade_decision(self, current_data: pd.DataFrame) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """
        Generate trading decision based on current market data
        
        Returns:
            Tuple[str, float, float]: (action, quantity, price)
        """
        try:
            # Check stop loss first if in position
            current_price = current_data['close'].iloc[-1]
            if self.position != 0 and self.check_stop_loss(current_price):
                return self._exit_position()
            
            # Update trailing stop if needed
            self._update_trailing_stop(current_price)
            
            # Prepare features and get prediction
            df = self.prepare_features(current_data.tail(self.lookback_period + 1))
            features = df[[
                'returns', 'sma_20', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_width', 'bb_position', 'price_above_upper', 'price_below_lower'
            ]].iloc[-1:]
            
            if features.isnull().any().any():
                return None, None, None
                
            prediction = self.model.predict(features)[0]
            confidence = self.model.predict_proba(features)[0][prediction]
            
            self.current_price = current_price
            
            # Define action based on prediction and current position
            action = None
            quantity = None
            
            if prediction == 1 and confidence > 0.6 and self.position <= 0:
                action = "BUY"
                quantity = self._calculate_position_size(confidence)
                self._initialize_stop_loss(current_price, "BUY")
            elif prediction == 0 and confidence > 0.6 and self.position >= 0:
                action = "SELL"
                quantity = self._calculate_position_size(confidence)
                self._initialize_stop_loss(current_price, "SELL")
            
            return action, quantity, current_price
            
        except Exception as e:
            self.logger.error(f"Error generating trade decision: {str(e)}")
            return None, None, None

    def _exit_position(self) -> Tuple[str, float, float]:
        """Exit current position"""
        action = "SELL" if self.position > 0 else "BUY"
        quantity = abs(self.position)
        self._reset_stop_loss()
        return action, quantity, self.current_price

    def _initialize_stop_loss(self, price: float, action: str) -> None:
        """Initialize stop loss for new position"""
        self.current_stop_loss = self.calculate_stop_loss(action, price)
        if action == "BUY":
            self.highest_price_since_entry = price
            self.lowest_price_since_entry = None
        else:
            self.lowest_price_since_entry = price
            self.highest_price_since_entry = None

    def _update_trailing_stop(self, current_price: float) -> None:
        """Update trailing stop loss if enabled"""
        if not self.trailing_stop or self.position == 0:
            return
            
        if self.position > 0:
            if current_price > self.highest_price_since_entry:
                self.highest_price_since_entry = current_price
                self.current_stop_loss = current_price * (1 - self.trailing_stop_pct)
        else:
            if current_price < self.lowest_price_since_entry:
                self.lowest_price_since_entry = current_price
                self.current_stop_loss = current_price * (1 + self.trailing_stop_pct)

    def _reset_stop_loss(self) -> None:
        """Reset stop loss variables"""
        self.current_stop_loss = None
        self.highest_price_since_entry = None
        self.lowest_price_since_entry = None

    def check_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss has been triggered"""
        if self.current_stop_loss is None:
            return False
            
        if self.position > 0:
            return current_price < self.current_stop_loss
        else:
            return current_price > self.current_stop_loss

    def calculate_stop_loss(self, action: str, entry_price: float) -> float:
        """
        Calculate stop loss price for a trade
        """
        if action == "BUY":
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)

    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on model confidence"""
        base_size = 100  # Base position size
        return round(base_size * confidence)

    def create_trade_record(self, entry_time: datetime, exit_time: datetime,
                          trade_type: str, size: float, entry_price: float,
                          exit_price: float) -> Dict:
        """Create a standardized trade record"""
        pnl = (exit_price - entry_price) * size if trade_type == "LONG" else (entry_price - exit_price) * size
        return {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'type': trade_type,
            'size': size,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'return': (pnl / (entry_price * size)) * 100,
            'stop_loss': self.current_stop_loss
        }
