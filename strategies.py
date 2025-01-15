from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from copy import deepcopy
from baseStrategy import BaseStrategy
import xgboost as xgb
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler


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
    def __init__(self, symbol: str, lookback_period: int = 60):
        super().__init__(symbol, lookback_period)
        # Set up logging
        import logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # Add handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Model parameters
        self.model = None
        self.feature_columns = [
            'returns_15', 'returns_30', 'returns_60',
            'trend_strength', 'atr_ratio', 'volume_trend',
            'rsi', 'rsi_trend', 'macd_hist',
            'momentum', 'trend_quality', 'price_strength'
        ]
        # Adjusted risk parameters for longer intraday holds
        self.stop_loss_pct = 0.005  # 0.5% stop loss
        self.trailing_stop_pct = 0.007  # 0.7% trailing stop
        self.profit_target_pct = 0.012  # 1.2% profit target
        self.max_position_holding_minutes = 240  # Maximum 4 hour hold
        self.position_entry_time = None
        
        # Trade frequency controls
        self.min_trades_spacing_minutes = 60  # Minimum time between trades
        self.last_trade_time = None
        self.min_setup_strength = 0.8  # Higher threshold for trade setup quality
        
    def _create_target(self, data: pd.DataFrame, forward_window: int = 20) -> pd.Series:
        """
        Create target variable for model training
        Returns 1 for profitable trades, 0 for unprofitable
        """
        future_returns = data['close'].shift(-forward_window) / data['close'] - 1
        # Adjusted for intraday - looking for moves >= 0.5%
        return (future_returns > 0.005).astype(int)

    def train_model(self, data: pd.DataFrame) -> None:
        """
        Train XGBoost model on historical data with enhanced features
        """
        try:
            df = self.prepare_features(data)
            
            # Create target variable
            df['target'] = self._create_target(df)
            
            # Remove last forward_window rows as they won't have valid targets
            df = df.iloc[:-20]
            
            # Handle missing values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill()
            
            # Select features and target
            X = df[self.feature_columns]
            y = df['target']
            
            # Check for sufficient data
            if len(X) < 100:  # Minimum required samples
                raise ValueError("Insufficient data for training")
            
            # Handle class imbalance
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y),
                y=y
            )
            weight_dict = dict(zip(np.unique(y), class_weights))
            
            # Initialize and train XGBoost model
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=weight_dict[1]/weight_dict[0],
                tree_method='hist',
                eval_metric=['auc', 'error'],
                random_state=42
            )
            
            # Train without splitting for intraday updates
            self.model.fit(X, y)
            
            # Log feature importance
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            })
            self.logger.info(f"Feature importance:\n{importance.sort_values('importance', ascending=False)}")
            
        except Exception as e:
            self.logger.error(f"Error in train_model: {str(e)}")
            raise

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions and probabilities from the model
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained")
                
            # Prepare features
            df = self.prepare_features(data)
            X = df[self.feature_columns].iloc[-1:]
            
            # Handle missing values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.ffill().bfill()
            
            # Generate predictions and probabilities
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            return predictions, probabilities
            
        except Exception as e:
            self.logger.error(f"Error in predict: {str(e)}")
            raise

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()


    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for swing-style intraday trading"""
        df = data.copy()
        
        # Price action features
        df['returns_15'] = df['close'].pct_change(15)
        df['returns_30'] = df['close'].pct_change(30)
        df['returns_60'] = df['close'].pct_change(60)
        
        # Trend detection (longer periods)
        for period in [20, 50, 100]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Trend strength
        df['trend_strength'] = (
            (df['ema_20'] > df['ema_50']).astype(int) + 
            (df['ema_50'] > df['ema_100']).astype(int)
        )
        
        # Volatility measures
        df['atr'] = self._calculate_atr(df, period=14)
        df['atr_ratio'] = df['atr'] / df['atr'].rolling(100).mean()
        
        # Volume analysis (longer periods)
        df['volume_ma_30'] = df['volume'].rolling(30).mean()
        df['volume_ma_60'] = df['volume'].rolling(60).mean()
        df['volume_trend'] = df['volume_ma_30'] / df['volume_ma_60']
        
        # RSI with standard period
        df['rsi'] = self._calculate_rsi(df['close'], period=14)
        df['rsi_ma'] = df['rsi'].rolling(window=10).mean()
        df['rsi_trend'] = (df['rsi'] > df['rsi_ma']).astype(int)
        
        # MACD (standard settings)
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Momentum and trend quality
        df['momentum'] = df['close'] / df['close'].shift(30) - 1
        df['trend_quality'] = (
            df['close'].rolling(20).std() / 
            df['close'].rolling(20).mean()
        )
        
        # Support/Resistance levels
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['r1'] = 2 * df['pivot'] - df['low']
        df['s1'] = 2 * df['pivot'] - df['high']
        
        # Price position relative to key levels
        df['price_strength'] = (
            (df['close'] > df['pivot']).astype(int) +
            (df['close'] > df['r1']).astype(int) -
            (df['close'] < df['s1']).astype(int)
        )
        
        return df
        
    def _calculate_setup_strength(self, row) -> float:
        """Calculate overall trade setup strength"""
        strength_factors = [
            1 if row['trend_strength'] >= 1 else 0.5,  # Relaxed trend requirement
            1 if 20 < row['rsi'] < 80 else 0.5,  # Wider RSI range
            1 if row['volume_trend'] > 0.8 else 0.6,  # Lower volume requirement
            1 if abs(row['momentum']) > 0.001 else 0.7,  # Lower momentum requirement
            1 if row['macd_hist'] * row['momentum'] > 0 else 0.5  # Alignment check
        ]
        return sum(strength_factors) / len(strength_factors)
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            df = self.prepare_features(data)
            df['setup_strength'] = df.apply(self._calculate_setup_strength, axis=1)
            
            # Initialize prediction columns with NaN
            df['prediction'] = np.nan
            df['confidence'] = np.nan
            df['signal'] = 0
            
            # Minimum bars needed for initial training
            min_training_bars = 100
            
            if len(df) < min_training_bars * 2:  # Need enough data for both training and testing
                self.logger.warning(f"Not enough data for training and testing. Need at least {min_training_bars * 2} bars")
                return df
            
            # Create target variable
            df['target'] = self._create_target(df)
            
            # Initial training window
            training_size = int(len(df) * 0.6)  # Use 60% of data for initial training
            
            # Train on initial window
            train_data = df.iloc[:training_size]
            self.train_model(train_data)
            
            if self.model is not None:
                # Generate predictions for remaining data
                for i in range(training_size, len(df)):
                    # Prepare features for current bar
                    current_features = df[self.feature_columns].iloc[i:i+1]
                    current_features = current_features.replace([np.inf, -np.inf], np.nan)
                    current_features = current_features.ffill().bfill()
                    
                    # Generate prediction for current bar
                    prediction = self.model.predict(current_features)[0]
                    probability = self.model.predict_proba(current_features)[0][1]
                    
                    # Store prediction and confidence
                    df.iloc[i, df.columns.get_loc('prediction')] = prediction
                    df.iloc[i, df.columns.get_loc('confidence')] = probability
                    
                    # Check conditions for signals
                    trend_ok = df['trend_strength'].iloc[i] >= 1
                    volume_ok = df['volume_trend'].iloc[i] > 0.8
                    
                    # Generate signals based on conditions
                    if (prediction == 1 and 
                        probability > 0.55 and 
                        trend_ok and 
                        volume_ok):
                        df.iloc[i, df.columns.get_loc('signal')] = 1
                        self.logger.info(f"Buy signal generated at index {df.index[i]} with confidence: {probability:.2f}")
                        
                    elif (prediction == 0 and 
                          probability > 0.55):
                        df.iloc[i, df.columns.get_loc('signal')] = -1
                        self.logger.info(f"Sell signal generated at index {df.index[i]} with confidence: {probability:.2f}")
                    
                    # Retrain model periodically (every 20 bars)
                    if (i - training_size) % 20 == 0 and i > training_size:
                        # Use expanding window for retraining
                        train_data = df.iloc[:i]
                        self.train_model(train_data)
            
            return df
                
        except Exception as e:
            self.logger.error(f"Error in calculate_signals: {str(e)}")
            raise
        
    def generate_trade_decision(self, current_data: pd.DataFrame) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        try:
            current_time = pd.Timestamp.now()
            current_price = current_data['close'].iloc[-1]
            
            # Check minimum time between trades
            if self.last_trade_time and (current_time - self.last_trade_time).total_seconds() / 60 < self.min_trades_spacing_minutes:
                return None, None, None
            
            # Position management
            if self.position != 0:
                # Check holding time
                if self.position_entry_time:
                    minutes_held = (current_time - self.position_entry_time).total_seconds() / 60
                    if minutes_held >= self.max_position_holding_minutes:
                        return self._exit_position()
                
                # Check profit target
                if self.position > 0 and current_price >= self.entry_price * (1 + self.profit_target_pct):
                    return self._exit_position()
                elif self.position < 0 and current_price <= self.entry_price * (1 - self.profit_target_pct):
                    return self._exit_position()
            
            # Signal processing
            signals = self.calculate_signals(current_data)
            latest_signal = signals['signal'].iloc[-1]
            
            if latest_signal != 0 and signals['setup_strength'].iloc[-1] >= self.min_setup_strength:
                action = "BUY" if latest_signal > 0 else "SELL"
                quantity = self._calculate_position_size(signals['confidence'].iloc[-1])
                self._initialize_stop_loss(current_price, action)
                self.position_entry_time = current_time
                self.last_trade_time = current_time
                return action, quantity, current_price
            
            return None, None, None
            
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
        """Calculate stop loss price for a trade"""
        if action == "BUY":
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)

    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on model confidence"""
        base_size = 100  # Base position size
        return round(base_size * confidence) 
