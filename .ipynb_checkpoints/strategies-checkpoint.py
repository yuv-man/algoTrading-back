from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from copy import deepcopy
from baseStrategy import BaseStrategy
import xgboost as xgb
from datetime import datetime
import logging
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score



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
    def __init__(self, symbol: str, params: dict = None):
        default_params = {
            'stop_loss_pct': 0.01,     # 1% stop loss
            'take_profit_pct': 0.02,    # 2% take profit
            'max_holding_bars': 30,     # Maximum holding period
            'min_bars_between_trades': 20,
            'position_size': 10000
        }
        params = {**default_params, **(params or {})}
        super().__init__(symbol, params)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.model = None
        self.feature_columns = [
            'returns_5', 'returns_10', 'returns_15',
            'volume_ratio', 'volume_ma_ratio',
            'high_low_ratio', 'close_to_high', 'close_to_low',
            'momentum', 'acceleration', 'open', 'close', 'high', 'low'
        ]

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features focused on price action and volume"""
        df = data.copy()
        
        # Returns at different timeframes
        for period in [5, 10, 15]:
            df[f'returns_{period}'] = df['close'].pct_change(period)
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_ma_ratio'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        
        # Price action features
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['close_to_high'] = (df['high'] - df['close']) / (df['high'] - df['low'])
        df['close_to_low'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Momentum and acceleration
        df['momentum'] = df['close'].pct_change(10)
        df['acceleration'] = df['momentum'].diff()
        
        return df

    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create labels based on actual achieved profit/loss"""
        df = data.copy()
        max_forward_return = pd.Series(index=df.index, dtype=float)
        min_forward_return = pd.Series(index=df.index, dtype=float)
        
        # Calculate maximum profit and drawdown within next 30 bars
        for i in range(len(df) - 30):
            future_prices = df['close'].iloc[i+1:i+31]
            max_price = future_prices.max()
            min_price = future_prices.min()
            current_price = df['close'].iloc[i]
            
            max_forward_return.iloc[i] = (max_price - current_price) / current_price
            min_forward_return.iloc[i] = (min_price - current_price) / current_price
        
        # Label as 1 if profit target can be hit before stop loss
        return ((max_forward_return > self.params['take_profit_pct']) & 
                (min_forward_return > -self.params['stop_loss_pct'])).astype(int)

    def train_model(self, data: pd.DataFrame) -> None:
        """Train model with focus on prediction accuracy"""
        try:
            df = self.calculate_features(data)
            df['target'] = self.create_labels(df)
            df = df.dropna()
            
            # Split into training and validation sets chronologically
            train_size = int(len(df) * 0.7)
            X_train = df[self.feature_columns][:train_size]
            y_train = df['target'][:train_size]
            X_val = df[self.feature_columns][train_size:]
            y_val = df['target'][train_size:]
            
            # Handle class imbalance
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            weight_dict = dict(zip(np.unique(y_train), class_weights))
            
            # Initialize base model to ensure we always have one
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                scale_pos_weight=weight_dict[1]/weight_dict[0],
                tree_method='hist',
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=1
            )
            
            # Grid search for best parameters
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.03, 0.1],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'min_child_weight': [1, 3, 5]
            }
            
            best_score = 0
            best_params = None
            
            # Fit base model first
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            base_pred = self.model.predict(X_val)
            best_score = precision_score(y_val, base_pred)
            
            # Try to find better parameters
            for params in self._param_combinations(param_grid):
                try:
                    model = xgb.XGBClassifier(
                        objective='binary:logistic',
                        scale_pos_weight=weight_dict[1]/weight_dict[0],
                        tree_method='hist',
                        **params
                    )
                    
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                    val_pred = model.predict(X_val)
                    score = precision_score(y_val, val_pred)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        self.model = model
                except Exception as e:
                    self.logger.warning(f"Error with params {params}: {str(e)}")
                    continue
            
            self.logger.info(f"Best validation precision: {best_score:.3f}")
            self.logger.info(f"Best parameters: {best_params if best_params else 'base parameters'}")
            
            # Log feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                self.logger.info(f"\nFeature importance:\n{importance}")
            
        except Exception as e:
            self.logger.error(f"Error in train_model: {str(e)}")
            raise

    def _param_combinations(self, param_grid):
        """Generate all combinations of parameters"""
        keys = param_grid.keys()
        values = param_grid.values()
        for instance in itertools.product(*values):
            yield dict(zip(keys, instance))

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on model predictions"""
        try:
            df = self.calculate_features(data)
            df['signal'] = 0
            
            if len(df) < 100:
                return df
            
            # Split into training and testing
            train_size = int(len(df) * 0.7)
            train_data = df.iloc[:train_size]
            test_data = df.iloc[train_size:]
            
            self.train_model(train_data)
            
            if self.model is not None:
                X_test = test_data[self.feature_columns]
                X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Generate predictions
                predictions = self.model.predict(X_test)
                probabilities = self.model.predict_proba(X_test)[:, 1]
                
                # Track position and generate signals
                in_position = False
                entry_price = None
                entry_idx = None
                bars_since_last_trade = self.params['min_bars_between_trades']
                
                for i in range(len(test_data)):
                    curr_idx = test_data.index[i]
                    
                    if in_position:
                        bars_held = i - entry_idx
                        curr_price = test_data['close'].iloc[i]
                        pnl = (curr_price - entry_price) / entry_price
                        
                        # Exit conditions
                        if (pnl <= -self.params['stop_loss_pct'] or 
                            pnl >= self.params['take_profit_pct'] or 
                            bars_held >= self.params['max_holding_bars']):
                            df.loc[curr_idx, 'signal'] = -1
                            in_position = False
                            bars_since_last_trade = 0
                    else:
                        bars_since_last_trade += 1
                        
                        if (bars_since_last_trade >= self.params['min_bars_between_trades'] and 
                            predictions[i] == 1 and 
                            probabilities[i] > 0.6):
                            df.loc[curr_idx, 'signal'] = 1
                            in_position = True
                            entry_price = test_data['close'].iloc[i]
                            entry_idx = i
                            bars_since_last_trade = 0
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in calculate_signals: {str(e)}")
            raise

    def initialize(self, data: pd.DataFrame):
        """Initialize strategy with historical data"""
        df = self.calculate_features(data)
        self.train_model(df)

    def generate_trade_decision(self, current_data: dict) -> Tuple[Optional[str], Optional[int], Optional[float]]:
        """Generate trade decision based on current market data"""
        try:
            df = pd.DataFrame([current_data])
            features = self.calculate_features(df)
            X = features[self.feature_columns].iloc[-1:]
            
            if self.model is None:
                return None, None, None
                
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0][1]
            
            if prediction == 1 and probability > 0.6:
                return "BUY", self.params['position_size'], current_data['close']
            elif prediction == 0 and probability > 0.6:
                return "SELL", self.params['position_size'], current_data['close']
                
            return None, None, None
            
        except Exception as e:
            self.logger.error(f"Error in generate_trade_decision: {str(e)}")
            return None, None, None


