import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpdates
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class Trend:
    """
    Class to store information about a market trend.
    
    Attributes:
        start_idx (int): Index where the trend pattern begins
        end_idx (int): Index where the trend ends
        trend_type (str): Type of trend ('uptrend' or 'downtrend')
        confirmation_idx (int): Index where the trend was confirmed
    """
    start_idx: int
    end_idx: int
    trend_type: str
    confirmation_idx: int


class StockTrendAnalyzer:
    def __init__(self, symbol, stock_data):
        """Initialize as before"""
        self.symbol = symbol
        self.data = stock_data
        self.peaks = []
        self.troughs = []
        self.uptrends = []
        self.downtrends = []
        self.data['candleType'] = ''
        self.data['reversalType'] = ''
        self.data['candleMove'] = ''

   
    def enforce_alternation(self, peaks, troughs):
        """
        Ensure strict alternation between peaks and troughs.
        Returns cleaned lists where peaks and troughs properly alternate.
        """
        # Combine peaks and troughs with their types and sort by index
        all_points = [(idx, 'peak') for idx in peaks] + [(idx, 'trough') for idx in troughs]
        all_points.sort(key=lambda x: x[0])
       
        # Initialize cleaned lists
        clean_peaks = []
        clean_troughs = []
       
        # Determine if we should start with peak or trough based on first two points
        if len(all_points) < 2:
            return np.array(clean_peaks), np.array(clean_troughs)
       
        # Find the first valid point
        if all_points[0][1] == all_points[1][1]:  # If first two are the same type
            i = 1
            while i < len(all_points) and all_points[i][1] == all_points[0][1]:
                i += 1
            if i < len(all_points):
                # Take the best point among the sequence of same type
                if all_points[0][1] == 'peak':
                    best_idx = max(range(i), key=lambda x: self.data['Close'].iloc[all_points[x][0]])
                else:
                    best_idx = min(range(i), key=lambda x: self.data['Close'].iloc[all_points[x][0]])
                all_points = [all_points[best_idx]] + all_points[i:]
            else:
                return np.array(clean_peaks), np.array(clean_troughs)
       
        # Process remaining points ensuring alternation
        current_type = all_points[0][1]
        last_added = all_points[0][0]
       
        if current_type == 'peak':
            clean_peaks.append(last_added)
        else:
            clean_troughs.append(last_added)
           
        for idx, point_type in all_points[1:]:
            if point_type != current_type:  # Different type than last point
                if point_type == 'peak':
                    clean_peaks.append(idx)
                else:
                    clean_troughs.append(idx)
                current_type = point_type
                last_added = idx
               
        return np.array(clean_peaks), np.array(clean_troughs)
   
    def identify_candle(self, i):
        """
        Identifies the type of candle for the current row based on the previous row.
       
        Args:
        - i: Index of the current row.
        """
        current_idx = self.data.index[i]      
        previous_idx = self.data.index[i - 1]
        current = self.data.loc[current_idx]
        previous = self.data.loc[previous_idx]
   
        def in_middle_third(candle):
            candle_range = candle['high'] - candle['low']
            mid_low = candle['low'] + candle_range / 3
            mid_high = candle['high'] - candle_range / 3
            return mid_low <= candle['open'] <= mid_high and mid_low <= candle['close'] <= mid_high
   
        def in_higher_third(candle):
            candle_range = candle['high'] - candle['low']
            high_threshold = candle['high'] - candle_range / 3
            return candle['open'] >= high_threshold and candle['close'] >= high_threshold
   
        def in_lower_third(candle):
            candle_range = candle['high'] - candle['low']
            low_threshold = candle['low'] + candle_range / 3
            return candle['open'] <= low_threshold and candle['close'] <= low_threshold
       
        # Doji
        if in_middle_third(current):  # Added colon here
            self.data.loc[current_idx, 'reversalType'] = 'Doji'
           
        # Hammer
        if in_higher_third(current):  # Added colon here
            self.data.loc[current_idx, 'reversalType'] = 'Hammer'
           
        # Inverted Hammer
        if in_lower_third(current):  # Added colon here
            self.data.loc[current_idx, 'reversalType'] = 'InverterHammer'
       

       

    def find_peaks_and_troughs(self):
        """
        Identifies peaks and troughs based on candlestick patterns and price movement.
        Strictly enforces alternation - must have a trough between peaks and a peak between troughs.
        """
        self.data = self.data.copy()
        self.data['reversalType'] = None  # Ensure column exists
        self.data['candleMove'] = None  # Track movement direction
    
        self.peaks = []
        self.troughs = []
        need_peak = False   # True if the next valid point must be a peak
        need_trough = False # True if the next valid point must be a trough
    
        # Handle the first candle
        first_idx = self.data.index[0]
        current = self.data.loc[first_idx]
        self.data.loc[first_idx, 'candleMove'] = 'up' if current['open'] < current['close'] else 'down'
    
        # Process remaining candles
        for i in range(1, len(self.data)):
            current_idx = self.data.index[i]
            previous_idx = self.data.index[i - 1]
    
            current = self.data.loc[current_idx]
            previous = self.data.loc[previous_idx]
    
            # Determine basic move direction
            if current['high'] > previous['high'] and current['low'] > previous['low']:
                self.data.loc[current_idx, 'candleMove'] = 'up'
            elif current['high'] < previous['high'] and current['low'] < previous['low']:
                self.data.loc[current_idx, 'candleMove'] = 'down'
    
            # Check for inside bar - mark and continue
            if current['high'] <= previous['high'] and current['low'] >= previous['low']:
                self.data.loc[current_idx, 'reversalType'] = 'insidebar'
                continue
    
            # Check for outside bar
            is_outside_bar = current['high'] > previous['high'] and current['low'] < previous['low']
            if is_outside_bar:
                if current['open'] > current['close']:
                    self.data.loc[current_idx, 'reversalType'] = 'RedOKR'
                else:
                    self.data.loc[current_idx, 'reversalType'] = 'GreenOKR'
    
            if i > 1:  # Need at least 3 bars for peak/trough detection
                prev_prev_idx = self.data.index[i - 2]
    
                # Skip if the previous bar was an inside bar
                if self.data.loc[previous_idx, 'reversalType'] == 'insidebar':
                    continue
    
                # Define peak and trough conditions
                peak_condition = (self.data.loc[previous_idx, 'candleMove'] == 'down' or
                                  self.data.loc[previous_idx, 'reversalType'] in ['Doji', 'InverterHammer', 'RedKR'])
                trough_condition = (self.data.loc[previous_idx, 'candleMove'] == 'up' or
                                    self.data.loc[previous_idx, 'reversalType'] in ['Doji', 'Hammer', 'GreenKR'])
    
                # Handle peaks
                if (peak_condition or (is_outside_bar and current['open'] > current['close'])):
                    if len(self.peaks) == 0 or need_peak:  # Can only add a peak if we need one
                        index_to_add = i - 2 if peak_condition else i
    
                        if not trough_condition:  # Ensure it's not also a trough
                            # Check for higher bars between last trough and this peak
                            if self.troughs:
                                last_trough_idx = self.troughs[-1]
                                highest_idx = max(range(last_trough_idx, index_to_add + 1),
                                                  key=lambda x: self.data.loc[self.data.index[x], 'high'])
                                index_to_add = highest_idx
    
                            self.peaks.append(index_to_add)
                            need_peak = False
                            need_trough = True
    
                # Handle troughs
                if (trough_condition or (is_outside_bar and current['open'] < current['close'])):
                    if len(self.troughs) == 0 or need_trough:  # Can only add a trough if we need one
                        index_to_add = i - 2 if trough_condition else i
    
                        if not peak_condition:  # Ensure it's not also a peak
                            # Check for lower bars between last peak and this trough
                            if self.peaks:
                                last_peak_idx = self.peaks[-1]
                                lowest_idx = min(range(last_peak_idx, index_to_add + 1),
                                                 key=lambda x: self.data.loc[self.data.index[x], 'low'])
                                index_to_add = lowest_idx
    
                            self.troughs.append(index_to_add)
                            need_trough = False
                            need_peak = True
    
        return self.peaks, self.troughs

   
    def identify_trends(self):
        """
        Identify market trends based on peaks and troughs pattern.
        Trends can continue with multiple peaks and troughs as long as termination rules aren't met.
        Termination rules are always checked against the most recent trough/peak.
        
        Returns:
            Tuple[List[Trend], List[Trend], pd.DataFrame]: Lists of uptrends, downtrends, and updated data
        """
        self.uptrends = []
        self.downtrends = []
        
        # Initialize new columns
        self.data['trend'] = None
        self.data['pivot_point'] = None
        
        current_trend = None
        trend_start_idx = None
        last_peak_idx = None
        last_trough_idx = None
        
        # Mark all peaks and troughs in pivot_point column
        for peak in self.peaks:
            self.data.loc[self.data.index[peak], 'pivot_point'] = 'peak'
        for trough in self.troughs:
            self.data.loc[self.data.index[trough], 'pivot_point'] = 'trough'
        
        for i in range(3, len(self.data)):
            # Look for potential uptrend pattern if not in downtrend
            if current_trend != 'downtrend':
                if current_trend != 'uptrend':
                    # Looking to start a new uptrend
                    recent_troughs = [t for t in self.troughs if t < i][-2:] if len([t for t in self.troughs if t < i]) >= 2 else []
                    recent_peaks = [p for p in self.peaks if p < i][-1:] if len([p for p in self.peaks if p < i]) >= 1 else []
                    
                    if len(recent_troughs) == 2 and len(recent_peaks) == 1:
                        first_trough, second_trough = recent_troughs
                        peak = recent_peaks[0]
                        
                        # Check pattern validity and no overlap
                        if (first_trough < peak < second_trough and 
                            self.data['low'].iloc[second_trough] > self.data['low'].iloc[first_trough] and 
                            self.data['close'].iloc[i] > self.data['high'].iloc[peak] and
                            self.data['open'].iloc[i] > self.data['open'].iloc[peak] and  # Higher bar check
                            not self._is_overlapping(first_trough, i)):
                            
                            trend = Trend(first_trough, None, 'uptrend', i)
                            self.uptrends.append(trend)
                            # Initialize trend marking up to current index
                            self.data.loc[self.data.index[first_trough:i+1], 'trend'] = 'uptrend'
                            current_trend = 'uptrend'
                            trend_start_idx = first_trough
                            last_trough_idx = second_trough
                else:
                    # Already in uptrend, check for new higher troughs
                    recent_troughs = [t for t in self.troughs if t > last_trough_idx and t < i]
                    if recent_troughs:
                        newest_trough = recent_troughs[-1]
                        if self.data['low'].iloc[newest_trough] > self.data['low'].iloc[last_trough_idx]:
                            # Update to the new higher trough
                            last_trough_idx = newest_trough
                    # Continue marking the trend
                    if current_trend == 'uptrend':
                        self.data.loc[self.data.index[i], 'trend'] = 'uptrend'
                            
            # Look for potential downtrend pattern if not in uptrend
            if current_trend != 'uptrend':
                if current_trend != 'downtrend':
                    # Looking to start a new downtrend
                    recent_peaks = [p for p in self.peaks if p < i][-2:] if len([p for p in self.peaks if p < i]) >= 2 else []
                    recent_troughs = [t for t in self.troughs if t < i][-1:] if len([t for t in self.troughs if t < i]) >= 1 else []
                    
                    if len(recent_peaks) == 2 and len(recent_troughs) == 1:
                        first_peak, second_peak = recent_peaks
                        trough = recent_troughs[0]
                        
                        # Check pattern validity and no overlap
                        if (first_peak < trough < second_peak and 
                            self.data['high'].iloc[second_peak] < self.data['high'].iloc[first_peak] and 
                            self.data['close'].iloc[i] < self.data['low'].iloc[trough] and
                            self.data['open'].iloc[i] < self.data['open'].iloc[trough] and  # Lower bar check
                            not self._is_overlapping(first_peak, i)):
                            
                            trend = Trend(first_peak, None, 'downtrend', i)
                            self.downtrends.append(trend)
                            # Mark the trend in data from start to end (will be updated when trend ends)
                            self.data.loc[self.data.index[first_peak:trend.end_idx+1 if trend.end_idx is not None else i], 'trend'] = 'downtrend'
                            current_trend = 'downtrend'
                            trend_start_idx = first_peak
                            last_peak_idx = second_peak
                else:
                    # Already in downtrend, check for new lower peaks
                    recent_peaks = [p for p in self.peaks if p > last_peak_idx and p < i]
                    if recent_peaks:
                        newest_peak = recent_peaks[-1]
                        if self.data['high'].iloc[newest_peak] < self.data['high'].iloc[last_peak_idx]:
                            # Update to the new lower peak
                            last_peak_idx = newest_peak
            
            # Check for trend termination
            if current_trend == 'uptrend' and last_trough_idx is not None:
                # Check for lower trough formation
                recent_troughs = [t for t in self.troughs if t > last_trough_idx]
                if recent_troughs and self.data['low'].iloc[recent_troughs[0]] < self.data['low'].iloc[last_trough_idx]:
                    # Update trend end to previous index (before confirmation)
                    self.uptrends[-1].end_idx = i - 1
                    # Clear trend marking after end_idx
                    self.data.loc[self.data.index[i:], 'trend'] = None
                    current_trend = None
                    last_trough_idx = None
                # Check if any price component breaks the last trough's low
                elif (self.data['close'].iloc[i] < self.data['low'].iloc[last_trough_idx] or
                      self.data['low'].iloc[i] < self.data['low'].iloc[last_trough_idx] or
                      self.data['open'].iloc[i] < self.data['low'].iloc[last_trough_idx]):
                    # Update trend end to previous index (before confirmation)
                    self.uptrends[-1].end_idx = i - 1
                    # Clear trend marking at confirmation index
                    self.data.loc[self.data.index[i], 'trend'] = None
                    current_trend = None
                    last_trough_idx = None
                    
            elif current_trend == 'downtrend' and last_peak_idx is not None:
                # Check for higher peak formation
                recent_peaks = [p for p in self.peaks if p > last_peak_idx]
                if recent_peaks and self.data['high'].iloc[recent_peaks[0]] > self.data['high'].iloc[last_peak_idx]:
                    # Update trend end to previous index (before confirmation)
                    self.downtrends[-1].end_idx = i - 1
                    # Update trend marking to match end_idx
                    self.data.loc[self.data.index[trend_start_idx:i-1+1], 'trend'] = 'downtrend'
                    self.data.loc[self.data.index[i:], 'trend'] = None
                    current_trend = None
                    last_peak_idx = None
                # Check if any price component breaks the last peak's high
                elif (self.data['close'].iloc[i] > self.data['high'].iloc[last_peak_idx] or
                      self.data['high'].iloc[i] > self.data['high'].iloc[last_peak_idx] or
                      self.data['open'].iloc[i] > self.data['high'].iloc[last_peak_idx]):
                    # Update trend end to previous index (before confirmation)
                    self.downtrends[-1].end_idx = i - 1
                    # Clear trend marking at confirmation index
                    self.data.loc[self.data.index[i], 'trend'] = None
                    current_trend = None
                    last_peak_idx = None
                    
        return self.uptrends, self.downtrends, self.data

    def apply_daily_trends_to_intraday(self, intraday_data):
        """
        Apply daily trends to intraday data and identify the last relevant peak/trough.
        """
        # Make a copy to avoid modifying original data
        intraday = intraday_data.copy()
        
        # Add new columns properly
        intraday['daily_trend'] = None
        intraday['last_daily_peak'] = np.nan
        intraday['last_daily_trough'] = np.nan
        
        # Process each intraday bar
        for i, (idx, row) in enumerate(intraday.iterrows()):
            current_ts = idx.timestamp()
            
            # Find active trends for this date
            active_uptrend = None
            active_downtrend = None
            
            # Check uptrends
            for trend in self.uptrends:
                start_ts = self.data.index.to_numpy()[trend.start_idx].astype(datetime)
                end_ts = self.data.index.to_numpy()[trend.end_idx].astype(datetime)
                print(start_ts, current_ts, end_ts)  # Debugging print
                if start_ts <= current_ts <= end_ts:  # Fix applied here
                    active_uptrend = trend
                    break
                    
            # Check downtrends
            for trend in self.downtrends:
                start_ts = self.data.index.to_numpy()[trend.start_idx].astype(datetime)
                end_ts = self.data.index.to_numpy()[trend.end_idx].astype(datetime)
                if start_ts <= current_ts <= end_ts:  # Fix applied here
                    active_downtrend = trend
                    break
             # This should now be reached
    
            # Set trend based on which is active (priority to most recent)
            if active_uptrend and active_downtrend:
                trend_value = 'uptrend' if active_uptrend.end_idx > active_downtrend.end_idx else 'downtrend'
            elif active_uptrend:
                trend_value = 'uptrend'
            elif active_downtrend:
                trend_value = 'downtrend'
            else:
                trend_value = None
                
            intraday.at[idx, 'daily_trend'] = trend_value
                
            # Find last peak and trough before current time
            daily_idx = self.data.index.searchsorted(idx)
            
            # Get peaks and troughs before current time
            prior_peaks = [p for p in self.peaks if p < daily_idx]
            prior_troughs = [t for t in self.troughs if t < daily_idx]
            
            # Set last peak and trough if available
            if prior_peaks:
                last_peak_idx = prior_peaks[-1]
                intraday.at[idx, 'last_daily_peak'] = self.data['high'].iloc[last_peak_idx]
                
            if prior_troughs:
                last_trough_idx = prior_troughs[-1]
                intraday.at[idx, 'last_daily_trough'] = self.data['low'].iloc[last_trough_idx]
        
        return intraday


    def _is_overlapping(self, start_idx: int, end_idx: int) -> bool:
        """
        Check if a new trend would overlap with existing trends.
        
        Args:
            start_idx: Start index of new trend
            end_idx: End index of new trend
            
        Returns:
            bool: True if overlapping with existing trends
        """
        # Check uptrends
        for trend in self.uptrends:
            if not (end_idx < trend.start_idx or start_idx > trend.end_idx):
                return True
                
        # Check downtrends
        for trend in self.downtrends:
            if not (end_idx < trend.start_idx or start_idx > trend.end_idx):
                return True
                
        return False

    # Rest of the class remains the same
    def visualize_trends(self):
        """
        Visualizes the price data with identified trends using matplotlib.
        Plots candlesticks and marks uptrends and downtrends.
        """
        import matplotlib.pyplot as plt
        from mplfinance.original_flavor import candlestick_ohlc
        import matplotlib.dates as mdates
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(15, 7))
        
        # Convert index to numbers for plotting
        self.data['date_num'] = mdates.date2num(self.data.index)
        
        # Prepare OHLC data
        ohlc = self.data[['date_num', 'open', 'high', 'low', 'close']].values
        
        # Plot candlesticks
        candlestick_ohlc(ax, ohlc, width=0.6/24, colorup='g', colordown='r')
        
        # Plot trends
        for trend in self.uptrends:
            start_date = self.data['date_num'].iloc[trend.start_idx]
            end_date = self.data['date_num'].iloc[trend.end_idx]
            ax.axvspan(start_date, end_date, alpha=0.2, color='g', label='Uptrend')
        
        for trend in self.downtrends:
            start_date = self.data['date_num'].iloc[trend.start_idx]
            end_date = self.data['date_num'].iloc[trend.end_idx]
            ax.axvspan(start_date, end_date, alpha=0.2, color='r', label='Downtrend')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Add peaks and troughs markers
        for peak_idx in self.peaks:
            ax.plot(self.data['date_num'].iloc[peak_idx], self.data['high'].iloc[peak_idx], 
                    '^', color='blue', markersize=10, label='Peak')
        
        for trough_idx in self.troughs:
            ax.plot(self.data['date_num'].iloc[trough_idx], self.data['low'].iloc[trough_idx], 
                    'v', color='orange', markersize=10, label='Trough')
        
        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.title('Price Chart with Identified Trends')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        plt.tight_layout()
        
        return fig
        
    def visualize_intraday_trends(self, prev_bar, last_peak, last_trough):
        """Create a visualization of the intraday stock price with candlesticks and trends."""
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Prepare data for candlestick chart
        df_ohlc = self.data.reset_index()
        df_ohlc['timestamp'] = df_ohlc['Datetime'].map(mpdates.date2num)  # Map intraday datetime to numeric format
        ohlc_data = df_ohlc[['timestamp', 'open', 'high', 'low', 'close']].values
        
        # Plot candlesticks
        candlestick_ohlc(ax, ohlc_data, width=0.001,  # Smaller width for high granularity
                         colorup='green', colordown='red', alpha=0.7)
        
        # Plot peaks and troughs
        dates_float = df_ohlc['timestamp'].values
        offset_factor = 0.001  # Offset for peaks/troughs
        peak_y_positions = self.data['high'].iloc[self.peaks] + (self.data['close'].iloc[self.peaks] * offset_factor)
        ax.plot(dates_float[self.peaks], peak_y_positions,
                'gv', label='Peaks', markersize=10)  # Green upward triangle for peaks
        
        trough_y_positions = self.data['low'].iloc[self.troughs] - (self.data['close'].iloc[self.troughs] * offset_factor)
        ax.plot(dates_float[self.troughs], trough_y_positions,
                'r^', label='Troughs', markersize=10)
        
        # Highlight trends
        for start_idx, end_idx in self.uptrends:
            ax.axvspan(dates_float[start_idx], dates_float[end_idx],
                       alpha=0.2, color='green', label='Uptrend')
        
        for start_idx, end_idx in self.downtrends:
            ax.axvspan(dates_float[start_idx], dates_float[end_idx],
                       alpha=0.2, color='red', label='Downtrend')
        
        # Calculate intraday price range
        intraday_high = self.data['high'].max()
        intraday_low = self.data['low'].min()
        price_range = intraday_high - intraday_low
        range_limit = 1.5 * price_range  # Define acceptable range for additional lines
        
        # Add lines for `prev_bar` if within range
        if abs(prev_bar['high'] - intraday_high) <= range_limit:
            ax.axhline(prev_bar['high'], color='green', linestyle='--', linewidth=1, label='Prev Bar High')
        if abs(prev_bar['Low'] - intraday_low) <= range_limit:
            ax.axhline(prev_bar['low'], color='red', linestyle='--', linewidth=1, label='Prev Bar Low')
        if abs(prev_bar['close'] - intraday_high) <= range_limit:
            ax.axhline(prev_bar['close'], color='blue', linestyle='-', linewidth=1, label='Prev Bar Close')
        if abs(prev_bar['open'] - intraday_high) <= range_limit:
            ax.axhline(prev_bar['open'], color='orange', linestyle='-', linewidth=1, label='Prev Bar Open')
        
        # Add bold lines for `last_peak` and `last_trough` if within range
        if abs(last_peak - intraday_high) <= range_limit:
            ax.axhline(last_peak, color='darkorange', linestyle='-', linewidth=2.5, label='Last Daily Peak')
        if abs(last_trough - intraday_low) <= range_limit:
            ax.axhline(last_trough, color='purple', linestyle='-', linewidth=2.5, label='Last Daily Trough')
        
        # Customize the plot
        ax.xaxis.set_major_formatter(mpdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mpdates.AutoDateLocator())
        plt.title(f'{self.symbol} Intraday Stock Price Trends')
        plt.xlabel('Date and Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt

    
    def get_trend_summary(self):
        """Generate a summary of identified trends."""
        summary = []
       
        for start_idx, end_idx in self.uptrends:
            start_price = self.data['close'].iloc[start_idx]
            end_price = self.data['close'].iloc[end_idx]
            change_pct = ((end_price - start_price) / start_price) * 100
           
            summary.append(
                f"Uptrend: Start: {self.data.index[start_idx].strftime('%Y-%m-%d')} "
                f"(Price: ${start_price:.2f}), End: {self.data.index[end_idx].strftime('%Y-%m-%d')} "
                f"(Price: ${end_price:.2f}), Change: {change_pct:.1f}%"
            )
       
        for start_idx, end_idx in self.downtrends:
            start_price = self.data['close'].iloc[start_idx]
            end_price = self.data['close'].iloc[end_idx]
            change_pct = ((end_price - start_price) / start_price) * 100
           
            summary.append(
                f"Downtrend: Start: {self.data.index[start_idx].strftime('%Y-%m-%d')} "
                f"(Price: ${start_price:.2f}), End: {self.data.index[end_idx].strftime('%Y-%m-%d')} "
                f"(Price: ${end_price:.2f}), Change: {change_pct:.1f}%"
            )
           
        return summary
