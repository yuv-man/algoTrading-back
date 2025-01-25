from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from typing import Dict, Any, Optional
import logging
from trading_engine import TradingEngine
from tradingApp import TradingApplication
import random #to delete
import time
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173"],  # Your frontend origin
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
socketio = SocketIO(app, cors_allowed_origins="*")

trading_engine = TradingEngine(socketio)
trading_app = TradingApplication()

def validate_dates(start_date: str, end_date: str) -> tuple[bool, Optional[str]]:
    """Validate date strings are in correct format and logical order."""
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        if end < start:
            return False, "End date must be after start date"
        return True, None
    except ValueError:
        return False, "Dates must be in YYYY-MM-DD format"

def validate_strategy_params(strategy_type: str, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate strategy parameters based on strategy type."""
    if strategy_type == 'SMA':
        required_params = {'short_window', 'long_window'}
        if not all(param in params for param in required_params):
            return False, f"Missing required parameters: {required_params}"
        if not all(isinstance(params[p], int) and params[p] > 0 for p in required_params):
            return False, "Window parameters must be positive integers"
    return True, None

@app.route('/api/register_strategy', methods=['POST'])
def register_strategy():
    """Register a new trading strategy with parameter validation."""
    logger.debug("Received request to register strategy")
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No JSON data provided'}), 400
        
        
        strategy_type = data.get('strategy_type')
        symbol = data.get('symbol')
        params = data.get('params', {})
        
        if not all([strategy_type, symbol]):
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields: strategy_type and symbol'
            }), 400
        
        # Validate strategy parameters
        is_valid, error_msg = validate_strategy_params(strategy_type, params)
        if not is_valid:
            return jsonify({'status': 'error', 'message': error_msg}), 400

        res = trading_app.register_strategy(symbol=symbol, strategy_type=strategy_type, params=params)
        print("results: ", res)
        logger.info(f"Strategy registered: {strategy_type} for {symbol}")
        return jsonify(res)
        
    except Exception as e:
        logger.error(f"Error registering strategy: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
def backtest():
    """Run strategy backtest with proper validation and error handling."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No JSON data provided'}), 400
        
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        period = data.get('period')
        interval = data.get('interval', '1d')
        if not period and not (start_date and end_date):
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields: start_date and end_date or period'
            }), 400
        
        # Validate dates
        if(start_date and end_date):
            is_valid, error_msg = validate_dates(start_date, end_date)
            if not is_valid:
                return jsonify({'status': 'error', 'message': error_msg}), 400

        if not trading_app.trading_engine.strategy:
            return jsonify({
                'status': 'error',
                'message': 'No strategy registered'
            }), 400
            
        results = trading_app.run_backtest(start_date=start_date, end_date=end_date, interval=interval, period=period)
        print('get results')
        return jsonify({
            'status': 'success',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error during backtest: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/get_stock_data', methods=['POST'])
def get_stock_data():
    """
    Fetch market data based on time parameters.
    Returns both daily and intraday data with trend analysis.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No JSON data provided'}), 400
        
        # Get parameters with defaults
        symbol = data.get('symbol', 'SPY')
        interval = data.get('interval', '5 mins')
        period = data.get('period')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # Validate parameter combinations
        if not period and not (start_date and end_date):
            return jsonify({
                'status': 'error',
                'message': 'Either period or both start_date and end_date must be provided'
            }), 400
            
        if period and (start_date or end_date):
            return jsonify({
                'status': 'error',
                'message': 'Cannot provide both period and date range'
            }), 400

        # Fetch data using trading_app
        try:
            result = trading_app.get_stock_data(
                symbol=symbol,
                interval=interval,
                period=period,
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            logger.error(f"Error in trading_app.get_stock_data: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Failed to fetch stock data: {str(e)}'
            }), 500

        # Helper function to safely get column value
        def get_column_value(row, column_name):
            """Safely get column value handling both cases and providing default None"""
            upper_name = column_name.upper()
            lower_name = column_name.lower()
            
            if upper_name in row:
                return row[upper_name]
            elif lower_name in row:
                return row[lower_name]
            return None

        # Process intraday data if available
        formatted_intraday = None
        if 'intraday_data' in result and result['intraday_data'] is not None and not result['intraday_data'].empty:
            start_date = result['intraday_data'].index[0].strftime('%Y-%m-%d %H:%M:%S')
            end_date = result['intraday_data'].index[-1].strftime('%Y-%m-%d %H:%M:%S')
            
            formatted_intraday = [
                {
                    'time': index.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': get_column_value(row, 'open'),
                    'high': get_column_value(row, 'high'),
                    'low': get_column_value(row, 'low'),
                    'close': get_column_value(row, 'close'),
                    'volume': get_column_value(row, 'volume'),
                    'daily_trend': get_column_value(row, 'daily_trend'),
                    'last_daily_peak': get_column_value(row, 'last_daily_peak'),
                    'last_daily_trough': get_column_value(row, 'last_daily_trough')
                }
                for index, row in result['intraday_data'].iterrows()
            ]

        # Process daily data
        formatted_daily = None
        if 'daily_data' in result and result['daily_data'] is not None and not result['daily_data'].empty:
            if not start_date:  # If not set by intraday data
                start_date = result['daily_data'].index[0].strftime('%Y-%m-%d %H:%M:%S')
                end_date = result['daily_data'].index[-1].strftime('%Y-%m-%d %H:%M:%S')
                
            formatted_daily = [
                {
                    'time': index.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': get_column_value(row, 'open'),
                    'high': get_column_value(row, 'high'),
                    'low': get_column_value(row, 'low'),
                    'close': get_column_value(row, 'close'),
                    'volume': get_column_value(row, 'volume'),
                    'daily_trend': get_column_value(row, 'trend'),
                    'pivot_point': get_column_value(row, 'pivot_point')
                }
                for index, row in result['daily_data'].iterrows()
            ]

        # Ensure we have at least one type of data
        if formatted_daily is None and formatted_intraday is None:
            return jsonify({
                'status': 'error',
                'message': 'No data available for the specified parameters'
            }), 404

        # Calculate total rows
        num_rows = (len(formatted_intraday) if formatted_intraday else 0) + \
                  (len(formatted_daily) if formatted_daily else 0)

        return jsonify({
            'status': 'success',
            'intraday_data': formatted_intraday,
            'daily_data': formatted_daily,
            'metadata': {
                'symbol': symbol,
                'interval': interval,
                'period': period,
                'start_date': start_date,
                'end_date': end_date,
                'total_rows': num_rows
            }
        })
            
    except Exception as e:
        logger.error(f"Unexpected error in get_stock_data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/start_trading', methods=['POST', 'OPTIONS'])
def start_trading():
    """Start live trading with connection management and error handling."""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'success'}), 200

    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400

        symbol = data.get('symbol', 'SPY')
        interval = data.get('interval', '15 mins')
        strategy_type = data.get('strategy_type')
        params = data.get('params')

        if not strategy_type:
            return jsonify({
                'status': 'error',
                'message': 'Strategy type is required'
            }), 400

        # Register strategy first
        trading_app.register_strategy(symbol, strategy_type, params)

        # Start trading in a background thread
        def run_trading():
            try:
                trading_app.start_trading(symbol, interval, strategy_type)
            except Exception as e:
                logger.error(f"Error in trading thread: {str(e)}")
                trading_app.trading_engine.stop_live_trading()

        trading_thread = threading.Thread(target=run_trading, daemon=True)
        trading_thread.start()
        
        logger.info(f"Live trading started for {symbol} with {strategy_type} strategy")
        return jsonify({
            'status': 'success',
            'message': 'Live trading started',
            'data': {
                'symbol': symbol,
                'interval': interval,
                'strategy': strategy_type
            }
        })
        
    except Exception as e:
        logger.error(f"Error starting trading: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/stop_trading', methods=['POST'])
def stop_trading():
    """Stop live trading with state validation."""
    try:
        if not trading_engine.is_trading:
            return jsonify({
                'status': 'error',
                'message': 'No active trading session'
            }), 400
        
        trading_engine.stop_live_trading()
        logger.info("Live trading stopped")
        return jsonify({
            'status': 'success',
            'message': 'Live trading stopped',
            'final_stats': {
                'total_trades': len(trading_engine.strategy.trades),
                'final_portfolio_value': trading_engine.strategy.portfolio_value
            }
        })
        
    except Exception as e:
        logger.error(f"Error stopping trading: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/get_strategies', methods=['GET'])
def get_strategies():
    """Get all strategies and their parameters"""
    try:
        
        strategies = trading_app.get_strategies()
        
        return jsonify({
            'status': 'success',
            'strategies': strategies
        })
        
    except Exception as e:
        logger.error(f"Error getting strategies: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
        

@app.route('/api/get_performance', methods=['GET'])
def get_performance():
    """Get current strategy performance with detailed metrics."""
    try:
        if not trading_engine.strategy:
            return jsonify({
                'status': 'error',
                'message': 'No strategy registered'
            }), 400
        
        performance_data = {
            'strategy_type': trading_engine.strategy.__class__.__name__,
            'symbol': trading_engine.strategy.symbol,
            'current_position': trading_engine.strategy.position,
            'portfolio_value': trading_engine.strategy.portfolio_value,
            'total_trades': len(trading_engine.strategy.trades),
            'trades': trading_engine.strategy.trades,
            'is_trading': trading_engine.is_trading,
            'last_update': datetime.now().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'performance': performance_data
        })
        
    except Exception as e:
        logger.error(f"Error getting performance: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/optimize_strategy', methods=['POST'])
def optimize_strategy():
    """
    Fetch market data based on time parameters.
    Returns both daily and intraday data with trend analysis.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No JSON data provided'}), 400
        
        # Get parameters with defaults
        symbol = data.get('symbol', 'SPY')
        interval = data.get('interval', '5 mins')
        period = data.get('period')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        strategy_type = data.get('strategy_type')
        optimize_target = data.get('optimize_target')
        param_ranges = data.get('param_ranges')
        initial_guess = data.get('initial_guess')
        
        # Validate parameter combinations
        if not period and not (start_date and end_date):
            return jsonify({
                'status': 'error',
                'message': 'Either period or both start_date and end_date must be provided'
            }), 400
            
        if period and (start_date or end_date):
            return jsonify({
                'status': 'error',
                'message': 'Cannot provide both period and date range'
            }), 400

        # Fetch data using trading_app
        try:
            result = trading_app.optimize_strategy(
                symbol=symbol,
                interval=interval,
                period=period,
                start_date=start_date,
                end_date=end_date,
                strategy_type=strategy_type,
                optimize_target=optimize_target,
                param_ranges=param_ranges,
                initial_guess=initial_guess
            )

            return jsonify({
            'status': 'success',
            'results': asdict(result)
            })
        except Exception as e:
            logger.error(f"Error in trading_app.optimize_strategy: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Failed to optimize data: {str(e)}'
            }), 500

               
    except Exception as e:
        logger.error(f"Unexpected error in optimize_strategy: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/orders', methods=['GET'])
def get_orders():
    """Get current strategy performance with detailed metrics."""
    try:
        orders = trading_engine.get_orders()
        json_orders = orders.to_json(orient='records', date_format='iso', double_precision=2)
        
        return jsonify({
            'status': 'success',
            'orders': json_orders
        })
        
    except Exception as e:
        logger.error(f"Error getting orders: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/watchlist', methods=['GET'])
def get_positions():
    """Get current positions."""
    try:     
        positions = trading_engine.get_positions_summary()
        
        return jsonify({
            'status': 'success',
            'watchlist': positions
        })
        
    except Exception as e:
        logger.error(f"Error getting positions: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/get_account_value', methods=['GET'])
def get_account_value():
    """Get current account value."""
    try: 
        account_value = trading_engine.get_account_info()
        
        return jsonify({
            'status': 'success',
            'account_value': account_value
        })
        
    except Exception as e:
        logger.error(f"Error getting account value: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/')
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'path': str(rule)
        })
    return jsonify(routes)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'data': 'Connected to trading websocket server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('subscribe')
def handle_subscribe(symbol):
    print(f'Client subscribed to {symbol}')
    emit('subscribed', {'symbol': symbol})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'status': 'error', 'message': 'Method not allowed'}), 405

def generate_trading_data():
    while True:
        trading_data = {
            'symbol': 'BTC/USD',
            'price': round(random.uniform(30000, 40000), 2),
            'volume': round(random.uniform(100, 1000), 2),
            'timestamp': int(time.time())
        }
        socketio.emit('trading_update', trading_data)
        time.sleep(1)  # Send updates every second

if __name__ == '__main__':
    import multiprocessing
    # Windows-specific multiprocessing support
    if sys.platform.startswith('win'):
        multiprocessing.freeze_support()
    
    try:
        # Start the trading data generator in a separate thread
        trading_thread = threading.Thread(target=generate_trading_data)
        trading_thread.daemon = True
        trading_thread.start()
        
        # Windows-specific Flask configuration
        socketio.run(app, 
                    host='127.0.0.1',
                    port=5001,
                    debug=True,
                    use_reloader=False)  # Disable reloader for Windows
                    
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Cleanup
        if hasattr(trading_app, 'app'):
            trading_app.disconnect()