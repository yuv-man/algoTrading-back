import json
import types
import inspect
import pandas as pd
from typing import Dict, Any, Type, Optional, Tuple
from pathlib import Path

class StrategyHandler:
    @staticmethod
    def create_strategy_from_json(strategy_json: str) -> Dict[str, Any]:
        """
        Creates a strategy class from JSON definition
        
        Args:
            strategy_json: JSON string containing strategy definition
            
        Returns:
            Dict containing either the created class or error information
        """
        result = {
            'success': False,
            'strategy_class': None,
            'class_source': None,
            'message': '',
            'errors': []
        }
        
        try:
            # Parse JSON
            strategy_data = json.loads(strategy_json)
            
            # Validate required fields
            required_fields = ['name', 'init_params', 'calculate_signals_body', 'generate_trade_decision_body']
            missing_fields = [field for field in required_fields if field not in strategy_data]
            if missing_fields:
                result['errors'].append(f"Missing required fields: {', '.join(missing_fields)}")
                return result
                
            # Create class source code
            class_source = f"""
class {strategy_data['name']}(BaseStrategy):
    def __init__(self, symbol: str, params: dict = None):
        {strategy_data.get('init_params', 'super().__init__(symbol, params)')}
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        {strategy_data['calculate_signals_body']}
        
    def generate_trade_decision(self, current_data: dict) -> Tuple[Optional[str], Optional[int], Optional[float]]:
        {strategy_data['generate_trade_decision_body']}
"""
            # Create a new module to hold the class
            module = types.ModuleType(strategy_data['name'])
            
            # Add necessary imports to the module
            exec("""
import pandas as pd
from typing import Dict, Optional, Tuple
from strategies import BaseStrategy
            """, module.__dict__)
            
            # Execute the class code in the module's namespace
            exec(class_source, module.__dict__)
            
            # Get the class from the module
            strategy_class = getattr(module, strategy_data['name'])
            
            result['success'] = True
            result['strategy_class'] = strategy_class
            result['class_source'] = class_source
            result['message'] = f"Successfully created strategy class: {strategy_data['name']}"
            
        except Exception as e:
            result['errors'].append(str(e))
            result['message'] = "Error creating strategy from JSON"
            
        return result

class StrategyValidator:
    def __init__(self, strategies_file_path: str = 'strategies.py'):
        self.strategies_file_path = Path(strategies_file_path)

    def validate_and_save_strategy(self, strategy_class: Type, class_source: str) -> Dict[str, Any]:
        """
        Validates strategy class format and saves it to strategies.py if valid
        
        Args:
            strategy_class: The strategy class to validate and save
            class_source: The source code of the strategy class
            
        Returns:
            Dict containing operation results and any error messages
        """
        result = {
            'success': False,
            'message': '',
            'errors': []
        }
        
        # Check if it inherits from BaseStrategy
        if not any(base.__name__ == 'BaseStrategy' for base in strategy_class.__bases__):
            result['errors'].append("Strategy must inherit from BaseStrategy")
            return result
        
        # Check required methods
        required_methods = ['__init__', 'calculate_signals', 'generate_trade_decision']
        for method_name in required_methods:
            if not hasattr(strategy_class, method_name):
                result['errors'].append(f"Missing required method: {method_name}")
                return result
        
        try:
            # Read existing file content
            with open(self.strategies_file_path, 'r') as file:
                existing_content = file.read()
            
            # Check if strategy already exists
            if f"class {strategy_class.__name__}" in existing_content:
                result['message'] = f"Strategy {strategy_class.__name__} already exists"
                return result
            
            # Append new strategy to file
            with open(self.strategies_file_path, 'a') as file:
                file.write(f"\n\n{class_source}")
            
            result['success'] = True
            result['message'] = f"Successfully added strategy: {strategy_class.__name__}"
            
        except Exception as e:
            result['errors'].append(str(e))
            result['message'] = "Error adding strategy to file"
            
        return result

# Usage example:
if __name__ == "__main__":
    # Example JSON input
    strategy_json = '''{
        "name": "SimpleMovingAverageStrategy",
        "init_params": "super().__init__(symbol, params)",
        "calculate_signals_body": "return data",
        "generate_trade_decision_body": "return None, None, None"
    }'''
    
    # First, create the strategy class from JSON
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