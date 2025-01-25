from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime
import logging
from bson.objectid import ObjectId

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MongoDBWrapper:
    def __init__(self):
        try:
            connection_string = "mongodb+srv://yuvalmandler:EZ9BnwbKebJtofZ4@cluster0.fqq12.mongodb.net/"
            self.client = MongoClient(connection_string)
            self.client.server_info() # Validate connection
            self.db = self.client['trading_db']
            self.positions = self.db['positions']
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise

    def disconnect(self):
       self.client.close()

    def update_position(self, position_data):
       position_id = position_data['_id']
       update = {
           '$set': {
               'symbol': position_data['symbol'],
               'interval': position_data['symbol'],
               'strategy': position_data['strategy'],
               'type': position_data['type'], 
               'entry_time': position_data['entry_time'],
               'status': position_data['status'],
               'profit': position_data['profit'],
               'last_updated': datetime.now()
           }
       }
       self.positions.update_one({'_id': position_id}, update)

    def get_positions(self):
       return list(self.positions.find({}))
    
    def close_position(self, symbol):
       self.positions.update_one(
           {'symbol': symbol},
           {'$set': {'status': 'Closed', 'close_time': datetime.now()}}
       )
    def insert_position(self, position_data):
        position = {
            'symbol': position_data['symbol'],
            'interval': position_data['symbol'],
           'strategy': position_data['strategy'],
           'type': position_data['type'], 
           'entry_time': position_data['entry_time'],
           'status': position_data['status'],
           'profit': position_data['profit'],
           'last_updated': datetime.now()
       }
        return self.positions.insert_one(position)
        
    def delete_position(self, position_id):
        if isinstance(position_id, str):
            position_id = ObjectId(position_id)
        result = self.positions.delete_one({'_id': position_id})
        return result.deleted_count

    