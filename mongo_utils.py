from pymongo import MongoClient
from secrets_utils import secrets

config = secrets.get('mongo_db', {})
uri = config.get('url')
db_name = config.get('db')
client = MongoClient(uri)

def get_collection(collection_name):
    db = client[db_name]
    return db[collection_name]