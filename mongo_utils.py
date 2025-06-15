from pymongo import MongoClient

from secrets_utils import secrets

config = secrets.get('mongo_db', {})
uri = config.get('url')
db_name = config.get('db')
client = MongoClient(uri)
something_db = client[db_name]
db = something_db["vcons"]