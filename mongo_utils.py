from pymongo import MongoClient
import secrets_utils

config = secrets_utils.secrets.get('mongo_db', {})
uri = config.get('url')
db_name = config.get('db')

client = MongoClient(
    uri,
    maxPoolSize=25,  # Increase from 1 to 25 for better concurrency
    minPoolSize=5,   # Keep 5 connections ready at minimum
    maxIdleTimeMS=30000,  # Reduce idle timeout to 30s
    waitQueueTimeoutMS=10000,  # Reduce wait timeout to 10s
    serverSelectionTimeoutMS=10000,  # Reduce server selection timeout
    socketTimeoutMS=30000,  # Reduce socket timeout to 30s
    connectTimeoutMS=10000,  # Reduce connection timeout to 10s
    retryWrites=True,
    w=1
)

something_db = client[db_name]
db = something_db["vcons"]