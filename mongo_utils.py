from pymongo import MongoClient
import secrets_utils

config = secrets_utils.secrets.get('mongo_db', {})
uri = config.get('url')
db_name = config.get('db')

client = MongoClient(
    uri,
    maxPoolSize=25,  # Increase from 1 to 25 for better concurrency
    minPoolSize=5,   # Keep 5 connections ready at minimum
    maxIdleTimeMS=300000,  # Increase idle timeout to 5 minutes
    waitQueueTimeoutMS=60000,  # Increase wait timeout to 60s
    serverSelectionTimeoutMS=60000,  # Increase server selection timeout to 60s
    socketTimeoutMS=300000,  # Increase socket timeout to 5 minutes
    connectTimeoutMS=60000,  # Increase connection timeout to 60s
    retryWrites=True,
    w=1  # Write to primary only, don't wait for replicas
)

something_db = client[db_name]
db = something_db["vcons"]
