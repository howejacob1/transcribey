from pymongo import MongoClient
import secrets_utils
import threading

# Semaphore to limit concurrent database operations to prevent connection pool exhaustion
_db_semaphore = threading.Semaphore(1)  # Allow max 25 concurrent DB operations

config = secrets_utils.secrets.get('mongo_db', {})
uri = config.get('url')
db_name = config.get('db')

client = MongoClient(
    uri,
    maxPoolSize=1,  # Handle more concurrent threads
    minPoolSize=1,   # Keep connections ready
    maxIdleTimeMS=120000,
    waitQueueTimeoutMS=120000,  # Increased timeout from 5s to 15s
    serverSelectionTimeoutMS=120000,  # Increased timeout
    socketTimeoutMS=120000,  # Increased socket timeout
    connectTimeoutMS=120000,  # Increased connection timeout
    retryWrites=True,
    w=1
)

something_db = client[db_name]
db = something_db["vcons"]