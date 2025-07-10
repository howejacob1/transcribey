from pymongo import MongoClient
import secrets_utils
import threading

# Semaphore to limit concurrent database operations to prevent connection pool exhaustion
_db_semaphore = threading.Semaphore(5)  # Allow max 25 concurrent DB operations (conservative limit)

config = secrets_utils.secrets.get('mongo_db', {})
uri = config.get('url')
db_name = config.get('db')

client = MongoClient(
    uri
    # maxPoolSize=10,  # Increased from 50 to handle more concurrent threads
    # minPoolSize=1,   # Increased minimum pool size
    # maxIdleTimeMS=30000,
    # waitQueueTimeoutMS=15000,  # Increased timeout from 5s to 15s
    # serverSelectionTimeoutMS=10000,  # Increased timeout
    # socketTimeoutMS=30000,  # Increased socket timeout
    # connectTimeoutMS=15000,  # Increased connection timeout
    # retryWrites=True,
    # w=1
)

something_db = client[db_name]
db = something_db["vcons"]