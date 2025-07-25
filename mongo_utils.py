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
    w=1
)

something_db = client[db_name]
db = something_db["vcons"]

def ensure_indexes():
    """Ensure all necessary indexes exist for optimal query performance"""
    try:
        # Create index on basename for fast basename queries
        db.create_index("basename", background=True)
        print("Ensured index on basename")
            # Legacy index for dialog.0.basename removed - using top-level basename only
        
        # Create index on done field for processing queries
        db.create_index("done", background=True)
        print("Ensured index on done")
        
        # Create index on processed_by field for reservation queries
        db.create_index("processed_by", background=True)
        print("Ensured index on processed_by")
        
        # Compound index for finding unprocessed items
        db.create_index([("done", 1), ("processed_by", 1)], background=True)
        print("Ensured compound index on done+processed_by")
        
    except Exception as e:
        print(f"Error creating indexes: {e}")

# Ensure indexes are created when module is imported
ensure_indexes()