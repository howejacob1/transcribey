from pymongo import MongoClient
import time
import logging
import toml
import os

# Load MongoDB config from .secrets.toml
SECRETS_PATH = os.path.join(os.path.dirname(__file__), '.secrets.toml')
if not os.path.exists(SECRETS_PATH):
    SECRETS_PATH = os.path.join(os.getcwd(), '.secrets.toml')
with open(SECRETS_PATH, 'r') as f:
    secrets = toml.load(f)

mongo_config = secrets.get('mongo_db', {})
MONGO_URI = mongo_config.get('url')
DB_NAME = mongo_config.get('db')
COLLECTION_NAME = mongo_config.get('collection')

if not MONGO_URI or not DB_NAME or not COLLECTION_NAME:
    raise ValueError("Missing MongoDB configuration in .secrets.toml. Please provide 'url', 'db', and 'collection' under the 'mongo_db' section.")

def get_mongo_collection(uri=MONGO_URI, db_name=DB_NAME, collection_name=COLLECTION_NAME):
    start_time = time.time()
    logging.info(f"Loading mongo collection '{collection_name}' from database '{db_name}' at '{uri}'")

    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    
    elapsed = time.time() - start_time
    logging.info(f"Loaded mongo collection in {elapsed:.4f} seconds.")
    return collection

if __name__ == '__main__':
    # Test connection and insert a test document
    collection = get_mongo_collection()
    result = collection.insert_one({'test': 'hello world'})
    print(f'Inserted test document with id: {result.inserted_id}')

# Add this function to print all vcons in the collection
def print_all_vcons():
    collection = get_mongo_collection()
    for doc in collection.find():
        print(doc)

# Add this function to delete all vcons in the collection
def delete_all_vcons():
    collection = get_mongo_collection()
    result = collection.delete_many({})
    print(f"Deleted {result.deleted_count} documents from the collection.") 