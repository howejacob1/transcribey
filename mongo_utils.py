from pymongo import MongoClient
import time
import logging

# You can change the URI if you want to use a remote MongoDB or authentication
MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'transcribey'
COLLECTION_NAME = 'vcons'

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