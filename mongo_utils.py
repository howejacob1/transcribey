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
VCONS_COLLECTION_NAME = mongo_config.get('vcons_collection', 'vcons')
VCONS_CACHE_COLLECTION_NAME = mongo_config.get('vcons_cache_collection', 'vcons-cache')

if not MONGO_URI or not DB_NAME or not VCONS_COLLECTION_NAME:
    raise ValueError("Missing MongoDB configuration in .secrets.toml. Please provide 'url', 'db', and 'vcons_collection' under the 'mongo_db' section.")

def get_mongo_collection(uri=MONGO_URI, db_name=DB_NAME, collection_name=VCONS_COLLECTION_NAME):
    start_time = time.time()
    logging.info(f"Loading mongo collection '{collection_name}' from database '{db_name}' at '{uri}'")

    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    
    elapsed = time.time() - start_time
    logging.info(f"Loaded mongo collection in {elapsed:.4f} seconds.")
    return collection

# For future use: get the vcons-cache collection
def get_vcons_cache_collection():
    return get_mongo_collection(collection_name=VCONS_CACHE_COLLECTION_NAME)

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

def delete_all_faqs():
    faqs_collection = get_mongo_collection(collection_name="faqs")
    result = faqs_collection.delete_many({})
    print(f"Deleted {result.deleted_count} documents from the faqs collection.")

def delete_all_vcons_cache():
    vcons_cache_collection = get_mongo_collection(collection_name=VCONS_CACHE_COLLECTION_NAME)
    result = vcons_cache_collection.delete_many({})
    print(f"Deleted {result.deleted_count} documents from the vcons-cache collection.")


def print_all_vcon_transcriptions():
    """
    Print all vCon transcriptions from the database. For each vCon, print the filename and all transcription texts.
    """
    collection = get_mongo_collection()
    for doc in collection.find():
        # Try to get filename from dialog or attachments
        filename = None
        if 'dialog' in doc and isinstance(doc['dialog'], list):
            for dlg in doc['dialog']:
                if isinstance(dlg, dict) and 'filename' in dlg:
                    filename = dlg['filename']
                    break
        if not filename and 'attachments' in doc and isinstance(doc['attachments'], list):
            for att in doc['attachments']:
                if isinstance(att, dict) and att.get('type') == 'audio' and 'body' in att:
                    filename = att['body']
                    break
        # Find all transcriptions in analysis
        transcriptions = []
        if 'analysis' in doc and isinstance(doc['analysis'], list):
            for analysis in doc['analysis']:
                if analysis.get('type') == 'transcription':
                    transcriptions.append(analysis.get('body'))
        if transcriptions:
            print(f"Filename: {filename if filename else '[unknown]'}")
            for t in transcriptions:
                print(f"  Transcription: {t}")
            print() 

def all_vcon_urls(collection):
    return [doc["filename"] for doc in collection.find({}, {"filename": 1})]

def clear_mongo_collections():
    delete_all_vcons()
    delete_all_vcons_cache()
    delete_all_faqs()