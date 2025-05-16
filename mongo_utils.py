from pymongo import MongoClient

# You can change the URI if you want to use a remote MongoDB or authentication
MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'transcribey'
COLLECTION_NAME = 'results'

def get_mongo_collection(uri=MONGO_URI, db_name=DB_NAME, collection_name=COLLECTION_NAME):
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    return collection

if __name__ == '__main__':
    # Test connection and insert a test document
    collection = get_mongo_collection()
    result = collection.insert_one({'test': 'hello world'})
    print(f'Inserted test document with id: {result.inserted_id}') 