import tqdm
import pymongo
from datetime import datetime

Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["openAlex20240517"]
collection = db["works"]
db_new = Client["openAlex20240517"]
collection_eco = db_new["works_SHS"]
docs = collection.find()


# UBI 
#keywords = ["economics"]
# SHS 
keywords = ["philosophy"," sociology", "history"," geography","psychology","economics","political science","art","business"]



def get_ubi(keywords):    
    try:
        collection_eco.create_index([ ("id",1) ])   
        collection_eco.create_index([ ("publication_year",1) ])         
    except:
        pass
    docs = collection.find({})
    list_of_insertion = []
    for doc in tqdm.tqdm(docs):
        n = 0
        for concept in doc["concepts"]:
            if concept["display_name"].lower() in keywords:
                n += 1
        if n > 0 :
            try:
                list_of_insertion.append(doc)
            except:
                continue
    
        if len(list_of_insertion) == 10000:
            collection_eco.insert_many(list_of_insertion)
            list_of_insertion = []            
    collection_eco.insert_many(list_of_insertion)
    list_of_insertion = []

get_ubi(keywords)


def check_for_dups(db_name, collection_name):
    Client = pymongo.MongoClient("mongodb://localhost:27017")
    db = Client[db_name]
    collection = db[collection_name]
    docs = collection.find({})
    ids = []
    for doc in tqdm.tqdm(docs):
        ids.append(doc["id"])
    if len(ids) != len(list(set(ids))):
        print("WOUAH WATCH OUT")
    else:
        print("RAS")
    return ids

test = check_for_dups(db_name = "UBI", collection_name = "works_UBI_20240517")

def delete_dups(db_name, collection_name):
    Client = pymongo.MongoClient("mongodb://localhost:27017")
    db = Client[db_name]
    collection = db[collection_name]
    docs = collection.find({})
    ids = []
    for doc in tqdm.tqdm(docs):
        ids.append(doc["id"])
    for id_ in ids:
        docs = list(collection.find({"id":id_}))
        if len(docs) > 1:
            docs.sort(key=lambda x: datetime.fromisoformat(x['updated_date']), reverse=True)
            # Keep the most recent document and delete the rest
            for doc_to_delete in docs[1:]:
                collection.delete_one({"_id": doc_to_delete["_id"]})

delete_dups(db_name = "UBI", collection_name = "works_UBI_20240517")

"""
n = 0
docs = collection_eco.find()
for doc in tqdm.tqdm(docs):
    if doc["type"]:
        if doc["type"] == "journal-article":
            n += 1
"""


import tqdm
import pymongo
from datetime import datetime

# Connect to MongoDB
client = pymongo.MongoClient('mongodb://localhost:27017')
db = client['UBI']
collection = db['works_UBI']


# Define a cursor to fetch all documents in the collection
cursor = collection.find({})

# Create a dictionary to track the most recent 'updated_date' for each 'id'
latest_dates = {}
latest_id = {}

# Iterate over the documents and identify and delete duplicates
for document in tqdm.tqdm(cursor):
    id_ = document['id']
    updated_date_str = document['updated_date']

    # Remove the time part and standardize the date format
    updated_date_str = updated_date_str.split('T')[0]

    # Convert the date to a Unix timestamp
    updated_date_unix = int(datetime.strptime(updated_date_str, '%Y-%m-%d').timestamp())

    if id_ in latest_dates:
        # If we've seen this 'id' before, compare the Unix timestamp 'updated_date_unix'
        if updated_date_unix > latest_dates[id_]:
            # Delete the previous duplicate document
            collection.delete_one({'_id': latest_id[id_]})
            # This document has a more recent 'updated_date', so update the 'latest_dates' dictionary
            latest_dates[id_] = updated_date_unix
            latest_id[id_] = document['_id']
        else:
            pass
            # This document is a duplicate with an older 'updated_date', so delete it
            collection.delete_one({'_id': document['_id']})
            
    else:
        # First time seeing this 'id', add it to the 'latest_dates' dictionary
        latest_dates[id_] = updated_date_unix
        latest_id[id_] = document['_id']

# Close the MongoDB connection
client.close()

