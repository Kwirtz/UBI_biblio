import tqdm
import pymongo


Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["openAlex"]
collection = db["works"]
collection_eco = db["works_UBI"]
docs = collection.find()


keywords = ["basic income"]



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

"""
n = 0
docs = collection_eco.find()
for doc in tqdm.tqdm(docs):
    if doc["type"]:
        if doc["type"] == "journal-article":
            n += 1
"""