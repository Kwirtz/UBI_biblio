import tqdm
import pymongo
from collections import defaultdict



#%%

# MongoDB connection
Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["openAlex_20230920"]
collection = db["concepts"]

# Get children of "Basic Income"
concepts_UBI = defaultdict(list)
docs = collection.find({})
for doc in tqdm.tqdm(docs):
    for concept in doc["ancestors"]:
        if concept["display_name"] == "Basic income":
            concepts_UBI[doc["level"]].append(doc["display_name"])
            
#%% Topics with basic income

Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["openAlex20240517"]
collection = db["topics"]

docs = collection.find()
topics_ids = {}

for doc in tqdm.tqdm(docs):
    keywords = [i.lower() for i in doc["keywords"]]
    if "income" in doc["display_name"].lower():
        topics_ids[doc["id"]] = doc["description"]
    