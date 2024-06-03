#%% Init

import tqdm
import json
import pymongo
import requests
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# MongoDB connection
Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["UBI"]
collection = db["works_UBI_20240517"]

list_papers = []
docs = collection.find()

for doc in tqdm.tqdm(docs):
    list_papers.append(doc["id"])
    if doc["referenced_works"]:
        for ref in doc["referenced_works"]:
            list_papers.append(ref)
            
list_papers = list(set(list_papers))

db_all = Client["openAlex20240517"]
collection_all = db_all["works_SHS"]
collection_new = db["works_UBI_global"]

data = []
for paper in tqdm.tqdm(list_papers):
    doc = collection_all.find_one({"id":paper})
    if doc:
        data.append(doc)

collection_new.insert_many(data)
