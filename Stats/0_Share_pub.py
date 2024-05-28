#%% Init

import tqdm
import pymongo
import pandas as pd
from collections import defaultdict

# MongoDB connection
Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["UBI"]
collection = db["works_UBI_20240517"]

db_OA = Client["openAlex20240517"]
collection_OA = db_OA["works"]

#%% get share evolution

year2n_pub = defaultdict(int)

docs = collection_OA.find()

for doc in tqdm.tqdm(docs):
    try:
        year = doc["publication_year"]
        if year != None:
            if year >1900 and year <2025:
                year2n_pub[doc["publication_year"]] += 1
    except:
        pass
    

year2n_pub_ubi = defaultdict(int)
docs = collection.find()

for doc in tqdm.tqdm(docs):
    try:
        year = doc["publication_year"]
        if year != None:
            if year >1900 and year <2025:
                year2n_pub_ubi[doc["publication_year"]] += 1
    except:
        pass
    

df1 = pd.DataFrame.from_dict(year2n_pub, orient='index', columns=['Value1'])
df2 = pd.DataFrame.from_dict(year2n_pub_ubi, orient='index', columns=['Value2'])

# Merge the DataFrames
df = pd.concat([df1, df2], axis=1)
df1.index = df1.index.astype(str)
df_sorted = df.sort_index()
df_sorted = df_sorted.fillna(0)

