#%% Init

import tqdm
import pymongo
import pandas as pd
from collections import defaultdict

# MongoDB connection
Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["UBI"]
collection = db["works_UBI_gobu_2"]

db_OA = Client["openAlex20240517"]
collection_OA = db_OA["works_SHS"]

#%% get share evolution

year2n_pub = defaultdict(lambda:defaultdict(int))

docs = collection_OA.find()

for doc in tqdm.tqdm(docs):
    try:
        year = doc["publication_year"]
        if year != None:
            if year >1900 and year <2021:
                year2n_pub[doc["publication_year"]]["n_pub_SHS"] += 1
                for concept in doc["concepts"]:
                    if concept["display_name"].lower() == "economics":
                        year2n_pub[doc["publication_year"]]["n_pub_SHS_eco"] += 1
    except:
        pass
    

year2n_pub_ubi = defaultdict(lambda:defaultdict(int))

docs = collection.find()

for doc in tqdm.tqdm(docs):
    try:
        year = doc["publication_year"]
        if year != None:
            if year >1900 and year <2021:
                year2n_pub_ubi[doc["publication_year"]]["n_ubi_SHS"] += 1
                for concept in doc["concepts"]:
                    if concept["display_name"].lower() == "economics":
                        year2n_pub_ubi[doc["publication_year"]]["n_ubi_SHS_eco"] += 1
    except:
        pass
    

df1 = pd.DataFrame.from_dict(year2n_pub, orient='index', columns=['n_pub_SHS','n_pub_SHS_eco'])
df2 = pd.DataFrame.from_dict(year2n_pub_ubi, orient='index', columns=['n_ubi_SHS','n_ubi_SHS_eco'])

# Merge the DataFrames
df = pd.concat([df1, df2], axis=1)
df1.index = df1.index.astype(str)
df_sorted = df.sort_index()
df_sorted = df_sorted.fillna(0)
df_sorted["year"] = df_sorted.index
df_sorted.to_csv("Data/Fig_intro.csv",index=False)
