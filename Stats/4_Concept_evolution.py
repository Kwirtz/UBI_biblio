#%% Init

import tqdm
import pymongo
import pandas as pd
from collections import defaultdict

# MongoDB connection
Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["UBI"]
collection = db["works_UBI_20240517"]

#%% Dynamic of keywords

check = ["basic income", "unconditional basic income", "universal basic income", "negative income tax", "guaranteed minimum income", "social dividend", "basic income guarantee"]


year_list = [] 

docs = collection.find()
for doc in tqdm.tqdm(docs):
    year = doc["publication_year"]
    if year:
        year_list.append(year)

year2keywords = {i:{j:0 for j in check} for i in year_list}

docs = collection.find()
for doc in tqdm.tqdm(docs):
    try:
        title = doc["title"]
    except:
        title = ""
    try:
        abstract = doc["abstract"]
    except:
        abstract = ""
    year = doc["publication_year"]
    if not title:
        title = ""
    if not abstract:
        abstract = ""
    text = title + " " + abstract
    text = text.lower()
    for keyword in check:
        if keyword in text:
            year2keywords[year][keyword] += 1
    