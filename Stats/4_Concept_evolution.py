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
collection = db["works_UBI_gobu_2"]

#%% Dynamic of keywords

check = ["state bonus", "minimum income", "national dividend", "social dividend", "basic minimum income", "basic income",
         "negative income tax", "minimum income guarantee", "guaranteed minimum income", "basic income guarantee", "demogrant", "guaranteed income", "credit income tax",
         "citizen’s basic income", "citizen’s income", "social credit",
         "unconditional basic income", "universal basic income", "guaranteed income", "social dividend", "basic income guarantee"]


year_list = [] 
list_papers = []
docs = collection.find()
for doc in tqdm.tqdm(docs):
    list_papers.append(doc["id"])
    year = doc["publication_year"]
    if year:
        year_list.append(year)

year2keywords = {i:{j:0 for j in check} for i in year_list}


for paper in tqdm.tqdm(list_papers):
    doc = collection.find_one({"id":paper})
    year = doc["publication_year"]
    try:
        title = doc["title"]
    except:
        title = ""
    try:
        abstract = doc["abstract"]
    except:
        abstract = ""
    if not title:
        title = ""
    if not abstract:
        abstract = ""
    text = title + " " #+ abstract
    text = text.lower()
    full_gram = ""

    if doc["has_fulltext"]:
        doc_id = doc["id"].split("/")[-1]
        response = requests.get("https://api.openalex.org/works/{}/ngrams?mailto=kevin-w@hotmail.fr".format(doc_id))
        ngrams = json.loads(response.content)["ngrams"]
        for gram in ngrams:
            if gram['ngram_tokens']<4:
                full_gram += gram["ngram"].lower() + " "
    for keyword in check:
        done = False
        if keyword in text and done == False:
            year2keywords[year][keyword] += 1
            done = True
        if keyword in full_gram and done == False:
            year2keywords[year][keyword] += 1
            done = True

            
            

# Convert the dictionary to a DataFrame
df = pd.DataFrame.from_dict(year2keywords, orient='index').reset_index()

# Rename the index column to 'year'
df.rename(columns={'index': 'year'}, inplace=True)
df_sorted = df.sort_values(by='year')

df_sorted.to_csv("Data/Fig1.csv",index=False)

#%% Special Eva


check = ["state bonus", "minimum income", "national dividend", "social dividend", "basic minimum income", "basic income",
         "negative income tax", "minimum income guarantee", "guaranteed minimum income", "basic income guarantee", "demogrant", "guaranteed income", "credit income tax",
         "citizen’s basic income", "citizen’s income", "social credit",
         "unconditional basic income", "universal basic income", "guaranteed income", "social dividend", "basic income guarantee"]

list_papers = []
docs = collection.find()
for doc in tqdm.tqdm(docs):
    if doc["publication_year"] <= 1950:
        list_papers.append(doc["id"])

list_of_insertion = []


for paper in tqdm.tqdm(list_papers):
    doc = collection.find_one({"id":paper})
    year = doc["publication_year"]
    try:
        title = doc["title"]
    except:
        title = ""
    try:
        abstract = doc["abstract"]
    except:
        abstract = ""
    if not title:
        title = ""
    if not abstract:
        abstract = ""
    text = title + " " #+ abstract
    text = text.lower()
    full_gram = ""

    if doc["has_fulltext"]:
        doc_id = doc["id"].split("/")[-1]
        response = requests.get("https://api.openalex.org/works/{}/ngrams?mailto=kevin-w@hotmail.fr".format(doc_id))
        ngrams = json.loads(response.content)["ngrams"]
        for gram in ngrams:
            if gram['ngram_tokens']<4:
                full_gram += gram["ngram"].lower() + " "
    for keyword in check:
        if keyword in text or keyword in full_gram:
            list_of_insertion.append([doc["id"],keyword,year])
            
            
            

# Convert the dictionary to a DataFrame
df = pd.DataFrame(list_of_insertion, columns=["id","keyword","year"])
df = df.sort_values("year")
df.to_csv("Data/special_Eva.csv",index=False)

