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
collection = db["works_UBI_gobu"]

#%% Dynamic of keywords

check = ["state bonus", "minimum income", "national dividend", "social dividend", "basic minimum income", "basic income",
         "negative income tax", "minimum income guarantee", "guaranteed minimum income", "basic income guarantee", "demogrant", "guaranteed income", "credit income tax",
         "citizen’s basic income", "citizen’s income", 
         "unconditional basic income", "universal basic income", "guaranteed minimum income", "social dividend", "basic income guarantee"]


# 1910-1940 1960-1970 2010 maintenant
check1 = ["state bonus", "minimum income", "national dividend", "social dividend", "social credit"]
check2 = ["negative income tax", "minimum income guarantee","minimum income", "guaranteed minimum income", "basic income guarantee", "demogrant", "guaranteed income", "credit income tax",
"citizen’s basic income", "citizen’s income"]
check3 = ["unconditional basic income", "universal basic income","basic income","ubi"]

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


