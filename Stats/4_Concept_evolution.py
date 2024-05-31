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

#%% Dynamic of keywords

check = ["state bonus", "minimum income", "national dividend", "social dividend", "basic minimum income", "basic income",
         "negative income tax", "minimum income guarantee", "guaranteed minimum income", "basic income guarantee", "demogrant", "guaranteed income", "credit income tax",
         "citizen’s basic income", "citizen’s income", "helicopter money", "quantitative easing",
         "unconditional basic income", "universal basic income", "negative income tax", "guaranteed minimum income", "social dividend", "basic income guarantee"]


year_list = [] 

docs = collection.find()
for doc in tqdm.tqdm(docs):
    year = doc["publication_year"]
    if year:
        year_list.append(year)

year2keywords = {i:{j:0 for j in check} for i in year_list}

docs = collection.find()
for doc in tqdm.tqdm(docs):
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
    text = title + " " + abstract
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

# Plot each keyword over the years
plt.figure(figsize=(14, 8))
for keyword in check:
    plt.plot(df_sorted.loc[(df_sorted["year"] > 1970) & (df_sorted["year"] < 2010), 'year'], 
         df_sorted.loc[(df_sorted["year"] > 1970) & (df_sorted["year"] < 2010), keyword], 
         marker='o', label=keyword)
plt.title('Keyword Trends Over Years')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig('Results/Figures/keyword_trend.png', format='png')
plt.show()


df_sorted[df_sorted["year"]<2000]['year']
doc = collection.find_one({"id":"https://openalex.org/W3123802184"})

