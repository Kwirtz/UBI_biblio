#%% Init

import tqdm
import pymongo
import pandas as pd
import matplotlib.pyplot as plt
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

# Convert the dictionary to a DataFrame
df = pd.DataFrame.from_dict(year2keywords, orient='index').reset_index()

# Rename the index column to 'year'
df.rename(columns={'index': 'year'}, inplace=True)
df_sorted = df.sort_values(by='year')

# Plot each keyword over the years
plt.figure(figsize=(14, 8))
for keyword in check:
    plt.plot(df_sorted['year'], df_sorted[keyword], marker='o', label=keyword)

plt.title('Keyword Trends Over Years')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig('Results/Figures/keyword_trend.png', format='png')
plt.show()
