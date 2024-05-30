#%% init
import tqdm
import pymongo
import itertools
import pycountry
import numpy as np
import pandas as pd
from collections import defaultdict

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['UBI']
collection = db['works_UBI_20240517']

#%% National Output

country_output = defaultdict(lambda:defaultdict(lambda:defaultdict(int)))
df = pd.DataFrame(columns=["country","year","n_total","n_solo_country","n_solo_author","n_collab"])

docs = collection.find()
for doc in tqdm.tqdm(docs):
    country_list = []
    authors = doc["authorships"]
    year = doc["publication_year"]
    for author in authors:
        try:
            country = author["institutions"][0]["country_code"]
            if country != None:
                country_list.append(country)
        except Exception as e:
            pass
    if len(authors) == len(country_list):
        if len(country_list) == 1:
            index = country_output[country_list[0]][year]
            index["n_total"] += 1
            index["n_solo_author"] += 1
        if len(country_list) > 1 and len(set(country_list)) == 1:
            index = country_output[country_list[0]][year]
            index["n_total"] += 1
            index["n_solo_country"] += 1
        if len(set(country_list)) > 1:
            for country in set(country_list):
                index = country_output[country][year]
                index["n_total"] += 1
                index["n_collab"] += 1                


records = []
for country, years in country_output.items():
    for year, values in years.items():
        record = {"country": country, "year": year}
        record.update(values)
        records.append(record)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(records)

# Sort the DataFrame by 'country' and 'year'
df_sorted = df.sort_values(by=['country', 'year'])

# Replace NaN values with 0
df_sorted.fillna(0, inplace=True)

# Define a function to map country codes to country names
def get_country_name(code):
    try:
        country = pycountry.countries.get(alpha_2=code)
        return country.name
    except AttributeError:
        return code  # Return code if no country found

# Apply the function to create a new column with country names
df_sorted['country_name'] = df_sorted['country'].apply(get_country_name)
columns = ['country_name',"country", 'year', 'n_total', 'n_solo_country', 'n_solo_author', 'n_collab']
df_sorted = df_sorted.reindex(columns=columns)


df_sorted.to_csv("Data/national_output.csv",index=False)


#%% International Output


docs = collection.find()

country_inter_output = defaultdict(lambda:defaultdict(lambda:defaultdict(int)))
n = 0
for doc in tqdm.tqdm(docs):
    country_list = []
    authors = doc["authorships"]
    year = doc["publication_year"]
    for author in authors:
        try:
            country = author["institutions"][0]["country_code"]
            if country != None:
                country_list.append(country)
        except Exception as e:
            pass
    if len(authors) == len(country_list):
        if len(set(country_list)) > 1:
            cooc = set(list(itertools.permutations(country_list, 2)))
            for comb in cooc:
                country_inter_output[comb[0]][year][comb[1]] += 1
    else:
        print(doc["id"])
        n+=1               

records = []

for country, years in country_inter_output.items():
    for year, values in years.items():
        for value in values:
            records.append([country,value,values[value],year])

df = pd.DataFrame(records)
df.columns=["source","target","weight","year"]
df_sorted = df.sort_values(by=['source', 'year'])

df_sorted.to_csv("Data/collab_output.csv")

# %%
