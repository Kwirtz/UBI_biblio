#%% Init

import re
import tqdm
import pymongo
import pandas as pd
import numpy as np
from collections import defaultdict, Counter


# MongoDB connection
Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["UBI"]
collection = db["works_UBI_global"]


#%% Journal overtime

journal_pub = defaultdict(lambda:defaultdict(int)) 

test = []

docs = collection.find()
for doc in tqdm.tqdm(docs):
    try:
        journal_pub[doc["primary_location"]["source"]][doc["publication_year"]] += 1
    except:
        test.append(doc["id"])
    
records = []
for journal, years in journal_pub.items():
    for year, value in years.items():
        records.append([journal,year,value])

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(records)
df.columns = ["Journal_name","Year","N_pub"]

# Group by "Journal_name" and sum "N_pub" for each group
grouped_df = df.groupby("Journal_name")["N_pub"].sum()

# Sort the grouped DataFrame in descending order and get the top 10
top_10_journals = grouped_df.sort_values(ascending=False).head(10)

print(top_10_journals)
grouped_df.sum()


journals = grouped_df.sort_values(ascending=False)
journals["Journals"] = journals.index
journals.to_csv("Data/Fig1_b.csv", index= False)

#%% 4x4 plot

journals = []

docs = collection.find()
for doc in tqdm.tqdm(docs):
    if doc["locations"]:
        for location in doc["locations"]:
            journal = False
            try:
                if location["source"]["type"] == "journal":
                    journal = True
            except:
                pass
            if journal:
                try:
                    journals.append(location["source"]["display_name"])
                except Exception as e:
                    print(str(e))
    
# Count the occurrences of each journal
journal_counts = Counter(journals)

# Get the 16 most common journals
most_common_journals = journal_counts.most_common(16)
most_common_journals = [i[0] for i in most_common_journals] 


df_journals = pd.DataFrame(index = most_common_journals, columns = ["n_theo","n_expe"]).fillna(0)

# Repeat dataframe for each year
df_temp = df_journals.reset_index().rename(columns={'index': 'journal'})
years = pd.DataFrame({'year': range(1900, 2021)})

df_temp['key'] = 1
years['key'] = 1
df_expanded = pd.merge(df_temp, years, on='key').drop('key', axis=1)
df_expanded = df_expanded[['journal', 'year', 'n_theo', 'n_expe']]


experimental_keywords = [
    "experiment", "experimental", "measurement", "measured", "observation", "observed",
    "empirical", "test", "testing", "simulation", "validation", "trial", "evaluation",
    "analysis", "sample", "sampling", "data collection", "experimental setup", "prototype",
    "implementation", "performance", "practical", "real-world", "field study", "case study",
    "survey", "questionnaire", "experimental results", "apparatus", "instrumentation", "laboratory", "clinical trial",
    "in vitro", "in vivo", "pilot study", "experimentation", "empirical study", "rct", "experience"
]


docs = collection.find()
for doc in tqdm.tqdm(docs):
    done = False
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
    if doc["locations"]:
        for location in doc["locations"]:
            journal = False
            try:
                if location["source"]["type"] == "journal":
                    journal = True
            except:
                pass
            if journal:
                try:
                    if location["source"]["display_name"] in most_common_journals:
                        if len([i for i in experimental_keywords if i in text]) > 0 and done == False:
                            condition = (df_expanded['journal'] == location["source"]["display_name"]) & (df_expanded['year'] == year)
                            df_expanded.loc[condition, "n_expe"] += 1
                            df_journals.at[location["source"]["display_name"],"n_expe"] += 1
                            done = True
                        elif done == False:
                            condition = (df_expanded['journal'] == location["source"]["display_name"]) & (df_expanded['year'] == year)
                            df_expanded.loc[condition, "n_theo"] += 1
                            df_journals.at[location["source"]["display_name"],"n_theo"] += 1
                            done = True
                except Exception as e:
                    print(str(e))
    
df_journals["Journal"] = df_journals.index

def clean_text(text):
    # Remove special characters
    clean_text = re.sub(r'[^\x00-\x7F]+', '', text)
    clean_text = clean_text.split("/")[0]
    return clean_text


df_expanded['journal'] = df_expanded['journal'].apply(clean_text)
df_expanded.to_csv("Data/Fig2_ab.csv",index=False,encoding="utf-8")
df_journals['Journal'] = df_journals['Journal'].apply(clean_text)
df_journals.to_csv("Data/Fig2_b.csv",index=False,encoding="utf-8")




