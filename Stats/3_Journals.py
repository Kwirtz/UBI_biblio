#%% Init

import tqdm
import pymongo
import pandas as pd
import numpy as np
from collections import defaultdict, Counter


# MongoDB connection
Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["UBI"]
collection = db["works_UBI_20240517"]


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


experimental_keywords = [
    "experiment", "experimental", "measurement", "measured", "observation", "observed",
    "empirical", "test", "testing", "simulation", "validation", "trial", "evaluation",
    "analysis", "sample", "sampling", "data collection", "experimental setup", "prototype",
    "implementation", "performance", "practical", "real-world", "field study", "case study",
    "survey", "questionnaire", "experimental results", "apparatus", "instrumentation", "laboratory", "clinical trial",
    "in vitro", "in vivo", "pilot study", "experimentation", "empirical study"
]


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
    




