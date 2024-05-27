#%% Init

import tqdm
import pymongo
import pandas as pd
from collections import defaultdict

journal_pub = defaultdict(lambda:defaultdict(int))

# MongoDB connection
Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["UBI"]
collection = db["works_UBI"]

test = []

docs = collection.find()
for doc in tqdm.tqdm(docs):
    try:
        journal_pub[doc["host_venue"]["display_name"]][doc["publication_year"]] += 1
    except:
        test.append(doc["id"])
        pass
    
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
