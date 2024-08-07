import re
import tqdm
import pickle
import pymongo
import pandas as pd

with open('Data/commu2papers.pkl', 'rb') as f:
    commu2papers = pickle.load(f)

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['UBI']
collection = db['works_UBI_gobu_2']


docs = collection.find()
data = []
for doc in tqdm.tqdm(docs):
    data.append({"id":doc["id"],"id_cleaned":doc["id_cleaned"],"year":doc["publication_year"]})

df = pd.DataFrame(data,columns=["id", "id_cleaned","year"])

#%% novelty evolution

indicators = ['foster','lee','wang',"uzzi"]
data = []

for indicator in tqdm.tqdm(indicators):
    if indicator == "wang":
        collection = db['output_{}_referenced_works_add_3_1_restricted50'.format(indicator)]
    else:
        collection = db['output_{}_referenced_works_add'.format(indicator)]
    docs = collection.find({})
    for doc in docs:
        if indicator=="wang":
            score = doc["referenced_works_add_wang_3_1_restricted50"]["score"]["novelty"]
        else:
            score = doc["referenced_works_add_{}".format(indicator)]["score"]["novelty"]
        id_cleaned = doc["id_cleaned"]
        data.append({"id_cleaned":id_cleaned,"score_{}".format(indicator):score})
    df_temp = pd.DataFrame(data)
    data = []
    df = df.merge(df_temp,how="inner",on="id_cleaned")
    

#%% Add disruptiveness

collection = db["output_disruptiveness"]
docs = collection.find()

data = []
for doc in tqdm.tqdm(docs):
    data.append({"id_cleaned":doc["id_cleaned"],
                 "DI1":doc["disruptiveness"]["DI1"],
                 "DI5":doc["disruptiveness"]["DI5"],
                 "DI5nok":doc["disruptiveness"]["DI5nok"],
                 "DI1nok":doc["disruptiveness"]["DI1nok"],
                 "Breadth":doc["disruptiveness"]["Breadth"],
                 "Depth":doc["disruptiveness"]["Depth"]})

df_temp = pd.DataFrame(data)
df = df.merge(df_temp,how="inner",on="id_cleaned")

df.to_csv("Data/Fig_creativity.csv",index=False)




df["Creative"] = df["score_foster"] +df["DI5nok"]
# Sort DataFrame by 'Count' column in descending order
df_sorted = df.sort_values(by='Creative', ascending=False)

# Calculate the number of rows that correspond to the top 10%
top_10_percent_count = int(0.1 * len(df_sorted))

# Select the top 10% of rows
top_10_percent_df = df_sorted.head(top_10_percent_count)

filtered_df = top_10_percent_df[top_10_percent_df['id'].isin(commu2papers[4])]
