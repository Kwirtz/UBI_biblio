import re
import tqdm
import pymongo
import pandas as pd

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['UBI']
collection = db['works_UBI_gobu']


docs = collection.find()
data = []
for doc in tqdm.tqdm(docs):
    data.append({"id_cleaned":doc["id_cleaned"],"year":doc["publication_year"]})

df = pd.DataFrame(data,columns=["id_cleaned","year"])

#%% novelty evolution

indicators = ['foster','lee','wang']
data = []

for indicator in tqdm.tqdm(indicators):
    if indicator == "wang":
        collection = db['output_{}_concepts_3_1_restricted50'.format(indicator)]
    else:
        collection = db['output_{}_concepts'.format(indicator)]
    docs = collection.find({})
    for doc in docs:
        if indicator=="wang":
            score = doc["concepts_wang_3_1_restricted50"]["score"]["novelty"]
        else:
            score = doc["concepts_{}".format(indicator)]["score"]["novelty"]
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

df.columns
