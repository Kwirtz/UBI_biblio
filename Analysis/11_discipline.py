import tqdm
import pickle
import pymongo
import numpy as np
import pandas as pd

# MongoDB connection
Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["UBI"]
collection = db["works_UBI_gobu_2"]


with open('Data/commu2papers.pkl', 'rb') as f:
    commu2papers = pickle.load(f)
    

list_of_insertion = []
docs = collection.find()
for doc in tqdm.tqdm(docs):
    year = doc["publication_year"]    
    if year >=1960 and year <= 2020:
        for commu in commu2papers:
            if doc["id"] in commu2papers[commu]:
                done = False
                for concept in doc["concepts"]:
                    if concept["level"] == 1 and done == False:
                        discipline = concept["display_name"]
                        done = True
                        list_of_insertion.append([commu,discipline,year])
                    
df = pd.DataFrame(list_of_insertion,columns=["commu","discipline","year"])
df["value"] = 1

df.to_csv("Data/commu2discipline.csv",index=False)
