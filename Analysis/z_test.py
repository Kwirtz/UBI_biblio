import tqdm
import pickle
import pymongo

# MongoDB connection
Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["UBI"]
collection = db["works_UBI_gobu_2"]

with open('Data/commu2papers.pkl', 'rb') as f:
    commu2papers = pickle.load(f)

docs = collection.find()
for doc in tqdm.tqdm(docs):
    if doc["id"] in commu2papers[4]:
        year = doc["publication_year"]
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
        if "africa" in text:
            print(doc["id"])
