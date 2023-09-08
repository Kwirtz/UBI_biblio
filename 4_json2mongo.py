import glob
import json
import pymongo

Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["openAlex"]
collection = db["works"]

for file in glob.glob("Data/works/*"):
    with open(file, 'r') as f:
        list_of_insertion = json.load(f)
    collection.insert_many(list_of_insertion)