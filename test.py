import json
import pymongo

with open('Data/batch_0.json', 'r') as f:
    test = json.load(f)
    
    
import requests

list_ids = []

cursor = '*'
while cursor:
    response = requests.get("https://api.openalex.org/works?filter=locations.source.issn:1932-0183", params = {
         'cursor': cursor,
         'per_page': 20
     })
    data = response.json()
    cursor = data['meta']['next_cursor']
    for doc in data["results"]:
        list_ids.append(doc["id"])
        
Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["UBI"]
collection = db["works_UBI_gobu"]

list_ids_gobu = []
docs = collection.find({})
for doc in docs:
    list_ids_gobu.append(doc["id"])
    
    
rest = [i.split("/")[-1] for i in list_ids if i not in list_ids_gobu]

for i in rest:
    response = requests.get("https://api.openalex.org/works/{}".format(i))
    data = response.json()
    print(data["title"],data["id"])


db = Client["openAlex20240517"]
collection = db["works_SHS"]

doc = collection.find_one({"id":"https://openalex.org/W4311881878"})


import re

def remove_special_characters(text):
    # Define a regex pattern to match special characters
    pattern = r'[^a-zA-Z0-9\s]'
    # Substitute special characters with an empty string
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

keywords = ["state bonus", "minimum income", "national dividend", "social dividend", "basic minimum income", "basic income", " ubi ",
         "negative income tax", "minimum income guarantee", "guaranteed minimum income", "basic income guarantee", "demogrant", "guaranteed income", "credit income tax",
         "citizen’s basic income", "citizen’s income", "unconditional basic income", "universal basic income", "negative income tax", "guaranteed minimum income", "social dividend", "basic income guarantee"]
     
n = 0
try:
    title = doc["title"]
except:
    title = ""
try:
    abstract = doc["abstract"]
except:
    abstract = ""
if not title:
    title = ""
if not abstract:
    abstract = ""
text = title #+ " " + abstract
text = text.lower()     
text = remove_special_characters(text)


text = "Unconditional Endowment and Acceptance of Taxes: A Lab-in-the-Field Experiment on UBI with Unemployed"
text =  text.lower()   
for keyword in keywords:
    if keyword in text:
        n += 1
        print(keyword)