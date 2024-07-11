#%% Init
import re
import tqdm
import json
import pymongo
import requests

def remove_special_characters(text):
    # Define a regex pattern to match special characters
    pattern = r'[^a-zA-Z0-9\s]'
    # Substitute special characters with an empty string
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

# MongoDB connection
Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["UBI"]
collection = db["works_UBI_general"]

list_papers = []
docs = collection.find()

for doc in tqdm.tqdm(docs):
    list_papers.append(doc["id"])
    if doc["referenced_works"]:
        for ref in doc["referenced_works"]:
            list_papers.append(ref)
            
list_papers = list(set(list_papers))

db_all = Client["openAlex20240517"]
collection_all = db_all["works_SHS"]
collection_new = db["works_UBI_global"]

data = []
for paper in tqdm.tqdm(list_papers):
    doc = collection_all.find_one({"id":paper})
    if doc:
        data.append(doc)

collection_new.insert_many(data)



check = ["state bonus", "minimum income", "national dividend", "social dividend", "basic minimum income", "basic income"," ubi ",
         "negative income tax", "minimum income guarantee", "guaranteed minimum income", "basic income guarantee", "demogrant", "guaranteed income", "credit income tax",
         "citizen’s basic income", "citizen’s income", 
         "unconditional basic income", "universal basic income", "guaranteed minimum income", "social dividend", "basic income guarantee"]


collection_new_new = db["works_UBI_gobu"]
docs = collection_new.find({})

list_papers = []
for doc in tqdm.tqdm(docs):
    list_papers.append(doc["id"])


list_insertion = []
for paper in tqdm.tqdm(list_papers):
    doc = collection_new.find_one({"id":paper})
    n = 0
    year = doc["publication_year"]

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
    text = title + " " #+ abstract
    text = text.lower()
    text = remove_special_characters(text)
    full_gram = ""
    if doc["has_fulltext"]:
        doc_id = doc["id"].split("/")[-1]
        response = requests.get("https://api.openalex.org/works/{}/ngrams?mailto=kevin-w@hotmail.fr".format(doc_id))
        ngrams = json.loads(response.content)["ngrams"]
        for gram in ngrams:
            if gram['ngram_tokens']<4:
                full_gram += gram["ngram"].lower() + " "
    for concept in doc["concepts"]:
        if concept["display_name"].lower() in check:
            n += 1
    full_text = text +  " " + full_gram
    for keyword in check:
        if keyword in full_text:
            n += 1 
    if n >0:
        list_insertion.append(doc)

collection_new_new.insert_many(list_insertion)
        
        
            
            
        

            