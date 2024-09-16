import re
import tqdm
import pymongo

Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["openAlex20240517"]
collection = db["works"]
db_new = Client["openAlex20240517"]
collection_eco = db_new["works_SHS"]


def remove_special_characters(text):
    # Define a regex pattern to match special characters
    pattern = r'[^a-zA-Z0-9\s]'
    # Substitute special characters with an empty string
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

# UBI 
#keywords = ["economics"]
# SHS 
keywords = ["philosophy"," sociology", "history"," geography","psychology","economics","political science","art","business"]



def get_ubi(keywords):    
    try:
        collection_eco.create_index([ ("id",1) ])   
        collection_eco.create_index([ ("publication_year",1) ])         
    except:
        pass
    docs = collection.find({})
    list_of_insertion = []
    for doc in tqdm.tqdm(docs):
        n = 0
        for concept in doc["concepts"]:
            if concept["display_name"].lower() in keywords:
                n += 1
        if n > 0 :
            try:
                list_of_insertion.append(doc)
            except:
                continue
    
        if len(list_of_insertion) == 10000:
            collection_eco.insert_many(list_of_insertion)
            list_of_insertion = []            
    collection_eco.insert_many(list_of_insertion)
    list_of_insertion = []

get_ubi(keywords)


def check_for_dups(db_name, collection_name):
    Client = pymongo.MongoClient("mongodb://localhost:27017")
    db = Client[db_name]
    collection = db[collection_name]
    docs = collection.find({})
    ids = []
    for doc in tqdm.tqdm(docs):
        ids.append(doc["id"])
    if len(ids) != len(list(set(ids))):
        print("WOUAH WATCH OUT")
    else:
        print("RAS")
    return ids

test = check_for_dups(db_name = "UBI", collection_name = "works_UBI_global")



#%% 

db = Client["openAlex20240517"]
collection = db["works_SHS"]
db_new = Client["UBI"]
collection_eco = db_new["works_UBI_general"]

keywords = ["state bonus", "national dividend", "social dividend", "basic minimum income", "basic income",
         "negative income tax", "minimum income guarantee", "guaranteed minimum income", "basic income guarantee", "demogrant", "guaranteed income", "credit income tax",
         "citizen’s basic income", "citizen’s income", 
         "unconditional basic income", "universal basic income", "guaranteed minimum income", "social dividend", "basic income guarantee"]

#keywords = ["basic income"]

def get_ubi_in_text(keywords):    
    try:
        collection_eco.create_index([ ("id",1) ])   
        collection_eco.create_index([ ("publication_year",1) ])         
    except:
        pass
    docs = collection.find({})
    list_of_insertion = []
    for doc in tqdm.tqdm(docs):
        n = 0
        for concept in doc["concepts"]:
            if concept["display_name"].lower() in keywords:
                n += 1
        
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

        for keyword in keywords:
            if keyword in text:
                print(keyword)
                n += 1
        
        if n > 0 :
            try:
                list_of_insertion.append(doc)
            except:
                continue
    
        if len(list_of_insertion) == 10000:
            collection_eco.insert_many(list_of_insertion)
            list_of_insertion = []            
    collection_eco.insert_many(list_of_insertion)
    list_of_insertion = []

get_ubi_in_text(keywords)

