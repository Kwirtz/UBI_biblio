#%% Init
import re
import tqdm
import json
import langid
import pymongo
import requests
from datetime import datetime

def remove_special_characters(text):
    # Define a regex pattern to match special characters
    pattern = r'[^a-zA-Z0-9\s]'
    # Substitute special characters with an empty string
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def delete_old_papers(db_name, collection_name):
    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client[db_name]
    collection = db[collection_name]
    
    # Define the query to match documents with publication_year before 1900
    query = {"publication_year": {"$lt": 1900}}
    
    # Perform the delete operation
    result = collection.delete_many(query)
    
    # Print the number of deleted documents
    print(f"Deleted {result.deleted_count} documents.")
            
def delete_latin_papers(db_name, collection_name):
    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client[db_name]
    collection = db[collection_name]
    
    # Fetch all documents from the collection
    docs = collection.find({})
    
    # Initialize a counter for the number of deleted documents
    delete_count = 0
    
    # Iterate through the documents and check the title language
    for doc in tqdm.tqdm(docs):
        title = doc["title"]
        if title:
            language, _ = langid.classify(title)
            if language == "la":  # 'la' is the language code for Latin
                collection.delete_one({"_id": doc["_id"]})
                delete_count += 1
    
    print(f"Deleted {delete_count} documents with Latin titles.")

def delete_duplicate_titles(db_name, collection_name):
    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client[db_name]
    collection = db[collection_name]
    
    # Fetch all documents from the collection
    docs = collection.find({})
    
    # Create a dictionary to track titles and their corresponding _id values
    title_dict = {}
    delete_count = 0
    
    for doc in tqdm.tqdm(docs):
        title = doc["title"]
        if title:
            if title in title_dict:
                # If title already exists in the dictionary, delete the current document
                #collection.delete_one({"_id": doc["_id"]})
                print(doc["id"])
                delete_count += 1
            else:
                # Otherwise, add the title to the dictionary
                title_dict[title] = doc["_id"]
    
    print(f"Deleted {delete_count} documents with duplicate titles.")

def check_for_doi_in_abstract(db_name, collection_name):
    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client[db_name]
    collection = db[collection_name]
    docs = collection.find({})
    
    ids = []
    for doc in tqdm.tqdm(docs):
        # Check if the abstract field contains "https://doi.org"
        if "abstract" in doc and "https://doi.org" in doc["abstract"]:
            # Update the document to set abstract to ""
            collection.update_one({"_id": doc["_id"]}, {"$set": {"abstract": ""}})
            ids.append(doc["id"])
    
    return ids


def delete_dups(db_name, collection_name):
    Client = pymongo.MongoClient("mongodb://localhost:27017")
    db = Client[db_name]
    collection = db[collection_name]
    docs = collection.find({})
    ids = []
    for doc in tqdm.tqdm(docs):
        ids.append(doc["id"])
    for id_ in ids:
        docs = list(collection.find({"id":id_}))
        if len(docs) > 1:
            docs.sort(key=lambda x: datetime.fromisoformat(x['updated_date']), reverse=True)
            # Keep the most recent document and delete the rest
            for doc_to_delete in docs[1:]:
                collection.delete_one({"_id": doc_to_delete["_id"]})




# Call the function to delete old papers
delete_old_papers(db_name="UBI", collection_name="works_UBI_gobu_2")
delete_duplicate_titles("UBI", "works_UBI_gobu_2")
delete_latin_papers(db_name="UBI", collection_name="works_UBI_gobu_2")
test = check_for_doi_in_abstract(db_name="UBI", collection_name="works_UBI_gobu_2")
delete_dups(db_name = "UBI", collection_name = "works_UBI_gobu_2")

# MongoDB connection
Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["UBI"]
collection = db["works_UBI"]

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
collection_new = db["works_UBI_global_2"]

data = []
for paper in tqdm.tqdm(list_papers):
    doc = collection_all.find_one({"id":paper})
    if doc:
        data.append(doc)

collection_new.insert_many(data)



check = ["state bonus", "national dividend", "social dividend", "basic minimum income", "basic income"," ubi ",
         "negative income tax", "minimum income guarantee", "guaranteed minimum income", "basic income guarantee", "demogrant", "guaranteed income", "credit income tax",
         "citizen’s basic income", "citizen’s income", 
         "unconditional basic income", "universal basic income", "guaranteed minimum income", "social dividend", "basic income guarantee"]


collection_new_new = db["works_UBI_gobu_2"]
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
    text = title + " " + abstract
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
        

        