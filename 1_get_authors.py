import re
import tqdm
import json
import pickle
import pymongo

def get_list_eco_researcher():
    
    Client = pymongo.MongoClient("mongodb://localhost:27017")
    db = Client["openAlex"]
    collection = db["works_eco"]
    
    list_aid = set()
    docs = collection.find()
    for doc in tqdm.tqdm(docs):
        for author in doc["authorships"]:
            try:
                id_ = int(re.findall(r'\d+',author["author"]["id"] )[0])
                list_aid.add(id_)
            except:
                continue
    list_aid = list(list_aid)
    with open('Data/list_aid_eco.pickle', 'wb') as handle:
        pickle.dump(list_aid, handle, protocol=pickle.HIGHEST_PROTOCOL)  

        
def get_eco_researcher_metadata():

    with open('Data/list_aid_eco.pickle', 'rb') as f:
        list_aid_eco = pickle.load(f)  
        
    Client = pymongo.MongoClient("mongodb://localhost:27017")
    db = Client["openAlex"]
    collection = db["authors"]
    
    list_of_insertion = []
    for author in tqdm.tqdm(list_aid_eco):
        doc = collection.find_one({"id_cleaned":author},{"_id":0})
        list_of_insertion.append(doc)   
        
    with open('Data/author.json', 'w') as fout:
        json.dump(list_of_insertion, fout) 
