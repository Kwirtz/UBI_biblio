import re
import tqdm
import pickle
import pymongo


Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["openAlex"]
collection = db["works"]

with open('Data/list_aid_ai.pickle', 'rb') as f:
    list_aid = pickle.load(f)  
    
dict_publication = {aid:[] for aid in list_aid}
docs = collection.find()


for doc in tqdm.tqdm(docs):
    for author in doc["authorships"]:
        try:
            id_ = int(re.findall(r'\d+',author["author"]["id"] )[0])
            dict_publication[id_].append(doc["id_cleaned"])
        except:
            continue

with open('Data/dict_publication.pickle', 'wb') as handle:
    pickle.dump(dict_publication, handle, protocol=pickle.HIGHEST_PROTOCOL)  