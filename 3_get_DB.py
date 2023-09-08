import json
import tqdm
import pickle
import pymongo


with open('Data/dict_publication.pickle', 'rb') as f:
    dict_publication = pickle.load(f)  

Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["openAlex"]
collection = db["works"]

new_pubs = []
for ind in tqdm.tqdm(dict_publication):
    new_pubs += dict_publication[ind]

pubs_to_get = list(set(new_pubs))


batch = 0
list_of_insertion = []
for pub in tqdm.tqdm(pubs_to_get):
    list_of_insertion.append(collection.find_one({"id_cleaned":pub},{"_id":0}))
    if len(list_of_insertion) % 100000 == 0:
        with open('Data/batch_{}.json'.format(batch), 'w') as fout:
            json.dump(list_of_insertion, fout)
        batch += 1
        list_of_insertion = []
        
with open('Data/batch_{}.json'.format(batch), 'w') as fout:
    json.dump(list_of_insertion, fout)
    


