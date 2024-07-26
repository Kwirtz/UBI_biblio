import tqdm
import pymongo


client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['UBI']
collection = db['works_UBI_gobu_2']
db_total = client['openAlex20240517']
collection_total = db_total['works']

list_papers = {}
docs = collection.find()
for doc in tqdm.tqdm(docs):
    if doc["referenced_works"]:
        list_papers[doc["id"]] = doc["referenced_works"]

for doc in tqdm.tqdm(list_papers):
    to_update = []
    for ref in list_papers[doc]:
        doc_temp = collection_total.find_one({"id":ref})
        try:
            issn = doc_temp["primary_location"]["source"]["issn_l"]
            year = doc_temp["publication_year"]
            if issn and year:
                to_update.append({"issn":issn ,"year":year}) 
        except:
            continue
    collection.find_one_and_update({"id":doc},{'$set': {
                      'referenced_works_add': to_update
                      }
                  }, upsert=False)
    
    

