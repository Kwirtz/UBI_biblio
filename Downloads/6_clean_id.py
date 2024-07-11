# create a cleaned_id to make query faster in Mongo (from str to int with less redudancy)

import re
import tqdm
import pymongo

client_name = 'mongodb://localhost:27017'
db_name = 'UBI'

def convert(client_name,db_name,collection_name,id_variable):
    Client = pymongo.MongoClient(client_name)
    db = Client[db_name]
    collection  = db[collection_name]
    collection.create_index([ ("id_cleaned",1) ])   
    docs = collection.find({},no_cursor_timeout=True)
    list_of_insertion = []
    for doc in tqdm.tqdm(docs):
        id_ = int(re.findall(r'\d+', doc[id_variable] )[0])
        list_of_insertion.append(
            pymongo.UpdateOne({id_variable: doc[id_variable]}, 
                               {"$set":{"id_cleaned": id_}})
            )
        if len(list_of_insertion) == 1000:
            collection.bulk_write(list_of_insertion)
            list_of_insertion = []
    collection.bulk_write(list_of_insertion)
    list_of_insertion = []    
    
if __name__ == "__main__":
    convert(client_name,db_name,"works_UBI_global","id")
