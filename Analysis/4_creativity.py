import re
import tqdm
import pymongo
import novelpy


#%% cooc

concept_cooc = novelpy.utils.cooc_utils.create_cooc(
                client_name="mongodb://localhost:27017",
                db_name = "UBI",    
                collection_name = "works_UBI_gobu",
                year_var="publication_year",
                var = "concepts",
                sub_var = "display_name",
                weighted_network = True, self_loop = True)

concept_cooc.main()

concept_cooc = novelpy.utils.cooc_utils.create_cooc(
                client_name="mongodb://localhost:27017",
                db_name = "UBI",    
                collection_name = "works_UBI_gobu",
                year_var="publication_year",
                var = "concepts",
                sub_var = "display_name",
                weighted_network = False, self_loop = False)

concept_cooc.main()


#%% Novelty indicators



# Foster et al.(2015) Meshterms_sample
for focal_year in tqdm.tqdm(range(1950,2021), desc = "Computing indicator for window of time"):
    try:
        Foster = novelpy.indicators.Foster2015( client_name="mongodb://localhost:27017",
                                           db_name = "UBI",    
                                           collection_name = "works_UBI_gobu",
                                           id_variable = 'id_cleaned',
                                           year_variable = 'publication_year',
                                           variable = "concepts",
                                           sub_variable = "display_name",
                                           focal_year = focal_year,
                                           community_algorithm = "Louvain",
                                           density = False)
        Foster.get_indicator()
    except:
        continue

# Lee et al.(2015) Meshterms_sample
for focal_year in tqdm.tqdm(range(1950,2021), desc = "Computing indicator for window of time"):
    try:
        Lee = novelpy.indicators.Lee2015(client_name="mongodb://localhost:27017",
                                           db_name = "UBI",    
                                           collection_name = "works_UBI_gobu",
                                           id_variable = 'id_cleaned',
                                           year_variable = 'publication_year',
                                           variable = "concepts",
                                           sub_variable = "display_name",
                                            focal_year = focal_year,
                                            density = True)
        Lee.get_indicator()
    except:
        continue


# Wang et al.(2017) Meshterms_sample
for focal_year in tqdm.tqdm(range(1950,2021)):
    try:
        Wang = novelpy.indicators.Wang2017( client_name="mongodb://localhost:27017",
                                            db_name = "UBI",    
                                            collection_name = "works_UBI_gobu",
                                            id_variable = 'id_cleaned',
                                            year_variable = 'publication_year',
                                            variable = "concepts",
                                            sub_variable = "display_name",
                                            focal_year = focal_year,
                                            time_window_cooc = 3,
                                            n_reutilisation = 1,
                                            density = True)
        Wang.get_indicator()
    except:
        continue


#%% Disruptiveness Indicators

client_name = 'mongodb://localhost:27017'
db_name = 'UBI'
collection_name = "works_UBI_gobu_2"

Client = pymongo.MongoClient(client_name)
db = Client[db_name]
collection  = db[collection_name]

docs = collection.find({},no_cursor_timeout=True)

list_of_insertion = []
for doc in tqdm.tqdm(docs):
    refs = doc["referenced_works"]
    refs_cleaned = [int(re.findall(r'\d+', i)[0]) for i in refs]
    list_of_insertion.append(
        pymongo.UpdateOne({"id_cleaned": doc["id_cleaned"]}, 
                           {"$set":{"referenced_cleaned": refs_cleaned}})
        )
    if len(list_of_insertion) == 1000:
        collection.bulk_write(list_of_insertion)
        list_of_insertion = []
collection.bulk_write(list_of_insertion)
list_of_insertion = []

clean = novelpy.utils.preprocess_disruptiveness.create_citation_network(client_name = 'mongodb://localhost:27017',
                                                                        db_name = 'UBI',
                                                                        collection_name = "works_UBI_gobu_2",
                                                                        id_variable = "id_cleaned",
                                                                        year_variable = "publication_year",
                                                                        variable = "referenced_cleaned"
                                                                        )
clean.id2citedby()
clean.update_db()

for focal_year in tqdm.tqdm(range(1950,2021)):
    try:
        disruptiveness = novelpy.Disruptiveness(
            client_name = 'mongodb://localhost:27017',
            db_name = 'UBI',
            collection_name = 'works_UBI_gobu_2_cleaned',
            focal_year = focal_year,
            id_variable = 'id_cleaned',
            variable = "citations",
            refs_list_variable ='refs',
            cits_list_variable = 'cited_by',
            year_variable = 'publication_year')
    
        disruptiveness.get_indicators()
    except:
        continue




