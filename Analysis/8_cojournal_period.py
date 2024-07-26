import tqdm
import novelpy
import pymongo
import pandas as pd
import matplotlib.pyplot as plt

ref_cooc = novelpy.utils.cooc_utils.create_cooc(
                 client_name = "mongodb://localhost:27017",
                 db_name = "UBI",
                 collection_name = "works_UBI_gobu_2",
                 year_var="publication_year",
                 var = "referenced_works_add",
                 sub_var = "issn",
                 time_window = range(1900,2021),
                 weighted_network = False, self_loop = False)

ref_cooc.main()

#%% Novelty indicators



# Uzzi et al.(2013) Meshterms_sample
for focal_year in tqdm.tqdm(range(1950,2021), desc = "Computing indicator for window of time"):
    try:
        Uzzi = novelpy.indicators.Uzzi2013(client_name="mongodb://localhost:27017",
                                           db_name = "UBI",    
                                           collection_name = "works_UBI_gobu_2",
                                           id_variable = 'id_cleaned',
                                           year_variable = 'publication_year',
                                           variable = "referenced_works_add",
                                           sub_variable = "issn",
                                               focal_year = focal_year,
                                               density = True)
        Uzzi.get_indicator()
    except:
        continue

# Foster et al.(2015) 
for focal_year in tqdm.tqdm(range(1950,2021), desc = "Computing indicator for window of time"):
    try:
        Foster = novelpy.indicators.Foster2015( client_name="mongodb://localhost:27017",
                                           db_name = "UBI",    
                                           collection_name = "works_UBI_gobu_2",
                                           id_variable = 'id_cleaned',
                                           year_variable = 'publication_year',
                                           variable = "referenced_works_add",
                                           sub_variable = "issn",
                                           focal_year = focal_year,
                                           community_algorithm = "Louvain",
                                           density = False)
        Foster.get_indicator()
    except:
        continue

# Lee et al.(2015) 
for focal_year in tqdm.tqdm(range(1950,2021), desc = "Computing indicator for window of time"):
    try:
        Lee = novelpy.indicators.Lee2015(client_name="mongodb://localhost:27017",
                                           db_name = "UBI",    
                                           collection_name = "works_UBI_gobu_2",
                                           id_variable = 'id_cleaned',
                                           year_variable = 'publication_year',
                                           variable = "referenced_works_add",
                                           sub_variable = "issn",
                                            focal_year = focal_year,
                                            density = True)
        Lee.get_indicator()
    except:
        continue


# Wang et al.(2017) 
for focal_year in tqdm.tqdm(range(1950,2021)):
    try:
        Wang = novelpy.indicators.Wang2017( client_name="mongodb://localhost:27017",
                                            db_name = "UBI",    
                                            collection_name = "works_UBI_gobu_2",
                                            id_variable = 'id_cleaned',
                                            year_variable = 'publication_year',
                                            variable = "referenced_works_add",
                                            sub_variable = "issn",
                                            focal_year = focal_year,
                                            time_window_cooc = 3,
                                            n_reutilisation = 1,
                                            density = True)
        Wang.get_indicator()
    except Exception as e:
        print(str(e))


#%% Get stats on novelty

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['UBI']
collection = db['works_UBI_gobu_2']


docs = collection.find()
data = []
for doc in tqdm.tqdm(docs):
    data.append({"id_cleaned":doc["id_cleaned"],"year":doc["publication_year"]})

df = pd.DataFrame(data,columns=["id_cleaned","year"])


indicators = ['foster','lee','wang',"uzzi"]
data = []

for indicator in tqdm.tqdm(indicators):
    if indicator == "wang":
        collection = db['output_{}_referenced_works_add_3_1_restricted50'.format(indicator)]
    else:
        collection = db['output_{}_referenced_works_add'.format(indicator)]
    docs = collection.find({})
    for doc in docs:
        if indicator=="wang":
            score = doc["referenced_works_add_wang_3_1_restricted50"]["score"]["novelty"]
        else:
            score = doc["referenced_works_add_{}".format(indicator)]["score"]["novelty"]
        id_cleaned = doc["id_cleaned"]
        data.append({"id_cleaned":id_cleaned,"score_{}".format(indicator):score})
    df_temp = pd.DataFrame(data)
    data = []
    df = df.merge(df_temp,how="inner",on="id_cleaned")
    
df.to_csv("Data/Fig_creativity.csv")

