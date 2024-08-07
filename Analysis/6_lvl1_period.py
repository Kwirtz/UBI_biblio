import tqdm
import pymongo
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['UBI']
collection_works_UBI_global = db['works_UBI_global']
collection_works_UBI_gobu = db['works_UBI_gobu']
db_concept = client['openAlex20240517']
collection_concept = db_concept['concepts']


lvl1 = []


docs = collection_concept.find()
for doc in tqdm.tqdm(docs):
    if doc["level"] == 1:
        lvl1.append(doc["display_name"])
        
years = range(1900, 2021)
co_occurrence_dict = {
    year: pd.DataFrame(0, index=lvl1, columns=lvl1) for year in years
}

# Function to update the co-occurrence matrix
def update_co_occurrence_matrix(year, concepts):
    combinations = itertools.combinations(concepts, 2)
    for concept1, concept2 in combinations:
        co_occurrence_dict[year].loc[concept1, concept2] += 1
        co_occurrence_dict[year].loc[concept2, concept1] += 1

def log_transform(matrix):
    return np.log1p(matrix)  # log1p is log(1 + x) to handle log(0)


# Iterate over each document in works_UBI_gobu
for doc in tqdm.tqdm(collection_works_UBI_gobu.find(), desc="Processing documents"):
    year = doc["publication_year"]
    if year and 1900 <= year <= 2020:
        references = doc['referenced_works']
        if references:
            for reference_id in references:
                referenced_doc = collection_works_UBI_global.find_one({"id": reference_id})
                if referenced_doc:
                    concepts = referenced_doc["concepts"]
                    lvl1_concepts_in_doc = [concept['display_name'] for concept in concepts if concept['level'] == 1]
                    if len(lvl1_concepts_in_doc) > 1:
                        update_co_occurrence_matrix(year, lvl1_concepts_in_doc)

correlations = []

for year in tqdm.tqdm(range(1945,2021)):
    past_matrix = pd.DataFrame(0, index=lvl1, columns=lvl1)
    for i in range(year-20,year-5):
        past_matrix += co_occurrence_dict[i]
    past_matrix = log_transform(past_matrix)
    current_matrix = log_transform(co_occurrence_dict[year])
    
    # Flatten the dataframes
    flat_df1 = past_matrix.values.flatten()
    flat_df2 = current_matrix.values.flatten()
    
    # Calculate the Pearson correlation coefficient
    correlation = np.corrcoef(flat_df1, flat_df2)[0, 1]
    correlations.append([correlation,year])

df = pd.DataFrame(correlations, columns = ["corr","year"])
    
