import tqdm
import pymongo
import pandas as pd

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['UBI']
collection = db['works_UBI_gobu_2']

# NUmber of papers (journal articles/books), number of journals, number of authors
# Number of authors, cited by, countries, references, number of concept spanned, open_access, preprint, journal_articles, books



#%% Number of papers


collection.count_documents({})

#%% Number of Journals

journals = []

docs = collection.find()
for doc in tqdm.tqdm(docs):
    if doc["locations"]:
        for location in doc["locations"]:
            journal = False
            try:
                if location["source"]["type"] == "journal":
                    journal = True
            except:
                pass
            if journal:
                try:
                    journals.append(location["source"]["display_name"])
                except Exception as e:
                    print(str(e))
                    
len(list(set(journals)))


#%% Number of Authors

author_list = []

docs = collection.find()
for doc in tqdm.tqdm(docs):
    authors = doc["authorships"]
    for author in authors:
        name = author["author"]["display_name"]
        author_list.append(name)
         
len(list(set(author_list)))

#%% full_tables

list_of_insertion = []

docs = collection.find()
for doc in tqdm.tqdm(docs):
    countries = doc["countries_distinct_count"]
    if countries == 0:
        countries = None
    institutions = doc["institutions_distinct_count"]
    if institutions == 0:
        institutions = None
    is_oa = doc["open_access"]["is_oa"]
    n_ref = doc["referenced_works_count"]
    if n_ref == 0:
        n_ref = None
    n_concepts = doc["concepts_count"]
    n_authors = doc["authors_count"]
    if n_authors == 0:
        n_authors = None
    n_citations = doc["cited_by_count"]
    is_journal = 0
    is_book = 0
    is_preprint = 0
    if doc["type"] == "preprint":
        is_preprint = 1
    if doc["type"] == "book":
        is_book = 1
    if doc["type"] == "article":
        is_journal = 1
    list_of_insertion.append([n_authors, n_citations, countries, n_ref, n_concepts, is_oa, is_preprint, is_journal, is_book])
    

df = pd.DataFrame(list_of_insertion, columns = ["Nb. Authors", "Nb. Citations", "Nb. Countries", "Nb. references",
                                                "Nb. Concepts", "Open Access", "Preprint","Journal article", "Book"])

df["Open Access"] = df["Open Access"].astype(int)      

def calculate_stats(series):
    return pd.Series({
        'Mean': series.mean(skipna=True),
        'SD': series.std(skipna=True),
        'Min': series.min(skipna=True),
        'Q1': series.quantile(0.25, interpolation='linear'),
        'Median': series.median(skipna=True),
        'Q3': series.quantile(0.75, interpolation='linear'),
        'Max': series.max(skipna=True)
    })

stats_df = df.apply(calculate_stats).T


# Assuming stats_df is your DataFrame
latex_table = stats_df.to_latex(float_format="%.2f")

# Print or save latex_table to a .tex file
print(latex_table)
