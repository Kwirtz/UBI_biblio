import tqdm
import json
import pickle
import pymongo
import requests
import pandas as pd


start_year = 1960
end_year = 2020

query = {
    'publication_year': {
        '$gte': start_year,
        '$lte': end_year
    }
}


client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['UBI']
collection = db['works_UBI_gobu_2']

df = pd.read_csv("Data/national_output_bi.csv")
country2commu = {}
for row in df.iterrows():
    country2commu[row[1]["country"]] = row[1]["Community"]






#keywords = ["basic income", "negative income tax", "minimum income guarantee"]
keywords = ["periodic", "cash payment", "individual", "universal", "unconditional", "regularly", "in kind"]

#%% Topics to keywords


with open('Data/commu2papers.pkl', 'rb') as f:
    commu2papers = pickle.load(f)


# Initialize an empty list to hold the final data
data = []


# Iterate over documents
docs = collection.find(query)
for doc in tqdm.tqdm(docs):
    if doc.get("has_fulltext"):
        for community, papers in commu2papers.items():
            if doc["id"] in papers:
                topic_community = community
        authors = doc["authorships"]
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
    
        full_gram = ""
        if doc.get("has_fulltext"):
            doc_id = doc["id"].split("/")[-1]
            response = requests.get(f"https://api.openalex.org/works/{doc_id}/ngrams?mailto=kevin-w@hotmail.fr")
            ngrams = json.loads(response.content).get("ngrams", [])
            for gram in ngrams:
                if gram['ngram_tokens'] < 4:
                    full_gram += gram["ngram"].lower() + " "
    
        text = (title + " " + abstract + " " + full_gram).lower()
    
        for keyword in keywords:
            if keyword in text:
                data.append([year, keyword, topic_community])

df_final = pd.DataFrame(data, columns=["year", "definition", "topic"])
df_pivot = df_final.pivot_table(index=["year", "definition"], columns="topic", aggfunc='size', fill_value=0)
df_pivot.columns = ["communities_" + str(col) for col in df_pivot.columns]
# Reset the index to move 'year' and 'type' back to columns
df_pivot.reset_index(inplace=True)

df_pivot.to_csv("Data/Fig_topics2keywords_v2.csv",index=False)

#%% Countries to keywords

# Initialize an empty list to hold the final data
data = []


# Iterate over documents
docs = collection.find(query)
for doc in tqdm.tqdm(docs):
    if doc.get("has_fulltext"):
        country_list = []
        authors = doc["authorships"]
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
    
        full_gram = ""
        if doc.get("has_fulltext"):
            doc_id = doc["id"].split("/")[-1]
            response = requests.get(f"https://api.openalex.org/works/{doc_id}/ngrams?mailto=kevin-w@hotmail.fr")
            ngrams = json.loads(response.content).get("ngrams", [])
            for gram in ngrams:
                if gram['ngram_tokens'] < 4:
                    full_gram += gram["ngram"].lower() + " "
    
        text = (title + " " + abstract + " " + full_gram).lower()
    
        # Collect countries
        for author in authors:
            try:
                country = author["institutions"][0]["country_code"]
                if country:
                    country_list.append(country2commu[country])
            except Exception as e:
                pass
    
        if len(set(country_list)) == 1:  # If all authors are from the same country
            for keyword in keywords:
                if keyword in text:
                    data.append([year, keyword, country_list[0]])

df_final = pd.DataFrame(data, columns=["year", "definition", "country"])
df_pivot = df_final.pivot_table(index=["year", "definition"], columns="country", aggfunc='size', fill_value=0)
df_pivot.columns = ["countries_" + str(col) for col in df_pivot.columns]
# Reset the index to move 'year' and 'type' back to columns
df_pivot.reset_index(inplace=True)
df_pivot.to_csv("Data/Fig_countries2keywords_v2.csv",index=False)

#%% Full definition

keywords = ["unconditional", "universal", "individual", "cash payment", "regular"]

with open('Data/commu2papers.pkl', 'rb') as f:
    commu2papers = pickle.load(f)


# Initialize an empty list to hold the final data
data = []

ids_fulldef = []

# Iterate over documents
docs = collection.find(query)
for doc in tqdm.tqdm(docs):
    if doc.get("has_fulltext"):
        for community, papers in commu2papers.items():
            if doc["id"] in papers:
                topic_community = community
        authors = doc["authorships"]
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
    
        full_gram = ""
        if doc.get("has_fulltext"):
            doc_id = doc["id"].split("/")[-1]
            response = requests.get(f"https://api.openalex.org/works/{doc_id}/ngrams?mailto=kevin-w@hotmail.fr")
            ngrams = json.loads(response.content).get("ngrams", [])
            for gram in ngrams:
                if gram['ngram_tokens'] < 4:
                    full_gram += gram["ngram"].lower() + " "
    
        text = (full_gram).lower()
        
        unconditional = 0
        universal = 0
        individual = 0
        cash_payment = 0
        regular = 0
        
        for i in ["unconditional","no conditions","without conditions","regardless of income"]:
            if len(i.split(" ")) > 1:
                if i in text:
                    unconditional = 1
            else:
                if i in text.split(" "):
                    unconditional = 1
        for i in ["universal","for everyone","across all demographics"]:
            if len(i.split(" ")) > 1:
                if i in text:
                    universal = 1
            else:
                if i in text.split(" "):
                    universal = 1
        for i in ["individual","per person","individually","per individual"]:
            if len(i.split(" ")) > 1:
                if i in text:
                    individual = 1
            else:
                if i in text.split(" "):
                    individual = 1
        for i in ["cash payment","cash","direct payment","monetary transfer"]:
            if len(i.split(" ")) > 1:
                if i in text:
                    cash_payment = 1
            else:
                if i in text.split(" "):
                    cash_payment = 1
        for i in ["regular","regular intervals","periodic payments","consistent"]:
            if len(i.split(" ")) > 1:
                if i in text:
                    regular = 1
            else: 
                if i in text.split(" "):
                    regular = 1     
        total_def = unconditional + universal + individual + cash_payment + regular
        if total_def == 5:
            keyword = "Full_def"
            ids_fulldef.append(str(doc["id"]) + " " + str(topic_community) + " " + str(year))
        elif total_def > 0 :
            keyword = "Partial_def"
        else:
            keyword = "no_def"
    
        data.append([year, keyword, topic_community])

df_final = pd.DataFrame(data, columns=["year", "definition", "topic"])
df_pivot = df_final.pivot_table(index=["year", "definition"], columns="topic", aggfunc='size', fill_value=0)
df_pivot.columns = ["communities_" + str(col) for col in df_pivot.columns]
# Reset the index to move 'year' and 'type' back to columns
df_pivot.reset_index(inplace=True)

df_pivot.to_csv("Data/Fig_topics2definitions_v2.csv",index=False)

with open("Data/ids_fulldef.txt", "w") as file:
    for item in ids_fulldef:
        file.write(f"{item}\n")