#%% Init



import re
import tqdm
import string
import pymongo
import pickle
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter

start_year = 1960
end_year = 2020

query = {
    'publication_year': {
        '$gte': start_year,
        '$lte': end_year
    }
}

lemmatizer = WordNetLemmatizer()

def clear_text(text):
    # Remove leading and trailing spaces
    text = text.strip()
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if len(token) > 1]
    
    # Remove punctuation from the text
    lemmatized_text = ''.join([char for char in ' '.join(lemmatized_tokens) if char not in string.punctuation])
    
    cleaned_text = ' '.join(lemmatized_text.split())
    return cleaned_text 

# MongoDB connection
Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["UBI"]
collection = db["works_UBI_gobu_2"]

with open('Data/commu2papers.pkl', 'rb') as f:
    commu2papers = pickle.load(f)

#%% Journal overtime

journal_pub = defaultdict(lambda:defaultdict(int)) 

test = []

docs = collection.find()
for doc in tqdm.tqdm(docs):
    try:
        journal_pub[doc["primary_location"]["source"]][doc["publication_year"]] += 1
    except:
        test.append(doc["id"])
    
records = []
for journal, years in journal_pub.items():
    for year, value in years.items():
        records.append([journal,year,value])

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(records)
df.columns = ["Journal_name","Year","N_pub"]

# Group by "Journal_name" and sum "N_pub" for each group
grouped_df = df.groupby("Journal_name")["N_pub"].sum()

# Sort the grouped DataFrame in descending order and get the top 10
top_10_journals = grouped_df.sort_values(ascending=False).head(10)

print(top_10_journals)
grouped_df.sum()


journals = grouped_df.sort_values(ascending=False)
journals["Journals"] = journals.index
journals.to_csv("Data/Fig1_b.csv", index= False)

#%% 4x4 plot

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
    
# Count the occurrences of each journal
journal_counts = Counter(journals)

# Get the 16 most common journals
most_common_journals = journal_counts.most_common(20)
most_common_journals = [i[0] for i in most_common_journals] 


df_journals = pd.DataFrame(index = most_common_journals, columns = ["n_theo","n_expe"]).fillna(0)

# Repeat dataframe for each year
df_temp = df_journals.reset_index().rename(columns={'index': 'journal'})
years = pd.DataFrame({'year': range(1900, 2021)})

df_temp['key'] = 1
years['key'] = 1
df_expanded = pd.merge(df_temp, years, on='key').drop('key', axis=1)
df_expanded = df_expanded[['journal', 'year', 'n_theo', 'n_expe']]


experimental_keywords = [
    "experiment", "simulation", "sample", "sampling", "field study", "case study", 
    "survey", "questionnaire", "laboratory", "pilot study", "experimentation", "empirical study", "rct", "experience", "randomized controlled trial",
    "evaluating","setting","scenario", "pilot", "design"
]




docs = collection.find()
for doc in tqdm.tqdm(docs):
    done = False
    try:
        title = doc["title"]
    except:
        title = ""
    try:
        abstract = doc["abstract"]
    except:
        abstract = ""
    year = doc["publication_year"]
    if not title:
        title = ""
    if not abstract:
        abstract = ""
    text = title + " " + abstract
    text = text.lower()    
    text = clear_text(text)
    if doc["locations"]:
        for location in doc["locations"]:
            journal = False
            try:
                if location["source"]["type"] == "repository":
                    journal = True
            except:
                pass
            if journal:
                try:
                    if location["source"]["display_name"] in most_common_journals:
                        if len([i for i in experimental_keywords if i in text.split(" ")]) > 0 and done == False:
                            condition = (df_expanded['journal'] == location["source"]["display_name"]) & (df_expanded['year'] == year)
                            df_expanded.loc[condition, "n_expe"] += 1
                            df_journals.at[location["source"]["display_name"],"n_expe"] += 1
                            done = True
                        elif done == False:
                            condition = (df_expanded['journal'] == location["source"]["display_name"]) & (df_expanded['year'] == year)
                            df_expanded.loc[condition, "n_theo"] += 1
                            df_journals.at[location["source"]["display_name"],"n_theo"] += 1
                            done = True
                except Exception as e:
                    print(str(e))
    
df_journals["Journal"] = df_journals.index

def clean_text(text):
    # Remove special characters
    clean_text = re.sub(r'[^\x00-\x7F]+', '', text)
    clean_text = clean_text.split("/")[0]
    return clean_text


df_expanded['journal'] = df_expanded['journal'].apply(clean_text)
df_expanded.to_csv("Data/Fig2_ab.csv",index=False,encoding="utf-8")
df_journals['Journal'] = df_journals['Journal'].apply(clean_text)
df_journals.to_csv("Data/Fig2_b.csv",index=False,encoding="utf-8")

#%% 4x4 plot expe Journal

experimental_keywords = [
    "experiment", "sample", "sampling", "field study", "case study", 
    "survey", "questionnaire", "laboratory", "pilot study", "experimentation", "empirical study", "rct", "experience", "randomized controlled trial", 
    ]




journals = []

docs = collection.find(query)
for doc in tqdm.tqdm(docs):
    done = False
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
    text = title + " " + abstract 
    text = text.lower()        
    text = clear_text(text)    
    if len([i for i in experimental_keywords if i in text.split(" ")]) == 0 and done == False:
        if doc["locations"]:
            for location in doc["locations"]:
                journal = False
                try:
                    if location["source"]["type"] == "journal":
                        journal = True
                    else:
                        pass
                        #print(location["source"]["type"],location["source"]["display_name"],doc["id"],"\n")
                except:
                    pass
                if journal:
                    try:
                        journals.append(location["source"]["display_name"])
                    except Exception as e:
                        print(str(e))
    
# Count the occurrences of each journal
journal_counts = Counter(journals)

# Get the 16 most common journals
most_common_journals = journal_counts.most_common(20)
most_common_journals = [i[0] for i in most_common_journals] 


df_journals = pd.DataFrame(index = most_common_journals, columns = ["n_theo","n_expe"]).fillna(0)

# Repeat dataframe for each year
df_temp = df_journals.reset_index().rename(columns={'index': 'journal'})
years = pd.DataFrame({'year': range(1900, 2021)})

df_temp['key'] = 1
years['key'] = 1
df_expanded = pd.merge(df_temp, years, on='key').drop('key', axis=1)
df_expanded = df_expanded[['journal', 'year', 'n_theo', 'n_expe']]


docs = collection.find(query)
for doc in tqdm.tqdm(docs):
    done = False
    try:
        title = doc["title"]
    except:
        title = ""
    try:
        abstract = doc["abstract"]
    except:
        abstract = ""
    year = doc["publication_year"]
    if not title:
        title = ""
    if not abstract:
        abstract = ""  
    text = title + " " + abstract 
    text = text.lower()        
    text = clear_text(text)     
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
                    if location["source"]["display_name"] in most_common_journals:
                        if len([i for i in experimental_keywords if i in text.split(" ")]) > 0 or len([i for i in experimental_keywords if i in location["source"]["display_name"].lower()]) > 0 and done == False:
                            condition = (df_expanded['journal'] == location["source"]["display_name"]) & (df_expanded['year'] == year)
                            df_expanded.loc[condition, "n_expe"] += 1
                            df_journals.at[location["source"]["display_name"],"n_expe"] += 1
                            done = True
                        elif done == False:
                            condition = (df_expanded['journal'] == location["source"]["display_name"]) & (df_expanded['year'] == year)
                            df_expanded.loc[condition, "n_theo"] += 1
                            df_journals.at[location["source"]["display_name"],"n_theo"] += 1
                            done = True
                except Exception as e:
                    print(str(e))
    
df_journals["Journal"] = df_journals.index

def clean_text(text):
    # Remove special characters
    clean_text = re.sub(r'[^\x00-\x7F]+', '', text)
    clean_text = clean_text.split("/")[0]
    return clean_text


df_expanded['journal'] = df_expanded['journal'].apply(clean_text)
df_expanded.to_csv("Data/Fig2_ab.csv",index=False,encoding="utf-8")
df_journals['Journal'] = df_journals['Journal'].apply(clean_text)
df_journals.to_csv("Data/Fig2_b.csv",index=False,encoding="utf-8")

#%% 4x4 plot expe repository

experimental_keywords = [
    "experiment", "sample", "sampling", "field study", "case study", 
    "survey", "questionnaire", "laboratory", "pilot study", "experimentation", "empirical study", "rct", "experience", "randomized controlled trial"
]


journals = []

docs = collection.find(query)
for doc in tqdm.tqdm(docs):
    done = False
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
    text = title + " " + abstract
    text = clear_text(text)     
    if len([i for i in experimental_keywords if i in text]) > 0 and done == False:
        if doc["locations"]:
            for location in doc["locations"]:
                journal = False
                try:
                    if location["source"]["type"] == "repository":
                        journal = True
                    else:
                        pass
                        #print(location["source"]["type"],location["source"]["display_name"],doc["id"],"\n")
                except:
                    pass
                if journal:
                    try:
                        journals.append(location["source"]["display_name"])
                    except Exception as e:
                        print(str(e))
    
# Count the occurrences of each journal
journal_counts = Counter(journals)

# Get the 16 most common journals
most_common_journals = journal_counts.most_common(20)
most_common_journals = [i[0] for i in most_common_journals] 


df_journals = pd.DataFrame(index = most_common_journals, columns = ["n_theo","n_expe"]).fillna(0)

# Repeat dataframe for each year
df_temp = df_journals.reset_index().rename(columns={'index': 'journal'})
years = pd.DataFrame({'year': range(1900, 2021)})

df_temp['key'] = 1
years['key'] = 1
df_expanded = pd.merge(df_temp, years, on='key').drop('key', axis=1)
df_expanded = df_expanded[['journal', 'year', 'n_theo', 'n_expe']]

n_list = []
docs = collection.find(query)
for doc in tqdm.tqdm(docs):
    n =0
    done = False
    try:
        title = doc["title"]
    except:
        title = ""
    try:
        abstract = doc["abstract"]
    except:
        abstract = ""
    year = doc["publication_year"]
    if not title:
        title = ""
    if not abstract:
        abstract = ""
    text = title + " " + abstract  
    text = clear_text(text)
    if doc["locations"]:
        for location in doc["locations"]:
            journal = False
            try:
                if location["source"]["type"] == "repository":
                    n += 1
                    journal = True
            except:
                pass
            if journal:
                try:
                    if location["source"]["display_name"] in most_common_journals:
                        if len([i for i in experimental_keywords if i in text.split(" ")]) > 0  or len([i for i in experimental_keywords if i in location["source"]["display_name"].lower()]) > 0 and done == False:
                            condition = (df_expanded['journal'] == location["source"]["display_name"]) & (df_expanded['year'] == year)
                            df_expanded.loc[condition, "n_expe"] += 1
                            df_journals.at[location["source"]["display_name"],"n_expe"] += 1
                            done = True
                        elif done == False:
                            if location["source"]["display_name"] == "AEA Randomized Controlled Trials":
                                print(doc["id"])
                            condition = (df_expanded['journal'] == location["source"]["display_name"]) & (df_expanded['year'] == year)
                            df_expanded.loc[condition, "n_theo"] += 1
                            df_journals.at[location["source"]["display_name"],"n_theo"] += 1
                            done = True
                except Exception as e:
                    print(str(e))
    n_list.append(n)
  
df_journals["Journal"] = df_journals.index

def clean_text(text):
    # Remove text inside parentheses
    text = re.sub(r'\([^)]*\)', '', text)
    # Remove special characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Remove text after "/"
    text = text.split("/")[0].strip()
    return text

df_journals['Journal'] = df_journals['Journal'].apply(clean_text)

df_journals.to_csv("Data/Fig2_c.csv",index=False,encoding="utf-8")
