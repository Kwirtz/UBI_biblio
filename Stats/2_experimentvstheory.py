#%% Init

import tqdm
import string
import pymongo
import pandas as pd
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()




# MongoDB connection
Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["UBI"]
collection = db["works_UBI_gobu_2"]

def clear_text(text):
    # Remove leading and trailing spaces
    text = text.strip()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if len(token) > 1]
    
    # Remove punctuation from the text
    lemmatized_text = ''.join([char.lower() for char in ' '.join(lemmatized_tokens) if char not in string.punctuation])
    
    cleaned_text = ' '.join(lemmatized_text.split())
    return cleaned_text 

"""
experimental_keywords = [
    "experiment", "experimental", "simulation", "trial", "sample", "sampling", "data collection", "experimental setup", "prototype",
    "implementation", "performance", "practical", "real-world", "field study", "case study",
    "survey", "questionnaire", "experimental results", "apparatus", "instrumentation", "laboratory", "clinical trial",
    "in vitro", "in vivo", "pilot study", "experimentation", "empirical study", "rct", "experience"
]
"""

experimental_keywords = [
    " experiment ", "simulation", "sample", "sampling", "experimental setup", "field study", "case study", "pilot",
    "survey", "questionnaire", "experimental results", "laboratory", "pilot study", "experimentation", "empirical study", "rct", "experience", "randomized controlled trial"
]

experimental_keywords = [
    "experiment", "sample", "sampling", "field study", "case study", 
    "survey", "questionnaire", "laboratory", "pilot study", "experimentation", "empirical study", "rct", "experience", "randomized controlled trial",
    "evaluating","setting","scenario", "pilot", "design"
]



#%% get share evolution

year2expe = defaultdict(int)
year2theory = defaultdict(int)
docs = collection.find()
for doc in tqdm.tqdm(docs):
    year = doc["publication_year"]
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
    for keyword in experimental_keywords:
        for word in clear_text(text).split(" "):
            if keyword == word:
                done = True
                if year<1950:
                    print(doc["id"],clear_text(text),keyword)
    if done == True:
        year2expe[year] += 1
    else:
        year2theory[year] += 1


df1 = pd.DataFrame.from_dict(year2expe, orient='index', columns=['expe'])
df2 = pd.DataFrame.from_dict(year2theory, orient='index', columns=['theoric'])


df = pd.concat([df1, df2], axis=1)
df_sorted = df.sort_index()
df_sorted = df_sorted.fillna(0)
df_sorted["share_expe"] = df_sorted["expe"]/(df_sorted["theoric"]+df_sorted["expe"])
df_sorted["share_expe"].mean()


df_sorted["Year"] = df_sorted.index
df_sorted.to_csv("Data/Fig2_a.csv",index=False)
