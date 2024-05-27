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
collection = db["works_UBI"]

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



#%% get share evolution

year2expe = defaultdict(int)
year2theory = defaultdict(int)
docs = collection.find()
for doc in tqdm.tqdm(docs):
    try:
        if "experiment" in clear_text(doc["title"]):
            year2expe[doc["publication_year"]] += 1
        else:
            year2theory[doc["publication_year"]] += 1
    except:
        pass

df1 = pd.DataFrame.from_dict(year2expe, orient='index', columns=['expe'])
df2 = pd.DataFrame.from_dict(year2theory, orient='index', columns=['theoric'])


df = pd.concat([df1, df2], axis=1)
df_sorted = df.sort_index()
df_sorted = df_sorted.fillna(0)
df_sorted["share_expe"] = df_sorted["expe"]/df_sorted["theoric"]
df_sorted["share_expe"].mean()


df_sorted["Year"] = df_sorted.index
df_sorted.to_csv("Data/Fig1_a.csv",index=False)
