#Bon alors pour les vagues, je te propose sans pause 1960 à 1980 puis 1981 à 2010 puis 2011
#si tu peux des plus grosses pauses entre les périodes, 1960-1975, 1980-2005 et 2010
#%% init
import tqdm
import nltk
import spacy
import pickle
import random
import pymongo
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
from scipy.spatial.distance import cdist
from langdetect import detect
from collections import Counter
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer



#%% Train
# Download the punkt tokenizer if not already downloaded

#nltk.download('punkt')
#nltk.download('stopwords')
# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['UBI']
collection = db['works_UBI_gobu_2']


with open('Data/commu2papers.pkl', 'rb') as f:
    commu2papers = pickle.load(f)



# Function to check if a text is in English
def is_english(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return False


#%% Own model

documents = []
documents_doi = []
documents_title = []
documents_id = []
documents_period = []
community = 2
# Iterate over the collection once
for doc in tqdm.tqdm(collection.find({"abstract": {"$exists": 1}})):
    if doc["id"] in commu2papers[community]:
        if doc["publication_year"] >= 1960 and doc["publication_year"] <= 2020:
            if is_english(doc["abstract"]) and is_english(doc["title"]):
                documents.append(doc["abstract"])
                documents_doi.append(doc["doi"])
                documents_title.append(doc["title"])
                documents_id.append(doc["id"])
                if doc["publication_year"] >= 1960 and doc["publication_year"] <= 1980:
                    documents_period.append("Period1")
                if doc["publication_year"] >= 1981 and doc["publication_year"] <= 2000:
                    documents_period.append("Period2")
                if doc["publication_year"] >= 2001 and doc["publication_year"] <= 2020:
                    documents_period.append("Period3")
         

# Tokenize the abstracts into words
tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_documents = [[word for word in doc if word not in stop_words] for doc in tokenized_documents]

# Flatten the list of filtered words
all_words = [word for doc in filtered_documents for word in doc]

# Filter words with frequency higher than 10
word_counts = Counter(all_words)
filtered_words = [word for word, count in word_counts.items() if count > 0]

# Filter tokenized documents to keep only relevant words
filtered_tokenized_documents = [[word for word in doc if word in filtered_words] for doc in filtered_documents]

# Train Word2Vec model
model = Word2Vec(sentences=filtered_tokenized_documents, vector_size=100, window=2, min_count=1, workers=4)

# Save the trained model
model.save("Data/word2vec_model.model")

# Get word vectors and words
words = list(model.wv.key_to_index.keys())
vectors = [model.wv[word] for word in words]

### Centroid docs
# Compute document centroids
document_centroids = []
for doc in filtered_tokenized_documents:
    # Filter out words not present in the Word2Vec model
    valid_words = [word for word in doc if word in model.wv]
    
    # Calculate the mean vector for the document
    if valid_words:
        centroid = np.mean([model.wv[word] for word in valid_words], axis=0)
        document_centroids.append(centroid)
    else:
        # If no valid words, add a zero vector as the centroid
        document_centroids.append(np.zeros(model.vector_size))

# Convert the list of centroids to a numpy array
document_centroids = np.array(document_centroids)

### Data Viz cluster doc

tsne = TSNE(n_components=2)
centroids_tsne = tsne.fit_transform(document_centroids)

# Create DataFrame for Plotly
df = pd.DataFrame({'X': centroids_tsne[:, 0], 'Y': centroids_tsne[:, 1],
                   'Cluster': documents_period, 'Abstract': documents, "DOI":documents_doi,"title":documents_title,"id_":documents_id})



# Create an interactive scatter plot
fig = px.scatter(df, x='X', y='Y', color='Cluster', hover_data=['DOI'], title='t-SNE Visualization of Document Centroids')

# Save the interactive plot as an HTML file
fig.write_html("Results/Figures/TSNE_visualization.html")


df.to_csv("Data/doc_centroid.csv",index=False)




#%% Spacy
community = 2
nlp = spacy.load("Data/en_core_sci_lg-0.5.3/en_core_sci_lg/en_core_sci_lg-0.5.3")
documents = []
documents_doi = []
documents_title = []
documents_id = []
documents_period = []
documents_centroid = []
# Iterate over the collection once
for doc in tqdm.tqdm(collection.find({"abstract": {"$exists": 1}})):
    if doc["id"] in commu2papers[community]:
        if doc["publication_year"] >= 1960 and doc["publication_year"] <= 2020:
            if is_english(doc["abstract"]) and is_english(doc["title"]):
                documents.append(doc["abstract"])
                documents_doi.append(doc["doi"])
                documents_title.append(doc["title"])
                documents_id.append(doc["id"])
                if doc["publication_year"] >= 1960 and doc["publication_year"] <= 1980:
                    documents_period.append("Period1")
                if doc["publication_year"] >= 1981 and doc["publication_year"] <= 2000:
                    documents_period.append("Period2")
                if doc["publication_year"] >= 2001 and doc["publication_year"] <= 2020:
                    documents_period.append("Period3")       
                tokens = nlp(doc["abstract"])
                article_title_centroid = np.sum([t.vector for t in tokens], axis=0) / len(tokens)
                #article_title_centroid = article_title_centroid.tolist()
                documents_centroid.append(article_title_centroid)

documents_centroid = np.array(documents_centroid)
tsne = TSNE(n_components=3)
centroids_tsne = tsne.fit_transform(documents_centroid)

# Create DataFrame for Plotly
df = pd.DataFrame({
    'X': centroids_tsne[:, 0],
    'Y': centroids_tsne[:, 1],
    'Z': centroids_tsne[:, 2],
    'Cluster': documents_period,
    'Abstract': documents,
    "DOI": documents_doi,
    "title": documents_title,
    "id_": documents_id
})

# Create an interactive 3D scatter plot
fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='Cluster', hover_data=['DOI'], title='t-SNE Visualization of Document Centroids')

# Save the interactive plot as an HTML file
fig.write_html("Results/Figures/TSNE_visualization_3D.html")


df.to_csv("Data/doc_centroid.csv",index=False)
