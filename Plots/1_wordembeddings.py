import tqdm
import pymongo
import webbrowser
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from langdetect import detect
from collections import Counter
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize

from langdetect import DetectorFactory
DetectorFactory.seed = 0

#%% Train
# Download the punkt tokenizer if not already downloaded

#nltk.download('punkt')
#nltk.download('stopwords')
# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['UBI']
collection = db['works_UBI']


# Function to check if a text is in English
def is_english(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return False

# Retrieve abstracts from MongoDB
documents = [doc["abstract"] for doc in collection.find({"abstract":{"$exists":1}}) if is_english(doc["abstract"])]
documents_doi = [doc["doi"] for doc in collection.find({"abstract":{"$exists":1}}) if is_english(doc["abstract"])]
documents_title = [doc["title"] for doc in collection.find({"abstract":{"$exists":1}}) if is_english(doc["abstract"])]
# Tokenize the abstracts into words
tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_documents = [[word for word in doc if word not in stop_words] for doc in tokenized_documents]

# Flatten the list of filtered words
all_words = [word for doc in filtered_documents for word in doc]

# Filter words with frequency higher than 10
word_counts = Counter(all_words)
filtered_words = [word for word, count in word_counts.items() if count > 10]

# Filter tokenized documents to keep only relevant words
filtered_tokenized_documents = [[word for word in doc if word in filtered_words] for doc in filtered_documents]

# Train Word2Vec model
model = Word2Vec(sentences=filtered_tokenized_documents, vector_size=100, window=5, min_count=1, workers=4)

# Save the trained model
model.save("word2vec_model.model")

# Get word vectors and words
words = list(model.wv.key_to_index.keys())
vectors = [model.wv[word] for word in words]


#%% Centroid docs
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

# Cluster documents using KMeans
k = 5  # You can change the number of clusters as needed
kmeans = KMeans(n_clusters=k, random_state= 12345)
cluster_labels = kmeans.fit_predict(document_centroids)

# Assign cluster labels to documents
clustered_documents = [{'abstract': doc, 'cluster': cluster,"doi":doi,"title":title} for doc, cluster, doi, title 
                       in zip(documents, cluster_labels,documents_doi,documents_title)]

# Print or store the results as needed
for document in clustered_documents:
    print(f"Cluster {document['cluster']}:\n{document['abstract']}\n{'-'*50}")

#%% Data Viz cluster doc

tsne = TSNE(n_components=2, random_state=12345)
centroids_tsne = tsne.fit_transform(document_centroids)

# Create DataFrame for Plotly
df = pd.DataFrame({'X': centroids_tsne[:, 0], 'Y': centroids_tsne[:, 1],
                   'Cluster': cluster_labels, 'Abstract': documents, "DOI":documents_doi,"title":documents_title})



# Create an interactive scatter plot
fig = px.scatter(df, x='X', y='Y', color='Cluster', hover_data=['DOI'], title='t-SNE Visualization of Document Centroids')

# Save the interactive plot as an HTML file
fig.write_html("Results/Figures/document_centroids_visualization.html")


df.to_csv("Data/doc_centroid.csv",index=False)

#%% Data Viz embedding space

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=12345)
vectors_tsne = tsne.fit_transform(vectors)

# Create a DataFrame for Plotly
df = pd.DataFrame({'Word': words, 'X': vectors_tsne[:, 0], 'Y': vectors_tsne[:, 1]})

# Create an interactive scatter plot
fig = px.scatter(df, x='X', y='Y', text='Word', title='t-SNE Visualization of Word Embeddings')

# Save the interactive plot as an HTML file
fig.write_html("word_embeddings_visualization.html")


# Create a DataFrame for Plotly
df = pd.DataFrame({'Word': words, 'X': vectors_tsne[:, 0], 'Y': vectors_tsne[:, 1]})

# Create an interactive scatter plot with names appearing on hover
fig = px.scatter(df, x='X', y='Y', text='Word', title='t-SNE Visualization of Word Embeddings',
                 hover_data=['Word'])

# Update the hover text to an empty string
fig.update_traces(text='')

# Save the interactive plot as an HTML file
fig.write_html("Results/Figures/word_embeddings_visualization.html")