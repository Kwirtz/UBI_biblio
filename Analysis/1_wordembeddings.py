#%% init
import tqdm
import nltk
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


# Function to check if a text is in English
def is_english(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return False

documents = []
documents_doi = []
documents_title = []
documents_id = []

# Iterate over the collection once
for doc in tqdm.tqdm(collection.find({"abstract": {"$exists": 1}})):
    if is_english(doc["abstract"]) and is_english(doc["title"]):
        documents.append(doc["abstract"])
        documents_doi.append(doc["doi"])
        documents_title.append(doc["title"])
        documents_id.append(doc["id"])
        
        
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
model.save("Data/word2vec_model.model")

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

# Test for best clustering

def test_K_means(document_centroids, documents, documents_doi, documents_title):
    cluster_values = [2, 3, 4, 5, 6,7,8,9,10,11,12]
    num_trials = 50
    
    data = []

    def get_membership_matrices(num_nodes, cluster_labels):
        num_communities = np.max(cluster_labels) + 1
        membership_matrix = np.zeros((num_nodes, num_communities))
        for node_idx, community in enumerate(cluster_labels):
            membership_matrix[node_idx, community] = 1
        return membership_matrix
    
    def get_vi(X, Y):
        N = X.shape[0]  
        p = np.sum(X, axis=0) / N  
        q = np.sum(Y, axis=0) / N  
        R = np.dot(X.T, Y) / N  
        
        VI = - R * (np.log(R / p[:, None]) + np.log(R / q[None, :]))
        VI[np.isnan(VI)] = 0  
        return np.sum(VI)  
    
    def run_kmeans_opti(document_centroids, k, num_trials):
        VIs = []
        membership_matrices = []
        num_nodes = len(document_centroids)
        for _ in range(num_trials):
            kmeans = KMeans(n_clusters=k)
            cluster_labels = kmeans.fit_predict(document_centroids)
            membership_matrix = get_membership_matrices(num_nodes, cluster_labels)
            membership_matrices.append(membership_matrix)
        for i, community1 in enumerate(membership_matrices):
            for j, community2 in enumerate(membership_matrices):
                if i != j:  # Avoid comparing the same membership matrix with itself
                    VI = get_vi(community1, community2)
                    VIs.append(VI)
        return np.mean(VIs)
        
    for k in tqdm.tqdm(cluster_values):
        VI = run_kmeans_opti(document_centroids, k, num_trials)
        data.append({"VI": VI, "k": k})
    
    return data

#results = test_K_means(document_centroids, documents, documents_doi, documents_title)
#print(results)

def calculate_wss(data, n_clusters):
    """ Calculate WSS (within-cluster sum of squares) for k clusters """
    kmeans = KMeans(n_clusters=n_clusters, random_state=12345)
    kmeans.fit(data)
    centers = kmeans.cluster_centers_
    wss = np.sum(np.min(cdist(data, centers, 'euclidean'), axis=1)) / data.shape[0]
    return wss

def generate_reference_data(data, n_refs=10):
    """ Generate reference datasets """
    ref_datasets = []
    for _ in range(n_refs):
        random_data = np.random.random_sample(size=data.shape)
        ref_datasets.append(random_data)
    return ref_datasets

def gap_statistic(data, k_range, n_refs=10):
    """ Calculate gap statistic for different k values """
    gaps = []
    s_k = np.zeros(len(k_range))
    wss_orig = []
    wss_refs = np.zeros((len(k_range), n_refs))
    
    # Calculate WSS for original data
    for i, k in enumerate(k_range):
        wss_orig.append(calculate_wss(data, k))
    
    # Generate reference datasets and calculate WSS for them
    ref_datasets = generate_reference_data(data, n_refs=n_refs)
    for j, ref_data in enumerate(ref_datasets):
        for i, k in enumerate(k_range):
            wss_refs[i, j] = calculate_wss(ref_data, k)
    
    # Calculate gaps
    for i, k in enumerate(k_range):
        log_wss_orig = np.log(wss_orig[i])
        log_wss_refs = np.log(wss_refs[i, :])
        gap = np.mean(log_wss_refs) - log_wss_orig
        s_k[i] = np.sqrt(1 + 1/n_refs) * np.std(log_wss_refs)
        gaps.append(gap)
    
    optimal_k = k_range[np.argmax(gaps - s_k)]
    return gaps, s_k, optimal_k

# Example usage:
k_range = range(1, 40)  # Adjust the range as needed
gaps, s_k, optimal_k = gap_statistic(document_centroids, k_range)


# Print the optimal number of clusters
print(f"Optimal number of clusters found by Gap Statistic: {optimal_k}")

# Plotting the gap values
plt.figure(figsize=(10, 6))
plt.errorbar(k_range, gaps, yerr=s_k, fmt='-o', capsize=5)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Gap Statistic')
plt.title('Gap Statistic vs. Number of clusters')
plt.show()

# Fit KMeans with the optimal number of clusters
kmeans = KMeans(n_clusters=6, random_state=12345)
cluster_labels = kmeans.fit_predict(document_centroids)

# Assuming documents, documents_doi, and documents_title are defined earlier
clustered_documents = [{'abstract': doc, 'cluster': cluster, 'doi': doi, 'title': title} 
                       for doc, cluster, doi, title in zip(documents, cluster_labels, documents_doi, documents_title)]

# Print or store the results as needed
for document in clustered_documents:
    print(f"Cluster {document['cluster']}:\n{document['abstract']}\n{'-'*50}")

#%% Data Viz cluster doc

tsne = TSNE(n_components=2)
centroids_tsne = tsne.fit_transform(document_centroids)

# Create DataFrame for Plotly
df = pd.DataFrame({'X': centroids_tsne[:, 0], 'Y': centroids_tsne[:, 1],
                   'Cluster': cluster_labels, 'Abstract': documents, "DOI":documents_doi,"title":documents_title,"id_":documents_id})



# Create an interactive scatter plot
fig = px.scatter(df, x='X', y='Y', color='Cluster', hover_data=['DOI'], title='t-SNE Visualization of Document Centroids')

# Save the interactive plot as an HTML file
fig.write_html("Results/Figures/TSNE_visualization.html")


df.to_csv("Data/doc_centroid.csv",index=False)

#%% Get papers info

papers2authors = defaultdict(list)
list_papers_ubi = []

docs = collection.find()
for doc in tqdm.tqdm(docs):
    list_papers_ubi.append(doc["id"])

    

docs = collection.find()
for doc in tqdm.tqdm(docs):
    authors = doc["authorships"]
    for author in authors:
        name = author["author"]["display_name"]
        papers2authors[doc["id"]].append(name)
        
commu2cited = {i:[] for i in df["Cluster"]}

commu2papers = df.groupby("Cluster")["id_"].apply(list).to_dict()

docs = collection.find()
for doc in tqdm.tqdm(docs):
    community_value = None
    if doc["referenced_works"]:
        for community in commu2papers:
            if doc["id"] in commu2papers[community]:
                community_value = community
        if community_value == None:
            print(doc["id"])
        for ref in doc["referenced_works"]:
            if ref in list_papers_ubi:
                commu2cited[community_value].append(ref)

    
commu2cited_authors = {i:[] for i in df["Cluster"]}

for commu in commu2cited:
    for paper in commu2cited[commu]:
        try:
            authors = papers2authors[paper]
            commu2cited_authors[commu] += authors
        except:
            pass
    
def get_top_authors(community_authors, top_n=10):
    top_authors_per_community = {}
    
    for community, authors in community_authors.items():
        # Count the frequency of each author in the community
        author_counts = Counter(authors)
        
        # Calculate the total number of authors in the community
        total_authors = sum(author_counts.values())
        
        # Get the top_n authors
        top_authors = author_counts.most_common(top_n)
        
        # Calculate the share for each top author
        top_authors_with_share = [(author, count, count / total_authors) for author, count in top_authors]
        
        # Store the results in the dictionary
        top_authors_per_community[community] = top_authors_with_share
        
    return top_authors_per_community
    

# Get the top 10 authors for each community
top_authors_per_community = get_top_authors(commu2cited_authors, top_n=10)


# Top 10 most cited paper per commu in our sample
commu2papers_UBI_only = {i:[] for i in df["Cluster"]}


for commu in commu2cited:
    for paper in commu2cited[commu]:
        commu2papers_UBI_only[commu].append(paper)

top_cited_per_community_sample = get_top_authors(commu2papers_UBI_only, top_n=10)


# Top 10 most cited paper per commu in OpenAlex

most_cited_global = defaultdict(lambda:defaultdict(int))

for community in commu2papers:
    for paper in commu2papers[community]:
        try:
            doc = collection.find_one({"id":paper})
            citations = doc["cited_by_count"]
            title = doc["title"]
            most_cited_global[community][doc["id"]] = citations
        except:
            pass

top_cited_per_community_global = {}
    
for community, papers in most_cited_global.items():
    # Sort papers by number of citations in descending order
    sorted_papers = sorted(papers.items(), key=lambda item: item[1], reverse=True)
    total_citations = sum(paper[1] for paper in papers.items())
    top_papers = sorted_papers[:10]
    
    # Calculate the share for each top paper
    top_papers_with_share = [(paper_id, count, count / total_citations) for paper_id, count in top_papers]

    # Select the top N papers
    top_cited_per_community_global[community] = top_papers_with_share
    
    
#%% Key terms per community

df['text'] = df['title'] + ' ' + df['Abstract']

# Preprocessing function
def preprocess(text, freq_threshold, word_counts):
    stop_words = set(stopwords.words('english'))
    
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [word for word in tokens if word_counts[word] >= freq_threshold]
    return ' '.join(tokens)

def tf_idf_gram(gram, freq_threshold=15):
    # Apply preprocessing
    df.dropna(subset=['text'], inplace=True)
    
    # Calculate word frequencies
    all_tokens = nltk.word_tokenize(' '.join(df['text']).lower())
    word_counts = Counter(all_tokens)
    
    df['text'] = df['text'].apply(lambda x: preprocess(x, freq_threshold, word_counts))
    
    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(gram, gram))
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    
    # Create a DataFrame with TF-IDF scores
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=df['id_'])
    
    # Add community information to TF-IDF DataFrame
    tfidf_df['Cluster'] = df['Cluster'].values
    
    # Function to get key terms for each community
    def get_key_terms(tfidf_df, community):
        community_df = tfidf_df[tfidf_df['Cluster'] == community].drop(columns=['Cluster'])
        mean_tfidf = community_df.mean(axis=0)
        # Remove the terms that are in the exclude_words list
        filter_out_substrings = ["basic", "income", "universal", "ubi", "unconditional", "state bonus", "minimum income", 
                         "national dividend", "social dividend", "basic minimum income", "basic income",
                         "negative income tax", "minimum income guarantee", "guaranteed minimum income", 
                         "basic income guarantee", "demogrant", "guaranteed income", "credit income tax",
                         "citizen’s basic income", "citizen’s income", "social credit",
                         "unconditional basic income", "universal basic income", "guaranteed income", 
                         "social dividend", "basic income guarantee"]

        # Create a boolean mask to filter out the words that contain any of the substrings
        mask = ~mean_tfidf.index.to_series().apply(lambda x: any(substring in x for substring in filter_out_substrings))
        
        # Filter the mean_tfidf DataFrame using the mask
        mean_tfidf = mean_tfidf[mask]
        
        top_terms = mean_tfidf.sort_values(ascending=False).head(20)
        return top_terms
        
    
    # Get key terms for each community
    communities = df['Cluster'].unique()
    key_terms = {community: get_key_terms(tfidf_df, community) for community in tqdm.tqdm(communities)}
    return key_terms


key_terms1 = tf_idf_gram(1,freq_threshold=100)
key_terms2 = tf_idf_gram(2,freq_threshold=100)
key_terms3 = tf_idf_gram(3,freq_threshold=100)

#%% Save info

#%%

data = []

for community in top_authors_per_community:
    for key_term1, key_term2, key_term3 in zip(key_terms1[community].index,
                                               key_terms2[community].index,
                                               key_terms3[community].index):
        data.append({ "community": community, "tf_idf_term1":key_term1,"tf_idf_term2":key_term2,"tf_idf_term3":key_term3})
    
df_tf_idf = pd.DataFrame(data, columns = ["community","tf_idf_term1", "tf_idf_term2", "tf_idf_term3"])
df_tf_idf.to_csv("Data/tf_idf_20_kmeans.csv", index = False)

#%%

data = []

for community in top_authors_per_community:
    for authors, papers_sample, papers_global ,key_term1, key_term2, key_term3 in zip(top_authors_per_community[community],
                                                     top_cited_per_community_sample[community],
                                                     top_cited_per_community_global[community],
                                                     key_terms1[community].index,
                                                     key_terms2[community].index,
                                                     key_terms3[community].index):
        data.append({"id_sample":papers_sample[0], "n_sample":papers_sample[1], "share_sample": papers_sample[2],
                     "id_global":papers_global[0], "n_global":papers_global[1], "share_global": papers_global[2],
                     "author_sample":authors[0], "community": community,
                     "tf_idf_term1":key_term1,"tf_idf_term2":key_term2,"tf_idf_term3":key_term3})
    
df_top10 = pd.DataFrame(data, columns = ["id_sample", "n_sample", "share_sample",
                                         "id_global", "n_global", "share_global",
                                         "author_sample","community","tf_idf_term1", "tf_idf_term2", "tf_idf_term3"])

def get_title(id_):
    try:
        doc = collection.find_one({"id":id_})["title"]
    except:
        doc = "none"
    return doc

def get_authors_year(id_):
    doc = collection.find_one({"id":id_})
    if doc == None:
        return None
    else:
        year = doc["publication_year"]
        text = ""
        if len(doc["authorships"]) > 2:
            text += doc["authorships"][0]["author"]["display_name"] +" et al. ({})".format(year)
        elif len(doc["authorships"]) == 2:
            text += doc["authorships"][0]["author"]["display_name"] +" et "+ doc["authorships"][1]["author"]["display_name"]+" ({})".format(year)
        elif len(doc["authorships"]) == 1:
            text += doc["authorships"][0]["author"]["display_name"] +" ({})".format(year)
        return text

df_top10['title_sample'] = df_top10['id_sample'].apply(get_title)
df_top10['title_global'] = df_top10['id_global'].apply(get_title)
df_top10['bib_sample'] = df_top10['id_sample'].apply(get_authors_year)
df_top10['bib_global'] = df_top10['id_global'].apply(get_authors_year)
    
column_order = ['community', 'id_sample', "n_sample", "share_sample", 'title_sample', 'bib_sample',
                'id_global',"n_global", "share_global", 'title_global','bib_global',
                'author_sample',"tf_idf_term1","tf_idf_term2","tf_idf_term3"]
df_top10 = df_top10[column_order]


#%% Add level 0 info

db_concept = client['openAlex20240517']
collection_concept = db_concept['concepts']

lvl0 = []


docs = collection_concept.find()
for doc in tqdm.tqdm(docs):
    if doc["level"] == 0:
        lvl0.append(doc["display_name"])

commu2discipline = {i:{j:0 for j in lvl0} for i in commu2papers}
n_papers = {i:0 for i in commu2papers}

for commu in commu2papers:
    for paper in commu2papers[commu]:
        n_papers[commu] += 1
        doc = collection.find_one({"id":paper})
        for concept in doc["concepts"]:
            if concept["display_name"] in lvl0:
                commu2discipline[commu][concept["display_name"]] += 1
        
        
# Prepare the data for the DataFrame
data = []
for community, disciplines in commu2discipline.items():
    total_papers = n_papers[community]
    # Sort and get the top 10 disciplines
    sorted_disciplines = sorted(disciplines.items(), key=lambda item: item[1], reverse=True)[:10]
    for discipline, count in sorted_disciplines:
        share = count / total_papers if total_papers > 0 else 0
        data.append({
            "community": community,
            "discipline": discipline,
            "n_discipline": count,
            "share_discipline": share
        })

df_discipline = pd.DataFrame(data)

df_top10["discipline"] = df_discipline["discipline"]
df_top10["n_discipline"] = df_discipline["n_discipline"]
df_top10["share_discipline"] = df_discipline["share_discipline"]

df_top10.to_csv("Data/table2.csv", index = False)

