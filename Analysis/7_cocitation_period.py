import tqdm
import nltk
import pymongo
import itertools
import numpy as np
import pandas as pd
import igraph as ig
from langdetect import detect
from collections import Counter
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['UBI']
collection = db['works_UBI_gobu']
db_total = client['openAlex20240517']
collection_total = db_total['works']

id_ref = []
periods = [range(1960,1970),range(1980,1990),range(1990,2000),range(2000,2010)]


docs = collection.find()
for doc in tqdm.tqdm(docs):
    id_ref += doc["referenced_works"]
    
    
frequency = Counter(id_ref)
id_ref = list(set(id_ref))
id2citations = dict(frequency)

"""
id2citations = {}
ref_not_found = []
for ref in tqdm.tqdm(id_ref):
    doc = collection_total.find_one({"id":ref})
    if doc == None:
        ref_not_found.append(ref)
        continue
    id2citations[ref] = doc["cited_by_count"]
"""   

def is_english(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return False
    
def preprocess(text, freq_threshold, word_counts):
    stop_words = set(stopwords.words('english'))
    #stop_words |= {"basic", "income", "universal", "ubi", "unconditional","state bonus", "minimum income", "national dividend", "social dividend", "basic minimum income", "basic income",
    #         "negative income tax", "minimum income guarantee", "guaranteed minimum income", "basic income guarantee", "demogrant", "guaranteed income", "credit income tax",
    #         "citizen’s basic income", "citizen’s income", "social credit",
    #         "unconditional basic income", "universal basic income", "guaranteed income", "social dividend", "basic income guarantee"}
    
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [word for word in tokens if word_counts[word] >= freq_threshold]
    return ' '.join(tokens)

# Function to get key terms for each community
def get_key_terms(tfidf_df, community):
    community_df = tfidf_df[tfidf_df['modularity_class'] == community].drop(columns=['modularity_class'])
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

def tf_idf_gram(gram, freq_threshold=15):
    # Apply preprocessing
    df_nodes.dropna(subset=['text'], inplace=True)
    
    # Calculate word frequencies
    all_tokens = nltk.word_tokenize(' '.join(df_nodes['text']).lower())
    word_counts = Counter(all_tokens)
    
    df_nodes['text'] = df_nodes['text'].apply(lambda x: preprocess(x, freq_threshold, word_counts))
    
    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(gram, gram))
    tfidf_matrix = vectorizer.fit_transform(df_nodes['text'])
    
    # Create a DataFrame with TF-IDF scores
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=df_nodes['Label'])
    
    # Add community information to TF-IDF DataFrame
    tfidf_df['modularity_class'] = df_nodes['Community'].values
    
    # Get key terms for each community
    communities = df_nodes['Community'].unique()
    key_terms = {community: get_key_terms(tfidf_df, community) for community in tqdm.tqdm(communities)}
    return key_terms
    

def build_co_citation_matrix(docs,id2citations):
    co_citation_matrix = {}
    for doc in docs:
        refs = doc["referenced_works"]
        for ref1, ref2 in itertools.combinations(refs, 2):
            if ref1 not in id2citations or ref2 not in id2citations:
                continue
            if ref1 not in co_citation_matrix:
                co_citation_matrix[ref1] = {}
            if ref2 not in co_citation_matrix[ref1]:
                co_citation_matrix[ref1][ref2] = 0
            co_citation_matrix[ref1][ref2] += 1
            if ref2 not in co_citation_matrix:
                co_citation_matrix[ref2] = {}
            if ref1 not in co_citation_matrix[ref2]:
                co_citation_matrix[ref2][ref1] = 0
            co_citation_matrix[ref2][ref1] += 1

    # Convert to DataFrame for easy manipulation
    matrix_df = pd.DataFrame.from_dict(co_citation_matrix, orient='index').fillna(0)
    matrix_df = matrix_df.apply(lambda x: x if x >= 2 else 0)
    return matrix_df



for period in periods:
    period_docs = []
    for year in tqdm.tqdm(period, desc=f"Processing period {period}"):
        docs = collection.find({"publication_year": year})
        period_docs.extend(docs)
    co_citation_matrix = build_co_citation_matrix(period_docs, id2citations)
    
    # Normalize the co-citation matrix
    citation_counts = np.array([id2citations.get(doc, 1) for doc in co_citation_matrix.index])
    normalized_matrix = co_citation_matrix.div(citation_counts, axis=0)

    # Calculate cosine similarity
    cosine_sim_matrix = cosine_similarity(normalized_matrix)

    # Convert to DataFrame for easier manipulation
    similarity_df = pd.DataFrame(cosine_sim_matrix, index=co_citation_matrix.index, columns=co_citation_matrix.index)
    # Convert the similarity DataFrame to a graph
    g = ig.Graph.Weighted_Adjacency(similarity_df.values.tolist(), mode='undirected', attr="weight")
    g.vs['label'] = similarity_df.index.tolist()
    communities = g.community_leiden(objective_function="modularity", resolution=1, n_iterations = 500)

    node_names = []
    community_memberships = []
    
    # Iterate over communities
    for idx, community in enumerate(communities):
        # Iterate over nodes in the community
        for node in community:
            node_names.append(g.vs[node]["label"])
            community_memberships.append(idx)
            
    node2community = pd.DataFrame({"Node": node_names, "Community": community_memberships})
    node2community["Community"].value_counts()
    
    
    
    # Get the list of nodes
    node_list = [(v.index, v["label"]) for v in g.vs]
    df_nodes = pd.DataFrame(node_list, columns=["id", "Label"])
    df_nodes = df_nodes.merge(node2community, left_on="Label", right_on="Node", how="inner")
    df_nodes = df_nodes.drop(columns=["Node"])
    
    # Identify communities with frequency less than 500
    community_freq = df_nodes['Community'].value_counts()
    communities_to_replace = community_freq[community_freq < 100].index
    df_nodes.loc[df_nodes['Community'].isin(communities_to_replace), 'Community'] = max(df_nodes['Community']) + 1
    unique_communities = sorted(df_nodes['Community'].unique())
    community_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_communities, start=1)}
    df_nodes['Community'] = df_nodes['Community'].map(community_mapping)
    
    
    documents = []
    for doc_id in tqdm.tqdm(df_nodes["Label"]):
        doc = collection_total.find_one({'id': doc_id}, {'title': 1})
        if doc:
            if "title" not in doc:
                doc["title"] = ""
            if is_english(doc["title"]) == False:
                doc["title"] = ""
        else:
            doc = {}
            doc["title"] = ""
            doc["id"] = doc_id
        documents.append(doc)
        
    # Merge retrieved documents with the DataFrame
    df_nodes['title'] = [doc['title'] for doc in documents]
    
    # Combine title and abstract
    df_nodes['text'] = df_nodes['title'] 
    
    
    
    
    key_terms1 = tf_idf_gram(1,freq_threshold=1)
    key_terms2 = tf_idf_gram(2,freq_threshold=1)
    key_terms3 = tf_idf_gram(3,freq_threshold=1)
    
    data = []
    
    for community in df_nodes["Community"].unique():
        for key_term1, key_term2, key_term3 in zip(key_terms1[community].index,
                                                   key_terms2[community].index,
                                                   key_terms3[community].index):
            data.append({ "community": community, "tf_idf_term1":key_term1,"tf_idf_term2":key_term2,"tf_idf_term3":key_term3})
        
    df_tf_idf = pd.DataFrame(data, columns = ["community","tf_idf_term1", "tf_idf_term2", "tf_idf_term3"])
    df_tf_idf.to_csv("Data/tf_idf_20_{}_{}.csv".format(period[0],period[-1]), index = False)