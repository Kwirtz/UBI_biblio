#%% Init

import tqdm
import nltk
import pymongo
import pandas as pd
import igraph as ig
from nltk.corpus import stopwords
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


# Download stopwords if not already downloaded
#nltk.download('stopwords')
#nltk.download('punkt')

# Initialize PorterStemmer
stemmer = PorterStemmer()

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['UBI']
collection = db['works_UBI_20240517']


#%% Get Citation network

list_of_papers = []
docs = collection.find()
for doc in tqdm.tqdm(docs):
    list_of_papers.append(doc["id"])    



data = []
docs = collection.find()
for doc in tqdm.tqdm(docs):
    if doc["referenced_works"]:
        for ref in doc["referenced_works"]:
            if ref in list_of_papers:
                data.append({"source":doc["id"],"target":ref})

df = pd.DataFrame(data,columns=["source","target"])


g = ig.Graph.TupleList(df.itertuples(index=False))

# Extract the largest connected component
if not g.is_connected():
    components = g.components()
    g = components.giant()



#communities = g.community_leiden(objective_function="modularity", resolution=0.3)
#communities = g.community_spinglass(gamma=1.2)
#communities = g.community_multilevel(resolution=0.205)
#communities = g.community_infomap()
#communities = g.community_edge_betweenness(clusters=6).as_clustering()
communities = g.community_fastgreedy()

node_names = []
community_memberships = []

# Iterate over communities
for idx, community in enumerate(communities):
    # Iterate over nodes in the community
    for node in community:
        node_names.append(g.vs[node]["name"])
        community_memberships.append(idx)
        
node2community = pd.DataFrame({"Node": node_names, "Community": community_memberships})
node2community["Community"].value_counts()

# Get the list of nodes
node_list = [(v.index, v["name"]) for v in g.vs]
df_nodes = pd.DataFrame(node_list, columns=["id", "label"])
df_nodes = df_nodes.merge(node2community, left_on="label", right_on="Node", how="inner")
df_nodes = df_nodes.drop(columns=["Node"])

# Get the edge list
edge_list = g.get_edgelist()
df_edges = pd.DataFrame(edge_list, columns=["Source", "Target"])

df_edges.to_csv("Data/Edges_citations.csv",index=False)
df_nodes.to_csv("Data/Nodes_citations.csv",index=False)

#%% Gephi community

"Resolution: 2.0"

#%% get stats for communities

df = pd.read_csv("Data/modularity.csv")
df = df.sort_values(by='modularity_class')
papers2authors = defaultdict(list)

docs = collection.find()
for doc in tqdm.tqdm(docs):
    authors = doc["authorships"]
    for author in authors:
        name = author["author"]["display_name"]
        papers2authors[doc["id"]].append(name)
        
commu2cited = {i:[] for i in df["modularity_class"]}

commu2papers = df.groupby("modularity_class")["Label"].apply(list).to_dict()

docs = collection.find()
for doc in tqdm.tqdm(docs):
    if doc["referenced_works"]:
        for community in commu2papers:
            if doc["id"] in commu2papers[community]:
                community_value = community
        for ref in doc["referenced_works"]:
            commu2cited[community_value].append(ref)
    
commu2cited_authors = {i:[] for i in df["modularity_class"]}

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
commu2papers_UBI_only = {i:[] for i in df["modularity_class"]}


for commu in commu2cited:
    for paper in commu2cited[commu]:
        if paper in list_of_papers:
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




#%% tf_idf of community

df = pd.read_csv("Data/modularity.csv")

documents = []
for doc_id in df['Label']:
    doc = collection.find_one({'id': doc_id}, {'title': 1, 'abstract': 1})
    if "abstract" not in doc:
        doc["abstract"] = ""
    if "title" not in doc:
        doc["title"] = ""
    documents.append(doc)

# Merge retrieved documents with the DataFrame
df['title'] = [doc['title'] for doc in documents]
df['abstract'] = [doc['abstract'] for doc in documents]

# Combine title and abstract
df['text'] = df['title'] #+ ' ' + df['abstract']

# Preprocessing function
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    #stop_words |= {"basic","income","universal","ubi","unconditional"}
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

def tf_idf_gram(gram):
    # Apply preprocessing
    df.dropna(subset=['text'], inplace=True)
    df['text'] = df['text'].apply(preprocess)
    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(gram, gram))
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    
    # Create a DataFrame with TF-IDF scores
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=df['Label'])
    
    # Add community information to TF-IDF DataFrame
    tfidf_df['modularity_class'] = df['modularity_class'].values
    
    # Function to get key terms for each community
    def get_key_terms(tfidf_df, community):
        community_df = tfidf_df[tfidf_df['modularity_class'] == community].drop(columns=['modularity_class'])
        mean_tfidf = community_df.mean(axis=0)
        top_terms = mean_tfidf.sort_values(ascending=False).head(10)  # Get top 10 terms
        return top_terms
    
    # Get key terms for each community
    communities = df['modularity_class'].unique()
    key_terms = {community: get_key_terms(tfidf_df, community) for community in communities}
    return key_terms

key_terms1 = tf_idf_gram(1)
key_terms2 = tf_idf_gram(2)
key_terms3 = tf_idf_gram(3)

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
    return collection.find_one({"id":id_})["title"]

df_top10['title_sample'] = df_top10['id_sample'].apply(get_title)
df_top10['title_global'] = df_top10['id_global'].apply(get_title)

column_order = ['community', 'id_sample', "n_sample", "share_sample", 'title_sample',
                'id_global',"n_global", "share_global", 'title_global', 'author_sample',"tf_idf_term1","tf_idf_term2","tf_idf_term3"]
df_top10 = df_top10[column_order]

df_top10.to_csv("Data/citation_network_top10.csv",index=False)

#%% Add level 0 info

db_concept = client['openAlex20240517']
collection_concept = db['concepts']

lvl0 = []
commu2discipline = {i:{j:0 for j in lvl0} for i in commu2papers}
n_papers = {i:0 for i in commu2papers}

docs = collection_concept.find()
for doc in tqdm.tqdm(docs):
    if doc["level"] == 0:
        lvl0.append(doc["display_name"])

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
            "number_of_papers": count,
            "share_of_papers": share
        })

df = pd.DataFrame(data)
