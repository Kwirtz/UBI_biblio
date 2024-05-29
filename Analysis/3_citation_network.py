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



#communities = g.community_leiden(objective_function="modularity", resolution=0.0000000000001)
#communities = g.community_spinglass(gamma=1.25, spins=20)
#communities = g.community_multilevel(resolution=0.205)
#q = g.modularity(communities)
communities = g.community_infomap()

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
len(set(node2community["Community"]))

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
        
        # Get the top_n authors
        top_authors = author_counts.most_common(top_n)
        
        # Store the results in the dictionary
        top_authors_per_community[community] = top_authors
    
    return top_authors_per_community

# Get the top 10 authors for each community
top_authors_per_community = get_top_authors(commu2cited_authors, top_n=10)

# Print the resulting dictionary
for community, authors in top_authors_per_community.items():
    print(f"Community {community} top authors:")
    for author, count in authors:
        print(f"{author}: {count}")


# Step 1: Filter the DataFrame
commu2papers_UBI_only = {i:[] for i in df["modularity_class"]}


for commu in commu2cited:
    for paper in commu2cited[commu]:
        if paper in list_of_papers:
            commu2papers_UBI_only[commu].append(paper)

top_papers_per_community = get_top_authors(commu2papers_UBI_only, top_n=10)
# Print the resulting dictionary
for community, authors in top_papers_per_community.items():
    print(f"\n Community {community} top papers cited:")
    for author, count in authors:
        try:
            doc = collection.find_one({"id":author})
            title = doc["title"]
            print(f"{title}: {count}")
        except:
            print(f" : {count}")

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
    stop_words |= {"basic","income","universal","ubi","unconditional"}
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

# Apply preprocessing
df.dropna(subset=['text'], inplace=True)
df['text'] = df['text'].apply(preprocess)
# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(3, 3))
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

# Print key terms for each community
for community, terms in key_terms.items():
    print(f"Community {community} key terms:")
    print(terms)

#%%
