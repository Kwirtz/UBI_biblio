#%% Init

import tqdm
import nltk
import json
import pickle
import langid
import random
import pymongo
import requests
import pandas as pd
import numpy as np
import igraph as ig
from langdetect import detect
from nltk.corpus import stopwords
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Download stopwords if not already downloaded
#nltk.download('stopwords')
#nltk.download('punkt')



# Initialize PorterStemmer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['UBI']
collection = db['works_UBI_gobu_2']
db_all = client['openAlex20240517']
collection_all = db_all['works']

list_of_papers = []
docs = collection.find()
for doc in tqdm.tqdm(docs):
    list_of_papers.append(doc["id"])    

start_year = 1900
end_year = 2024

query = {
    'publication_year': {
        '$gte': start_year,
        '$lte': end_year
    }
}

query = {}

#%% Get Citation network


data = []
docs = collection.find(query)
for doc in tqdm.tqdm(docs):
    if doc["publication_year"] > 1900:
        if doc["referenced_works"]:
            for ref in doc["referenced_works"]:
                if ref in list_of_papers:
                    data.append({"source":doc["id"],"target":ref})

df = pd.DataFrame(data,columns=["source","target"])


g = ig.Graph.TupleList(df.itertuples(index=False), directed=False)

# Extract the largest connected component
if not g.is_connected():
    components = g.components()
    g = components.giant()


# Simplify the graph to remove multi-edges and self-loops
g = g.simplify(multiple=True, loops=True)


#%%

for i in tqdm.tqdm(range(19,20)):
    random.seed(i)
    communities = g.community_leiden(objective_function="modularity", resolution=0.5)
    #communities = g.community_spinglass(gamma=0.9)
    #communities = g.community_multilevel(resolution=0.4)  
    #communities = g.community_infomap()
    #communities = g.community_edge_betweenness(clusters=4).as_clustering()
    #communities = g.community_fastgreedy().as_clustering()
    
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
    
    
    # Get the edge list
    edge_list = g.get_edgelist()
    df_edges = pd.DataFrame(edge_list, columns=["Source", "Target"])
    
    df_edges.to_csv("Data/Edges_citations.csv",index=False)
    df_nodes.to_csv("Data/Nodes_citations.csv",index=False)
    
#%% get stats for communities refs

    #collection = db['works_UBI_gobu_2']
    
    list_papers_ubi = []
    
    docs = collection.find()
    for doc in tqdm.tqdm(docs):
        list_papers_ubi.append(doc["id"])
    
    
    df = df_nodes
    df["modularity_class"] = df["Community"]
    df = df.sort_values(by='modularity_class')
    df = df[df['Label'].isin(list_papers_ubi)]
    
    
    papers2authors = defaultdict(list)
    
    
    docs = collection.find()
    for doc in tqdm.tqdm(docs):
        authors = doc["authorships"]
        for author in authors:
            name = author["author"]["display_name"]
            papers2authors[doc["id"]].append(name)
            
    commu2cited = {i:[] for i in df["modularity_class"]}
    
    commu2papers = df.groupby("modularity_class")["Label"].apply(list).to_dict()
    with open('Data/commu2papers.pkl', 'wb') as f:
        pickle.dump(commu2papers, f)
    docs = collection.find()
    for doc in tqdm.tqdm(docs):
        community_value = None
        if doc["referenced_works"]:
            for community in commu2papers:
                if doc["id"] in commu2papers[community]:
                    community_value = community
            if community_value != None:
                for ref in doc["referenced_works"]:
                    if ref in ["https://openalex.org/W3147703848","https://openalex.org/W1676072121"]:
                        pass
                    commu2cited[community_value].append(ref)
        
    commu2cited_authors = {i:[] for i in df["modularity_class"]}
    
    for commu in commu2cited:
        for paper in commu2cited[commu]:
            try:
                authors = papers2authors[paper]
                commu2cited_authors[commu] += authors
            except:
                pass
        
    def get_top_authors(community_authors, top_n=10, normalize= False):
        top_authors_per_community = {}
        
        for community, authors in community_authors.items():
            # Count the frequency of each author in the community
            author_counts = Counter(authors)
            
            if normalize:
                for id_ in author_counts:
                    doc = collection_all.find_one({"id":id_})
                    try:
                        author_counts[id_] /= (doc["cited_by_count"]+author_counts[id_])
                    except:
                        author_counts[id_] = 0
            # Calculate the total number of authors in the community
            total_authors = sum(author_counts.values())
            
            # Get the top_n authors
            top_authors = author_counts.most_common(top_n)
            """
            for i in top_authors:
                if i[0] == "https://openalex.org/W1559924327":
                    print(community)
            """
            # Calculate the share for each top author
            top_authors_with_share = [(author, count, count / total_authors) for author, count in top_authors]
    
            # Ensure the list has exactly top_n elements, padding with None if necessary
            while len(top_authors_with_share) < top_n:
                top_authors_with_share.append((None, None, None))
            
            # Store the results in the dictionary
            top_authors_per_community[community] = top_authors_with_share
            
        return top_authors_per_community
        
    
    # Get the top 10 authors for each community
    top_authors_per_community = get_top_authors(commu2cited_authors, top_n=20)
    
    
    # Top 10 most used pqper per commu in our sample
    commu2papers_UBI_only = {i:[] for i in df["modularity_class"]}
    
    
    for commu in commu2cited:
        for paper in commu2cited[commu]:
            if paper in list_papers_ubi and paper not in ["https://openalex.org/W3147703848","https://openalex.org/W1676072121"]:
                commu2papers_UBI_only[commu].append(paper)
    
    top_cited_per_community_sample = get_top_authors(commu2papers_UBI_only, top_n=20)
    
    
    #%% Global Top 10 most cited paper per commu in OpenAlex
    
    most_cited_global = defaultdict(lambda:defaultdict(int))
    
    for community in commu2papers:
        for paper in commu2papers[community]:
            try:
                if paper not in ["https://openalex.org/W3147703848","https://openalex.org/W1676072121"]:
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
        top_papers = sorted_papers[:20]
        
        # Calculate the share for each top paper
        top_papers_with_share = [(paper_id, count, count / total_citations) for paper_id, count in top_papers]
    
        # Select the top N papers
        top_cited_per_community_global[community] = top_papers_with_share
    
    
    #Most used ref, Most used authors by our sample
    #Most cited paper in community for UBI, Most cited paper in community for global
    
    papers_cited_UBI = []
    
    docs = collection.find()
    for doc in tqdm.tqdm(docs):
        if doc["referenced_works"]:
            papers_cited_UBI += doc["referenced_works"]
    
    papers2cited_UBI = Counter(papers_cited_UBI)
    
    most_cited_UBI = defaultdict(lambda:defaultdict(int))
    for community, papers in commu2papers.items():
        for paper in papers:
            if paper not in ["https://openalex.org/W3147703848","https://openalex.org/W1676072121"]:
                most_cited_UBI[community][paper] = papers2cited_UBI[paper]
    
    top_cited_per_community_UBI = {}
        
    for community, papers in most_cited_UBI.items():
        # Sort papers by number of citations in descending order
        sorted_papers = sorted(papers.items(), key=lambda item: item[1], reverse=True)
        total_citations = sum(paper[1] for paper in papers.items())
        top_papers = sorted_papers[:20]
        
        # Calculate the share for each top paper
        top_papers_with_share = [(paper_id, count, count / total_citations) for paper_id, count in top_papers]
    
        # Select the top N papers
        top_cited_per_community_UBI[community] = top_papers_with_share

    
    
    
    #%% tf_idf of community
    
    
    stop_words = ["full-text", "http", "doi", "not available for this content", "pdf", "url", "access", "vol", "ha" , "de"]
    """
    def is_english(text):
        try:
            lang = detect(text)
            return lang == 'en'
        except:
            return False
    """
    def is_english(text):
        if text:
            lang, _ = langid.classify(text)
            return lang == 'en'
        else:
            return False
    
    def is_not_korean(text):
        if text:
            lang, _ = langid.classify(text)
            return lang != 'ko'
        else:
            return False            
        
    documents = []
    for doc_id in df['Label']:
        doc = collection.find_one({'id': doc_id}, {'title': 1, 'abstract': 1})
        if "abstract" not in doc:
            doc["abstract"] = ""
        else:
            for word in stop_words:
                if word in doc["abstract"].lower():
                    doc["abstract"] = ""
        if "title" not in doc:
            doc["title"] = ""
        if is_english(doc["title"]) == False:
            doc["abstract"] = ""
            doc["title"] = ""
        documents.append(doc)
    
    # Merge retrieved documents with the DataFrame
    df['title'] = [doc['title'] for doc in documents]
    df['abstract'] = [doc['abstract'] for doc in documents]
    
    # Combine title and abstract
    df['text'] = df['title'] + ' ' + df['abstract']
    
    # Preprocessing function
    def preprocess(text, freq_threshold, word_counts):
        stop_words = set(stopwords.words('english'))
        #stop_words |= {"basic", "income", "universal", "ubi", "unconditional","state bonus", "minimum income", "national dividend", "social dividend", "basic minimum income", "basic income",
        #         "negative income tax", "minimum income guarantee", "guaranteed minimum income", "basic income guarantee", "demogrant", "guaranteed income", "credit income tax",
        #         "citizen’s basic income", "citizen’s income", "social credit",
        #         "unconditional basic income", "universal basic income", "guaranteed income", "social dividend", "basic income guarantee"}
        
        tokens = nltk.word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        tokens = [word for word in tokens if word_counts[word] >= freq_threshold]
        
        return ' '.join(tokens)
    
    def tf_idf_gram(gram, freq_threshold=15):
        # Apply preprocessing
        df.dropna(subset=['text'], inplace=True)
        
        # Calculate word frequencies
        all_tokens = nltk.word_tokenize(' '.join(df['text']).lower())
        all_tokens = [lemmatizer.lemmatize(token) for token in all_tokens]
        word_counts = Counter(all_tokens)
        
        df['text'] = df['text'].apply(lambda x: preprocess(x, freq_threshold, word_counts))
        
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
            # Remove the terms that are in the exclude_words list
            filter_out_substrings = ["welfare","social","policy","economic","basic", "income", "universal", "ubi", "unconditional", "state bonus", "minimum income", 
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
            mean_tfidf = mean_tfidf[mean_tfidf.index.to_series().apply(is_not_korean)]
            top_terms = mean_tfidf.sort_values(ascending=False).head(20)
            return top_terms
        
        # Get key terms for each community
        communities = df['modularity_class'].unique()
        key_terms = {community: get_key_terms(tfidf_df, community) for community in tqdm.tqdm(communities)}
        return key_terms
    
    
    key_terms1 = tf_idf_gram(1,freq_threshold=10)
    key_terms2 = tf_idf_gram(2,freq_threshold=10)
    key_terms3 = tf_idf_gram(3,freq_threshold=10)
    
    
    #%%
    
    data = []
    
    for community in df['modularity_class'].unique():
        for authors, papers_sample, papers_UBI, papers_global ,key_term1, key_term2, key_term3 in zip(top_authors_per_community[community],
                                                                                          top_cited_per_community_sample[community],
                                                                                          top_cited_per_community_UBI[community],
                                                                                          top_cited_per_community_global[community],
                                                                                          key_terms1[community].index,
                                                                                          key_terms2[community].index,
                                                                                          key_terms3[community].index):
            data.append({"id_most_used_ref":papers_sample[0], "n_most_used_ref":papers_sample[1], "share_most_used_ref": papers_sample[2],
                         "author_most_used":authors[0], "community": community,
                         "id_most_cited_UBI":papers_UBI[0], "n_most_cited_UBI":papers_UBI[1], "share_most_cited_UBI": papers_UBI[2],
                         "id_most_cited_global":papers_global[0], "n_most_cited_global":papers_global[1], "share_most_cited_global": papers_global[2],
                         "tf_idf_term1":key_term1,"tf_idf_term2":key_term2,"tf_idf_term3":key_term3})
    df_top10 = pd.DataFrame(data, columns = ["id_most_used_ref", "n_most_used_ref", "share_most_used_ref","author_most_used",
                                             "id_most_cited_UBI", "n_most_cited_UBI", "share_most_cited_UBI",
                                             "id_most_cited_global", "n_most_cited_global", "share_most_cited_global",
                                             "community","tf_idf_term1", "tf_idf_term2", "tf_idf_term3"])
    
    def get_title(id_):
        try:
            doc = collection_all.find_one({"id":id_})["title"]
        except:
            doc = None
        return doc
    
    def get_authors_year(id_):
        if id_ != None:
            doc = collection_all.find_one({"id":id_})
            year = doc["publication_year"]
            text = ""
            if len(doc["authorships"]) > 2:
                text += doc["authorships"][0]["author"]["display_name"] +" et al. ({})".format(year)
            elif len(doc["authorships"]) == 2:
                text += doc["authorships"][0]["author"]["display_name"] +" et "+ doc["authorships"][1]["author"]["display_name"]+" ({})".format(year)
            elif len(doc["authorships"]) == 1:
                text += doc["authorships"][0]["author"]["display_name"] +" ({})".format(year)
        else:
            text = None
        return text
    
    df_top10['title_most_used_ref'] = df_top10['id_most_used_ref'].apply(get_title)
    df_top10['bib_most_used_ref'] = df_top10['id_most_used_ref'].apply(get_authors_year)
    df_top10['title_most_cited_global'] = df_top10['id_most_cited_global'].apply(get_title)
    df_top10['bib_most_cited_global'] = df_top10['id_most_cited_global'].apply(get_authors_year)
    df_top10['title_most_cited_UBI'] = df_top10['id_most_cited_UBI'].apply(get_title)
    df_top10['bib_most_cited_UBI'] = df_top10['id_most_cited_UBI'].apply(get_authors_year)
    
    #%% Add level 0 info
    
    db_concept = client['openAlex20240517']
    collection_concept = db_concept['concepts']
    
    lvl1 = []
    
    
    docs = collection_concept.find()
    for doc in tqdm.tqdm(docs):
        if doc["level"] == 0:
            lvl1.append(doc["display_name"])
    
    commu2discipline = {i:{j:0 for j in lvl1} for i in commu2papers}
    n_papers = {i:0 for i in commu2papers}
    
    for commu in commu2papers:
        for paper in commu2papers[commu]:
            n_papers[commu] += 1
            doc = collection.find_one({"id":paper})
            for concept in doc["concepts"]:
                if concept["display_name"] in lvl1:
                    commu2discipline[commu][concept["display_name"]] += 1
            
            
    # Prepare the data for the DataFrame
    data = []
    for community, disciplines in commu2discipline.items():
        total_papers = n_papers[community]
        # Sort and get the top 10 disciplines
        sorted_disciplines = sorted(disciplines.items(), key=lambda item: item[1], reverse=True)[:15]
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
    
    column_order = ["community","tf_idf_term1", "tf_idf_term2", "tf_idf_term3",
                    "discipline","n_discipline","share_discipline",
                    "author_most_used", 'title_most_used_ref','bib_most_used_ref',
                    'title_most_cited_UBI','bib_most_cited_UBI',
                    'title_most_cited_global','bib_most_cited_global',
                    "id_most_used_ref", "n_most_used_ref", "share_most_used_ref",
                    "id_most_cited_UBI", "n_most_cited_UBI", "share_most_cited_UBI",
                    "id_most_cited_global", "n_most_cited_global", "share_most_cited_global"]
    df_top10 = df_top10[column_order]
    
    df_top10.to_csv("Data/table1_{}.csv".format(i), index = False)


    #%%
    df_prop_concept = pd.DataFrame(0, index=list(commu2papers), columns = ["basic income", "negative income tax", "minimum income guarantee","periodic", "cash payment", "individual", "universal", "unconditional", "regularly", "in kind","Total"])
    for commu in commu2papers:
        papers = commu2papers[commu]
        df_prop_concept.at[commu,"Total"] = len(commu2papers[commu])
        for paper in tqdm.tqdm(papers):
            doc = collection.find_one({"id":paper})
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
            text = title + abstract
            text = text.lower()
            full_gram = ""
            if doc["has_fulltext"]:
                doc_id = doc["id"].split("/")[-1]
                #response = requests.get("https://api.openalex.org/works/{}/ngrams?mailto=kevin-w@hotmail.fr".format(doc_id))
                #ngrams = json.loads(response.content)["ngrams"]
                ngrams = []
                for gram in ngrams:
                    if gram['ngram_tokens']<4:
                        full_gram += gram["ngram"].lower() + " "
            for keyword in ["basic income", "negative income tax", "minimum income guarantee","periodic", "cash payment", "individual", "universal", "unconditional", "regularly", "in kind"]:
                text_temp = text
                full_gram_temp = full_gram
                done = False
                if len(keyword.split(" ")) == 1:
                    text_temp = text_temp.split(" ")
                    full_gram_temp = full_gram_temp.split(" ")
                if keyword in text_temp and done == False:
                    df_prop_concept.at[commu,keyword] += 1
                    done = True
                if keyword in full_gram_temp and done == False:
                    df_prop_concept.at[commu,keyword] += 1
                    done = True
                if keyword == "minimum income guarantee" and done == False:
                    if "guaranteed minimum income" in text_temp and done == False:
                        df_prop_concept.at[commu,keyword] += 1
                        done = True
                    if "guaranteed minimum income" in full_gram_temp and done == False:
                        df_prop_concept.at[commu,keyword] += 1
                        done = True


    for col in ["basic income", "negative income tax", "minimum income guarantee","periodic", "cash payment", "individual", "universal", "unconditional", "regularly", "in kind"]:
        df_prop_concept["share " + col] = df_prop_concept[col]/df_prop_concept["Total"]
    df_prop_concept["community"] = df_prop_concept.index
    df_prop_concept.to_csv("Data/commu2keywords_tan.csv",index=False)

    list_of_insertion = []
    keywords_to_check = ["periodic", "cash payment", "individual", "universal", "unconditional", "regularly", "in kind", "basic income", "negative income tax","minimum_income_guarantee"]
    for commu in commu2papers:
        papers = commu2papers[commu]
        freq_dict = {i:[] for i in keywords_to_check}
        for paper in tqdm.tqdm(papers):
            doc = collection.find_one({"id":paper})
            full_gram = ""
            if doc["has_fulltext"]:
                doc_id = doc["id"].split("/")[-1]
                #response = requests.get("https://api.openalex.org/works/{}/ngrams?mailto=kevin-w@hotmail.fr".format(doc_id))
                #ngrams = json.loads(response.content)["ngrams"]
                ngrams = []
                temp_freq_dict = {i:0 for i in keywords_to_check}
                for gram in ngrams:
                    if gram['ngram_tokens']<4:
                        for keyword in keywords_to_check:
                            if keyword in gram["ngram"].lower():  
                                temp_freq_dict[keyword] += gram["term_frequency"]
                        if "guaranteed minimum income" in gram["ngram"].lower() :
                            temp_freq_dict["minimum_income_guarantee"] += gram["term_frequency"]
                for keyword in temp_freq_dict:
                    freq_dict[keyword].append(temp_freq_dict[keyword])
        to_insert = []
        for keyword in freq_dict:
            if freq_dict[keyword]:
                to_insert.append(np.mean(freq_dict[keyword]))
            else:
                to_insert.append(0)
        list_of_insertion.append(to_insert)
        

    df_freq_concept = pd.DataFrame(list_of_insertion, index=list(commu2papers), columns = keywords_to_check )
    df_freq_concept["community"] = df_freq_concept.index
    df_freq_concept.to_csv("data/commu2freq.csv",index=False)

    #%% Commu2period prop
    n_period1 = 0 
    n_period2 = 0
    n_period3 = 0
    df_period_comm = pd.DataFrame(0, index=df['modularity_class'].unique(),columns=["Period1","Period2","Period3"])
    # Iterate over the collection once
    for doc in tqdm.tqdm(collection.find()):
        period = None
        if doc["publication_year"] >= 1960 and doc["publication_year"] <= 1980:
            period = "Period1"
            n_period1 += 1
        if doc["publication_year"] >= 1981 and doc["publication_year"] <= 2010:
            period = "Period2"
            n_period2 += 1
        if doc["publication_year"] >= 2011 and doc["publication_year"] <= 2020:
            period = "Period3"
            n_period3 += 1
        if period:
            for community, papers in commu2papers.items():
                if doc["id"] in papers:
                    df_period_comm.at[community,period] += 1
                    
    df_period_comm["Period1"] = df_period_comm["Period1"]/n_period1
    df_period_comm["Period2"] = df_period_comm["Period2"]/n_period2
    df_period_comm["Period3"] = df_period_comm["Period3"]/n_period3
    df_period_comm["Community"] = df_period_comm.index
    df_period_comm[["Period1", "Period2", "Period3"]] = df_period_comm[["Period1", "Period2", "Period3"]].round(2)
    df_period_comm.to_csv("Data/First_look_period.csv",index=False)
    
    #%% Commu yearly publications
    # Define the year range
    start_year = 1960
    end_year = 2022
    
    query = {
        'publication_year': {
            '$gte': start_year,
            '$lte': end_year
        }
    }

    df_year_comm = pd.DataFrame(0, index=range(1960,2023),columns=[f'community_{col}' for col in df['modularity_class'].unique()])
    for doc in tqdm.tqdm(collection.find(query)):
        for community, papers in commu2papers.items():
            if doc["id"] in papers:
                df_year_comm.at[doc["publication_year"],"community_"+str(community)] += 1
                if community == 4 and doc["publication_year"] <1980:
                    print(doc["id"])
        # Calculate the sum of each row
    row_sums = df_year_comm.sum(axis=1)
    
    # Divide each value in the DataFrame by the sum of its row
    df_year_comm_normalized = df_year_comm.div(row_sums, axis=0)
    df_year_comm_normalized["year"] = df_year_comm_normalized.index
    df_year_comm_normalized.to_csv("Data/Fig_commu_evolution.csv",index=False)


































#%% get stats for communities refs

    #collection = db['works_UBI_gobu_2']

    periods = [range(1960,1981)]
    for period in periods:

        start_year = period[0]
        end_year = period[-1]
        query = {
            'publication_year': {
                '$gte': start_year,
                '$lte': end_year
            }
        } 
            
        df = df_nodes
        df["modularity_class"] = df["Community"]
        df = df.sort_values(by='modularity_class')
        df = df[df['Label'].isin(list_papers_ubi)]
        
        
        papers2authors = defaultdict(list)
        
        
        docs = collection.find(query)
        for doc in tqdm.tqdm(docs):
            authors = doc["authorships"]
            for author in authors:
                name = author["author"]["display_name"]
                papers2authors[doc["id"]].append(name)
                
        commu2cited = {i:[] for i in df["modularity_class"]}
        commu2papers_period = {i:[] for i in df["modularity_class"]}
        commu2papers = df.groupby("modularity_class")["Label"].apply(list).to_dict()
        docs = collection.find(query)
        for doc in tqdm.tqdm(docs):
            community_value = None
            if doc["referenced_works"]:
                for community in commu2papers:
                    if doc["id"] in commu2papers[community]:
                        community_value = community
                if community_value != None:
                    for ref in doc["referenced_works"]:
                        commu2cited[community_value].append(ref)
                    commu2papers_period[community_value].append(doc["id"])
            
        commu2cited_authors = {i:[] for i in df["modularity_class"]}
        
        for commu in commu2cited:
            for paper in commu2cited[commu]:
                try:
                    authors = papers2authors[paper]
                    commu2cited_authors[commu] += authors
                except:
                    pass
            
        def get_top_authors(community_authors, top_n=10, normalize= False):
            top_authors_per_community = {}
            
            for community, authors in community_authors.items():
                # Count the frequency of each author in the community
                author_counts = Counter(authors)
                
                if normalize:
                    for id_ in author_counts:
                        doc = collection_all.find_one({"id":id_})
                        try:
                            author_counts[id_] /= (doc["cited_by_count"]+author_counts[id_])
                        except:
                            author_counts[id_] = 0
                # Calculate the total number of authors in the community
                total_authors = sum(author_counts.values())
                
                # Get the top_n authors
                top_authors = author_counts.most_common(top_n)
                """
                for i in top_authors:
                    if i[0] == "https://openalex.org/W1559924327":
                        print(community)
                """
                # Calculate the share for each top author
                top_authors_with_share = [(author, count, count / total_authors) for author, count in top_authors]
        
                # Ensure the list has exactly top_n elements, padding with None if necessary
                while len(top_authors_with_share) < top_n:
                    top_authors_with_share.append((None, None, None))
                
                # Store the results in the dictionary
                top_authors_per_community[community] = top_authors_with_share
                
            return top_authors_per_community
            
        
        # Get the top 10 authors for each community
        top_authors_per_community = get_top_authors(commu2cited_authors, top_n=15)
        
        
        # Top 10 most used pqper per commu in our sample
        commu2papers_UBI_only = {i:[] for i in df["modularity_class"]}
        
        
        for commu in commu2cited:
            for paper in commu2cited[commu]:
                if paper in list_papers_ubi:
                    commu2papers_UBI_only[commu].append(paper)
        
        top_cited_per_community_sample = get_top_authors(commu2papers_UBI_only, top_n=15)
        
        
            
        
        #%% Create df
        data = []
     
        for community in df['modularity_class'].unique():
            for authors, papers_sample  in zip(top_authors_per_community[community],
                                                                          top_cited_per_community_sample[community]):
                data.append({"id_most_used_ref":papers_sample[0], "n_most_used_ref":papers_sample[1], "share_most_used_ref": papers_sample[2],
                              "author_most_used":authors[0], "community": community})
        df_top10 = pd.DataFrame(data, columns = ["id_most_used_ref", "n_most_used_ref", "share_most_used_ref","author_most_used",
                                                  "community"])   
        
        df_top10['title_most_used_ref'] = df_top10['id_most_used_ref'].apply(get_title)
        df_top10['bib_most_used_ref'] = df_top10['id_most_used_ref'].apply(get_authors_year)

        db_concept = client['openAlex20240517']
        collection_concept = db_concept['concepts']
        
        lvl1 = []
        
        
        docs = collection_concept.find()
        for doc in tqdm.tqdm(docs):
            if doc["level"] == 0:
                lvl1.append(doc["display_name"])
        
        commu2discipline = {i:{j:0 for j in lvl1} for i in commu2papers}
        n_papers = {i:0 for i in commu2papers}
        
        for commu in commu2papers:
            for paper in commu2papers[commu]:
                n_papers[commu] += 1
                doc = collection.find_one({"id":paper})
                for concept in doc["concepts"]:
                    if concept["display_name"] in lvl1:
                        commu2discipline[commu][concept["display_name"]] += 1
                
                
        # Prepare the data for the DataFrame
        data = []
        for community, disciplines in commu2discipline.items():
            total_papers = n_papers[community]
            # Sort and get the top 10 disciplines
            sorted_disciplines = sorted(disciplines.items(), key=lambda item: item[1], reverse=True)[:15]
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
        
        column_order = ["community",
                        "discipline","n_discipline","share_discipline",
                        "author_most_used", 'title_most_used_ref','bib_most_used_ref',
                        "id_most_used_ref", "n_most_used_ref", "share_most_used_ref"]
        df_top10 = df_top10[column_order]
        
        df_top10.to_csv("Data/table1_{}_{}_{}.csv".format(i,start_year,end_year), index = False)
