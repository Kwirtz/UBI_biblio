# https://python.igraph.org/en/stable/api/igraph.Graph.html#community_spinglass

import tqdm
import random
import pickle
import pymongo
import numpy as np
import pandas as pd
import igraph as ig
from collections import defaultdict

# Load the CSV file
df = pd.read_csv('Data/collab_output.csv')

# Aggregate weights by summing them over the years
aggregated_df = df.groupby(['source', 'target']).agg({'weight': 'sum'}).reset_index()

# Create the graph
g = ig.Graph.TupleList(aggregated_df.itertuples(index=False), directed=False, edge_attrs=['weight'])

# Extract the largest connected component
if not g.is_connected():
    components = g.components()
    g = components.giant()

#%% Opti spin-glass

def get_membership_matrices(graph, communities):
    num_nodes = len(graph.vs)
    num_communities = len(communities)
    membership_matrix = np.zeros((num_nodes, num_communities))
    n_community = 0
    for community in communities:
        for i, node in enumerate(graph.vs):
            if i in community:
                membership_matrix[i, n_community] = 1
        n_community += 1 
    
    return membership_matrix

def get_vi(X, Y):
    # Function to compute VI score (same as before)
    N = X.shape[0]  
    p = np.sum(X, axis=0) / N  
    q = np.sum(Y, axis=0) / N  
    R = np.dot(X.T, Y) / N  
    
    VI = - R * (np.log(R / p[:, None]) + np.log((R / q[None, :])))
    VI[np.isnan(VI)] = 0  
    return np.sum(VI)  

# Function to run Spin Glass algorithm and compute VI score for a given gamma
def run_spin_glass_opti(graph, gamma, num_trials):
    VIs = []
    membership_matrices = []
    for _ in range(num_trials):
        communities = graph.community_spinglass(gamma=gamma, spins=20)
        membership_matrix = get_membership_matrices(graph, communities)
        membership_matrices.append(membership_matrix)
    for i, community1 in enumerate(membership_matrices):
        for j, community2 in enumerate(membership_matrices):
            VI = get_vi(community1, community2)
            VIs.append(VI)
    return np.mean(VIs)

# Create the graph from your data
# Example: g = ig.Graph.TupleList(aggregated_df.itertuples(index=False), edge_attrs=['weight'])

# Define the range of gamma values to explore
gamma_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0]
num_trials = 100

# Dictionary to store VI scores for each gamma
data = []

# Loop through each gamma value
for gamma in tqdm.tqdm(gamma_values):
    VI = run_spin_glass_opti(g, gamma, num_trials)
    data.append({"VI": VI, "Gamma": gamma})

df = pd.DataFrame(data, columns=["VI","Gamma"])
df

#%% spin-glass
random.seed(9)
communities = g.community_spinglass(gamma=0.8, spins=20)

node_names = []
community_memberships = []

# Iterate over communities
for idx, community in enumerate(communities):
    # Iterate over nodes in the community
    for node in community:
        node_names.append(g.vs[node]["name"])
        community_memberships.append(idx)

# Create DataFrame
df = pd.DataFrame({"Node": node_names, "Community": community_memberships})
df["Community"].value_counts()

df_national = pd.read_csv("Data/national_output.csv")

merged_df = df_national.merge(df, left_on="country", right_on="Node", how="inner")
merged_df = merged_df.drop(columns=["Node"])
merged_df.to_csv("Data/national_output_bi.csv",index=False)

agg_columns = ['n_total', 'n_solo_country', 'n_solo_author', 'n_collab']
keep_columns = ['country', 'Community', 'unique_authors']

# Aggregate the data
result = merged_df.groupby('country_name').agg({
    **{col: 'sum' for col in agg_columns},
    **{col: 'first' for col in keep_columns}
}).reset_index()

result.to_csv("Data/soule.csv",index= False)

#%%
country2commu = {}

for row in merged_df.iterrows():
    country2commu[row[1]["country"]] = row[1]["Community"]

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['UBI']
collection = db['works_UBI_gobu_2']

with open('Data/commu2papers.pkl', 'rb') as f:
    commu2papers = pickle.load(f)

countries2df = {i:pd.DataFrame(0, index=range(1960,2021),columns=[f'community_{col}' for col in commu2papers]) for i in merged_df["Community"].unique()}


start_year = 1960
end_year = 2020

query = {
    'publication_year': {
        '$gte': start_year,
        '$lte': end_year
    }
}

docs = collection.find(query)
for doc in tqdm.tqdm(docs):
    for community, papers in commu2papers.items():
        if doc["id"] in papers:
            topic_community = community
    country_list = []
    authors = doc["authorships"]
    year = doc["publication_year"]
    for author in authors:
        try:
            country = author["institutions"][0]["country_code"]
            if country != None:
                country_list.append(country2commu[country])
        except Exception as e:
            pass
    """
    for commu in country_list:
        countries2df[commu].at[year,"community_"+str(topic_community)] += 1
    """
    if len(set(country_list)) == 1:
        countries2df[country_list[0]].at[year,"community_"+str(topic_community)] += 1

    
for country_commu in countries2df:
    df_year_comm = countries2df[country_commu]
    row_sums = df_year_comm.sum(axis=1)
    
    # Divide each value in the DataFrame by the sum of its row
    df_year_comm_normalized = df_year_comm.div(row_sums, axis=0)
    df_year_comm_normalized["year"] = df_year_comm_normalized.index
    df_year_comm_normalized.to_csv("Data/Fig_commu_evolution_{}.csv".format(country_commu),index=False)


topic2countries = {"community_"+str(i):pd.DataFrame(0, index=range(1960,2021),columns=[f'countries_{col}' for col in countries2df]) for i in commu2papers}

for country_commu in countries2df:
    df_year_comm = countries2df[country_commu]
    row_sums = df_year_comm.sum(axis=1)
    
    # Divide each value in the DataFrame by the sum of its row
    df_year_comm_normalized = df_year_comm#.div(row_sums, axis=0)
    for topic_commu in df_year_comm_normalized.columns:
        topic2countries[topic_commu]["countries_"+str(country_commu)] = df_year_comm_normalized[topic_commu]

it = 0

for topic_commu in df_year_comm_normalized.columns:
    column_order = ["countries_0","countries_1","countries_2","countries_3"]
    topic2countries[topic_commu] = topic2countries[topic_commu][column_order]
    topic2countries[topic_commu]["Topics"] = topic_commu
    topic2countries[topic_commu]["year"] = topic2countries[topic_commu].index
    if it == 0:
        it += 1
        df_test = topic2countries[topic_commu]
    else:
        df_test = pd.concat([df_test,topic2countries[topic_commu]])
    topic2countries[topic_commu].to_csv("Data/Fig_topics2countries_evolution_{}.csv".format(topic_commu),index=False)
    
df_test.to_csv("Data/Fig_topics2countries_evolution.csv",index=False)

#%%