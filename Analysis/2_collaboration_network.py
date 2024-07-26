# https://python.igraph.org/en/stable/api/igraph.Graph.html#community_spinglass

import tqdm
import numpy as np
import pandas as pd
import igraph as ig

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

communities = g.community_spinglass(gamma=1.5, spins=20)

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
merged_df.to_csv("Data/national_output.csv",index=False)


