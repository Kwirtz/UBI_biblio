import tqdm
import pymongo
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['UBI']
collection = db['works_UBI_gobu_2']


df_country = pd.read_csv("Data/National_output.csv")
df_topics = pd.read_csv("Data/doc_centroid.csv")
unique_clusters = list(df_topics["Cluster"].unique())
unique_clusters.sort()

for cluster in unique_clusters:
    solo_column = f"Cluster_{cluster}_solo"
    inter_column = f"Cluster_{cluster}_inter"
    
    # Initialize these new columns with a default value, e.g., 0 or NaN
    df_country[solo_column] = 0
    df_country[inter_column] = 0

docs = collection.find({})
for doc in tqdm.tqdm(docs):
    year = doc["publication_year"]
    if doc["id"] in list(df_topics["id_"]):
        cluster = df_topics.loc[df_topics["id_"] == doc["id"], "Cluster"].values[0]
        if doc["countries_distinct_count"] == 1:
            for author in doc["authorships"]:
                try:
                    country = author["countries"][0]
                except:
                    continue
            row_index = df_country[(df_country["country"] == country) & (df_country["year"] == year)].index
            df_country.loc[row_index, "Cluster_{}_solo".format(cluster)] += 1            
        else:
            list_countries = []
            for author in doc["authorships"]:
                try:
                    country = author["countries"][0]
                    list_countries.append(country)
                except:
                    continue
            list_countries = list(set(list_countries))
            for country in list_countries:
                row_index = df_country[(df_country["country"] == country) & (df_country["year"] == year)].index
                df_country.loc[row_index, "Cluster_{}_inter".format(cluster)] += 1                       
            
# Group by 'Country' and sum the values
df_country_grouped = df_country.groupby(["country","country_name"], as_index=False).sum()


df_country_grouped["Total_intra"] = 0  
df_country_grouped["Total_inter"] = 0  

for cluster in unique_clusters:
    solo_column = f"Cluster_{cluster}_solo"
    inter_column = f"Cluster_{cluster}_inter"
    df_country_grouped["Total_intra"] += df_country_grouped[solo_column]
    df_country_grouped["Total_inter"] += df_country_grouped[inter_column]
    
for cluster in unique_clusters:
    solo_column = f"Cluster_{cluster}_solo"
    inter_column = f"Cluster_{cluster}_inter"    
    prop_intra_column = f"Cluster_{cluster}_solo_prop"
    prop_inter_column = f"Cluster_{cluster}_inter_prop"
    df_country_grouped[prop_intra_column] = df_country_grouped[solo_column]/df_country_grouped["Total_intra"]
    df_country_grouped[prop_inter_column] = df_country_grouped[inter_column]/df_country_grouped["Total_inter"]


df_country_grouped = df_country_grouped.dropna()
indicator_list = []
for row in df_country_grouped.iterrows():
    temp = 0
    for cluster in unique_clusters:
        prop_intra_column = f"Cluster_{cluster}_solo_prop"
        prop_inter_column = f"Cluster_{cluster}_inter_prop"
        temp += np.abs(row[1][prop_intra_column]-row[1][prop_inter_column])
    indicator_list.append(temp)

plt.figure(figsize=(10, 6))
sns.kdeplot(indicator_list, shade=True)
plt.title('Kernel Density Estimate (KDE) of Indicator List')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()