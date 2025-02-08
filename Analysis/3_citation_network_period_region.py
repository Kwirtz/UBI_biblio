# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:21:01 2024

@author: kevin
"""

#Bon alors pour les vagues, je te propose sans pause 1960 à 1980 puis 1981 à 2010 puis 2011
#si tu peux des plus grosses pauses entre les périodes, 1960-1975, 1980-2005 et 2010

#%% Init

import tqdm
import pickle
import random
import pymongo
import pandas as pd
import igraph as ig
from collections import defaultdict, Counter
import json

# Download stopwords if not already downloaded
#nltk.download('stopwords')
#nltk.download('punkt')

# Specify the path to your pickle file

# Open and load the pickle file
with open('Data/commu2papers.pkl', 'rb') as file:
    commu2papers = pickle.load(file)

merged_df = pd.read_csv("Data/national_output_bi.csv")

country2commu = {}

for row in merged_df.iterrows():
    country2commu[row[1]["country"]] = row[1]["Community"]

with open('Data/country2commu.json', "w") as json_file:
    json.dump(country2commu, json_file, indent=4)

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client['UBI']
collection = db['works_UBI_gobu_2']



# Define the year range
start_year = 1960
end_year = 1981

for goal_region in set(country2commu.values()):
    print(goal_region)
    query = {
        'publication_year': {
            '$gte': start_year,
            '$lte': end_year
        }
    }
    
    
    list_of_papers = []
    docs = collection.find(query)
    for doc in tqdm.tqdm(docs):
        region_list = []
        authors = doc["authorships"]
        for author in authors:
            try:
                country = author["institutions"][0]["country_code"]
                if country != None:
                    region_list.append(country2commu[country])
            except Exception as e:
                pass
        for region in list(set(region_list)):
            if region == goal_region:
                list_of_papers.append(doc["id"])   
        """
        if len(set(region_list)) == 1 and region_list[0] == goal_region:
            list_of_papers.append(doc["id"])   
        """
    list_papers_ubi = []
    
    docs = collection.find()
    for doc in tqdm.tqdm(docs):
        list_papers_ubi.append(doc["id"])

      
    #%% get stats for communities refs
    # Most used ref and authors
    # Most cited by UBI ref and authors
    # Most cited by global and authors
    
    papers2authors = defaultdict(list)
    
    
    docs = collection.find()
    for doc in tqdm.tqdm(docs):
        authors = doc["authorships"]
        for author in authors:
            name = author["author"]["display_name"]
            papers2authors[doc["id"]].append(name)
            
    commu2cited = {i:[] for i in commu2papers}      
    commu2cited_authors = {i:[] for i in commu2papers}
    
    
    docs = collection.find()
    for doc in tqdm.tqdm(docs):
        if doc["id"] in list_of_papers:
            community_value = None
            if doc["referenced_works"]:
                for community in commu2papers:
                    if doc["id"] in commu2papers[community]:
                        community_value = community
                        print(doc["id"],community_value)
                if community_value != None:
                    for ref in doc["referenced_works"]:
                        commu2cited[community_value].append(ref)
        
     
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
    
            # Ensure the list has exactly top_n elements, padding with None if necessary
            while len(top_authors_with_share) < top_n:
                top_authors_with_share.append((None, None, None))
            
            # Store the results in the dictionary
            top_authors_per_community[community] = top_authors_with_share
            
        return top_authors_per_community
        
    
    # Get the top 10 authors for each community
    top_authors_per_community = get_top_authors(commu2cited_authors, top_n=15)
    
    # Top 10 most used pqper per commu 
    commu2papers_UBI_only = {i:[] for i in commu2papers}
    
    
    for commu in commu2cited:
        for paper in commu2cited[commu]:
            if paper in list_papers_ubi:
                commu2papers_UBI_only[commu].append(paper)
    
    top_cited_per_community_sample = get_top_authors(commu2papers_UBI_only, top_n=15)
    
    #%% Global Top 10 most cited paper per commu in OpenAlex
    
    most_cited_global = defaultdict(lambda:defaultdict(int))
    
    for community in commu2papers:
        for paper in commu2papers[community]:
            if paper in list_of_papers:
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
        top_papers = sorted_papers[:15]
        
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
            if paper in list_of_papers:
                most_cited_UBI[community][paper] = papers2cited_UBI[paper]
    
    top_cited_per_community_UBI = {}
        
    for community, papers in most_cited_UBI.items():
        # Sort papers by number of citations in descending order
        sorted_papers = sorted(papers.items(), key=lambda item: item[1], reverse=True)
        total_citations = sum(paper[1] for paper in papers.items())
        top_papers = sorted_papers[:15]
        
        # Calculate the share for each top paper
        top_papers_with_share = [(paper_id, count, count / total_citations) for paper_id, count in top_papers]
    
        # Select the top N papers
        top_cited_per_community_UBI[community] = top_papers_with_share
    
    
    
    #%%
    
    data = []
    
    for community in top_cited_per_community_UBI:
        for authors, papers_sample, papers_UBI, papers_global in zip(top_authors_per_community[community],
                                                                                          top_cited_per_community_sample[community],
                                                                                          top_cited_per_community_UBI[community],
                                                                                          top_cited_per_community_global[community]):
            data.append({"community": community, "id_most_used_ref":papers_sample[0], "n_most_used_ref":papers_sample[1], "share_most_used_ref": papers_sample[2],
                         "author_most_used":authors[0],
                         "id_most_cited_UBI":papers_UBI[0], "n_most_cited_UBI":papers_UBI[1], "share_most_cited_UBI": papers_UBI[2],
                         "id_most_cited_global":papers_global[0], "n_most_cited_global":papers_global[1], "share_most_cited_global": papers_global[2]})
    df_top10 = pd.DataFrame(data, columns = ["id_most_used_ref", "n_most_used_ref", "share_most_used_ref","author_most_used",
                                             "id_most_cited_UBI", "n_most_cited_UBI", "share_most_cited_UBI",
                                             "id_most_cited_global", "n_most_cited_global", "share_most_cited_global",
                                             "community"])
    
    def get_title(id_):
        try:
            doc = collection.find_one({"id":id_})["title"]
        except:
            doc = None
        return doc
    
    def get_authors_year(id_):
        if id_ != None:
            doc = collection.find_one({"id":id_})
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
    print(goal_region)
    print("Data/table_{}_{}_region{}.csv".format(start_year,end_year,goal_region))
    df_top10.to_csv("Data/table_{}_{}_region{}.csv".format(start_year,end_year,goal_region))
        
