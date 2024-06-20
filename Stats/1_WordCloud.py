#%% Init

import tqdm
import nltk
import string
import pymongo
import pandas as pd
from langdetect import detect
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from nltk.corpus import stopwords
from nltk import bigrams, trigrams
from langdetect import DetectorFactory
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download the NLTK stopwords dataset
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')

DetectorFactory.seed = 0
lemmatizer = WordNetLemmatizer()

# MongoDB connection
Client = pymongo.MongoClient("mongodb://localhost:27017")
db = Client["UBI"]
collection = db["works_UBI_gobu"]


#%% Function to check if a text is in English

def is_english(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return False

def clear_text(text):
    # Remove leading and trailing spaces
    text = text.strip()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if len(token) > 1]
    
    # Remove punctuation from the text
    lemmatized_text = ''.join([char for char in ' '.join(lemmatized_tokens) if char not in string.punctuation])
    
    cleaned_text = ' '.join(lemmatized_text.split())
    return cleaned_text 

#%% Function for word cloud

stop_words = set(stopwords.words('english'))
take_not_journals = {"full-text", "http", "doi", "not available for this content", "pdf", "url", "access", "vol"}

def wordcloud_OpenAlex(Concepts=True, Title=False, Abstract=False,
                       Gram = 1, Stop_words=stop_words, query = {}, only_en=True, Years=None):
    docs = collection.find(query)
    list_text = []

    for doc in tqdm.tqdm(docs):
        if only_en and is_english(doc["title"]) == False:
            continue
        # Extract title, abstract, and concepts from the document if they exist
        if Title:
            title = doc.get("title", "")
        else:
            title = None
        if Abstract:
            abstract = doc.get("abstract", "").lower()
            for word in take_not_journals:
                if word in abstract:
                    abstract = ""            
        else:
            abstract = None
        if Concepts:
            concepts = doc.get("concepts", [])
        else:
            concepts = None
        
        # Concatenate title and abstract
        if title and abstract:
            text = title + " " + abstract
        elif title:
            text = title
        elif abstract:
            text = abstract
        else:
            text = ""
        # Iterate through concepts
        if Concepts:
            for concept in concepts:
                if concept.get("level", 0) > 2:
                    keyword = concept.get("display_name", "")
                    text += " " + keyword

        # Remove stopwords from the list of words
        cleaned_text = clear_text(text)
        filtered_text = " ".join([word.lower() for word in cleaned_text.split(" ") if word.lower() not in Stop_words])
        #if len(filtered_text.split(" ")) < Gram:
        #    continue 
        if Gram == 1:
            list_text.append(filtered_text)
        elif Gram == 2:
            list_text.extend(['_'.join(bigram) for bigram in bigrams(filtered_text.split(" "))])
        elif Gram == 3:
            list_text.extend(['_'.join(trigram) for trigram in trigrams(filtered_text.split(" "))])

    input_word_cloud = " ".join(list_text)
    wordcloud = WordCloud(width=1600, height=800, background_color="white", collocations=False).generate(input_word_cloud)

    # Plot the WordCloud image
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("Results/Figures/wordcloud_{}_{}_{}.pdf".format(str(Gram),str(Title),str(Abstract)), format="pdf", dpi=300)
    return list_text

test = wordcloud_OpenAlex(Gram=3,Concepts=False, Title=True, Abstract = True)

#%% Total
# Fetch keywords from MongoDB
keywords_list = []
docs = collection.find({})

for doc in docs:
    concepts = doc["concepts"]
    for concept in concepts:
        if concept["level"] > 2:
            keyword = concept["display_name"]
            keywords_list.append(keyword)


# Create a string from the list of keywords
keywords_text = " ".join([item for item in keywords_list if "income" not in item.lower() and "basic" not in item.lower()])

wordcloud = WordCloud(width=800, height=400, background_color="white", collocations=False).generate(keywords_text)

# Plot the WordCloud image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#%% Title + abstract + keywords

stop_words = set(stopwords.words('english'))
docs = collection.find({})
list_text = []

for doc in tqdm.tqdm(docs):
    # Extract title, abstract, and concepts from the document if they exist
    title = doc.get("title", "")
    abstract = doc.get("abstract", "")
    concepts = doc.get("concepts", [])
    
    # Concatenate title and abstract
    if title and abstract:
        text = title + " " + abstract
    elif title:
        text = title
    elif abstract:
        text = abstract
    else:
        text = ""
    # Iterate through concepts
    for concept in concepts:
        if concept.get("level", 0) > 2:
            keyword = concept.get("display_name", "")
            keywords_list.append(keyword)
    
    # Add title, abstract, and concepts to the text
    text += " ".join([item for item in keywords_list if "income" not in item.lower() and "basic" not in item.lower()])

    # Remove stopwords from the list of words
    filtered_text = " ".join([word.lower() for word in text.split(" ") if word.lower() not in stop_words and word not in string.punctuation])
    list_text.append(filtered_text)

input_word_cloud = " ".join(list_text)
wordcloud = WordCloud(width=800, height=400, background_color="white", collocations=False).generate(input_word_cloud)

# Plot the WordCloud image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#%% WordCloud per year

years = collection.distinct("publication_year",{"abstract": {"$exists": True}})
keywords_dict = {i:[] for i in years}


for year in tqdm.tqdm(years):    
    docs = collection.find({"publication_year":year,
                            "abstract":{"$exists":1}
                            })
    for doc in docs:
        abstract = doc["abstract"]
        # Tokenize the string into words
        words = word_tokenize(abstract)
        # Load the English stopwords
        stop_words = set(stopwords.words('english'))
        # Remove stopwords from the list of words
        filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word not in string.punctuation]
        keywords_dict[year] += filtered_words
        
df = pd.DataFrame(list(keywords_dict.items()), columns=["Year", "Words"])


# Initialize Plotly figure
fig = go.Figure()

# Create Word Cloud for each year
for year in tqdm.tqdm(df['Year'].unique()):
    text_for_year = df[df['Year'] == year]['Words'].explode().astype(str).str.cat(sep=' ')  # Convert Series of lists to a single string   
    # Create Word Cloud using matplotlib
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_for_year)
    
    # Display Word Cloud using Plotly
    img = plt.imshow(wordcloud, interpolation='bilinear')
    fig.add_trace(
        go.Image(z=img.to_rgba(wordcloud), visible=False, name=str(year))
    )

# Add slider
steps = [
    {"args": [{"visible": [False] * i + [True] + [False] * (len(df['Year'].unique()) - i - 1)}],
     "label": str(year),
     "method": "update"}
    for i, year in enumerate(df['Year'].unique())
]

sliders = [dict(active=0, pad={"t": 1}, steps=steps)]

# Update layout
fig.update_layout(
    sliders=sliders,
    showlegend=False
)

# Save the figure to an HTML file
fig.write_html("Results/Figures/wordcloud_slider.html")