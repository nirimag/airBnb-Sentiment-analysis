# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 14:52:31 2017

@author: nirimag
"""


import pandas as pd
import re
from bs4 import BeautifulSoup 
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
#from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

data = pd.read_csv(r"C:\Users\nirimag\Google Drive\Text Analytics\Analysis\3. classification\classification results\binary_classification\labelled_combined_v1.csv",index_col=None)
########case 4
group=data.groupby(['listing_id','listing_class'],as_index=False).comments.sum()
group=group[group['listing_class']=='listing']
#group['listing_id']=group.index
group.reset_index(drop=True,inplace=True)


###########cleaning###############
def review_to_words( raw_review ):

     # 1. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review) 
    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split()    
    stops = set(stopwords.words("english"))    
    # 3. Remove stop words
    meaningful_words = [w for w in words if not w in stops]    
    return( " ".join( meaningful_words ))    

# Initialize an empty list to hold the clean reviews
clean_reviews = []
num_reviews = group["comments"].size
##call method for cleaning
for i in range(0,num_reviews):
    clean_reviews.append( review_to_words( group["comments"][i] ))  
    
#############sentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
sentiment_score = [] # output as sentiment.csv
for i in range(len(group)):
    sentences = clean_reviews[i]
    ss = sid.polarity_scores(sentences)
    sentiment_score.append([ss['compound'],ss['pos'],ss['neg'],ss['neu']])

##################export sentiment_score and group



