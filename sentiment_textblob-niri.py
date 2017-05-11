# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 23:49:27 2017

@author: nirimag
"""

import pandas as pd

train= pd.read_csv(r"D:\Text analytics\project\data.csv",index_col=None)

###########cleaning###############
def review_to_words( raw_review ):

     # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    
    stops = set(stopwords.words("english"))
    
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    
    return( " ".join( meaningful_words ))
    
# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]


train["polarity"] = ""
train["p_pos"] = ""
train["p_neg"] = ""

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import Blobber

tb = Blobber(analyzer=NaiveBayesAnalyzer())


#############test############
blob = tb(train['comments'][61])
blob.sentiment.classification
blob.sentiment.p_pos
blob.sentiment.p_neg

############full run##############
for i in range(0,num_reviews):
    blob = tb(train['comments'][i])
    train["polarity"][i]= blob.sentiment.classification
    train["p_pos"][i]= blob.sentiment.p_pos
    train["p_neg"][i]= blob.sentiment.p_neg
#
#
#
writer = pd.ExcelWriter('sentiment.xlsx', engine='xlsxwriter')
train.to_excel(writer,'Sheet1')
writer.save()
#
#