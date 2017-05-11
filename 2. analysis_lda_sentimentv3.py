# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 00:31:26 2017

@author: nirimag
"""
# load libraries
import numpy as np
import pandas as pd
from time import time

# load data (csv stored in the same folder)

# reviews by sentences 
data1 = pd.read_csv('./Processed/data_sent.csv', header=0, encoding='utf-8').values.tolist()

# to add index column
#data1['index']=range(1,len(data1)+1)
#cols = ['index','listing_id','comments']
#data1 = data1.ix[:,cols].values.tolist()

# names dataset
names = pd.read_csv('./Original/namelist.csv', header = 0,encoding='utf-8')
names = names.Names.astype(str).str.replace('\[|\]|\'', '').values.tolist()
names = [w.lower() for w in names]
    
#====================#
#        LDA         # 
#====================#

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
import gensim

stoplist = stopwords.words('english')
# add on to stoplist
stoplist += ['also','although','us','really','never','however','especially','since','every','etc', 'cc','would','seattle','everything','something','nothing','could', 'btw']

# *preprocessing
# NOTE:  uses data1 - by sentences 
alltokens1 = []
allwords1 = []
corpus1 = []
corpustag1 = []
t0 = time()
for i in range(len(data1)):
    print(50*'=')
    print('Processing row %i' %i)
    print('Words tokenization...')
    token = word_tokenize(data1[i][1])
    alltokens1.append([data1[i][0], token])
    print('Regex in progress...')
    words_only = [w for w in token if re.search('^[A-Za-z]+$', w)]
    allwords1.append([data1[i][0],words_only])
    print('Lowercase in progress...')
    lower = [w.lower() for w in words_only]
    print('Stopwords removal...')
    words = [w for w in lower if w not in stoplist]
    corpus1.append(words)
    corpustag1.append([data1[i][0],words])
print("done in %0.3fs." % (time() - t0))

# to replace all names to 'hosts'
t0 = time()
for i in range(len(corpustag1)):
    for k in range(len(corpustag1[i][1])):
        for j in names: 
            if corpustag1[i][1][k] == j: 
                corpustag1[i][1][k] = 'host'
print("done in %0.3fs." % (time() - t0))

#==================================================================================#    
# [NOUNS only] build lda on the entire dataset - get topics for each sentence
# NOTE:  nouns is from data1 
#==================================================================================# 
# pos-tagging

#== retrieve nouns only #
pos = []
nouns =[]    
for i in range(len(corpustag1)):    
    tagged=nltk.pos_tag(corpustag1[i][1])
    pos.append([corpustag1[i][0],tagged])
    noun = [word for word,tag in tagged if tag == 'NNP' or tag == 'NNPS' or tag == 'NN' or tag == 'NNPS']
    nouns.append([corpustag1[i][0],noun])
    print(i)

#== remove adj, adv, prepo #
pos = []
nouns =[]    
for i in range(len(corpustag1)):    
    tagged=nltk.pos_tag(corpustag1[i][1])
    pos.append([corpustag1[i][0],tagged])
    noun = [word for word,tag in tagged if tag != 'JJ' or tag != 'JJR' or tag != 'JJS' 
            or tag != 'IN' or tag != 'FW' or tag != 'RB' or tag != 'RBR' or tag != 'RBS']
    nouns.append([corpustag1[i][0],noun])
    print(i)
#==#

nouns1 =[] # nouns,without id - to build dictionary
for i in range(len(nouns)):
    nouns1.append(nouns[i][1])
noundic = gensim.corpora.Dictionary(nouns1) #len: 14697

t0 = time()
nounvecs = [noundic.doc2bow(doc) for doc in nouns1] #len: 447580
nounlda100 = gensim.models.ldamodel.LdaModel(corpus=nounvecs, id2word=noundic, num_topics=100, iterations=100)
nounlda50 = gensim.models.ldamodel.LdaModel(corpus=nounvecs, id2word=noundic, num_topics=50, iterations=100)
nounlda15 = gensim.models.ldamodel.LdaModel(corpus=nounvecs, id2word=noundic, num_topics=15, iterations=100)
print("done in %0.3fs." % (time() - t0))

nountop100 = nounlda100.show_topics(100,20)
nountop50 = nounlda50.show_topics(50,20)
nountop15 = nounlda15.show_topics(15,20)

#====================#
# SENTIMENT ANALYSIS # 
#====================#

#=================================================#
# to test sentiment analysis using textblob       #
#=================================================#
from textblob import TextBlob
score = [] 
for i in range(len(data1)):
    score.append([data1[i][0],data1[i][1],TextBlob(data1[i][1]).sentiment.polarity])
    print(i)

# niri's mtd
from textblob.sentiments import NaiveBayesAnalyzer 
score1 = []
t0 = time()
for i in range(len(data1)):
    score.append([data1[i][0],data1[i][1],TextBlob(data1[i][1],analyzer=NaiveBayesAnalyzer()).sentiment])
    print(i)
print("done in %0.3fs." % (time() - t0))

#=========================================#
# to test sentiment analysis using vader  #
#=========================================#
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
sentiment_score = [] # output as sentiment.csv
for i in range(len(data1)):
    sentence = data1[i][1]
    print(i)
    ss = sid.polarity_scores(sentence)
    sentiment_score.append([data1[i][0],data1[i][1],ss['compound'],ss['pos'],ss['neg'],ss['neu']])

#=========================================#
# file export - if required               #
# change parameters accordingly           #
#=========================================#
import csv
t0 = time()
with open('data_idx.csv', 'w', encoding='utf-8') as outcsv:   
     #configure writer to write standard csv file
     writer = csv.writer(outcsv, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
     writer.writerow(['index','listing_id','comments'])
     for i in range(len(data1)):
         #Write item to outcsv
         writer.writerow([data1[i][0],data1[i][1],data1[i][2]])
print("done in %0.3fs." % (time() - t0))