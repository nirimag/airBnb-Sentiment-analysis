# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:40:30 2017

@author: nirimag
"""

""" Sentiment Analysis """

import numpy as np
import pandas as pd
from time import time
from nltk import word_tokenize

rawdata = pd.read_csv('labelled_combined_new.csv', header=0, encoding='utf-8')
cols = ['host_label','location_label','listing_label']
data = rawdata.drop(cols,axis=1)

data1 = data.values.tolist()

# listing_id, comments, labels
output = []
for i in range(len(data1)):
    output.append([data1[i][0],data1[i][1],word_tokenize(data1[i][2])])

# grouping sentences with the same labels together - will be exported 
host_label = []
listing_label = []
location_label = []
na_label = []
for i in range(len(output)):
    for j in output[i][2]:
        if j == "listing":
            listing_label.append([output[i][0], output[i][1], output[i][2]])
        elif j == "location": 
            location_label.append([output[i][0], output[i][1], output[i][2]])
        elif j == "host": 
            host_label.append([output[i][0], output[i][1], output[i][2]])
        elif j == "na":
            na_label.append([output[i][0], output[i][1], output[i][2]])

#=========================================#
# sentiment analysis using vader  #
#=========================================#
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# note: unreadable excel file once exported - check

sid = SentimentIntensityAnalyzer()
host_sentiment_score = [] 
for i in range(len(host_label)):
    sentence = host_label[i][1]
    print(i)
    ss = sid.polarity_scores(sentence)
    host_sentiment_score.append([host_label[i][0],host_label[i][1],ss['compound'],ss['pos'],ss['neg'],ss['neu']])

location_sentiment_score = [] 
for i in range(len(location_label)):
    sentence = location_label[i][1]
    print(i)
    ss = sid.polarity_scores(sentence)
    location_sentiment_score.append([location_label[i][0],location_label[i][1],ss['compound'],ss['pos'],ss['neg'],ss['neu']])

listing_sentiment_score = [] 
for i in range(len(listing_label)):
    sentence = listing_label[i][1]
    print(i)
    ss = sid.polarity_scores(sentence)
    listing_sentiment_score.append([listing_label[i][0],listing_label[i][1],ss['compound'],ss['pos'],ss['neg'],ss['neu']])

na_sentiment_score = [] 
for i in range(len(na_label)):
    sentence = na_label[i][1]
    print(i)
    ss = sid.polarity_scores(sentence)
    na_sentiment_score.append([na_label[i][0],na_label[i][1],ss['compound'],ss['pos'],ss['neg'],ss['neu']])

#=========================================#
# file export - if required               #
# change parameters accordingly           #
#=========================================#
import csv
t0 = time()
with open('na_sentiment_score(new).csv', 'w', encoding='utf-8') as outcsv:   
     #configure writer to write standard csv file
     writer = csv.writer(outcsv, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
     writer.writerow(['listing_id', 'comments','cpd','pos','neg','neu'])
     for i in range(len(na_sentiment_score)):
         #Write item to outcsv
         writer.writerow([na_sentiment_score[i][0],na_sentiment_score[i][1],na_sentiment_score[i][2],na_sentiment_score[i][3],na_sentiment_score[i][4],na_sentiment_score[i][5]])
print("done in %0.3fs." % (time() - t0))
    