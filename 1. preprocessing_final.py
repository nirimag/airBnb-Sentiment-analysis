# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:25:52 2017

@author: nirimag
"""

""" Text Analytics Project """

import numpy as np
import pandas as pd
from time import time

input_file = './Original/reviews.csv'

# load csv
review = pd.read_csv(input_file, header=0, encoding='utf-8')

# keeping only listing id & comments
review_subset = review.loc[:,('listing_id','comments')]

# converts from dataframe to a list 
review_test = review_subset.values.tolist() 
#review_test[0][1] #takes the comment of the first review

# to check for empty review fields (WIP)
count_empty = 0
allreviews = []
for i in range(len(review_test)):
    if type(review_test[i][1]) == float:
        count_empty += 1
    else:
        allreviews.append([review_test[i][0],review_test[i][1]])
print('Number of reviews:', len(allreviews))
print(count_empty)

import nltk
from nltk.tokenize import sent_tokenize

# split reviews by sentences and tag to respective listing_id
split = [] # len: 450,969
t0 = time()
for i in range(len(allreviews)): 
    sentences = sent_tokenize(allreviews[i][1])
    if len(sentences) == 1:
        split.append([allreviews[i][0],allreviews[i][1]])
    else: 
        for j in range(len(sentences)):
            split.append([allreviews[i][0],sentences[j]])
print("done in %0.3fs." % (time() - t0))

from textblob import TextBlob

# separating into eng and non-eng
output = []
foreign = []
error_lst = []
t0 = time()
for i in range(len(split)):
    try: 
        lan = TextBlob(split[i][2]).detect_language()
        if lan == 'en':
            output.append([split[i][0],split[i][1],split[i][2]])
            print(i)
        else: 
            foreign.append([split[i][0],split[i][1],split[i][2]])
    except: 
        error_lst.append([split[i][0],split[i][1],split[i][2]])
print("done in %0.3fs." % (time() - t0))

#==============================================================================
# load processed files directly from folder
#==============================================================================

errorlst = pd.read_csv('errorlst.csv', header=0, encoding='utf-8').values.tolist()
output = pd.read_csv('./Results/1. preprocess - new/output.csv', header=0, encoding='utf-8').values.tolist()
foreign = pd.read_csv('foreign.csv', header=0, encoding='utf-8').values.tolist()
foreign_err = pd.read_csv('foreign_err.csv', header=0, encoding='utf-8').values.tolist()

# to translate non-eng and append to final list
translated = []
err = []
t0=time()
for i in range(len(foreign)):
    t0 = time()
    try: 
        lan = TextBlob(foreign[i][1]).detect_language()
        processed = TextBlob(foreign[i][1]).translate(from_lang=lan, to='en')
        translated.append([foreign[i][0], processed]) # note: reviews in textblob format, but not reflected csv export file 
    except: 
        print(i, foreign[i][0], foreign[i][1])
        err.append([foreign[i][0], foreign[i][1]])
print("done in %0.3fs." % (time() - t0))

translated = pd.read_csv('./Results/1. preprocess - new/translated.csv', header=0, encoding='utf-8').values.tolist()
split = [] # len of 3960 translated split 
t0 = time()
for i in range(len(translated)): 
    sentences = sent_tokenize(translated[i][1])
    if len(sentences) == 1:
        split.append([translated[i][0],translated[i][1]])
    else: 
        for j in range(len(sentences)):
            split.append([translated[i][0],sentences[j]])
print("done in %0.3fs." % (time() - t0))

fullyprocessed = output + split #len: 447580; exported as data

# export to csv - if required; to change parameters accordingly 
#==============================================================================
# t0 = time()   
# with open('output.csv', 'w', encoding = 'utf-8') as outcsv:
#     #configure writer to write standard csv file
#     writer = csv.writer(outcsv, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
#     writer.writerow(['listing_id', 'comments'])
#     for i in range(len(output)):
#         try: 
#         #Write item to outcsv
#             writer.writerow([output[i][0],output[i][1]])
#         except: 
#             pass
# print("done in %0.3fs." % (time() - t0))
#==============================================================================
