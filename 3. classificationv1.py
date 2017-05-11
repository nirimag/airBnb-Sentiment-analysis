# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 11:48:03 2017

@author: nirimag
"""

""" Classification """
# PREPROCESSING WITH STEMMING & STOPWORDS REMOVAL FROM CUSTOMISED LIST

# console 6: listing
# console 5: host 400k label

import numpy as np
import pandas as pd
from time import time

rawdata = pd.read_csv('labels.csv', header=0, encoding='utf-8')

# to combine labelled set & output to csv
#==============================================================================
bing = pd.read_csv('set1 bing_labelled.csv', header=0, encoding='utf-8')
ray = pd.read_csv('set2 raymond_labelled.csv', header=0, encoding='utf-8')
ser = pd.read_csv('set3 serene_labelled.csv', header=0, encoding='utf-8') 
franky = pd.read_csv('set4 franky.csv', header=0, encoding='utf-8') 
niri = pd.read_csv('set5 niri.csv', header=0, encoding='utf-8') 
shanna = pd.read_csv('set6 shanna_labelled.csv', header=0, encoding='utf-8')
 
frames = [bing,ray,ser,franky,niri,shanna]
rawdata = pd.concat(frames)
rawdata.to_csv('combined.csv', sep=',', encoding='utf-8', index=False)
#==============================================================================

train,test = np.split(rawdata.sample(frac=1),[int(0.7*len(rawdata))])
train.sort(columns='labels',inplace=True)
test.sort(columns='labels', inplace=True)

train['idx']=range(0,len(train))
cols = ['idx','index','listing_id','comments','labels']
train = train.ix[:,cols].values.tolist()

test['idx']=range(0,len(test))
cols = ['idx','index','listing_id','comments','labels']
test = test.ix[:,cols].values.tolist()

amenities_cnt = []
attraction_cnt = []
food_cnt = []
multiple_cnt =[]
listing_cnt = []
location_cnt =[]
host_cnt = []
na_cnt = []
for i in range(len(train)): 
    if train[i][4] == 'amenities':
        amenities_cnt.append(train[i][0])
    elif train[i][4] == 'attraction':
        attraction_cnt.append(train[i][0])
    elif train[i][4] == 'food':
        food_cnt.append(train[i][0])
    elif train[i][4] == 'multiple':
        multiple_cnt.append(train[i][0])
    elif train[i][4] == 'listing':
        listing_cnt.append(train[i][0])
    elif train[i][4] == 'location':
        location_cnt.append(train[i][0])
    elif train[i][4] == 'host':
        host_cnt.append(train[i][0])
    elif train[i][4] == 'na':
        na_cnt.append(train[i][0])

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
import gensim
from gensim import corpora
from nltk.stem.porter import *
stemmer = PorterStemmer()

stoplist = stopwords.words('english')
# add on to stoplist
stoplist += ['also','although','us','really','never','however','especially','since','every',
             'etc', 'cc','would','seattle','everything','something','nothing','could', 'btw',
             'already','b','someone']

#==============================================================================
# preparing training data
#==============================================================================
processed = []
for i in range(len(train)):
    print('='*50)
    print('processing row %i' %i)
    print('words tokenisation...')
    tokens = word_tokenize(train[i][3])
    lower = [w.lower() for w in tokens]
    words = [w for w in lower if re.search('^[a-z]+$',w)]
    stopwords_rem = [w for w in words if w not in stoplist]
    stemmed = [stemmer.stem(w) for w in stopwords_rem]
    processed.append([train[i][0],train[i][1],train[i][2],stemmed,train[i][4]])
#    processed.append([train[i][0],train[i][1],train[i][2],stopwords_rem,train[i][4]])

words =[]
for i in range(len(processed)): 
    words.append(processed[i][3])

dictionary = corpora.Dictionary(words)
vecs = [dictionary.doc2bow(doc) for doc in words]

data_as_dict = [{id:1 for (id, tf_value) in vec} for vec in vecs] # for naive bayes

data_as_dict = [{id:tf_value for (id, tf_value) in vec} for vec in vecs] # for maxent/decision tree

amenities = [(d, 'amenities') for d in data_as_dict[min(amenities_cnt):max(amenities_cnt)+1]]
attraction = [(d, 'attraction') for d in data_as_dict[min(attraction_cnt):max(attraction_cnt)+1]]
food = [(d, 'food') for d in data_as_dict[min(food_cnt):max(food_cnt)+1]]
multiple = [(d, 'multiple') for d in data_as_dict[min(multiple_cnt):max(multiple_cnt)+1]]
host = [(d, 'host') for d in data_as_dict[min(host_cnt):max(host_cnt)+1]]
listing = [(d, 'listing') for d in data_as_dict[min(listing_cnt):max(listing_cnt)+1]]
location = [(d, 'location') for d in data_as_dict[min(location_cnt):max(location_cnt)+1]]
na = [(d, 'na') for d in data_as_dict[min(na_cnt):max(na_cnt)+1]]
all_train = amenities + attraction + food + host + listing + location + multiple + na
#all_train = host + na

nbclassifier = nltk.NaiveBayesClassifier.train(all_train)

meclassifier = nltk.MaxentClassifier.train(all_train)
dtclassifier = nltk.DecisionTreeClassifier.train(all_train)

#==============================================================================
# preparing test data 
#==============================================================================
amenities_cnt1 = []
attraction_cnt1 = []
food_cnt1 = []
multiple_cnt1 =[]
listing_cnt1 = []
location_cnt1 =[]
host_cnt1 = []
na_cnt1 = []
for i in range(len(test)): 
    if test[i][4] == 'amenities':
        amenities_cnt1.append(test[i][0])
    elif test[i][4] == 'attraction':
        attraction_cnt1.append(test[i][0])
    elif test[i][4] == 'food':
        food_cnt1.append(test[i][0])
    elif test[i][4] == 'multiple':
        multiple_cnt1.append(test[i][0])
    elif test[i][4] == 'listing':
        listing_cnt1.append(test[i][0])
    elif test[i][4] == 'location':
        location_cnt1.append(test[i][0])
    elif test[i][4] == 'host':
        host_cnt1.append(test[i][0])
    elif test[i][4] == 'na':
        na_cnt1.append(test[i][0])

processed1 = []
for i in range(len(test)):
    print('='*50)
    print('processing row %i' %i)
    print('words tokenisation...')
    tokens = word_tokenize(test[i][3])
    lower = [w.lower() for w in tokens]
    words = [w for w in lower if re.search('^[a-z]+$',w)]
    stopwords_rem = [w for w in words if w not in stoplist]
    stemmed = [stemmer.stem(w) for w in stopwords_rem]
    processed1.append([test[i][0],test[i][1],test[i][2],stemmed,test[i][4]])
#    processed1.append([test[i][0],test[i][1],test[i][2],stopwords_rem,test[i][4]])

words1 =[]
for i in range(len(processed1)): 
    words1.append(processed1[i][3])
    
test_vecs = [dictionary.doc2bow(doc) for doc in words1]

testdata_asdict = [{id:1 for (id, tf_value) in vec} for vec in test_vecs] # for naive bayes

testdata_asdict = [{id:tf_value for (id, tf_value) in vec} for vec in test_vecs] # for maxent/decision tree

amenities1 = [(d, 'amenities') for d in testdata_asdict[min(amenities_cnt1):max(amenities_cnt1)+1]]
attraction1 = [(d, 'attraction') for d in testdata_asdict[min(attraction_cnt1):max(attraction_cnt1)+1]]
food1 = [(d, 'food') for d in testdata_asdict[min(food_cnt1):max(food_cnt1)+1]]
multiple1 = [(d, 'multiple') for d in testdata_asdict[min(multiple_cnt1):max(multiple_cnt1)+1]]
host1 = [(d, 'host') for d in testdata_asdict[min(host_cnt1):max(host_cnt1)+1]]
listing1 = [(d, 'listing') for d in testdata_asdict[min(listing_cnt1):max(listing_cnt1)+1]]
location1 = [(d, 'location') for d in testdata_asdict[min(location_cnt1):max(location_cnt1)+1]]
na1 = [(d, 'na') for d in testdata_asdict[min(na_cnt1):max(na_cnt1)+1]]
all_test = amenities1 + attraction1 + food1 + host1 + listing1 + location1 + multiple1 + na1
#all_test = host1 + na1

nbaccuracy= nltk.classify.accuracy(nbclassifier, all_test)

meaccuracy= nltk.classify.accuracy(meclassifier, all_test)
dtaccuracy= nltk.classify.accuracy(dtclassifier, all_test)

print(nbaccuracy)
print(meaccuracy)
print(dtaccuracy)

#==============================================================================
# trying on full dataset
#==============================================================================
fulldata = pd.read_csv('data_idx.csv', header=0, encoding='utf-8').values.tolist()

processed_data = []
for i in range(len(fulldata)):
    print('='*50)
    print('processing row %i' %i)
    print('words tokenisation...')
    tokens = word_tokenize(fulldata[i][2])
    lower = [w.lower() for w in tokens]
    words = [w for w in lower if re.search('^[a-z]+$',w)]
    stopwords_rem = [w for w in words if w not in stoplist]
    stemmed = [stemmer.stem(w) for w in stopwords_rem]
    processed_data.append([fulldata[i][0],fulldata[i][1],stemmed])
#    processed_data.append([fulldata[i][0],fulldata[i][1],stopwords_rem])

processed_words =[]
for i in range(len(processed_data)): 
    processed_words.append(processed_data[i][2])
    
data_vecs = [dictionary.doc2bow(doc) for doc in processed_words]
data_vecs_asdict1 = [{id:1 for (id, tf_value) in vec} for vec in data_vecs] # for naive bayes
data_vecs_asdict2 = [{id:tf_value for (id, tf_value) in vec} for vec in data_vecs] # max ent/ decision tree 

# to test - output predicted label
# have to go back to data_idx.csv to check manually 
#num = [1237,12939,122393,8377]
#for i in num:
#    print('='*50)
#    print('Sentence:', fulldata[i][2])
#    print('- Naive Bayes predicted_label:', nbclassifier.classify(data_vecs_asdict1[i]),'\n',
#    '- Max Ent predicted_label:', meclassifier.classify(data_vecs_asdict2[i]),'\n',
#    '- Decision Tree predicted_label:', dtclassifier.classify(data_vecs_asdict2[i]))
 
# to output labelled containing labels predicted by the 3 classifiers 
labelled = []
for k in range(len(fulldata)):
    nb_label = nbclassifier.classify(data_vecs_asdict1[k])
    me_label = meclassifier.classify(data_vecs_asdict2[k])
    dt_label = dtclassifier.classify(data_vecs_asdict2[k])
    labelled.append([fulldata[k][0],fulldata[k][1],fulldata[k][2],nb_label,me_label,dt_label])
    
#=========================================#
# file export - if required               #
# change parameters accordingly           #
#=========================================#
import csv
t0 = time()
with open('labelled_host1.csv', 'w', encoding='utf-8') as outcsv:   
     #configure writer to write standard csv file
     writer = csv.writer(outcsv, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
     writer.writerow(['index','listing_id','comments','nb_label','me_label','dt_label'])
     for i in range(len(labelled)):
         #Write item to outcsv
         writer.writerow([labelled[i][0],labelled[i][1],labelled[i][2],labelled[i][3],labelled[i][4],labelled[i][5]])
print("done in %0.3fs." % (time() - t0))