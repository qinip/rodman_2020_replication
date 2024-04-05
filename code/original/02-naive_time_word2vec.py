#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:05:48 2019

@author: emma
"""
##       word2vec        ##
##  Naive Time Analysis  ##

import numpy
import os
import pickle
import csv
import math
from gensim.models import Word2Vec
from sklearn.utils import resample
import random

random.seed(3919)

## If not using an interpreter, you can use __file__ to set the directory to the master
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))

## If __file__ is not defined, manually set path to location of master replication directory
dir_path = ''
os.chdir(dir_path)

############################# 
# Loading all files/eras of texts into a master list of word2vec-ready sentences

os.chdir('./data')
eras = ['1855', '1880', '1905', '1930', '1955', '1980', '2005']
data = []

for era in eras:
    path = 'processed_' + era + 'era.txt'
    fp = open(path)
    x = fp.read() 
    data.append(x.split('\n'))

list_of_lists = []
    
for i in data:
    sentence_lists = []
    for article in range(0, len(i)):
        x = i[article].split()
        sentence_lists.append(x)
    list_of_lists.append(sentence_lists)
    
## Iterating over lists of sentences by era to do era-by-era word2vec modeling

#==============================================================================
#  Note: to replicate the word2vec model results, run lines 60-138.
#  This is computationally and time intensive (11:08:28). To replicate using
#  model outputs from the paper, proceed to line 142 and load pickled model
#  outputs.
#==============================================================================

n_bootstraps = 200
gender_similarity = []
intl_similarity = []
german_similarity = []
race_similarity = []
afam_similarity =[]
equality_top10 = []

#==============================================================================
#  Add'l. note: because the output of interest a single word, resampling very small 
#  corpora with a very sparse topic may result in some samples that do not 
#  contain articles with that word -- i.e. as occurs here with "Germany" in the 
#  earliest era in this corpus. This loop appends NAs in those cases. In order
#  to get a sample size of at least 100 cosine similarity values, a larger
#  number of bootstraps is run than the desired final total (here, 200).
#==============================================================================

for j in range(0, len(list_of_lists)):
    sim_stats_gender = []
    sim_stats_intl = []
    sim_stats_german = []
    sim_stats_race = []
    sim_stats_afam = []
    for k in range(n_bootstraps):
        sentence_samples = resample(list_of_lists[j])
        model = Word2Vec(sentence_samples, size = 100, min_count = 0, iter = 200, 
                     sg = 1, hs = 0, negative = 5, window = 10, workers = 4)
        while True:
            try:
                sim_stats_gender.append(model.similarity('equality','gender'))
                break
            except KeyError:
                sim_stats_gender.append('NA')
                break
        while True:
            try:
                sim_stats_intl.append(model.similarity('equality','treaty'))
                break
            except KeyError:
                sim_stats_intl.append('NA')
                break
        while True:
            try:
                sim_stats_german.append(model.similarity('equality','german'))
                break
            except KeyError:
                sim_stats_german.append('NA')
                break
        while True:
            try:
                sim_stats_race.append(model.similarity('equality','race'))
                break
            except KeyError:
                sim_stats_race.append('NA')
                break
        while True:
            try:
                sim_stats_afam.append(model.similarity('equality','african_american'))
                break
            except KeyError:
                sim_stats_afam.append('NA')
                break
        run = k+1
        era = j+1
        print("Finished with run %d out of %d for era %d." % (run, n_bootstraps, era))
    print("*******Finished with era %d.*******" % (era))
    gender_similarity.append(sim_stats_gender)
    intl_similarity.append(sim_stats_intl)
    german_similarity.append(sim_stats_german)
    race_similarity.append(sim_stats_race)
    afam_similarity.append(sim_stats_afam)

stat_types = [gender_similarity, intl_similarity, german_similarity, 
              race_similarity, afam_similarity]

## Saving model output

with open('naive_model_output.pickle', 'wb') as f:
    pickle.dump(stat_types, f)

## Loading model output

with open('naive_model_output.pickle', 'rb') as f:
    stat_types = pickle.load(f)

## Finding means (and, optionally, CIs) 

means_wCI = []

for t in stat_types:
    stats = []
    for e in range(0, len(t)):
        test = [score for score in t[e] if score != 'NA']
        test = numpy.asarray(test)
        test_mean = test[0:99].mean()                   #Specify the number of samples to use
        test_error = 1.95 * (test.std()/math.sqrt(100)) #Specify the number of samples you used
        test_ci = (test_mean - test_error, test_mean + test_error)
        test_stats = test_mean#, test_ci                #Uncomment to add CIs
        stats.append(test_stats)
    means_wCI.append(stats)
        
## Exporting as a .csv file for visualization in R

means_wCI.insert(0, eras)

with open('naive_mean_output.csv', 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(means_wCI)


        
        
        
