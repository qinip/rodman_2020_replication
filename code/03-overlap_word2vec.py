"""
Created on Sun Mar 10 17:05:48 2019
@author: emma

Modified on April 2 2024
@author: eddy, minh
"""

##           word2vec          ##
##  Overlapping Time Analysis  ##

import numpy
import os
import math
from gensim.models import Word2Vec
import pickle
import csv
import random
from sklearn.utils import resample

random.seed(6801)

## If not using an interpreter, you can use __file__ to set the directory to the master
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))

## If __file__ is not defined, manually set path to location of master replication folder
dir_path = '/Users/zhiqiangji/GitRepos/Rodman_demo/word2vec_time/word2vec_time'
os.chdir(dir_path)

############################# 
## Loading all files/eras of texts into a master list of word2vec-ready sentences

os.chdir('./data')
eras = ['1855', '1880', '1905', '1930', '1955', '1980', '2005']
data = []

# for era in eras:
#     path = 'processed_' + era + 'era.txt'
#     fp = open(path)
#     x = fp.read() 
#     data.append(x.split('\n'))

for era in eras:
    path = 'processed_' + era + 'era.txt'
    with open(path) as fp:
        x = fp.read() 
        data.append(x.split('\n'))

list_of_lists = []
    
for i in data:
    sentence_lists = []
    for article in range(0, len(i)):
        x = i[article].split()
        sentence_lists.append(x)
    list_of_lists.append(sentence_lists)
    
## Appending overlap onto each corpus

end_pieces = []     #Create a list of articles from the end of each era

for j in range(0, (len(list_of_lists)-1)):
    test = list_of_lists[j]
    cutpoint = int(round(.9 * len(test)))
    overlap = test[cutpoint:]
    end_pieces.append(overlap)

begin_pieces = []   #Create a list of articles from the beginning of each era

for j in range(1, len(list_of_lists)):
    test = list_of_lists[j]
    cutpoint = int(round(.1 * len(test)))
    overlap = test[:cutpoint]
    begin_pieces.append(overlap)

for j in range(0, (len(list_of_lists)-1)):   #Append beginning pieces to previous era
    list_of_lists[j] = list_of_lists[j] + begin_pieces[j]
        
for j in range(0, (len(list_of_lists)-1)):   #Append ending pieces to following era
    list_of_lists[j+1] = list_of_lists[j+1] + end_pieces[j]
    
## word2vec analyses, by era   

#==============================================================================
#  Note: to replicate the word2vec model results, run lines 91-159.
#  This is computationally and time intensive (3-4 hours). To replicate using
#  model outputs from the paper, proceed to line 161 and load pickled model
#  outputs.
#============================================================================== 
    
n_bootstraps = 200
gender_similarity = []
intl_similarity = []
german_similarity = []
race_similarity = []
afam_similarity =[]

for j in range(0, len(list_of_lists)):
    sim_stats_gender = []
    sim_stats_intl = []
    sim_stats_german = []
    sim_stats_race = []
    sim_stats_afam = []
    for k in range(n_bootstraps):
        sentence_samples = resample(list_of_lists[j])
        model = Word2Vec(sentence_samples, vector_size = 100, min_count = 0, epochs = 200, 
                     sg = 1, hs = 0, negative = 5, window = 10, workers = 4)
        while True:
            try:
                sim_stats_gender.append(model.wv.similarity('equality','gender'))
                break
            except KeyError:
                sim_stats_gender.append('NA')
                break
        while True:
            try:
                sim_stats_intl.append(model.wv.similarity('equality','treaty'))
                break
            except KeyError:
                sim_stats_intl.append('NA')
                break
        while True:
            try:
                sim_stats_german.append(model.wv.similarity('equality','german'))
                break
            except KeyError:
                sim_stats_german.append('NA')
                break
        while True:
            try:
                sim_stats_race.append(model.wv.similarity('equality','race'))
                break
            except KeyError:
                sim_stats_race.append('NA')
                break
        while True:
            try:
                sim_stats_afam.append(model.wv.similarity('equality','african_american'))
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

with open('overlap_model_output.pickle', 'wb') as f:
    pickle.dump(stat_types, f)

## Loading model output

with open('overlap_model_output.pickle', 'rb') as f:
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

with open('overlap_mean_output.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(means_wCI)