"""
Created on Sun Mar 10 17:05:48 2019
@author: emma

Modified on April 2 2024
@author: eddy, minh
"""

##                  word2vec                   ##
##   Trained & Aligned from Beginning to End   ##

import csv
import pickle
import numpy as np
import os
import math
import random
from word2vec_functions import *

random.seed(6801)

## If not using an interpreter, you can use __file__ to set the directory to the master
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))

## If __file__ is not defined, manually set path to location of master replication folder
dir_path = '/Users/zhiqiangji/GitRepos/Rodman_demo/word2vec_time/word2vec_time'
os.chdir(dir_path)

os.chdir('./code')
import word2vec_functions
os.chdir('..')

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
    
    
## Aligning the time slices and producing cosine similarity statistics    

#==============================================================================
#  Note: to replicate the word2vec model results, run lines 70-108.
#  This is computationally and time intensive (3-4 hours). To replicate using
#  model outputs from the paper, proceed to line 110 and load pickled model
#  outputs.
#==============================================================================      
            
iterations = 200

# specifies word2vec ready sentences in list of lists of list format;
# second input is number of iterations; output is cosine similarities for all
# models except the first one (the basis) 

stats = word2vec_functions.iterate_model_stats(list_of_lists, iterations)                                              

gender_similarity = []
intl_similarity = []
german_similarity = []
race_similarity = []
afam_similarity =[]              
                            
for j in range(0, len(eras)-1):
    gender = []
    intl = []
    german = []
    race = []
    afam = []
    for i in range(0, iterations):
        gender = gender + stats[j][i][0]
        intl = intl + stats[j][i][1]
        german = german + stats[j][i][2]
        race = race + stats[j][i][3]
        afam = afam + stats[j][i][4]
    gender_similarity.append(gender)
    intl_similarity.append(intl)
    german_similarity.append(german)
    race_similarity.append(race)
    afam_similarity.append(afam) 

stat_types = [gender_similarity, intl_similarity, german_similarity, 
              race_similarity, afam_similarity]
              
## Saving model output

with open('aligned_model_output.pickle', 'wb') as f:
    pickle.dump(stat_types, f, protocol=pickle.HIGHEST_PROTOCOL)

## Loading model output

with open('aligned_model_output.pickle', 'rb') as f:
    stat_types = pickle.load(f, encoding='latin1')           
              
## Generating means and confidence intervals of cosine similarity scores for 1880-2005 eras
# Note: not 1855, since that model is the "basis." Using scores from naive model for 1855
# in R visualization code.
              
means_wCI = []

for t in stat_types:
    type_stat = []
    for e in range(0, len(t)):
        test = [score for score in t[e] if score != 'NA']
        test = np.asarray(test)
        test_mean = test[0:99].mean()                   #Specify the number of samples to use
        test_error = 1.95 * (test.std()/math.sqrt(100)) #Specify the number of samples you used
        test_ci = (test_mean - test_error, test_mean + test_error)
        test_stats = test_mean#, test_ci                #Uncomment to add CIs
        type_stat.append(test_stats)
    means_wCI.append(type_stat)


## Exporting as a .csv file for visualization in R

means_wCI.insert(0, eras[1:])

# with open('aligned_mean_output.csv', 'wb') as f:
#         writer = csv.writer(f, delimiter=',')
#         writer.writerows(means_wCI)

with open('aligned_mean_output.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(means_wCI)
    
