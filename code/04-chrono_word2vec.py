"""
Created on Sun Mar 10 17:05:48 2019
@author: emma

Modified on April 2 2024
@author: eddy, minh
"""

##             word2vec            ##
##  Chronologically Trained Model  ##

import numpy
import pickle
import csv
import os
import math
import random
from gensim.models import Word2Vec

random.seed(6801)

## If not using an interpreter, you can use __file__ to set the directory to the master
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))

## If __file__ is not defined, manually set path to location of master replication folder
dir_path = '/Users/zhiqiangji/GitRepos/Rodman_demo/word2vec_time/word2vec_time'
os.chdir(dir_path)

os.chdir('./code')
from word2vec_functions import chrono_train
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
    
full_corpus = [item for sublist in list_of_lists for item in sublist]

## Modeling the full corpus and saving model output

#==============================================================================
#  Note: to replicate the word2vec model results, run lines 69-139.
#  This is computationally and time intensive (2-3 hours). To replicate using
#  model outputs from the paper, proceed to line 141 and load pickled model
#  outputs.
#============================================================================== 

start_model = Word2Vec(full_corpus, vector_size = 100, min_count = 0, epochs = 200, 
                     sg = 1, hs = 0, negative = 5, window = 10, workers = 4)

start_model.wv.similarity('equality','gender')
start_model.wv.similarity('equality','treaty')
start_model.wv.similarity('equality','german')
start_model.wv.similarity('equality','race')
start_model.wv.similarity('equality','african_american')
start_model.wv.similarity('equality','social')

start_model.save("model1_of_fullcorpus.model")

## Iteratively modeling each era with previous era model

# Run chrono_train function over each era to model starting with previous era, 
# saving cosine similarity outputs

results_all = []

results_1855 = chrono_train(100, list_of_lists[0], "model1_of_fullcorpus.model", "model2_of_1855.model")
results_all.append(results_1855)
    
results_1880 = chrono_train(100, list_of_lists[1], "model2_of_1855.model", "model3_of_1880.model")
results_all.append(results_1880)

results_1905 = chrono_train(100, list_of_lists[2], "model3_of_1880.model", "model4_of_1905.model")
results_all.append(results_1905)

results_1930 = chrono_train(100, list_of_lists[3], "model4_of_1905.model", "model5_of_1930.model")
results_all.append(results_1930)

results_1955 = chrono_train(100, list_of_lists[4], "model5_of_1930.model", "model6_of_1955.model")
results_all.append(results_1955)

results_1980 = chrono_train(100, list_of_lists[5], "model6_of_1955.model", "model7_of_1980.model")
results_all.append(results_1980)

results_2005 = chrono_train(100, list_of_lists[6], "model7_of_1980.model", "model8_of_2005.model")
results_all.append(results_2005)

# Set up the basic parameters for all loops

gender_similarity = []
intl_similarity = []
german_similarity = []
race_similarity = []
afam_similarity =[]
social_similarity = []

for i in range(0, len(results_all)):
    gender = results_all[i][0]
    gender_similarity.append(gender)
    intl = results_all[i][1]
    intl_similarity.append(intl)
    german = results_all[i][2]
    german_similarity.append(german)
    race = results_all[i][3]
    race_similarity.append(race)
    afam = results_all[i][4]
    afam_similarity.append(afam)
    social = results_all[i][5]
    social_similarity.append(social)

stat_types = [gender_similarity, intl_similarity, german_similarity, 
              race_similarity, afam_similarity, social_similarity]

## Saving model output

with open('chrono_model_output_320.pickle', 'wb') as f:
    pickle.dump(stat_types, f, protocol=pickle.HIGHEST_PROTOCOL) 

## Loading model output

with open('chrono_model_output.pickle', 'rb') as f:
    stat_types = pickle.load(f, encoding='latin1')               
    
print(len(stat_types)) 
## Finding means

means_wCI = []

for t in stat_types:
    stats = []
    for e in range(0, len(t)):
        test = [score for score in t[e] if score != 'NA']
        test = numpy.asarray(test)
        test_mean = test[0:99].mean()                   #Specify the number of samples to use
        test_error = 1.95 * (test.std()/math.sqrt(100)) #Specify the number of samples you used
        test_ci = (test_mean - test_error, test_mean + test_error)
        test_stats = test_mean#, test_ci
        stats.append(test_stats)
    means_wCI.append(stats)
   
## Exporting as a .csv file for visualization in R

means_wCI.insert(0, eras)

# with open('chrono_mean_output.csv', 'wb') as f:
#         writer = csv.writer(f, delimiter=',')
#         writer.writerows(means_wCI)
with open('chrono_mean_output.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(means_wCI)

## Exporting "social"-"equality" means and CIs

social = []
for e in range(0, len(stat_types[5])):
        test = [score for score in stat_types[5][e] if score != 'NA']
        test = numpy.asarray(test)
        test_mean = test[0:99].mean()                   #Specify the number of samples to use
        test_error = 1.95 * (test.std()/math.sqrt(100)) #Specify the number of samples you used
        test_stats = (test_mean, test_mean - test_error, test_mean + test_error)
        social.append(test_stats)

# with open('chrono_social_output.csv', 'wb') as f:
#         writer = csv.writer(f, delimiter=',')
#         writer.writerows(social)

with open('chrono_social_output.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(social)

