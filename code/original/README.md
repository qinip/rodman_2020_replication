# A Timely Intervention
This repository contains the replication materials for the article "A Timely Intervention: Tracking the Changing Meanings of Political Concepts with Word Vectors", to be published in _Political Analysis_, by Emma Rodman.

—--

## Setup and Benchmarks

**OS and software**
Original code run on MAC OSX 10.12.6, 8gb RAM, 1 cpu (1.6 GHz), 4 cores
R version 3.5.3
Python 2.7

**Required Packages**

`Python`:
- csv
- pickle
- numpy 
- os
- copy
- math
- random
- gensim
- sklearn

`R`:
- dplyr
- tidyr
- ggplot2
- quanteda
- readtext
- topicmodels
- devtools
- ReadMe (see below)
- ggthemes
- RColorBrewer
- stargazer

Note: `ReadMe` is currently unavailable via CRAN for the most recent version of `R` (3.5.3). Source files for `ReadMe` and `VA` (a required dependency also unavailable from CRAN) are available in the `./code/` directory, as well as from Gary King's [website](https://gking.harvard.edu/readme). To install from source after setting your working directory to the master "word2vec_time" folder:

```r
install.packages("./code/VA_0.9-2.12.tar.gz", repos = NULL, type="source")
install.packages("./code/readme_0.99837.tar.gz", repos = NULL, type="source")
```

**Benchmarks**
Estimates of run times for the scripts to produce the results of the main analyses are outlined below.

Run time per script (in order of replication):
- `01-gold_standard_model.R` = 5:18:42
- `02-naive_time_word2vec.py` = 11:08:28
- `03-overlap_word2vec.py` = 11:49:48
- `04-chrono_word2vec.py` = 5:25:54
- `05-aligned_word2vec.py` = 9:32:07
- `06-figures.R` = 0:00:11

## Instructions (in brief)

- Run `01-gold_standard_model.R` in order to produce the gold standard supervised model that generates document-category proportions by era.
- Run `02-naive_time_word2vec.py` in order to produce bootstrapped cosine similarities between equality and the five topic words using the naive time implementation of word2vec.
- Run `03-overlap_word2vec.py` in order to produce bootstrapped cosine similarities between equality and the five topic words using the overlapping time implementation of word2vec.
- Run `04-chrono_word2vec.py` in order to produce bootstrapped cosine similarities between equality and the five topic words, plus 'equality' and 'social', using the chronologically trained implementation of word2vec.
- Run `05-aligned_word2vec.py` in order to produce bootstrapped cosine similarities between equality and the five topic words using the aligned implementation of word2vec.
- Run `06-figures.R` in order to generate Figures 1 through 4, as well as Table 2.

## Data

The `./data/` directory contains the necessary data to replicate the analytical figures and tables in the article. Below, I describe each of the datasets in this directory:

- `NYT_complete_2016_1851.csv`: the master data set of articles exported from the New York Times Article API. 
- `aligned_mean_output.csv`: the mean cosine similarity values between 'equality' and the five topic words from the aligned `word2vec` model.
- `aligned_model_output.pickle`: the pickled cosine similarity scores for each pair of words for each model run. To be loaded to replicate exact article results or to avoid running full `word2vec` model from scratch.
- `cat_proportions.RData`: loads the document-category proportions for each era for each iteration of `ReadMe`. To be loaded to replicate exact article results or to avoid running full `ReadMe` models from scratch.
- `chrono_mean_output.csv`: the mean cosine similarity values between 'equality' and the five topic words from the chronologically trained `word2vec` model.
- `chrono_model_output.pickle`: the pickled cosine similarity scores for each pair of words for each model run. In addition to the five main pairs, also includes 'equality'-'social' pair. To be loaded to replicate exact article results or to avoid running full `word2vec` model from scratch.
- `chrono_social_output.csv`: the mean cosine similarity values (and confidence intervals around those values) for the 'equality'-'social' pair from the chronologically trained `word2vec` model.
- `model1_of_fullcorpus.model`, `model2_of_1855.model`, `model3_of_1880.model`, `model4_of_1905.model`, `model5_of_1930.model`, `model6_of_1955.model`, `model7_of_1980.model`, `model8_of_2005`: chronologically trained `word2vec` models for each era, used to initialize the subsequent era. Included here for replication purposes if one wants to replicate only a single era of the chronologically trained `word2vec` results.
- `naive_mean_output.csv`: the mean cosine similarity values between 'equality' and the five topic words from the naive `word2vec` model.
- `naive_model_output.pickle`: the pickled cosine similarity scores for each pair of words for each model run. To be loaded to replicate exact article results or to avoid running full `word2vec` model from scratch.
- `overlap_mean_output.csv`: the mean cosine similarity values between 'equality' and the five topic words from the overlapping `word2vec` model.
- `overlap_model_output.pickle`: the pickled cosine similarity scores for each pair of words for each model run. To be loaded to replicate exact article results or to avoid running full `word2vec` model from scratch.
- `processed_1855era.txt`, `processed_1880era.txt`, `processed_1905era.txt`, `processed_1930era.txt`, `processed_1955era.txt`, `processed_1980era.txt`, `processed_2005era.txt`: text files of corpus by era with selective stemming applied, for use with Python scripts for `word2vec` models.
- `supervised_category_means.csv`: mean proportion of documents in each category in each era, generated from the `R` package `ReadMe`.
- `trainingset_category_codes.csv`: hand coded NYT articles for use as the training set of documents for production of gold standard model using `ReadMe`

## Code

The `./code/` directory contains separate scripts to replicate each `ReadMe` and `word2vec` model. It also contains a script for figures and tables from the article. The `./figures/` directory contains a copy of each of the figures generated by this latter script; the `./tables/` directory contains a copy of Table 2.

- `word2vec_functions.py`: A set of Python functions used in the other scripts.
- `01-gold_standard_model.R`: Code to produce the gold standard model using the `R` package `ReadMe`; generates proportions of documents in each codebook category and outputs data for the five categories analyzed in the paper. To exactly replicate the results in the paper, load the `cat_proportions.RData` file rather than running ReadMe from scratch; otherwise your results will slightly vary from the paper results.
- `02-naive_time_word2vec.py`: Code to produce the naive time implementation of the `word2vec` model; generates bootstrapped cosine similarities for the five word pairs analyzed in the paper. To exactly replicate the results in the paper, load the `naive_model_output.pickle` file rather than running word2vec from scratch; otherwise your results will slightly vary from the paper results.
- `03-overlap_word2vec.py`: Code to produce the overlapping time implementation of the `word2vec` model; generates bootstrapped cosine similarities for the five word pairs analyzed in the paper. To exactly replicate the results in the paper, load the `overlap_model_output.pickle` file rather than running word2vec from scratch; otherwise your results will slightly vary from the paper results.
- `04-chrono_word2vec.py`: Code to produce the chronologically trained implementation of the `word2vec` model; generates bootstrapped cosine similarities for the five word pairs analyzed in the paper, as well as bootstrapped means and confidence intervals for the "social"-"equality" pair analyzed in Section 7 of the article.
- `05-aligned_word2vec.py`: Code to produce the aligned implementation of the `word2vec` model; generates bootstrapped cosine similarities for the five word pairs analyzed in the paper.
- `06-figures.R`: Code to produce the four figures in the article, as well as Table 2 of model assessment statistics.

