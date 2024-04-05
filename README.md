# Replication: A Timely Intervention: Tracking the Changing Meanings of Political Concepts with Word Vectors

## Table of Contents
- [Introduction](#Introduction)
- [Software Environment](#Software-Environment)
- [Codes](#codes)
- [Data](#data)
- [Output](#output)


## Introduction

This repository offers the replication materials for the study by Emma Rodman, published in 2020. The paper, titled "A Timely Intervention: Tracking the Changing Meanings of Political Concepts with Word Vectors," appears in Volume 28 of *Political Analysis*, on pages 87 to 111.

>**Abstract**: Word vectorization is an emerging text-as-data method that shows great promise for automating the analysis of semantics—here, the cultural meanings of words—in large volumes of text. Yet successes with this method have largely been confined to massive corpora where the meanings of words are presumed to be fixed. In political science applications, however, many corpora are comparatively small and many interesting questions hinge on the recognition that meaning changes over time. Together, these two facts raise vexing methodological challenges. Can word vectors trace the changing cultural meanings of words in typical small corpora use cases? I test four time-sensitive implementations of word vectors (word2vec) against a gold standard developed from a modest data set of 161 years of newspaper coverage. I find that one implementation method clearly outperforms the others in matching human assessments of how public dialogues around equality in America have changed over time. In addition, I suggest best practices for using  word2vec  to study small corpora for time series questions, including bootstrap resampling of documents and pretraining of vectors. I close by showing that  word2vec  allows granular analysis of the changing meaning of words, an advance over other common text-as-data methods for semantic research questions.

**Access the Paper**: The official publication can be found online at Political Analysis, [DOI: 10.1017/pan.2019.23](https://doi.org/10.1017/pan.2019.23)

**Replication Materials**: The authors have generously provided all materials necessary for replicating the analyses presented in their article. These materials are hosted on the *Political Analysis* Dataverse within the Harvard Dataverse Network. You can access the dataset directly via [DOI: 10.7910/DVN/CGNX3M](https://doi.org/10.7910/DVN/CGNX3M).

## Software-Environment

### 1. Required packages

- **Python**: `csv`, `pickle`, `numpy`, `os`, `copy`, `math`, `random`, `gensim`, `sklearn`, `reticulate`
- **R**: `dplyr`, `tidyr`, `ggplot2`, `quanteda`, `readtext`, `topicmodels`, `devtools`, `ReadMe`, `ggthemes`, `RColorBrewer`, `stargazer`

### 2. Note on `ReadMe` package
The [ReadMe package for R](https://gking.harvard.edu/readme) and its required dependency `VA` are currently unavailable via CRAN for the most recent version of `R`. There are two ways to install these two packages. (**Windows Users** must ensure that Python is installed before installing `ReadMe`.) 

1. **Install from local files:** The author has kindly provided source files of these two packages for local sourcing. You can find them in the `code` folder in the materials for replication provided by the author. 
    To install from source after setting your working directory to the master folder of your local cloned of this repository:  

  ```r
  install.packages("./code/original/VA_0.9-2.12.tar.gz", repos = NULL, type="source")
  install.packages("./code/original/readme_0.99837.tar.gz", repos = NULL, type="source")
  ```

2. **Install from Github:** We recommend installing with `devtools`. You can also install `VA` and `ReadMe` from Github, given that you have the `devtools` installed:
	
   ```r
    library(devtools)  
    install_github("iqss-research/VA-package")  
    install_github("iqss-research/ReadMeV1")
	```
	
3. **Use `reticulate` to set Python and Conda environment:** Along with other packages, it is essential to designate the Python and/or Conda environment before running the `ReadMe` package. Here is an example:
	
	```r
	library(reticulate)
	reticulate::use_python("~/your/python_path/python")
	reticulate::use_condaenv("~/anaconda3/envs/your_env_path")
	```
	The original study was conducted using a Python 2.7 environment. In order to execute the portion of code from the original `01-gold_standard_model.R` script within a modern Python 3 environment, it is necessary to modify the call to the `undergrad()` function by adding the argument `python3=TRUE`. This adjustment ensures compatibility and proper execution in the updated programming environment.
	```r
	undergrad.results <- undergrad(control = data, stem = F, strip.tags = F, ignore.case = T, python3 = T, table.file = "tablefile.txt", threshold = .0001)
	```
### 3. Python compatibility
The author's four Python scripts were developed using Python 2.7 and `Gensim 3` for training Word2Vec models. However, these scripts contain syntax incompatible with the now mainstream Python 3 and  `Gensim 4`. To ensure compatibility, please utilize the modified Python scripts provided with this replication. 
                                   
## Codes
For a complete replication of the study, you will need to run seven scripts that are stored within the `code` directory. They are not aggregated into a single notebook due to their distinct model-training tasks and extensive running time. This collection comprises two `.R` scripts and five `.py` scripts. Additionally, you can find the original scripts provided by the author in the `code/original/` subdirectory. This subdirectory also houses the author's codebook and other documentation. 

## Data
This replication used the same data files as the original study. Due to the large data size, please download the original data from the Harvard Dataverse and put the `data` folder from decompressed files into your project root folder. 

Please note that the `data` folder contains both the documents used for training and the models generated along the way. The generated models from this replication, along with text data for training, code, and output, can be downloaded [here](https://1drv.ms/u/s!AjoR-7ptawqCqI9rTcKbMWHRxqUILw?e=7ngp7N).

Text data files used in this replication include a human-labeled training set produced by the author and her undergraduate coders. For more detailed information about the labeling of the training set used in the original study, please consult the codebook and documentation in the `code/original/` subdirectory.

## Output

- All the tables and figures generated during this replication process are placed in the `output` directory. The corresponding tables or figures generated in the original article are also included in the `output/original/` subdirectory for comparison. 

- The slides for our presentation and the report are located in the main folder of this repository.
                                
  	
