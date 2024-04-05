##################################
# Equality in the New York Times #
#        Supervised Model        #
##################################

#Load packages

library(quanteda)
library(readtext)
library(dplyr)
library(tidyr)
library(topicmodels)
library(devtools)
library(ReadMe)      # Not currently available via CRAN for R 3.5.3 (see README)
library(reticulate)
reticulate::use_python("~/your_python_path")
reticulate::use_condaenv("~/anaconda3/envs/your_conda_env")
set.seed(6801)

## Set working directory to the location of the master "word2vec_time" folder

setwd("~/your_project_path/word2vec_time")

## Load text data

raw_text <- readtext("./data/NYT_complete_2016_1851.csv",
                     text_field = "materials")
raw_text$doc_id <- as.integer(seq(from = 1, to = length(raw_text$doc_id), by = 1))
raw_text$trainingset <- rep_len(c(rep(0, 5), 1), length(raw_text$doc_id))          #dim: (3105, 17)

## Add category codes to training set articles

coded_text <- readtext("./data/trainingset_category_codes.csv",
                       text_field = "text")
coded_text <- select(coded_text, -doc_id, -text) %>%    #remove some variables
  rename(doc_id = doc_id.1) %>%                         #rename variable for matching
  full_join(raw_text, by = "doc_id") %>%                #append codes to full text data
  select(doc_id, text, year, category) %>%              #select relevant variables
  mutate(trainingset = 0) %>%                           #add new trainingset variable for ReadMe
  rename(truth = category)                              #rename for ReadMe conventions
coded_text$trainingset[1:401] <- 1                      #recode trainingset variable for training texts

## get the ratio of trainingset in coded_text

# table(coded_text$trainingset)  # Test:2704, Training:401

## Producing data for ReadMe, separating by era

setwd("./data")
for (i in 1:nrow(coded_text)) {
  write(coded_text$text[i], paste0(coded_text$year[i], "_document", coded_text$doc_id[i], ".txt"))
}

training_set <- coded_text %>% filter(trainingset == 1) %>%           #Creates separate trained set
  mutate(filename = paste0(year, "_document", doc_id, ".txt")) %>%    #Appends file name to each text
  select(filename, truth, trainingset)                                #Selects relevant variables

quartercent <- seq(from = 1855, to = 2016, by = 25)       #Creates sequence of decades

quartercent_split <- function(x){                         #Function that splits test data into decades
  quarter <- coded_text %>% 
    filter(year >= x & year < x+25 & trainingset == 0) %>%
    mutate(filename = paste0(year, "_document", doc_id, ".txt")) %>%
    select(filename, truth, trainingset)
  return(quarter)
}

data_by_quarter <- lapply(quartercent, quartercent_split)      #Apply function defined above

## Create a table to show the number of documents by era

doc_count <- lapply(data_by_quarter, function(x) nrow(x))
eras <- c(1855, 1880, 1905, 1930, 1955, 1980, 2005)
doc_table <- data.frame(Era = eras, DocumentCount = unlist(doc_count))
doc_tibble <- as_tibble(doc_table)
transposed_tibble <- pivot_wider(doc_tibble, names_from = Era, values_from = DocumentCount)
knitr::kable(transposed_tibble)

## Bootstrapping ReadMe models (by era) to generate mean proportions

# NOTE: run the code in this section if you want to run the ReadMe function to replicate the
# process of generating 300 category proportion estimates for eras 2-7. If you choose to do this, be
# aware it is time and computationally intensive (approx. run time ~= 2 hours). Otherwise, simply load 
# the bootstrapped category proportions on line 95. 

start_time <- Sys.time()
samples <- 300
cat_results <- NULL
 
for (k in 2:length(data_by_quarter)) {
  props <- data.frame()
  while(nrow(props) < samples) {
    readme.results <- NULL
    test <- sample_frac(as.data.frame(data_by_quarter[k]), replace = T)
    train <- training_set
    data <- bind_rows(train, test)
    undergrad.results <- undergrad(control = data, stem = F, strip.tags = F, ignore.case = T, python3=TRUE,
                                   table.file = "tablefile.txt", threshold = .0001)
    undergrad.preprocess <- preprocess(undergrad.results)
    try(readme.results <- readme(undergrad.preprocess))
    try(props <- bind_rows(props, readme.results$est.CSMF))
  }
  cat_results <- bind_rows(cat_results, props)
  print(paste0("Completed era ", k, "."))
}  
 
save(cat_results, file = "cat_proportions.RData")

## Reading in document category proportions, averaging by era

load("cat_proportions.RData")

props_mean <- cat_results %>% 
  mutate(era = rep(2:7, each = 300)) %>%
  group_by(era) %>%
  summarize_all(mean) %>%
  gather("category", "mean", 2:16) %>%
  filter(category == 20 | category == 40 | category == 41 | category == 60 | category == 61)

write.csv(props_mean, file = "supervised_category_means.csv")
                      
end_time <- Sys.time()
execution_time <- end_time - start_time
cat(paste0("Total time spent for generating mean proportions: ", execution_time, "\n"))

