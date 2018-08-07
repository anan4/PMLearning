# My Submission to Prediction Assignment Writeup
## Approach
For consulting purposes, the readme file in this repository contains the statement of this assignment as it is publieshed in Coursera.
To solve the assignment I will follow the next steps:
1. Environment preparation
2. Data cleaning and analysis
3. Model training
4. Final conclussions
## Environment preparation
First thing to do is to prepare the environment: 

library(caret)
library(parallel)
library(doParallel)
library(randomForest)
library(rpart)
library(rpart.plot)
library(corrplot)

## Data cleaning and analysis
Then, it's time to download the data from the source:

urltrain <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

urltest  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

pmltraining <- read.csv(url(urltrain))

pmltesting  <- read.csv(url(urltest))

Analyzing the data, you can observe that there are 160 features but a lot of them with empty values or NA values.
Also you can observe that the first 6 features are  the first 6 features are only for identification but useless for analysis.
I remove all of them:

pmltraining <- pmltraining[,-(1:6)]

pmltesting <- pmltesting[,-(1:6)]

validcols <- which(colSums(is.na(pmltraining) | pmltraining == "")==0)

pmltraining <- pmltraining[,validcols]

With this, we have reduced the datased to 54 features.
Once I have a valid data set, I create a partition of the pmltraining dataset, one for training (70%) and one for testing (30%) the models:

inTrain  <- createDataPartition(pmltraining$classe, p=0.7, list=FALSE)

training <- pmltraining[inTrain, ]

testing  <- pmltraining[-inTrain, ]

Finally, I will check if I can reduce any feature more by checking correlation:



Thus, I have a dataset ready for testing

## Model training
