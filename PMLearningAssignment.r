# Environment preparation
library(caret)
library(parallel)
library(doParallel)
library(randomForest)
library(rpart)
library(rpart.plot)
library(corrplot)

# Obtain the data
urltrain <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urltest  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
pmltraining <- read.csv(url(urltrain))
pmltesting  <- read.csv(url(urltest))

# Data cleaning: use only numeric columns with no empty/NA values
pmltraining <- pmltraining[,-(1:6)]
validcols <- which(colSums(is.na(pmltraining) | pmltraining == "")==0)
pmltraining <- pmltraining[,validcols]
pmltesting <- pmltesting[,validcols]

# Create data partition for training and testing
inTrain  <- createDataPartition(pmltraining$classe, p=0.7, list=FALSE)
training <- pmltraining[inTrain, ]
testing  <- pmltraining[-inTrain, ]

# Random forest model
fitrf <- randomForest(classe ~ ., data=training, keep.forest=TRUE,proximity=TRUE,ntree=200)

# randomForest(formula = classe ~ ., data = training, method = "rf",      keep.forest = TRUE, proximity = TRUE, ntree = 200) 
# Type of random forest: classification
# Number of trees: 200
# No. of variables tried at each split: 7
# 
# OOB estimate of  error rate: 0.31%
# Confusion matrix:
#   A    B    C    D    E  class.error
# A 3905    0    0    0    1 0.0002560164
# B    8 2647    3    0    0 0.0041384500
# C    0    9 2386    1    0 0.0041736227
# D    0    0   17 2235    0 0.0075488455
# E    0    0    0    3 2522 0.0011881188

# Predicitons with random forest
predrf <- predict(fitrf,newdata=testing)
cmrf <- confusionMatrix(predrf, testing$classe)
cmrf$overall[1]

# Accuracy 0.9972812 

# Decission Tree model
fitrp <- rpart(classe ~ ., data=training, method="class", maxsurrogate=0)
rpart.plot(fitrp,varlen = -1,cex=0.5)

# Predictions with decission tree
predrp <- predict(fitrp, newdata=testing, type="class")
cmrp <- confusionMatrix(predrp, testing$classe)
cmrp$table

# Reference
# Prediction    A    B    C    D    E
# A 1522  276   45  115   99
# B   36  596   34   22   87
# C   10   60  829  139   79
# D   86  144   57  637  118
# E   20   63   61   51  699

cmrp$overall[1]

# Accuracy 0.7277825

# Generalized Boosted Model
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
fitgbm  <- train(classe ~ ., data=training, method = "gbm",trControl = controlGBM, verbose = FALSE)

# Predicition with GBM
predgbm <- predict(fitgbm, newdata=testing)
cmgbm <- confusionMatrix(predgbm, testing$classe)
cmgbm$table

# Reference
# Prediction    A    B    C    D    E
# A 1672   14    0    1    0
# B    2 1110    4    2    4
# C    0   13 1018   12    2
# D    0    2    2  949    9
# E    0    0    2    0 1067

cmgbm$overall[1]

# Accuracy 0.9882753

# Testing the best model (RF) with the testing dataset
predict(fitrf,newdata = pmltesting)

# 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
# B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B
