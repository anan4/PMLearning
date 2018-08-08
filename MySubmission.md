# My Submission to Prediction Assignment Writeup
## Approach
For consulting purposes, the readme file in this repository contains the statement of this assignment as it is publieshed in Coursera. To solve the assignment I will follow the next steps:
1.	Environment preparation
2.	Data cleaning and analysis
3.	Model training
4.	Final conclusions

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

Analyzing the data, you can observe that there are 160 features, but a lot of them with empty values or NA values.
Also you can observe that the first 6 features are only for identification but useless for evaluating the model.
I remove all of them:
pmltraining <- pmltraining[,-(1:6)]
pmltesting <- pmltesting[,-(1:6)]
validcols <- which(colSums(is.na(pmltraining) | pmltraining == "")==0)
pmltraining <- pmltraining[,validcols]

With this, I have reduced the dataset to 54 features.
Finally, I will check if I can reduce any feature more by checking correlation:
corMatrix <- cor(training[, -54])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", tl.cex = 0.8, tl.col = rgb(0, 0, 0))
 
The darkest colours indicate higher correlation. As correlations are quite few, I consider it is not necessary to perform a PCA (Principal Components Analysis).
Once I have a valid dataset, I create a partition of the pmltraining dataset, one for training (70%) and one for testing (30%) the models:
inTrain  <- createDataPartition(pmltraining$classe, p=0.7, list=FALSE)
training <- pmltraining[inTrain, ]
testing  <- pmltraining[-inTrain, ]
Thus, I have a dataset ready for testing.
## Model training
### Random Tree
First of all I have to admit that I have had a lot of performance problems running the models, specially using the command “train” in the caret package (including applying the mentor’s recommendation about this issue with the parallel execution). This has delayed and limited my analysis.
The first model I have tried is random forest, limited to 200 trees for performance issues and also because is high enough.
fitrf <- randomForest(classe ~ ., data=training, method="rf",keep.forest=TRUE,proximity=TRUE,ntree=200)
With the following output: 
Type of random forest: classification
 Number of trees: 200
 No. of variables tried at each split: 7
 
 OOB estimate of  error rate: 0.31%
 Confusion matrix:
 A 	B    	C   	D    	E  	class.error
 A	3905	0 	0	0	1 	0.0002560164
 B    	8 	2647    	3    	0    	0 	0.0041384500
 C    	0    	9 	2386    	1    	0 	0.0041736227
 D    	0    	0   	17 	2235    	0 	0.0075488455
 E    	0    	0    	0    	3 	2522 	0.0011881188

The results are very good; we proceed with prediction to evaluate the accuracy:
predrf <- predict(fitrf,newdata=testing)
cmrf <- confusionMatrix(testing$classe,predrf)
cmrf$overall[1]

Accuracy 0.9972812 

### Decision Tree
Secondly, I tried a decision tree model and see if the performance is better than the previous one:
fitrp <- rpart(classe ~ ., data=training, method="class", maxsurrogate=0)
Then I plot the tree:
rpart.plot(fitrp,varlen = -1,cex=0.5)

 
It’s time to predict and check the accuracy:
predrp <- predict(fitrp, newdata=testing, type="class")
cmrp <- confusionMatrix(predrp, testing$classe)

> cmrp$table
          Reference
Prediction    A    B    C    D    E
         A 1522  276   45  115   99
         B   36  596   34   22   87
         C   10   60  829  139   79
         D   86  144   57  637  118
         E   20   63   61   51  699

> cmrp$overall[1]
 Accuracy 0.7277825 

This model is worse the previous one in terms of accuracy in the predictions.

### Generalized Boosted Model
Finally I train a GBM model to see if it is a better model than the others:
	controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
fitgbm  <- train(classe ~ ., data=training, method = "gbm",trControl = controlGBM, verbose = FALSE)

With the following output:
> fitgbm
Stochastic Gradient Boosting 

13737 samples
   53 predictor
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Cross-Validated (5 fold, repeated 1 times) 
Summary of sample sizes: 10990, 10988, 10991, 10989, 10990 
Resampling results across tuning parameters:

  interaction.depth  n.trees  Accuracy   Kappa    
  1                   50      0.7572212  0.6918927
  1                  100      0.8311104  0.7863001
  1                  150      0.8710765  0.8368465
  2                   50      0.8847624  0.8541168
  2                  100      0.9373219  0.9206765
  2                  150      0.9629457  0.9531159
  3                   50      0.9325169  0.9145683
  3                  100      0.9709537  0.9632467
  3                  150      0.9854402  0.9815799

I predicted with the testing dataset to check the accuracy:

predgbm <- predict(fitgbm, newdata=testing)
cmgbm <- confusionMatrix(predgbm, testing$classe)

> cmgbm$table
          Reference
Prediction    A    B    C    D    E
         A 1672   14    0    1    0
         B    2 1110    4    2    4
         C    0   13 1018   12    2
         D    0    2    2  949    9
         E    0    0    2    0 1067

> cmgbm$overall[1]
 Accuracy 0.9882753 

We can observe that the model is very god but not so good as the random forest one.

Final conclusion
As you can observe in the previous analysis, the random forest model is the most accurate of the three models that I have evaluate. Then, this will be our model to predict the classe feature in the testing dataset:
> predTrf <- predict(fitrf,newdata = pmltesting)
> predTrf
 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
 B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 

