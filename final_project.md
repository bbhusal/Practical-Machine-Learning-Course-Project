
---
## "Practical Maching Learning Project"

 "Bikram Bhusal"
 "November 11, 2019"

## Overview
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, we  use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.The data consists of a Training data and a Test data

The goal of your project is to predict the manner in which they did the exercise. This is the “classe” variable in the training set.

## Getting Data

```{r,echo=TRUE}
trainD<-read.csv ("pml-training.csv",header = TRUE)
validD<-read.csv ("pml-testing.csv",header = TRUE)
dim(trainD)
dim(validD)
```
some useful packages:
```{r,echo=TRUE}
library(caret)
library(corrplot)
```

## Cleaning and Preparing the Data:
Note along the cleaning process we display the dimension of the reduced dataset
Here,
```{r,echo=TRUE}
# Removing the variables that contains missing values. 
trainD<- trainD[, colSums(is.na(trainD)) == 0]
validD <- validD[, colSums(is.na(validD)) == 0]
dim(trainD)
dim(validD)
```
and
```{r,echo=TRUE}
# removing the first seven variables as they have little impact on the outcome classe
trainD <- trainD[, -c(1:7)]
validD <- validD[, -c(1:7)]
dim(trainD)
dim(validD)
```

Preparing the datasets for prediction:
 
```{r,echo=TRUE}
set.seed(123)
R_training <- createDataPartition(y=trainD$classe, p=0.7, list=F)
training <- trainD[R_training, ]
testing <- trainD[-R_training, ]
```
Further, cleaning the data by removing the variables that are nearly zero variance
```{r,echo=TRUE}
# remove variables with nearly zero variance
nzv <- nearZeroVar(training)
training <- training[, -nzv]
testing <- testing[, -nzv]
dim(training)
dim(testing)
```
After this cleaning we are down now to 53 variables

## Model Building

Begin with buildilding a model using classification trees. 

Here is a classification tree:
```{r,echo=TRUE}
library(rattle)
library(rpart)
```

```{r,echo=TRUE}
set.seed(345)
modelfit_tree <- rpart(classe ~ ., data=training, method="class")
fancyRpartPlot(modelfit_tree)
```
![github-large](https://github.com/bbhusal/Practical-Machine-Learning-Course-Project/blob/master/Rplot.png)

We then validate the model “modelfit_tree” on the testData to find out how well it performs by looking at the accuracy variable:

```{r,echo=TRUE}
predict_tree <- predict(modelfit_tree, testing, type = "class")
cm_tree <- confusionMatrix(predict_tree, testing$classe)
cm_tree
```
We see that the accuracy rate of the model is low: 0.7315 and therefore the out-of-sample-error is about 0.27 which is considerable.

Random Forests model
```{r, echo=TRUE}
library(randomForest)
```
Fitting the model
```{r,echo=TRUE}
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modRF1 <- train(classe ~ ., data=training, method="rf", trControl=controlRF)
modRF1$finalModel
```
I see that it decided to use 500 trees and try 27 variables at each split.

Model Evaluation:
```{r,echo=TRUE}
predictRF1 <- predict(modRF1, newdata=testing)
cmrf <- confusionMatrix(predictRF1, testing$classe)
cmrf
```
The accuracy rate using the random forest is very high: Accuracy : 0.9922 and therefore the out-of-sample-error is equal to 0.0078.This is an excellent result, so rather than trying additional algorithms, I will use Random Forests to predict on the test set.

## Predictions
All that is left is to use this model to predict the classes of the validation data:

```{r,echo=TRUE}
pred_val <- predict(modRF1, validD)
pred_val
```
