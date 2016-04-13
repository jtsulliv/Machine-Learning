# EXPLORING THE TITANIC DATASET USING THE FOLLOWING CLASSIFICATION MODELS
# RECURSIVE PARTITIONING TREES
# CONDITIONOAL INFERENCE TREES 
# KNN
# LOGISTIC REGRESSION
# NAIVE BAYES CLASSIFIER


rm(list=ls())  # CLEAR VARIABLES

# SET THE WORKING DIRECTORY
setwd("~/Desktop/Data Science/Learning/ML with R Cookbook")

train.data = read.csv("train.csv", na.strings=c("NA", ""))  # READ THE FILE
str(train.data)                                             # LOOK AT THE VARIABLE TYPES

# Survived AND Pclass ARE int NUMERIC TYPES, BUT THESE NEED TO BE factor TYPES
# BECAUSE THEY ARE CATEGORICAL
train.data$Survived = factor(train.data$Survived)
train.data$Pclass = factor(train.data$Pclass)
str(train.data)

# NEED TO DETECT MISSING VALUES FOR AGE
sum(is.na(train.data$Age) == TRUE)  # NUMBER OF MISSING VALUES FOR Age IS 177
sum(is.na(train.data$Age) == TRUE) / length(train.data$Age) # PERCENTAGE OF MISSING VALUES IS 19.8%


# GETTING A PERCENTAGE OF MISSING VALUES OF ALL ATTRIBUTES
sapply(train.data, function(df) {
  sum(is.na(df) == TRUE) / length(df)
})



# IMPUTING MISSING VALUES FOR Embarked
table(train.data$Embarked, useNA = "always")
train.data$Embarked[which(is.na(train.data$Embarked))] = 'S' # ASSIGN THE MISSING VALUES TO THE MOST PROBABLE PORT
table(train.data$Embarked, useNA = "always")

# DISCOVERING THE TITLES FOR THE Name VARIABLE
train.data$Name = as.character(train.data$Name) #CHANGING TO CHARACTER TYPE
table_words = table(unlist(strsplit(train.data$Name, "\\s+"))) 
sort(table_words [grep('\\.',names(table_words))],decreasing=TRUE)

# DISCOVERING MISSING Name TITLES
library(stringr)
tb = cbind(train.data$Age, str_match(train.data$Name, "[a-zA-Z]+\\."))
table(tb[is.na(tb[,1]),2])

# IMPUTING MISSING VALUES AGE BASED ON THE MEAN VALUES FOR DIFFERENT TITLES
# FINDING THE MEAN
mean.mr = mean(train.data$Age[grepl(" Mr\\.", train.data$Name) & !is.na(train.data$Age)])
mean.mrs = mean(train.data$Age[grepl(" Mrs\\.", train.data$Name) & !is.na(train.data$Age)])
mean.dr = mean(train.data$Age[grepl(" Dr\\.", train.data$Name) & !is.na(train.data$Age)])
mean.miss = mean(train.data$Age[grepl(" Miss\\.", train.data$Name) & !is.na(train.data$Age)])
mean.master = mean(train.data$Age[grepl(" Master\\.", train.data$Name) & !is.na(train.data$Age)])
# ASSIGNING THE MEAN
train.data$Age[grepl(" Mr\\.", train.data$Name) & is.na(train.data$Age)] = mean.mr
train.data$Age[grepl(" Mrs\\.", train.data$Name) & is.na(train.data$Age)] = mean.mrs
train.data$Age[grepl(" Dr\\.", train.data$Name) & is.na(train.data$Age)] = mean.dr
train.data$Age[grepl(" Miss\\.", train.data$Name) & is.na(train.data$Age)] = mean.miss
train.data$Age[grepl(" Master\\.", train.data$Name) & is.na(train.data$Age)] = mean.master

train.data = subset(train.data, select = -Cabin) # REMOVING Cabin DATA; TOO MANY MISSING VALUES TO IMPUTE



# EXPLORING AND VISUALIZING DATA---------------------------------------------------------------------------------------
# PERFORMING AN EXPLORATORY ANALYSIS, WHICH INVOLVES USING A VISUALIZATION PLOT 
# AND AN AGGREGATION METHOD TO SUMMARIZE THE DATA CHARACTERISTICS

# barplot(table(train.data$Survived), main="Passenger Survival", names = c("Perished","Survived"))
# barplot(table(train.data$Pclass), main = "Passenger Class", names = c("first","second","third"))
# barplot(table(train.data$Sex), main = "Passenger Gender")
# hist(train.data$Age, main="Passenger Age", xlab = "Age")
# barplot(table(train.data$SibSp), main="Passenger Siblings")
# barplot(table(train.data$Parch), main="Passenger Parch")
# hist(train.data$Fare, main="Passenger Fare", xlab = "Fare")
# barplot(table(train.data$Embarked), main="Port of Embarkation")
# 
# counts = table(train.data$Survived, train.data$Sex)
# barplot(counts, col=c("darkblue","red"),)

train.child = train.data$Survived[train.data$Age  <  13]
length(train.child[which(train.child==1)])/length(train.child)

# SPLITTING THE DATA
split.data = function(data, p = 0.7, s = 666){
  set.seed(s)
  index = sample(1:dim(data)[1])
  train = data[index[1:floor(dim(data)[1]*p)], ]
  test = data[index[((ceiling(dim(data)[1]*p))+1):dim(data)[1]], ]
  return(list(train = train, test = test))
}

allset = split.data(train.data, p = 0.7)
trainset = allset$train
testset = allset$test


#
#
#
# THE FOLLOWING CLASSIFICATION METHODS ARE INVESTIGATED
# 1) RECURSIVE PARTITIONING TREE
# 2) CONDITIONAL INFERENCE TREE
# 3) KNN
# 4) LOGISTIC REGRESSION
# 5) NAIVE BAYES
#
# SEE PAGE 185 OF ML COOKBOOK FOR PROS AND CONS
# 



#----------------------------------------------------------------------------------------------------------
# RECURSIVE PARTITIONING TREES
library(rpart)
rp.survived = rpart(Survived ~ Pclass + Sex + Age + SibSp + Fare + Parch + Embarked, data = trainset)
printcp(rp.survived) # PRINTING THE COMPLEXITY PARAMETER
#plotcp(rp.survived)  # PLOTTING THE COST COMPLEXITY PARAMETERS
summary(rp.survived)

# # VISUALIZING THE PARTITIONING TREE
# plot(rp.survived, margin = 0.1)
# text(rp.survived, all=TRUE, use.n = TRUE)
# plot(rp.survived, uniform=TRUE, branch = 0.6, margin = 0.1)
# text(rp.survived, all=TRUE, use.n=TRUE)

# MEASURING PERFORMANCE
rp.predictions = predict(rp.survived, testset, type="class")

library(caret)
confusionMatrix(table(rp.predictions, testset$Survived)) #84.64 ACCURATE

# PRUNING THE TREE TO AVOID OVERFITTING
min(rp.survived$cptable[,"xerror"])
which.min(rp.survived$cptable[,"xerror"])
survived.cp = rp.survived$cptable[3,"CP"]
survived.cp
prune.tree = prune(rp.survived, cp=survived.cp)

predictions.cp = predict(prune.tree, testset, type="class")
confusionMatrix(table(predictions.cp, testset$Survived)) # 82.4% ACCURATE
#...PRUNING THE TREE DID NOT IMPROVE TEST ACCURACY, BUT IT MAY HAVE MADE THE MODEL MORE ROBUST
#----------------------------------------------------------------------------------------------------------





#----------------------------------------------------------------------------------------------------------
# CONDITIONAL INFERENCE TREE
library(party)
ctree.model = ctree(Survived ~ Pclass + Sex + Age + SibSp + Fare + Parch + Embarked, data = trainset)

# MEASURING PERFORMANCE
ctree.predict = predict(ctree.model, testset)
confusionMatrix(table(ctree.predict, testset$Survived)) # 85.39% ACCURATE
#----------------------------------------------------------------------------------------------------------






#----------------------------------------------------------------------------------------------------------
# K-NEAREST NEIGHBOR (KNN)
library(class)
trainset_knn = subset(trainset, select= -c(Pclass,Name,Sex,Ticket,Embarked)) # REMOVING NON-NUMERIC variables
str(trainset_knn)
testset_knn = subset(testset, select= -c(Pclass,Name,Sex,Ticket,Embarked)) # REMOVING NON-NUMERIC variables
str(testset_knn)

surv.knn = knn(trainset_knn[,!  names(trainset_knn) %in% c("Survived")], testset_knn[,! names(testset_knn) %in% c("Survived")], trainset_knn$Survived, k=3)
summary(surv.knn)
confusionMatrix(table(testset$Survived, surv.knn)) # 72.66% ACCURACY
#---------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------------------------------------------------
# LOGISTIC REGRESSION 
fit = glm(Survived ~ Sex + SibSp, data=trainset, family=binomial) # DROPPED VARIABLES NOT SIGNIFICANT (i.e. p > 0.05)
summary(fit)
pred = predict(fit, testset, type="response")
pred_class = rep(1,length(pred))
pred_class[pred<=.5] = 0
tb = table(testset$Survived, pred_class)
confusionMatrix(tb) # 82.02% ACCURACY
#---------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------------------------------------------------
# NAIVE BAYES CLASSIFIER 
library(e1071)
classifier = naiveBayes(trainset[, !names(trainset) %in% c("Survived")], trainset$Survived)
bayes.table = table(predict(classifier, testset[,!names(testset) %in% c("Survived")]), testset_knn$Survived)
bayes.table
confusionMatrix(bayes.table) # 81.27% ACCURATE
#---------------------------------------------------------------------------------------------------------





#---------------------------------------------------------------------------------------------------------
