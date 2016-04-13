# Titanic Dataset
# SVM Model

rm(list=ls())  # CLEAR VARIABLES

# SET THE WORKING DIRECTORY
# You will need to change this to the directory where you put this code, train.csv, and test.csv
setwd("~/Desktop/Data Science/Portfolio/Kaggle Competitions/Titanic/Submittals")

train.data = read.csv("train.csv", na.strings=c("NA", ""))  # READ THE FILE
str(train.data)                                             # LOOK AT THE VARIABLE TYPES

library(e1071)
library(caret)
library(rpart)

#Pre-processing the data---------------------------------------------------------------------------

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
train.data = subset(train.data, select = -Ticket) # REMOVING Ticket Data
#---------------------------------------------------------------------------------------------------------------



# K-fold using the e1071 package

tuned = tune.svm(Survived~PassengerId +Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = train.data, gamma = 10^-2, cost = 10^2, tunecontrol = tune.control(cross = 10))
summary(tuned)

tuned$performances

svmfit = tuned$best.model
table(train.data[,c("Survived")], predict(svmfit))

# Accuracy estimate: 0.84




#--------------------------------------------------------------------------------------------------------------
# Making predictions on the test set

test.data = read.csv("test.csv", na.strings=c("NA", ""))  # READ THE FILE

#Pre-processing the data---------------------------------------------------------------------------

# Survived AND Pclass ARE int NUMERIC TYPES, BUT THESE NEED TO BE factor TYPES
# BECAUSE THEY ARE CATEGORICAL

test.data$Pclass = factor(test.data$Pclass)


# NEED TO DETECT MISSING VALUES FOR AGE
sum(is.na(test.data$Age) == TRUE)  # NUMBER OF MISSING VALUES FOR Age IS 177
sum(is.na(test.data$Age) == TRUE) / length(test.data$Age) # PERCENTAGE OF MISSING VALUES IS 19.8%


# GETTING A PERCENTAGE OF MISSING VALUES OF ALL ATTRIBUTES
sapply(test.data, function(df) {
  sum(is.na(df) == TRUE) / length(df)
})




# DISCOVERING THE TITLES FOR THE Name VARIABLE
test.data$Name = as.character(test.data$Name) #CHANGING TO CHARACTER TYPE
table_words = table(unlist(strsplit(test.data$Name, "\\s+"))) 
sort(table_words [grep('\\.',names(table_words))],decreasing=TRUE)

# DISCOVERING MISSING Name TITLES
library(stringr)
tb = cbind(test.data$Age, str_match(test.data$Name, "[a-zA-Z]+\\."))
table(tb[is.na(tb[,1]),2])

# IMPUTING MISSING VALUES AGE BASED ON THE MEAN VALUES FOR DIFFERENT TITLES
# FINDING THE MEAN
mean.mr = mean(test.data$Age[grepl(" Mr\\.", test.data$Name) & !is.na(test.data$Age)])
mean.mrs = mean(test.data$Age[grepl(" Mrs\\.", test.data$Name) & !is.na(test.data$Age)])
mean.ms = mean(test.data$Age[grepl(" Ms\\.", test.data$Name) & !is.na(test.data$Age)])
mean.miss = mean(test.data$Age[grepl(" Miss\\.", test.data$Name) & !is.na(test.data$Age)])
mean.master = mean(test.data$Age[grepl(" Master\\.", test.data$Name) & !is.na(test.data$Age)])
# ASSIGNING THE MEAN
test.data$Age[grepl(" Mr\\.", test.data$Name) & is.na(test.data$Age)] = mean.mr
test.data$Age[grepl(" Mrs\\.", test.data$Name) & is.na(test.data$Age)] = mean.mrs
test.data$Age[grepl(" Ms\\.", test.data$Name) & is.na(test.data$Age)] = mean.ms
test.data$Age[grepl(" Miss\\.", test.data$Name) & is.na(test.data$Age)] = mean.miss
test.data$Age[grepl(" Master\\.", test.data$Name) & is.na(test.data$Age)] = mean.master



test.data = subset(test.data, select = -Cabin) # REMOVING Cabin DATA; TOO MANY MISSING VALUES TO IMPUTE
test.data = subset(test.data, select = -Ticket) 
test.data$Age[89] = median(train.data$Age)
test.data$Fare[153] = median(train.data$Fare)

#---------------------------------------------------------------------------------------------------------------


# GETTING A PERCENTAGE OF MISSING VALUES OF ALL ATTRIBUTES
sapply(test.data, function(df) {
  sum(is.na(df) == TRUE) / length(df)
})


test.fit = predict(svmfit, newdata = test.data)


# Create dataframe for submission
id <- test.data$PassengerId

solution <- data.frame(PassengerId = id, Survived = test.fit)



# Write to csv file
write.csv(solution, file = "my_solution", row.names = FALSE, quote = F)

