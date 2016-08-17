# XBGoost classifier for Red Hat 
# https://www.kaggle.com/cartographic/predicting-red-hat-business-value/r-starter-around-0-98-auc

# Pulling in the necessary libraries
library(xgboost)
library(data.table)
library(Matrix)
library(FeatureHashing)


rm(list=ls())  # CLEAR VARIABLES

# Set your working directory here
setwd("~/Desktop/Machine-Learning/Kaggle-Competitions/Red-Hat")


# DATA PREPARATION----------------------------------------------------------------------------
# Reading in the people data 
people <- fread("people.csv") 
p_logi <- names(people)[which(sapply(people, is.logical))]     # identifyting logical features

# Changing True/False to 1/0
for (col in p_logi) set(people, j = col, 
                        value = as.integer(people[[col]]))

# Reading in the activity data
act_train <- fread("act_train.csv")

# Merging the people and activity data 
X_train <- merge(act_train, people, by = "people_id", all.x = T)

# Creating the outcome vector
Y_train <- X_train$outcome

# Dropping the outcome column from the merged training set
X_train[ , outcome := NULL]   
#---------------------------------------------------------------------------------------------




# PROCESS CATEGORICAL FEATURES WITH FEATURE HASHING-------------------------------------------
#
#
# Feature Hashing is a way to encode categorical features into a sparse matrix.  Numeric features 
# are automatically included in the matrix as is.  

b <- 2 ^ 22     # This is the hash size
f <- ~. - people_id - activity_id - date.x - date.y - 1    # Dropping some features

DF_train <- hashed.model.matrix(f, X_train, hash.size = b)
#----------------------------------------------------------------------------------------------






# XGBOOST MODEL--------------------------------------------------------------------------------
#
#

set.seed(75786)


unique_p <- unique(X_train$people_id)                   # Creating a unique vector of people_id
valid_p <- unique_p[sample(1:length(unique_p), 30000)]  # Creating a vector of 30000 people for validation set

valid <- which(X_train$people_id %in% valid_p)
model <- (1:length(X_train$people_id))[-valid]


# XGBoost hyperparameters
param <- list(objective = "binary:logistic",
              eval_metric = "auc",
              booster = "gblinear",
              eta = 0.03)


# Creating DMatrix for XGBoost 
dmodel  <- xgb.DMatrix(DF_train[model, ], label = Y_train[model])
dvalid  <- xgb.DMatrix(DF_train[valid, ], label = Y_train[valid])



m1 <- xgb.train(data = dmodel, param, nrounds = 100,
                watchlist = list(model = dmodel, valid = dvalid),
                print_every_n = 10)
#-------------------------------------------------------------------------------------------------




# Submission--------------------------------------------------------------------------------------

dtrain <- xgb.DMatrix(DF_train, label = Y_train)

m2 <- xgb.train(data = dtrain, param, nrounds = 100, 
                watchlist = list(train = dtrain), 
                print_every_n = 10)


test <- fread("act_test.csv", showProgress = F)
d2   <- merge(test, people, by = "people_id", all.x = T)

X_test <- hashed.model.matrix(f, d2, hash.size = b)
dtest  <- xgb.DMatrix(X_test)

out <- predict(m2, dtest)
sub <- data.frame(activity_id = d2$activity_id, outcome = out)
write.csv(sub, file = "sub.csv", row.names = F)
summary(sub$outcome)





