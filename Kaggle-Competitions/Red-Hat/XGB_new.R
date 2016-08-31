# Xgboost + data leak 



library(data.table)
library(FeatureHashing)
library(xgboost)
library(dplyr)
library(Matrix)

# Reading in the data

train=fread('act_train.csv', verbose = FALSE, showProgress = FALSE) %>% as.data.frame() # SILENCE
test=fread('act_test.csv', verbose = FALSE, showProgress = FALSE) %>% as.data.frame() # SILENCE


#people data frame
people=fread('people.csv', verbose = FALSE, showProgress = FALSE) %>% as.data.frame() # SILENCE


cat(Sys.time())
cat("Processing data\n")

people$char_1<-NULL #unnecessary duplicate to char_2
names(people)[2:length(names(people))]=paste0('people_',names(people)[2:length(names(people))])

p_logi <- names(people)[which(sapply(people, is.logical))]
for (col in p_logi) set(people, j = col, value = as.numeric(people[[col]]))

#reducing group_1 dimension
people$people_group_1[people$people_group_1 %in% names(which(table(people$people_group_1)==1))]='group unique'


#reducing char_10 dimension
#unique.char_10=
#  rbind(
#    select(train,people_id,char_10),
#    select(test,people_id,char_10)) %>% group_by(char_10) %>% 
#  summarize(n=n_distinct(people_id)) %>% 
#  filter(n==1) %>% 
#  select(char_10) %>%
#  as.matrix() %>% 
#  as.vector()

#train$char_10[train$char_10 %in% unique.char_10]='type unique'
#test$char_10[test$char_10 %in% unique.char_10]='type unique'

d1 <- merge(train, people, by = "people_id", all.x = T)
d2 <- merge(test, people, by = "people_id", all.x = T)
Y <- d1$outcome
d1$outcome <- NULL

row.train=nrow(train)
gc(verbose=FALSE)

D=rbind(d1,d2)
D$i=1:dim(D)[1]


###uncomment this for CV run
#set.seed(120)
#unique_p <- unique(d1$people_id)
#valid_p  <- unique_p[sample(1:length(unique_p), 40000)]
#valid <- which(d1$people_id %in% valid_p)
#model <- (1:length(d1$people_id))[-valid]

test_activity_id=test$activity_id
rm(train,test,d1,d2);gc(verbose=FALSE)


# Feature Engineering
# putting in the date features

# Changing the date strings to dates 
D$date = as.Date(D$date)
D$people_date = as.Date(D$people_date)

# Putting in new features for the dates as difference in days between the current date and the baseline date
min_date = min(D$date, D$people_date)
D$date.days = as.numeric(D$date - min_date)                            # char days
D$people_date.days = as.numeric(D$people_date - min_date)

# Putting in new feature for month
D$date.month = format(D$date, "%m")
D$people_date.month = format(D$people_date, "%m")

# Putting in new feature for year
D$date.year = format(D$date, "%y")
D$people_date.year = format(D$people_date, "%y")

# Putting in new feature for group_1 + month + year
D$date.1 = paste(D$people_group_1, D$date.month, D$date.year, sep = "_")
#D$date.2 = paste(D$people_group_1, D$people_date.month, D$people_date.year, sep = "_")

#replacing all group_date features with a unique group
D[grep('unique', D$date.1),]$date.1<-"unique_date_group"
#[grep('unique', D$date.2),]$date.2<-"unique_date_group"



# Adding supervised ratio for date.1 feature
Dtrain<-cbind(D[1:row.train,],Y)
sumoutcomes<-aggregate(Y~date.1, data=Dtrain, sum)
countoutcomes<-aggregate(Y~date.1, data=Dtrain, length)
supervisedratio<-cbind(data.frame(date.1=sumoutcomes$date.1), data.frame(SVratio1=sumoutcomes$Y/countoutcomes$Y))
rm(sumoutcomes, countoutcomes)

D<-merge(D, supervisedratio, by='date.1', all.x=TRUE) 
D<-D[order(D$i),]
rm(supervisedratio)
averatio<-sum(Y)/length(Y)
D[is.na(D$SVratio1),]$SVratio1<-averatio



# adding a feature called "act_count.Freq" that shows the frequency of the activities 
#D<-transform(D, act_count=table(people_id)[people_id])
#D$act_count.people_id<-NULL
#D$act_count.Freq<-log(D$act_count.Freq)

#adding features for total number of each type of activity:
#activity_counts=as.data.frame.matrix(table(D$people_id, D$activity_category))
#names(activity_counts)=c("total_type1", "total_type2", "total_type3", "total_type4", "total_type5", "total_type6", "total_type7")
#activity_counts$people_id=row.names(activity_counts)
#D<-merge(D, activity_counts, by="people_id")



# Changing categorical features to integers 
char.cols=c('activity_category','people_group_1',
            'char_1','char_2','char_3','char_4','char_5','char_6','char_7','char_8','char_9','char_10',
            'people_char_2','people_char_3','people_char_4','people_char_5','people_char_6','people_char_7','people_char_8','people_char_9','date.month','people_date.month',
            'date.year','people_date.year','date.1')
for (f in char.cols) {
  if (class(D[[f]])=="character") {
    levels <- unique(c(D[[f]]))
    D[[f]] <- as.numeric(factor(D[[f]], levels=levels))
  }
}


# Creating sparse matrix for the categorical feautres; "one-hot-encoding"
D.sparse=
  cBind(sparseMatrix(D$i,D$activity_category),
        sparseMatrix(D$i,D$people_group_1),
        sparseMatrix(D$i,D$char_1),
        sparseMatrix(D$i,D$char_2),
        sparseMatrix(D$i,D$char_3),
        sparseMatrix(D$i,D$char_4),
        sparseMatrix(D$i,D$char_5),
        sparseMatrix(D$i,D$char_6),
        sparseMatrix(D$i,D$char_7),
        sparseMatrix(D$i,D$char_8),
        sparseMatrix(D$i,D$char_9),
        sparseMatrix(D$i,D$people_char_2),
        sparseMatrix(D$i,D$people_char_3),
        sparseMatrix(D$i,D$people_char_4),
        sparseMatrix(D$i,D$people_char_5),
        sparseMatrix(D$i,D$people_char_6),
        sparseMatrix(D$i,D$people_char_7),
        sparseMatrix(D$i,D$people_char_8),
        sparseMatrix(D$i,D$people_char_9),
        sparseMatrix(D$i,D$date.month),
        sparseMatrix(D$i,D$people_date.month),
        sparseMatrix(D$i,D$date.year),
        sparseMatrix(D$i,D$people_date.year),
        sparseMatrix(D$i,D$date.1)
        
  )


# Cobmining the sparse matrix with the numerical features
D.sparse=
  cBind(D.sparse,
        D$people_char_10,
        D$people_char_11,
        D$people_char_12,
        D$people_char_13,
        D$people_char_14,
        D$people_char_15,
        D$people_char_16,
        D$people_char_17,
        D$people_char_18,
        D$people_char_19,
        D$people_char_20,
        D$people_char_21,
        D$people_char_22,
        D$people_char_23,
        D$people_char_24,
        D$people_char_25,
        D$people_char_26,
        D$people_char_27,
        D$people_char_28,
        D$people_char_29,
        D$people_char_30,
        D$people_char_31,
        D$people_char_32,
        D$people_char_33,
        D$people_char_34,
        D$people_char_35,
        D$people_char_36,
        D$people_char_37,
        D$people_char_38,
        D$date.days,
        D$people_date.days,
        D$SVratio1
        )
      
# Creating train/test sets 
train.sparse=D.sparse[1:row.train,]
test.sparse=D.sparse[(row.train+1):nrow(D.sparse),]



# Creating DMatrix for Xgboost
dtrain  <- xgb.DMatrix(train.sparse, label = Y)
gc(verbose=FALSE)
xgb.DMatrix.save(dtrain, "dtrain.data")
gc(verbose=FALSE)
#rm(dtrain) #avoid getting through memory limits
#gc(verbose=FALSE)
zip("dtrain.data.zip", "dtrain.data", flags = "-m9X", extras = "", zip = Sys.getenv("R_ZIPCMD", "zip"))



dtest  <- xgb.DMatrix(test.sparse)
gc(verbose=FALSE)
xgb.DMatrix.save(dtest, "dtest.data")
gc(verbose=FALSE)
zip("dtest.data.zip", "dtest.data", flags = "-m9X", extras = "", zip = Sys.getenv("R_ZIPCMD", "zip"))
#file.remove("dtest.data")



#cat("Re-creating SVMLight format\n")
#dtrain  <- xgb.DMatrix(train.sparse, label = Y) #recreate train sparse to run under the memory limit of 8589934592 bytes
gc(verbose=FALSE)


param <- list(objective = "binary:logistic", 
              eval_metric = "auc",
              booster = "gblinear", 
              eta = 0.02,
              subsample = 0.7,
              colsample_bytree = 0.7,
              min_child_weight = 0,
              max_depth = 10)

# ###uncomment this for CV run
# #
# #dmodel  <- xgb.DMatrix(train.sparse[model, ], label = Y[model])
# #dvalid  <- xgb.DMatrix(train.sparse[valid, ], label = Y[valid])
# #
# #set.seed(120)
# #m1 <- xgb.train(data = dmodel
# #                , param
# #                , nrounds = 500
# #                , watchlist = list(valid = dvalid, model = dmodel)
# #                , early.stop.round = 20
# #                , nthread=11
# #                , print_every_n = 10)
# 
# #[300]	valid-auc:0.979167	model-auc:0.990326



cat(Sys.time())
cat("XGBoost\n")

set.seed(120)
m2 <- xgb.train(data = dtrain, 
                param, nrounds = 305,
                watchlist = list(train = dtrain),
                print_every_n = 101)


cat(Sys.time())
cat("Predicting on test data\n")

# Predict
out <- predict(m2, dtest)
sub <- data.frame(activity_id = test_activity_id, outcome = out)
write.csv(sub, file = "model_sub.csv", row.names = F)

#0.98035








cat("Cleaning up...\n")
remove(list = ls())
gc(verbose=FALSE)



## Leak from Loiso (0.987)

cat(Sys.time())
cat("Doing Loiso's magic leak\n")

cat("Working on people\n")
# load and transform people data ------------------------------------------
ppl <- fread("people.csv")

### Recode logic to numeric
p_logi <- names(ppl)[which(sapply(ppl, is.logical))]

for (col in p_logi) {
  set(ppl, j = col, value = as.numeric(ppl[[col]]))
}
rm(p_logi)

### transform date
ppl[,date := as.Date(as.character(date), format = "%Y-%m-%d")]

# load activities ---------------------------------------------------------

cat("Working on data\n")
# read and combine
activs <- fread("act_train.csv")
TestActivs <- fread("act_test.csv")
TestActivs$outcome <- NA
activs <- rbind(activs,TestActivs)
rm(TestActivs)

cat("Merging\n")
# Extract only required variables
activs <- activs[, c("people_id","outcome","activity_id","date"), with = F]

# Merge people data into actvitities
d1 <- merge(activs, ppl, by = "people_id", all.x = T)

# Remember, remember the 5th of November and which is test
testset <- which(ppl$people_id %in% d1$people_id[is.na(d1$outcome)])
d1[, activdate := as.Date(as.character(date.x), format = "%Y-%m-%d")]

rm(activs)

# prepare grid for prediction ---------------------------------------------

cat("Creating interaction\n")
# Create all group_1/day grid
minactivdate <- min(d1$activdate)
maxactivdate <- max(d1$activdate)
alldays <- seq(minactivdate, maxactivdate, "day")
allCompaniesAndDays <- data.table(
  expand.grid(unique(
    d1$group_1[!d1$people_id %in% ppl$people_id[testset]]), alldays
  )
)


## Nicer names
colnames(allCompaniesAndDays) <- c("group_1","date.p")

## sort it
setkey(allCompaniesAndDays,"group_1","date.p")

## What are values on days where we have data?
meanbycomdate <- d1[
  !d1$people_id %in% ppl$people_id[testset],
  mean(outcome),
  by = c("group_1","activdate")
  ]

## Add them to full data grid
allCompaniesAndDays <- merge(
  allCompaniesAndDays,
  meanbycomdate,
  by.x = c("group_1","date.p"), by.y = c("group_1","activdate"),
  all.x = T
)


# design function to interpolate unknown values ---------------------------

interpolateFun <- function(x){
  
  # Find all non-NA indexes, combine them with outside borders
  borders <- c(1, which(!is.na(x)), length(x) + 1)
  # establish forward and backward - looking indexes
  forward_border <- borders[2:length(borders)]
  backward_border <- borders[1:(length(borders) - 1)]
  
  # prepare vectors for filling
  forward_border_x <- x[forward_border]
  forward_border_x[length(forward_border_x)] <- abs(
    forward_border_x[length(forward_border_x) - 1] - 0.1
  ) 
  backward_border_x <- x[backward_border]
  backward_border_x[1] <- abs(forward_border_x[1] - 0.1)
  
  # generate fill vectors
  forward_x_fill <- rep(forward_border_x, forward_border - backward_border)
  backward_x_fill <- rep(backward_border_x, forward_border - backward_border)
  forward_x_fill_2 <- rep(forward_border, forward_border - backward_border) - 
    1:length(forward_x_fill)
  backward_x_fill_2 <- 1:length(forward_x_fill) -
    rep(backward_border, forward_border - backward_border)
  
  #linear interpolation
  vec <- (forward_x_fill + backward_x_fill)/2
  
  x[is.na(x)] <- vec[is.na(x)]
  return(x)
}


# apply and submit --------------------------------------------------------

cat("Applying interaction interpolation\n")
allCompaniesAndDays[, filled := interpolateFun(V1), by = "group_1"]

d1 <- merge(
  d1,
  allCompaniesAndDays,
  all.x = T,all.y = F,
  by.x = c("group_1","activdate"),
  by.y = c("group_1","date.p")
)


cat("Predicting leak\n")
## Create prediction file and write
testsetdt <- d1[
  d1$people_id %in% ppl$people_id[testset],
  c("activity_id","filled"), with = F
  ]

cat("There are ", sum(is.na(testsetdt$filled)), " unknown observations.\nThere are ", length(testsetdt[testsetdt$filled == 0.5, filled]), " uncertain observations (equal to 0.5 - anokas' idea).\n", sep = "")
cat("Adjusting leak with NA when unsure using anokas' idea (0.5 leakage => raddar's)\n")
testsetdt[testsetdt$filled == 0.5, "filled"] <- NA
cat("There are now in total ", sum(is.na(testsetdt$filled)), " unknown observations.\n", sep = "")

write.csv(testsetdt,"Submission.csv", row.names = FALSE)

cat("Cleaning up...\n")

remove(list = ls())
gc(verbose=FALSE)



## Combo

# Not elegant but fast enough for our taste, could go sqldf one-liner >_>


cat(Sys.time())
cat("Combining two submissions on targeted observations (NAs)\n")

submit1 <- fread("Submission.csv")
submit2 <- fread("model_sub.csv")
submit3 <- merge(submit1[is.na(submit1$filled), ], submit2, by = "activity_id", all.x = T)
submit4 <- merge(submit1, submit3, by = "activity_id", all.x = T)
submit4$filled.x[is.na(submit4$filled.x)] <- submit4$outcome[is.na(submit4$filled.x)]
submit5 <- data.frame(activity_id = submit4$activity_id, outcome = submit4$filled.x, stringsAsFactors = FALSE)
write.csv(submit5, file="erica_831_2.csv", row.names=FALSE)




