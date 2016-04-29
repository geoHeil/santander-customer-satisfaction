source('R/michi/init.R')

# load data sets (assuming these have been downloaded and extracted to the data directory)
train <- fread('data/train.csv', colClasses='numeric')
test <- fread('data/test.csv', colClasses='numeric')

# split train dataset into calibration and validation subset (80%/20%)
idx <- as.logical(rbinom(nrow(train), size=1, prob=0.8))
idx.cal <- which(idx)
idx.val <- which(!idx)

# helper method to drop columns which have only one distinct value
dropUniqueColumns <- function(dt) {
  is_unique <- sapply(dt, function(col) uniqueN(col) == 1)
  dt[, !is_unique, with=F]
}

# train GBM model on calibration dataset (runs for about 1 minute)
m1 <- gbm(TARGET ~ .,            # train TARGET against all other variables
          data = dropUniqueColumns(train[idx.cal]), # use calibration dataset; but drop unique columns to avoid warning messages
          distribution = 'bernoulli', # we model probabilities for a binary dependent variable
          train.fraction = 0.9,  # keep aside data at each step to determine optimal no. of trees
          n.trees = 200,         # number of trees
          shrinkage = 0.05,      # learning rate
          verbose = T,           # be verbose while training the model
          n.minobsinnode = 10,   # min. number of observations within a leave to consider a split
          interaction.depth = 4) # max. number of variables considered for each tree

# determine optimal number of trees
(best.iter <- gbm.perf(m1))

# make predictions for whole training dataset
est <- predict(m1, newdata=train, n.trees=best.iter, type='response')

# estimate AUC error measure
Epi::ROC(est[idx.cal], train[idx.cal, TARGET])  # 0.839 for 80% calibration subset
Epi::ROC(est[idx.val], train[idx.val, TARGET])  # 0.825 for 20% validation subset

# train new model on whole training dataset and create submission file
m2 <- gbm(TARGET ~ .,
          data = dropUniqueColumns(train),
          distribution = 'bernoulli',
          train.fraction = 0.9,
          n.trees = 200,
          shrinkage = 0.05,
          verbose = T,
          n.minobsinnode = 10,
          interaction.depth = 4)
test$TARGET <- predict(m2, newdata=test, n.trees=gbm.perf(m2), type='response')
write.csv(test[, .(ID, TARGET)], 'data/michi-submission.csv', row.names=F, quote=F)

