source('R/michi/init.R')

# Use Caret for grid search of hyperparameters in parallel
# see http://topepo.github.io/caret/training.html

# load data sets (assuming these have been downloaded and extracted to the data directory)
train <- fread('data/train.csv', colClasses='numeric')
test <- fread('data/test.csv', colClasses='numeric')

# register parallel backend (only works for Unix-based systems)
doMC::registerDoMC(cores = parallel::detectCores())

# caret expects dependent variable to be a factor
train[, TARGET := as.factor(ifelse(TARGET==0, 'zero', 'one'))]

tc <- trainControl(              # 5-fold cross-validation
  verboseIter = TRUE,
  method = 'cv',
  number = 5,
  summaryFunction = twoClassSummary,
  classProbs = TRUE)

grid.small <- expand.grid(       # a small grid for demo purposes
  interaction.depth = c(2),
  n.trees = (1:20) * 10,
  shrinkage = c(0.1, 0.05),
  n.minobsinnode = c(10, 20))

grid.large <- expand.grid(       # full 3x3x3 factorial design of hyper-parameters
  interaction.depth = c(2, 3, 4),
  n.trees = (1:30) * 10,
  shrinkage = c(0.02, 0.05, 0.1),
  n.minobsinnode = c(5, 10, 20))

# run hyper-parameter search for demo-purposes
gbmFit1 <- caret::train(
  TARGET ~ .,
  data = train[1:2000],  # use a small subset for demo purposes
  method = 'gbm',
  trControl = tc,
  distribution = 'bernoulli',
  verbose = T,
  metric = 'ROC',               # use AUC for comparing models
  tuneGrid = grid.small
)

trellis.par.set(caretTheme())
plot(gbmFit1)
print(gbmFit1$bestTune)

# full hyper-parameter search (will run for hours on 32-core EC2 instance)
gbmFit2 <- caret::train(
  TARGET ~ .,
  data = train,  # use full training set
  method = 'gbm',
  trControl = tc,
  distribution = 'bernoulli',
  verbose = T,
  metric = 'ROC',               # use AUC for comparing models
  tuneGrid = grid.large
)
trellis.par.set(caretTheme())
plot(gbmFit1)
print(gbmFit1$bestTune)
