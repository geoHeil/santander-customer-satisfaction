
# set seed for RNG to make results reproducible
set.seed(1)

# prevent R from writing numbers in scientific format
options(scipen=500)

# install/load required R packages
pkgs <- c('data.table', 'gbm', 'Epi', 'caret', 'doMC', 'pROC')
nil <- lapply(pkgs, function(pkg) {
  if (!pkg %in% installed.packages()) {
    install.packages(pkg, dependencies = TRUE)
  }
  library(pkg, character.only = TRUE)
})
