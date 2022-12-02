# Prediction
# in questo file unisco gli algoritmi di previsione allenati fino ad ora
# Nel markdown sono presenti LRM, Ranger, SVM
# in questo file: 
# Logistic Regression
# Random Forest
# SVM
# XGBoost

### Logistic Regresion##########################################################
# Logistic Regression
# Model estimate

library(tidymodels)
library(tidyverse)

# data
setwd("C:/Users/Davide Zulato/Desktop/Sim_Data")
data_bias <- read.csv("BIASED_data_29_11_22")
data_bias <- data_bias %>% 
  mutate(Y = factor(Y)) %>% 
  dplyr::select(-X) # elimino id e trtgroup (dummy)

df <- data_bias %>% dplyr::select(-Y) # prova a togliere S e i vari G
correlation <- cor(df)
corrplot::corrplot(correlation,method = "square",type="lower")

## remove columns with only 0s
data_bias <- 
  data_bias %>% 
  select_if(.predicate = ~n_distinct(.) > 1) 

## set type as factor
data_bias <- 
  data_bias %>% 
  mutate(Y= as.factor(Y)) # response as factor

## FIRST SPLIT
## create initial train/test split
data_split <- initial_split(data_bias, prop = 3/4, strata = "Y")
data_train <- training(data_split)
data_test <- testing(data_split)

## Logistic regression
## logistic model
logit_fit <- 
  logistic_reg(mode = "classification") %>%
  set_engine(engine = "glm") %>% 
  fit(Y ~ ., data = data_train)

# suppose fairness throug unawareness
logit_fit2 <- 
  logistic_reg(mode = "classification") %>%
  set_engine(engine = "glm") %>% 
  fit(Y ~ ., data = dplyr::select(data_train, -S))
# remove sensitive feature and treatment groups (assigned randomly)
# fit(Y ~ ., data = select(data_train, -X)) se non pulito

## PREDICTION
## get prediction on train set
pred_logit_train <- predict(logit_fit, new_data = data_train)
pred_logit_train2 <- predict(logit_fit2, new_data = data_train) # no S
## get prediction on test set
pred_logit_test <- predict(logit_fit, new_data = data_test)
pred_logit_test2 <- predict(logit_fit2, new_data = data_test)
# pred_logit_test <- predict(logit_fit, new_data = select(data_test, -X))
## get probabilities on test set
prob_logit_test <- predict(logit_fit, new_data = data_test, type="prob")
prob_logit_test2 <- predict(logit_fit2, new_data = data_test, type="prob")

## Model performance
metrics(bind_cols(data_test, pred_logit_test), truth = Y, estimate = .pred_class)
metrics(bind_cols(data_test, pred_logit_test2), truth = Y, estimate = .pred_class)

# more than accuracy and kappa
# multimetric <- metric_set(accuracy, bal_accuracy, sens, yardstick::spec, precision, recall, ppv, npv)
multimetric <- metric_set(accuracy,bal_accuracy,yardstick::spec,yardstick::precision,yardstick::recall)
multimetric(bind_cols(data_test, pred_logit_test), truth = Y, estimate = .pred_class)  
multimetric(bind_cols(data_test, pred_logit_test2), truth = Y, estimate = .pred_class)  

bind_cols(data_test$Y, prob_logit_test2,pred_logit_test2 )
female_predictions <- pred_logit_test2[data_test$S=="1",]
male_predictions <- pred_logit_test2[data_test$S=="0",]
prop.table(table(male_predictions)); prop.table(table(female_predictions))

# confusion matrix
conf_matrix <- table(pred_logit_test$.pred_class,data_test$Y)
# confusion matrix by S_a and S_d
conf_matrix_male <- table(pred_logit_test$.pred_class[which(data_test$S==0)],data_test$Y[which(data_test$S==0)])
conf_matrix_female <- table(pred_logit_test$.pred_class[which(data_test$S==1)],data_test$Y[which(data_test$S==1)])
# accuracy
sum(diag(conf_matrix))/sum(conf_matrix) 



# ROC  
roc_auc(bind_cols(data_test, prob_logit_test), truth = Y, .pred_0)

# this function works without 1-AUC
roc.logistic.reg <- AUC::roc(predictions = prob_logit_test2$.pred_1,labels = data_test$Y)
plot(roc.logistic.reg,main="ROC curve for Logistic regression")


## CROSS VALIDATION
## create 5 times a 10 fold cv
cv_train <- vfold_cv(data_train, v = 10, repeats = 5, strata = "Y")

# We define some functions to fit the model, get the predictions and the probabilities.
logit_mod <- 
  logistic_reg(mode = "classification") %>%
  set_engine(engine = "glm")

## compute mod on kept part
cv_fit <- function(splits, mod, ...) {
  res_mod <-
    fit(mod, Y ~ ., data = analysis(splits), family = binomial)
  return(res_mod)
}

## get predictions on holdout sets
cv_pred <- function(splits, mod){
  # Save the 10%
  holdout <- assessment(splits)
  pred_assess <- bind_cols(truth = holdout$Y, predict(mod, new_data = holdout))
  return(pred_assess)
}

## get probs on holdout sets
cv_prob <- function(splits, mod){
  holdout <- assessment(splits)
  prob_assess <- bind_cols(truth = as.factor(holdout$Y), 
                           predict(mod, new_data = holdout, type = "prob"))
  return(prob_assess)
}

# We apply these functions on the cv data.

res_cv_train <- 
  cv_train %>% 
  mutate(res_mod = map(splits, .f = cv_fit, logit_mod), ## fit model
         res_pred = map2(splits, res_mod, .f = cv_pred), ## predictions
         res_prob = map2(splits, res_mod, .f = cv_prob)) ## probabilities

# We can compute the model preformance for each fold:

res_cv_train %>% 
  mutate(metrics = map(res_pred, multimetric, truth = truth, estimate = .pred_class)) %>% 
  unnest(metrics) %>% 
  ggplot() + 
  aes(x = id, y = .estimate) +
  geom_point() + 
  facet_wrap(~ .metric, scales = "free_y")

# Model performance on each fold
# ROC curve on each fold:
# same Roc but 1-AUC, looks right but investigate whi 1-ROC
res_cv_train %>% 
  mutate(roc = map(res_prob, roc_curve, truth = truth, .pred_0)) %>% 
  unnest(roc) %>% 
  ggplot() +
  aes(x = 1 - specificity, y = sensitivity, color = id2) +
  geom_path() +
  geom_abline(lty = 3) + facet_wrap(~id) +
  ggtitle("Logistic model ROC","repeated CV")

# ROC curves on each fold

# Note that Cohen’s kappa coefficient (κ) is a similar measure to accuracy, 
# but is normalized by the accuracy that would be expected by chance alone and 
# is very useful when one or more classes have large frequency distributions.
# The higher the value, the better.

# ROC CURVE PER GROUP
# asse x 1-specificità, asse y sensibilità
# it could be interesting to look at the ROC curve per group
roc_data_female <- roc_curve(bind_cols(data_test[which(data_test$S==1),], prob_logit_test[which(data_test$S==1),]), truth = Y, .pred_0)
roc_data_male <- roc_curve(bind_cols(data_test[which(data_test$S==0),], prob_logit_test[which(data_test$S==0),]), truth = Y, .pred_0)

win.graph()
roc_data_female$FPR <- (1-roc_data_female$specificity)
roc_data_male$FPR <- (1-roc_data_male$specificity)
plot(roc_data_female[,4:3],type="l",col="firebrick2",main="ROC curve Logistic regression by group",
     sub="Per-group for S",xlim=c(0,1),ylim=c(0,1))
lines(roc_data_male[,4:3],type="l",col="blue")
abline(coef=c(0,1),lty=2) # diagonal as random guess
legend("bottomright",c("ROC S=1","ROC S=0"),col=c("firebrick2","blue"),lty=1)



### Random Forest###############################################################
# Random forest with tuning
# rand_forest(
# mode = "classification",
# engine = "ranger",
# mtry = tune(),
# trees = ,
# min_n = tune()
# )
rm(list=ls())
library(tidyverse)
library(tidymodels)
library(modeldata)
library(fairmodels)
library(ranger)
library(DALEX)

# speed up computation with parallel processing
library(doParallel)
all_cores <- parallel::detectCores(logical = FALSE)
registerDoParallel(cores = all_cores)

# Step 1 data
setwd("C:/Users/Davide Zulato/Desktop/Sim_Data")
data_bias <- read.csv("BIASED_data_29_11_22")
data_bias <- data_bias %>% 
  mutate(Y = factor(Y)) %>%  # outcome as factor
  dplyr::select(-X) # elimino id e trtgroup (dummy)

## create initial train/test split
data_split <- initial_split(data_bias, prop = 3/4, strata = "Y")
data_train <- training(data_split)
data_test <- testing(data_split)

# mtry() di default sqrt(m)
#Step 2: Preprocessing
pred.var <- c( "S","interview","Github_account","proxy","proxy2","age","X_score","score","simpson_score1","simpson_score2")

preprocessing_recipe <- 
  recipes::recipe(Y ~ ., data = training(data_split)) %>%
  # convert categorical variables to factors
  recipes::step_string2factor(all_nominal()) %>%
  # combine low frequency factor levels
  recipes::step_other(all_nominal(), threshold = 0.01) %>%
  # remove no variance predictors which provide no predictive information 
  recipes::step_nzv(all_nominal()) %>%
  prep()

#Step 3: Splitting for Cross Validation
data_cv_folds <- 
  recipes::bake(
    preprocessing_recipe, 
    new_data = training(data_split)
  ) %>%  
  rsample::vfold_cv(v = 10) # 10 folds cross validation

# Step 4 Random forest model specification
rf_model <- rand_forest(
  mode = "classification",
  engine = "ranger",
  mtry = 3, # tune the number of variables at each split
  trees = 1000,# number of trees contained in the ensemble
  min_n = tune() # minimum number of data points in a node to be split further.
)%>%
  set_engine("ranger")

# Step 5: Grid Specification, hiperparameters to tune
# https://www.tidymodels.org/start/tuning/
# mtry e min_n
rf_params <- 
  dials::parameters(
    min_n() # only parameter tu tune if we use default mtry()=3
  )

rf_grid <- 
  dials::grid_regular( # regular grid for hyperparameter optimization
    rf_params, 
    levels = 20 # sometimes returns an error
  )
knitr::kable(head(rf_grid))

#Step 6: Define the Workflow
rf_wf <- 
  workflows::workflow() %>%
  add_model(rf_model) %>% 
  add_formula(Y ~ .)

#Step 7: Tune the Model
rf_tuned <- 
  tune::tune_grid(
    object = rf_wf,
    resamples = data_cv_folds,
    grid = rf_grid,
    metrics = metric_set(accuracy, roc_auc),
    control = control_grid(verbose = TRUE)
  )

# best metric accuracy
rf_tuned %>%
  tune::show_best(metric = "accuracy") %>%
  knitr::kable()
# best metric roc_auc
rf_tuned %>%
  tune::show_best(metric = "roc_auc") %>%
  knitr::kable()

rf_best_params <- rf_tuned %>%
  tune::select_best("accuracy")
knitr::kable(rf_best_params)

rf_model_final <- rf_model %>% 
  finalize_model(rf_best_params)
## Random forest model
## rf model senza iperparametri
# it is possible to use the min_n tuned (12)
set.seed(876101)
rf_fit <- 
  rand_forest(mode = "classification",trees =1000,mtry = 3,min_n = 12) %>%
  set_engine(engine = "ranger") %>% 
  fit(Y ~ ., data = data_train)

## PREDICTION
## get prediction on training set
pred_rf_train <- predict(rf_fit, new_data = data_train)
## get prediction on test set
pred_rf_test <- predict(rf_fit, new_data = data_test)
# pred_rf_test <- predict(logit_fit, new_data = select(data_test, -X))
## get probabilities on test set
prob_rf_test <- predict(rf_fit, new_data = data_test, type="prob")

## Model performance
metrics(bind_cols(data_test, pred_rf_test), truth = Y, estimate = .pred_class)

# more than accuracy and kappa
# multimetric <- metric_set(accuracy, bal_accuracy, sens, yardstick::spec, precision, recall, ppv, npv)
multimetric <- metric_set(accuracy,bal_accuracy,yardstick::spec,yardstick::precision,yardstick::recall)
multimetric(bind_cols(data_test, pred_rf_test), truth = Y, estimate = .pred_class)  

bind_cols(data_test$Y, prob_rf_test,pred_rf_test)
female_predictions <- pred_rf_test[data_test$S=="1",]
male_predictions <- pred_rf_test[data_test$S=="0",]
prop.table(table(male_predictions)); prop.table(table(female_predictions))


# confusion matrix
# keep working on it
table(pred_rf_test$.pred_class,data_test$Y)
male_test_data <- data_test[which(data_test$S==0),]
female_test_data <- data_test[which(data_test$S==1),]
# confusion matrix by S_a and S_d
table(pred_rf_test$.pred_class[which(data_test$S==0)],male_test_data$Y)
table(pred_rf_test$.pred_class[which(data_test$S==1)],female_test_data$Y)
# this model hardly classifies woman with positive outcome

# ROC  
## continue with this approach or try other flows
# this function works without 1-AUC
roc.rf <- AUC::roc(predictions = prob_rf_test$.pred_1,labels = data_test$Y)
plot(roc.rf,main="ROC curve for RF")

# ROC by S
roc.rf_female <- AUC::roc(predictions = prob_rf_test$.pred_1[which(data_test$S==1)],labels = female_test_data$Y)
roc.rf_male <- AUC::roc(predictions = prob_rf_test$.pred_1[which(data_test$S==0)],labels = male_test_data$Y)
plot(roc.rf_male,main="ROC curve for RF by S_a and S_d")
plot(roc.rf_female,add=T,col=2)

# try with ggplot
roc.rf <- as.data.frame(cbind(roc.rf$cutoffs,roc.rf$fpr,roc.rf$tpr))
colnames(roc.rf) <- c("cutoffs","fpr","tpr")
# sensitivity: true positive rate
# specificity: true negative rate
roc.rf %>%  
  ggplot(aes(x =  fpr, y = tpr)) + # check
  geom_path() +
  geom_abline(lty = 3) + 
  coord_equal()

#########################################
# FAIRNESS WITH RANDOM FOREST CLASSIFIER
# making model
set.seed(876101)
rf_fit
# making explainer
data_bias$Y <- as.numeric(data_bias$Y)-1 # check if it is useful
# data_bias$Y <- as.numeric(data_bias$Y)
rf_explainer <- DALEX::explain(rf_fit,
                               data = data_bias[,-11],
                               y = data_bias$Y,
                               colorize = FALSE)
model_performance(rf_explainer)

fobject <- fairness_check(rf_explainer, 
                          protected  = factor(data_bias$S), 
                          privileged = "0", 
                          colorize = FALSE)
plot_density(fobject)

# cutoff manipulation
fobject_cut2 <- fairness_check(rf_explainer, 
                               protected  = factor(data_bias$S), 
                               privileged = "0",
                               cutoff = 0.2, # custom cutoff, default 0.5
                               colorize = FALSE)
plot_density(fobject_cut2)

plot(fobject)
plot(fobject_cut2) # excessive accuracy equality ratio could be related with
# too many female with a negative outcome

## reweighting the data with  KAMIRAN e CALDERS (2011) 
# i also made some custom weights, they works 
weights <- reweight(protected = factor(data_bias$Y), y = data_bias$Y)
# c(0.3845,0.6155)
set.seed(876101)
rf_model_weighted  <- ranger(Y ~. ,
                             data = data_bias, # all dataset
                             case.weights = weights, # see the help
                             classification = T,
)
rf_explainer_w <- DALEX::explain(rf_model_weighted,
                                 data = data_bias[,-11],
                                 y = data_bias$Y,
                                 label = "rf_weighted",
                                 verbose = FALSE)
fobject <- fairness_check(rf_explainer_w,rf_explainer,
                          protected = factor(data_bias$S), 
                          privileged = "0",
                          verbose = FALSE)
plot(fobject)
plot_density(fobject)

# stacked metric plot, useful fo choosing the best model
sm <- stack_metrics(fobject)
plot(sm)


## disparate impact remover
library("ggplot2")

set.seed(876101)

# let's start with the score with mean grater for men
ggplot(data_bias, aes(score, fill = factor(S))) +
  geom_density(alpha = 0.5)

fixed_data <- disparate_impact_remover(
  data = data_bias,
  protected = factor(data_bias$S),
  features_to_transform = "score",
  lambda = 0.8
)

ggplot(fixed_data, aes(score, fill =factor(S))) +
  geom_density(alpha = 0.5)

# lambda 1 gives identical distribution, lambda 0 (almost) original distributions

fixed_data_unchanged <- disparate_impact_remover(
  data = data_bias,
  protected = factor(data_bias$S),
  features_to_transform = "score",
  lambda = 0
)

ggplot(fixed_data_unchanged, aes(score, fill = factor(S))) +
  geom_density(alpha = 0.5)


fixed_data_fully_changed <- disparate_impact_remover(
  data = data_bias,
  protected = factor(data_bias$S),
  features_to_transform = "score",
  lambda = 1
)

ggplot(fixed_data_fully_changed, aes(score, fill = factor(S))) +
  geom_density(alpha = 0.5) +
  facet_wrap(S ~ ., nrow = 2)

# all cutoffs
ac <- all_cutoffs(fobject)
plot(ac)

# ceteris paribus cutoff
cpc <- ceteris_paribus_cutoff(fobject, subgroup = "1")
win.graph()
plot(cpc)

################################################################################
### SVM ########################################################################
# SVM
# SVM prediction for HR dataset
rm(list=ls())
library(tidyverse)
library(tidymodels)
library(modeldata)
setwd("C:/Users/Davide Zulato/Desktop/Sim_Data")
# make sure to read the right version
data_bias <- read.csv("BIASED_data_29_11_22")
data_bias <- data_bias %>% 
  mutate(Y = factor(Y)) %>% 
  dplyr::select(-X) # remove ids, useless for prediction

# speed up computation with parallel processing
library(doParallel)
all_cores <- parallel::detectCores(logical = FALSE)
registerDoParallel(cores = all_cores)

#Step 1: Initial Data Split
set.seed(876101)
data_split <- rsample::initial_split(data_bias, prop = 3/4, strata = Y)
# training set
data_train <- training(data_split)
# test set
data_test <- testing(data_split)

#Step 2: Preprocessing
pred.var <- c( "S","interview","Github_account","proxy","proxy2","age","X_score","score","simpson_score1","simpson_score2")


preprocessing_recipe <- 
  recipes::recipe(Y ~ ., data = training(data_split)) %>%
  # convert categorical variables to factors
  recipes::step_string2factor(all_nominal()) %>%
  # combine low frequency factor levels
  recipes::step_other(all_nominal(), threshold = 0.01) %>%
  # remove no variance predictors which provide no predictive information 
  recipes::step_nzv(all_nominal()) %>%
  prep()

 
#Step 3: Splitting for Cross Validation
data_cv_folds <- 
  recipes::bake(
    preprocessing_recipe, 
    new_data = training(data_split)
  ) %>%  
  rsample::vfold_cv(v = 10)

# step 4
set.seed(420)

roc_res <- metric_set(roc_auc) # accuracy, a classification metric

# Let's fit a radial basis function support vector machine to the palmers penguins 
# and tune the SVM cost parameter (cost()) and the ?? parameter in the kernel 
# function (rbf_sigma):

svm_spec <-
  svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")
# Now, let's set up our workflow() and feeding it our svm model

svm_wflow <- 
  workflow() %>% 
  add_model(svm_spec) %>% 
  add_recipe(preprocessing_recipe)
# Let's zoom in on the default parameter values for our two tuning parameters:

cost()

rbf_sigma()

# We can change them:

svm_param <- 
  svm_wflow %>% 
  hardhat::extract_parameter_set_dials()%>%  # parameters() deprecated
  update(rbf_sigma = rbf_sigma(c(-7, -1)))

# initial grid

start_grid <- 
  svm_param %>% 
  update(
    cost = cost(c(-6, 1)),
    rbf_sigma = rbf_sigma(c(-6, -4))) %>% 
  grid_regular(levels = 2)

set.seed(876101)

svm_initial <- 
  svm_wflow %>% 
  tune_grid(resamples = data_cv_folds, grid = start_grid, metrics = roc_res)
collect_metrics(svm_initial)

#Step 7: Tune the Model
svm_tuned <- 
  tune::tune_grid(
    object = svm_wflow,
    resamples = data_cv_folds,
    grid = start_grid,
    metrics = metric_set(accuracy, roc_auc),
    control = control_grid(verbose = TRUE)
  )

svm_tuned %>%
  tune::show_best(metric = "accuracy") %>%
  knitr::kable()
collect_metrics(svm_tuned)

svm_tuned %>%
  tune::show_best(metric = "roc_auc") %>%
  knitr::kable()
collect_metrics(svm_tuned)

# We can see that there's one point in which the performance is better.
# There results can be fed into iterative tuning functions as initial values

svm_best_params <- svm_tuned %>%
  tune::select_best("accuracy")
knitr::kable(svm_best_params)

svm_best_params <- svm_tuned %>%
  tune::select_best("roc_auc")
knitr::kable(svm_best_params)

svm_model_final <- svm_spec %>% 
  finalize_model(svm_best_params)

#Step 8: Evaluate Performance on Test Data
train_processed <- bake(preprocessing_recipe,  new_data = training(data_split))
train_prediction <- svm_model_final %>%
  # fit the model on all the training data
  fit(
    formula = Y ~ ., 
    data    = train_processed
  ) %>%
  # predict the sale prices for the training data
  predict(new_data = train_processed) %>%
  # predict_proba(new_data = train_processed)
  bind_cols(training(data_split))

# predicted probabilities
train_prob_prediction <- svm_model_final %>%
  # fit the model on all the training data
  fit(
    formula = Y ~ ., 
    data    = train_processed
  ) %>%
  # predict the sale prices for the training data
  predict(new_data = train_processed,type="prob") %>%
  # predict_proba(new_data = train_processed)
  bind_cols(training(data_split))


svm_score_train <- 
  train_prediction %>%
  yardstick::metrics(Y, .pred_class) %>%
  mutate(.estimate = format(round(.estimate, 2), big.mark = ","))
knitr::kable(svm_score_train)

test_processed  <- bake(preprocessing_recipe, new_data = testing(data_split))
test_prediction <- svm_model_final %>%
  # fit the model on all the training data
  fit(
    formula = Y ~ ., 
    data    = train_processed
  ) %>%
  # use the training model fit to predict the test data
  predict(new_data = test_processed) %>%
  bind_cols(testing(data_split))

# predicted probabilities on test set
test_prob_prediction <- svm_model_final %>%
  # fit the model on all the training data
  fit(
    formula = Y ~ ., 
    data    = train_processed
  ) %>%
  # use the training model fit to predict the test data
  predict(new_data = test_processed, type="prob") %>%
  bind_cols(testing(data_split))

# measure the accuracy of our model using `yardstick`
svm_score <- 
  test_prediction %>%
  yardstick::metrics(Y, .pred_class) %>%
  mutate(.estimate = format(round(.estimate, 2), big.mark = ","))
knitr::kable(svm_score)

data_prediction_residual <- test_prediction %>%
  arrange(.pred_class) %>%
  mutate(residual = ifelse(Y==.pred_class,1,0)) %>%
  dplyr::select(.pred_class, residual,Y)
prop.table(table(data_prediction_residual$residual))# accuracy
conf_matrix <- table(test_prediction$.pred_class,test_prediction$Y)
sum(diag(conf_matrix))/sum(conf_matrix) # accuracy

# confusion matrix by group
conf_matrix_male <- table(test_prediction$.pred_class[which(data_test$S==0)],test_prediction$Y[which(data_test$S==0)])
conf_matrix_female <- table(test_prediction$.pred_class[which(data_test$S==1)],test_prediction$Y[which(data_test$S==1)])


## ROC curve 
# this function works without 1-AUC
roc.svm <- AUC::roc(predictions = test_prob_prediction$.pred_1,labels = data_test$Y)
plot(roc.svm,main="ROC curve for SVM")
roc.svm

# ROC  
roc_auc(train_prob_prediction, truth = Y, .pred_0)

roc_data <- roc_curve(train_prob_prediction, truth = Y, .pred_1)
roc_data <- roc_curve(train_prob_prediction, truth = Y, .pred_0)
roc_data %>%  
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) + 
  ggtitle("ROC curve", "SVM model")+
  coord_equal()


roc_auc(test_prob_prediction,truth = Y, .pred_0)
# ROC CURVE PER GROUP
# asse x 1-specificità, asse y sensibilità
# it could be interesting to look at the ROC curve per group
roc_data_female <- roc_curve(bind_cols(data_test[which(data_test$S==1),], .pred_0=test_prob_prediction[which(data_test$S==1),]$.pred_0), truth = Y, .pred_0)
roc_data_male <- roc_curve(bind_cols(data_test[which(data_test$S==0),], .pred_0= test_prob_prediction[which(data_test$S==0),]$.pred_0), truth = Y, .pred_0)

win.graph()
roc_data_female$FPR <- (1-roc_data_female$specificity)
roc_data_male$FPR <- (1-roc_data_male$specificity)
plot(roc_data_female[,4:3],type="l",col="firebrick2",main="ROC curve SVM by group",
     sub="Per-group for S in test set",xlim=c(0,1),ylim=c(0,1))
lines(roc_data_male[,4:3],type="l",col="blue")
abline(coef=c(0,1),lty=2) # diagonal as random guess
legend("bottomright",c("ROC S=S_d","ROC S=S_d"),col=c("firebrick2","blue"),lty=1)

#################################################
## MODEL EXPLANATION FOR SVM from chapter 18 tmvr

library(DALEXtra)
vip_features <- c("S","interview","Github_account","proxy","proxy2","age","X_score","score","simpson_score1","simpson_score2")

vip_train <- 
  data_train %>% 
  select(all_of(vip_features))

# fit with the tuned model
svm_fit <- svm_model_final %>%
  # fit the model on all the training data
  fit(
    formula = Y ~ ., 
    data    = train_processed
  ) 

# data_train$Y <-as.numeric(data_train$Y)-1
explainer_svm <- 
  explain_tidymodels(
    svm_fit, 
    data = vip_train, # train without Y
    y = data_train$Y,
    label = "SVM",
    verbose = TRUE
  )

## LOCAL EXPLANATION
# loro sono interessanti
data_train[which(data_train$S==1 & data_train$Y==0),] 
Sara<- data_train[508,]
Sara
# Sara <- Sara[,1:11] # eventually omit treatment group
svm_breakdown <- predict_parts(explainer = explainer_svm, new_observation = Sara)
svm_breakdown

predict_parts(
  explainer = explainer_svm, 
  new_observation = Sara,
  order = svm_breakdown$variable_name
)

#  compute SHAP average attributions for this individual
set.seed(876101)
shap_duplex <- 
  predict_parts(
    explainer = explainer_svm, 
    new_observation = Sara, 
    type = "shap",
    B = 20
  )

# plot the shapley value
library(forcats)
shap_duplex %>%
  group_by(variable) %>%
  mutate(mean_val = mean(contribution)) %>%
  ungroup() %>%
  mutate(variable = fct_reorder(variable, abs(mean_val))) %>%
  ggplot(aes(contribution, variable, fill = mean_val > 0)) +
  geom_col(data = ~distinct(., variable, mean_val), 
           aes(mean_val, variable), 
           alpha = 0.5) +
  geom_boxplot(width = 0.5) +
  theme(legend.position = "none") +
  scale_fill_viridis_d() +
  ggtitle("Shapley values for S=1 & Y=0 individual","SVM model")+
  labs(y = NULL)

## GLOBAL EXPLANATION
set.seed(876101)
logit <- function(x) exp(x)/(1+exp(x))
custom_loss <- function(observed, predicted){
  sum((observed - logit(predicted))^2)
}
attr(custom_loss, "loss_name") <- "Logit residuals"
vip_svm <- model_parts(explainer_svm,loss_function = custom_loss)
# default loss function:1-AUC
vip_svm1 <- model_parts(explainer_svm,loss_function = loss_default(explainer_svm$model_info$type))
# vip_svm2 <- model_parts(explainer_svm, loss_function = loss_root_mean_square)
head(vip_svm) # warning and NAs if wrong loss function
win.graph()
plot(vip_svm)

## Fairness with fairmodels
library(fairmodels)
# making model
set.seed(876101)
svm_fit # fitted model already tuned
# making explainer
data_bias$Y <- as.numeric(data_bias$Y)-1 # check if it is useful
# data_bias$Y <- as.numeric(data_bias$Y) explainer needs the response as numeric
svm_explainer <- DALEX::explain(svm_fit,
                                data = data_bias[,-11],
                                y = data_bias$Y,
                                colorize = FALSE)
model_performance(svm_explainer)

fobject <- fairness_check(svm_explainer, 
                          protected  = factor(data_bias$S), 
                          privileged = "0", 
                          colorize = FALSE)
plot_density(fobject)
plot(fobject)
# cutoff manipulation
fobject_cut2 <- fairness_check(svm_explainer, 
                               protected  = factor(data_bias$S), 
                               privileged = "0",
                               cutoff = 0.2, # custom cutoff, default 0.5
                               colorize = FALSE)
plot_density(fobject_cut2)
plot(fobject_cut2)

### XGBoost ####################################################################
# XGboost prediction for HR dataset
rm(list=ls())
library(tidyverse)
library(tidymodels)
library(modeldata)
setwd("C:/Users/Davide Zulato/Desktop/Sim_Data")
# make sure to read the right version
data_bias <- read.csv("BIASED_data_29_11_22")
# data_bias <- read.csv("BIASED_data_11_11_22.csv")
data_bias <- data_bias %>% 
  mutate(Y = factor(Y)) %>% 
  dplyr::select(-X) # remove ids, useless for prediction

# speed up computation with parallel processing
library(doParallel)
all_cores <- parallel::detectCores(logical = FALSE)
registerDoParallel(cores = all_cores)

#Step 1: Initial Data Split
set.seed(876101)
data_split <- rsample::initial_split(data_bias, prop = 3/4, strata = Y)
# training set
data_train <- training(data_split)
# test set
data_test <- testing(data_split)

#Step 2: Preprocessing
pred.var <- c( "S","interview","Github_account","proxy","proxy2","age","X_score","score","simpson_score1","simpson_score2")


preprocessing_recipe <- 
  recipes::recipe(Y ~ ., data = training(data_split)) %>%
  # convert categorical variables to factors
  recipes::step_string2factor(all_nominal()) %>%
  # combine low frequency factor levels
  recipes::step_other(all_nominal(), threshold = 0.01) %>%
  # remove no variance predictors which provide no predictive information 
  recipes::step_nzv(all_nominal()) %>%
  prep()


# according to tmwr center, scale and PCA should improve xgboost performances
# but the accuracy, AUC looks lower
preprocessing_recipe2 <- 
  recipes::recipe(Y ~ ., data = training(data_split)) %>%
  # convert categorical variables to factors
  recipes::step_string2factor(all_nominal()) %>%
  # combine low frequency factor levels
  recipes::step_other(all_nominal(), threshold = 0.01) %>%
  # remove no variance predictors which provide no predictive information 
  recipes::step_nzv(all_nominal()) %>%
  # Center and scale for PCA
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors()) %>%
  # PCA (improves XGBoost performances)
  step_pca(interview, proxy,proxy2,X_score,score,simpson_score1,simpson_score2)%>%
  prep()

#Step 3: Splitting for Cross Validation
data_cv_folds <- 
  recipes::bake(
    preprocessing_recipe, 
    new_data = training(data_split)
  ) %>%  
  rsample::vfold_cv(v = 10)

#Step 4: XGBoost Model Specification
xgboost_model <- 
  parsnip::boost_tree(
    mode = "classification", # classification task
    trees = 1000,
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    loss_reduction = tune()
  ) %>%
  set_engine("xgboost")

#Step 5: Grid Specification, hiperparameters to tune
xgboost_params <- 
  dials::parameters(
    min_n(),
    tree_depth(),
    learn_rate(),
    loss_reduction()
  )

xgboost_grid <- 
  dials::grid_max_entropy(
    xgboost_params, 
    size = 20
  )
knitr::kable(head(xgboost_grid))

#Step 6: Define the Workflow
xgboost_wf <- 
  workflows::workflow() %>%
  add_model(xgboost_model) %>% 
  add_formulìèèèèèè+a(Y ~ .)

#Step 7: Tune the Model
xgboost_tuned <- 
  tune::tune_grid(
    object = xgboost_wf,
    resamples = data_cv_folds,
    grid = xgboost_grid,
    metrics = metric_set(accuracy, roc_auc),
    control = control_grid(verbose = TRUE)
  )

xgboost_tuned %>%
  tune::show_best(metric = "accuracy") %>%
  knitr::kable()

xgboost_best_params <- xgboost_tuned %>%
  tune::select_best("accuracy")
knitr::kable(xgboost_best_params)

xgboost_best_params <- xgboost_tuned %>%
  tune::select_best("roc_auc")
knitr::kable(xgboost_best_params)

xgboost_model_final <- xgboost_model %>% 
  finalize_model(xgboost_best_params)
#min_n:10| tree_depth:3| learn_rate:0.0705904| loss_reduction:0.0662725|.config:Preprocessor1_Model14 |



# With the data set BIASED_data_29_11_22.csv
xgboost_model_final <- 
  parsnip::boost_tree(
    mode = "classification", # classification task
    trees = 1000,
    min_n = 10,
    tree_depth = 3,
    learn_rate = 0.0705904,
    loss_reduction = 0.0662725
  ) %>%
  set_engine("xgboost")

#Step 8: Evaluate Performance on Test Data
train_processed <- bake(preprocessing_recipe,  new_data = training(data_split))
train_prediction <- xgboost_model_final %>%
  # fit the model on all the training data
  fit(
    formula = Y ~ ., 
    data    = train_processed
  ) %>%
  # predict the sale prices for the training data
  predict(new_data = train_processed) %>%
  # predict_proba(new_data = train_processed)
  bind_cols(training(data_split))

# predicted probabilities
train_prob_prediction <- xgboost_model_final %>%
  # fit the model on all the training data
  fit(
    formula = Y ~ ., 
    data    = train_processed
  ) %>%
  # predict the sale prices for the training data
  predict(new_data = train_processed,type="prob") %>%
  # predict_proba(new_data = train_processed)
  bind_cols(training(data_split))


xgboost_score_train <- 
  train_prediction %>%
  yardstick::metrics(Y, .pred_class) %>%
  mutate(.estimate = format(round(.estimate, 2), big.mark = ","))
knitr::kable(xgboost_score_train)

test_processed  <- bake(preprocessing_recipe, new_data = testing(data_split))
test_prediction <- xgboost_model_final %>%
  # fit the model on all the training data
  fit(
    formula = Y ~ ., 
    data    = train_processed
  ) %>%
  # use the training model fit to predict the test data
  predict(new_data = test_processed) %>%
  bind_cols(testing(data_split))

# predicted probabilities on test set
test_prob_prediction <- xgboost_model_final %>%
  # fit the model on all the training data
  fit(
    formula = Y ~ ., 
    data    = train_processed
  ) %>%
  # use the training model fit to predict the test data
  predict(new_data = test_processed, type="prob") %>%
  bind_cols(testing(data_split))

# measure the accuracy of our model using `yardstick`
xgboost_score <- 
  test_prediction %>%
  yardstick::metrics(Y, .pred_class) %>%
  mutate(.estimate = format(round(.estimate, 2), big.mark = ","))
knitr::kable(xgboost_score)

data_prediction_residual <- test_prediction %>%
  arrange(.pred_class) %>%
  mutate(residual = ifelse(Y==.pred_class,1,0)) %>%
  dplyr::select(.pred_class, residual,Y)
prop.table(table(data_prediction_residual$residual))# accuracy
# confusion matrix
conf_matrix <- table(test_prediction$.pred_class,test_prediction$Y)
# confusion matrix by S_a and S_d
conf_matrix_male <- table(test_prediction$.pred_class[which(data_test$S==0)],test_prediction$Y[which(data_test$S==0)])
conf_matrix_female <- table(test_prediction$.pred_class[which(data_test$S==1)],test_prediction$Y[which(data_test$S==1)])
# accuracy
sum(diag(conf_matrix))/sum(conf_matrix) 

# ggplot(data_prediction_residual, aes(x = .pred_class, y = residual)) +
#  geom_point() +
#  xlab("Predicted Sale Price") +
#  ylab("Residual (%)") +
#  scale_x_continuous(labels = scales::dollar_format()) +
#  scale_y_continuous(labels = scales::percent)

## ROC curve 
# this function works without 1-AUC
roc.xgboost <- AUC::roc(predictions = test_prob_prediction$.pred_1,labels = data_test$Y)
plot(roc.xgboost,main="ROC curve for XGBoost")
roc.xgboost

# ROC  
roc_auc(train_prob_prediction, truth = Y, .pred_1)
roc_auc(train_prob_prediction, truth = Y, .pred_0)

roc_data <- roc_curve(train_prob_prediction, truth = Y, .pred_1)
roc_data <- roc_curve(train_prob_prediction, truth = Y, .pred_0)
roc_data %>%  
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) + 
  ggtitle("ROC curve", " XGBoost model")+
  coord_equal()


roc_auc(test_prob_prediction,truth = Y, .pred_0)
# ROC CURVE PER GROUP
# asse x 1-specificità, asse y sensibilità
# it could be interesting to look at the ROC curve per group
roc_data_female <- roc_curve(bind_cols(data_test[which(data_test$S==1),], .pred_0=test_prob_prediction[which(data_test$S==1),]$.pred_0), truth = Y, .pred_0)
roc_data_male <- roc_curve(bind_cols(data_test[which(data_test$S==0),], .pred_0= test_prob_prediction[which(data_test$S==0),]$.pred_0), truth = Y, .pred_0)


roc_data_female$FPR <- (1-roc_data_female$specificity)
roc_data_male$FPR <- (1-roc_data_male$specificity)
plot(roc_data_female[,4:3],type="l",col="firebrick2",main="ROC curve XGBoost by group",
     sub="Per-group for S in test set",xlim=c(0,1),ylim=c(0,1))
lines(roc_data_male[,4:3],type="l",col="blue")
abline(coef=c(0,1),lty=2) # diagonal as random guess
legend("bottomright",c("ROC S=1","ROC S=0"),col=c("firebrick2","blue"),lty=1)

#################################################################################
## MODEL EXPLANATION FOR XGBOOST from chapter 16 

library(DALEXtra)
vip_features <- c("S","interview","Github_account","proxy","proxy2","age","X_score","score","simpson_score1","simpson_score2")

vip_train <- 
  data_train %>% 
  dplyr::select(all_of(vip_features))

# use the tuned model
xgb_fit <- xgboost_model_final %>%
  # fit the model on all the training data
  fit(
    formula = Y ~ ., 
    data    = train_processed
  ) 

## Create an explainer

explainer_xgb <- 
  explain_tidymodels(
    xgb_fit, 
    data = vip_train, 
    y = data_train$Y,
    label = "xgboost",
    verbose = FALSE
  )

## LOCAL EXPLANATION
# loro sono interessanti
data_train[which(data_train$S==1 & data_train$Y==0),] 
Sara<- data_train[969,]
Sara
# Sara <- Sara[,1:11] # omitt treatment group
xgb_breakdown <- predict_parts(explainer = explainer_xgb, new_observation = Sara)
xgb_breakdown

predict_parts(
  explainer = explainer_xgb, 
  new_observation = Sara,
  order = xgb_breakdown$variable_name
)

#  compute SHAP average attributions for this individual
set.seed(876101)
shap_duplex <- 
  predict_parts(
    explainer = explainer_xgb, 
    new_observation = Sara, 
    type = "shap",
    B = 20
  )

# plot the shapley value
library(forcats)
shap_duplex %>%
  group_by(variable) %>%
  mutate(mean_val = mean(contribution)) %>%
  ungroup() %>%
  mutate(variable = fct_reorder(variable, abs(mean_val))) %>%
  ggplot(aes(contribution, variable, fill = mean_val > 0)) +
  geom_col(data = ~distinct(., variable, mean_val), 
           aes(mean_val, variable), 
           alpha = 0.5) +
  geom_boxplot(width = 0.5) +
  theme(legend.position = "none") +
  scale_fill_viridis_d() +
  ggtitle("SHAP XGBoost model for individual S=1")+
  labs(y = NULL)


## GLOBAL EXPLANATION
set.seed(876101)
logit <- function(x) exp(x)/(1+exp(x))
custom_loss <- function(observed, predicted){
  sum((observed - logit(predicted))^2)
}
attr(custom_loss, "loss_name") <- "Logit residuals"
vip_xgb <- model_parts(explainer_xgb,loss_function = custom_loss)
# vip_xgb2 <- model_parts(explainer_xgb, loss_function = loss_root_mean_square)
head(vip_xgb)
plot(vip_xgb)

## 
# making explainer
# create a fairness object

# Model accuracy is ..., what about bias? are men assigned 
# to a better annual income?
str(data_train)
data_train$Y <- as.numeric(data_train$Y)-1
# in caso di errore convertire in numeric la y nell'oggetto explainer
explainer_xgb$y <- as.numeric(explainer_xgb$y)-1 # prima non serviva 

fobject <- fairness_check(explainer_xgb, 
                          protected  = data_train$S, 
                          privileged = "0", 
                          colorize = FALSE)

print(fobject, colorize = FALSE)

# how big is the BIAS?
plot(fobject)
plot_density(fobject)
