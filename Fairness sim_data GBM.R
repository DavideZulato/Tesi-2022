# Analysis of my synthetic data with fairmodels
# Fairmodels advanced Tutorial
# https://modeloriented.github.io/fairmodels/articles/Advanced_tutorial.html
# steps:
# 1 - fit the model
# 2 - create global and local explanations
# 3 - Bias mitigation
# integra il lavoro con Responsible AI, tidy models etc...

library(tidymodels)
library(tidyverse)

## MODEL EXPLANATION AND BIAS
library(fairmodels)

# data
setwd("C:/Users/Davide Zulato/Desktop/Sim_Data")
data_bias <- read.csv("BIASED_data_4_11_22.csv")
data_bias <- data_bias %>% 
  mutate(Y = factor(Y)) %>% 
  mutate(G1=ifelse(trtGrp==1,1,0))%>%
  mutate(G2=ifelse(trtGrp==2,1,0))%>% 
  mutate(G3=ifelse(trtGrp==3,1,0))%>%
  dplyr::select(-X,-trtGrp) # elimino id e trtgroup (dummy)

# the response is a factor
df<- data_bias %>% dplyr::select(-Y) # prova a togliere S e i vari G
correlation <- cor(df)
corrplot::corrplot(correlation,method = "square",type="lower")
# protected variable will be S, target variable = Y
# consider the correlation between the protected feature and the other variables

library(gbm) # for generalized boosted regression modeling
library(DALEX) # for model explanation at global and local level

df$Y   <- as.numeric(data_bias$Y) -1 # 0 (60%) if bad and 1(40%) if good
prop.table(table(data_bias$Y[which(data_bias$S==1)])) # for S=1
prop.table(table(data_bias$Y[which(data_bias$S==0)])) # for S=0
protected     <- df$S
df <- df[colnames(df) != "S"] # sex not specified
# kind of fairness through unawareness

# making model
set.seed(876101)
# Generalized Boosted Regression Modeling
# gradient boosted model with bernoulli loss function
gbm_model <-gbm(Y ~. , data = df, distribution = "bernoulli")

# making explainer
gbm_explainer <- DALEX::explain(gbm_model,
                         data = df[,-13],
                         y = df$Y,
                         colorize = FALSE)

DALEX::model_performance(gbm_explainer)

# Model accuracy is 78%, what about bias? are men assigned 
# to a better outcome?

fobject <- fairness_check(gbm_explainer, 
                          protected  = protected, 
                          privileged = "0", 
                          colorize = FALSE)

print(fobject, colorize = FALSE)

# how big is the BIAS?
plot(fobject)

## BIAS MITIGATION STRATEGIES

# pre-processing techniques
# changing the data before the model is trained

protected <- as.factor(protected)

data_fixed <- disparate_impact_remover(data = df, protected = protected, 
                                       features_to_transform = c("score", "proxy",
                                                                 "simpson_score1",
                                                                 "simpson_score2"))

set.seed(876101)
gbm_model     <- gbm(Y ~. , data = data_fixed, distribution = "bernoulli")
gbm_explainer_dir <- DALEX::explain(gbm_model,
                             data = data_fixed[,-13], # output specified later
                             y = df$Y,
                             label = "gbm_dir",
                             verbose = FALSE)

# now we will compare old explainer and new one

fobject <- fairness_check(gbm_explainer, gbm_explainer_dir,
                          protected = protected, 
                          privileged = "0",
                          verbose = FALSE)
plot(fobject)

## REWEIGHTING
# kamiran and Calderas 2011
# see also Responsible AI

weights <- reweight(protected = protected, y = df$Y)
# i could also customize the weights 


set.seed(876101)
gbm_model     <- gbm(Y ~. ,
                     data = df,
                     weights = weights, # weights Kamiran, Calders 2011
                     distribution = "bernoulli")

gbm_explainer_w <- DALEX::explain(gbm_model,
                           data = df[,-13],
                           y = df$Y,
                           label = "gbm_weighted",
                           verbose = FALSE)

fobject <- fairness_check(fobject, gbm_explainer_w, verbose = FALSE)

plot(fobject)

# take as metric of interest the statistical parity ratio

## RESAMPLING
# see also resampling in data mining and APM

# to obtain probs we will use simple linear regression
probs <- glm(Y ~., data = df, family = binomial())$fitted.values

uniform_indexes      <- resample(protected = protected,
                                 y = df$Y)
preferential_indexes <- resample(protected = protected,
                                 y = df$Y,
                                 type = "preferential",
                                 probs = probs)

set.seed(876101)
gbm_model     <- gbm(Y ~. ,
                     data = df[uniform_indexes,],
                     distribution = "bernoulli")

gbm_explainer_u <- DALEX::explain(gbm_model,
                           data = df[,-13],
                           y = df$Y,
                           label = "gbm_uniform",
                           verbose = FALSE)

set.seed(876101)
gbm_model     <- gbm(Y ~. ,
                     data = df[preferential_indexes,],
                     distribution = "bernoulli")

gbm_explainer_p <- DALEX::explain(gbm_model,
                           data = df[,-13],
                           y = df$Y,
                           label = "gbm_preferential",
                           verbose = FALSE)

fobject <- fairness_check(fobject, gbm_explainer_u, gbm_explainer_p, 
                          verbose = FALSE)
plot(fobject)

# preferential sampling, compared to uniform sampling (random obs from particular
# subgroup) is best at mitigating Statistical parity ratio 

## POST-PROCESSING TECHNIQUES
# changing the outputof model after model is generated

# ROC pivot

# we will need normal explainer 
set.seed(876101)
gbm_model <-gbm(Y ~. , data = df, distribution = "bernoulli")
gbm_explainer <- DALEX::explain(gbm_model,
                         data = df[,-13],
                         y = df$Y,
                         verbose = FALSE)

gbm_explainer_r <- roc_pivot(gbm_explainer,
                             protected = protected,
                             privileged = "0")


fobject <- fairness_check(fobject, gbm_explainer_r, 
                          label = "gbm_roc",  # label as vector for explainers
                          verbose = FALSE) 

plot(fobject)

print(fobject, colorize = FALSE)

# Cutoff manipolation

set.seed(876101)
gbm_model <-gbm(Y ~. , data = df, distribution = "bernoulli")
gbm_explainer <- DALEX::explain(gbm_model,
                         data = df[,-13],
                         y = df$Y,
                         verbose = FALSE)

# test fairness object
fobject_test <- fairness_check(gbm_explainer, 
                               protected = protected, 
                               privileged = "0",
                               verbose = FALSE) 

# win.graph()
plot(ceteris_paribus_cutoff(fobject_test, subgroup = "1"))

# it is possible to minimize all metrics or only few metrics of our interest

plot(ceteris_paribus_cutoff(fobject_test,
                            subgroup = "1",
                            fairness_metrics = c("ACC","TPR","STP")))

fc <- fairness_check(gbm_explainer, fobject,
                     label = "gbm_cutoff",
                     cutoff = list("1"= 0.25), # in the S 1 is the disadvantaged class
                     verbose = FALSE)

plot(fc)

print(fc, colorize=FALSE)

## TRADEOFF BETWEEN BIAS AND ACCURACY

# there is significant tradeoff between bias and accuracy, one way to visualize 
# it is to use "performance_and_fairness" function

paf <- performance_and_fairness(fc, fairness_metric = "STP",
                                performance_metric = "accuracy")

win.graph()
plot(paf,main="tradeoff accuracy and bias gbm models")

# While performing standard model development developers usually split the data 
# into train-test subsets. check fairness on a test set.
# check fairness in the test set

data_split <- initial_split(data_bias, prop = 3/4, strata = "Y")
data_train <- training(data_split)
data_test <- testing(data_split)

data_test$Y <- as.numeric(data_test$Y) -1 
protected_test <- data_test$S

data_test <- data_test[colnames(data_test) != "S"]

# on test
gbm_explainer_test <- DALEX::explain(gbm_model,
                              data = data_test[,-10],
                              y = data_test$Y,
                              verbose = FALSE)

# the objects are tested on different data, so we cannot compare them on one plot
fobject_train <- fairness_check(gbm_explainer, 
                                protected = protected, 
                                privileged = "0", 
                                verbose = FALSE)

fobject_test  <- fairness_check(gbm_explainer_test, 
                                protected = protected_test, 
                                privileged = "0", 
                                verbose = FALSE)

library(patchwork) # with patchwork library we will nicely compare the plots

plot(fobject_train) + plot(fobject_test)
plot(fobject_train, main="Fairness Train") + plot(fobject_test, main="Fairness Test")

# suggestion from Jakub Wiśniewski
# It is also good idea to combine few techniques (for example minimizing once
# with weights and then with cutoff). fairness_check interface is flexible and 
# allows combining models that were trained on different features, encodings etc. 
