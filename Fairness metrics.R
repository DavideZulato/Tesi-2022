# Simple prediction exercise with logistic regression
# the function reproduces the python code in p.20/21 "Responsible AI"
# The fairness measures will be evaluated on the test set

library(tidymodels)
library(tidyverse)
setwd("C:/Users/Davide Zulato/Desktop/Sim_Data")
data_bias <- read.csv("BIASED_data_29_11_22.csv")

data_bias <- 
  data_bias %>% 
  mutate(Y= as.factor(Y)) %>%# response as factor
  dplyr::select(-X)

# data split
set.seed(876101)
data_split <- initial_split(data_bias, prop = 3/4, strata = "Y")
data_train <- training(data_split)
data_test <- testing(data_split)

logit_fit <- 
  logistic_reg(mode = "classification") %>%
  set_engine(engine = "glm") %>% 
  fit(Y ~ ., data = data_train)

y_actual <- data_test$Y
y_pred_prob <- predict(logit_fit, new_data = data_test, type = "prob") 
y_pred_binary <-  predict(logit_fit, new_data = data_test)
y_pred_binary <- y_pred_binary$.pred_class
X_test <- data_bias %>% dplyr::select(-Y)
protected_group <- "S"
adv_val <- 0 # privileged class
disadv_val <- 1  # unprivileged class
# Fairness Metrics
# fairness performance metrics for a model to compare S_a and S_d
conf_matrix <- table(y_pred_binary,y_actual)
tn <- conf_matrix[1,1]
fn <- conf_matrix[1,2]
fp <- conf_matrix[2,1]
tp <- conf_matrix[2,2]

## Function for fairness performance metrics
fair_metrics <- function(y_actual,y_pred_prob,y_pred_binary,X_test,protected_group,adv_val,disadv_val){
  # y_actual <- data_bias$Y
  # y_pred_prob <- predict(logit_fit, new_data = data_bias, type = "prob") 
  # y_pred_binary <-  predict(logit_fit, new_data = data_bias)
  # X_test <- data_bias %>% dplyr::select(-Y)
  # protected_group <- "S"
  # adv_val <- 0 # privileged class
  # disadv_val <- 1  # unprivileged class
  # confusion matrix by S_a and S_d
  conf_matrix_adv <- table(y_pred_binary[X_test[protected_group]==adv_val],
                           y_actual[X_test[protected_group]==adv_val])
  tn_adv <- conf_matrix_adv[1,1]
  fn_adv <- conf_matrix_adv[1,2]
  fp_adv <- conf_matrix_adv[2,1]
  tp_adv <- conf_matrix_adv[2,2]
  # disadvantaged
  conf_matrix_disadv <- table(y_pred_binary[X_test[protected_group]==disadv_val],
                              y_actual[X_test[protected_group]==disadv_val])
  tn_disadv <- conf_matrix_disadv[1,1]
  fn_disadv <- conf_matrix_disadv[1,2]
  fp_disadv <- conf_matrix_disadv[2,1]
  tp_disadv <- conf_matrix_disadv[2,2]
  
  # Equal opportunity: S_d and S_a same FNR
  FNR_adv <- (fn_adv)/(fn_adv+tp_adv)
  FNR_disadv <- (fn_disadv)/(fn_disadv+tp_disadv)
  EOpp_diff = abs(FNR_disadv-FNR_adv)
  
  # Predictive equality S_d and S_a same FPR
  FPR_adv <- (fp_adv)/(fp_adv+tn_adv)
  FPR_disadv <- (fp_disadv)/(fp_disadv+tn_disadv)
  pred_eq_diff = abs(FPR_disadv-FPR_adv)
  
  # Equalized odds S_d and S_a same TPR+FPR
  TPR_adv <- (tp_adv)/(tp_adv+fn_adv)
  TPR_disadv <- (tp_disadv)/(tp_disadv+fn_disadv)
  EOdds_diff = abs((TPR_disadv + FPR_disadv)- ((TPR_adv + FPR_adv)))
  
  # Predictive parity S_d and S_a same PPV/precision (TP/TP+FP)
  prec_adv <- (tp_adv)/(tp_adv+fp_adv)
  prec_disadv <- (tp_disadv)/(tp_disadv+fp_disadv)
  prec_difference = abs(prec_disadv-prec_adv)
  
  # Demographic parity y_hat=1/total instances
  demo_parity_adv <- (tp_adv+fp_adv)/(tn_adv+fp_adv+fn_adv+tp_adv)
  demo_parity_disadv <- (tp_disadv+fp_disadv)/(tn_disadv+fp_disadv+fn_disadv+tp_disadv)
  demo_parity_diff <- abs(demo_parity_disadv-demo_parity_adv)
  
  # average of difference in FPR and TPR for S_a and S_d
  AOD = 0.5*((FPR_disadv-FPR_adv)+(TPR_disadv+TPR_adv))
  
  # Treatment equality
  TE_adv = fn_adv/fp_adv
  TE_disadv = fn_disadv/fp_disadv
  TE_diff = abs(TE_disadv-TE_adv)
  
  return(list("Equal opportunity difference"=EOpp_diff,
              "Predictive equality difference"=pred_eq_diff,
              "Equalized odds difference"=EOdds_diff,
              "PPV Difference"=prec_difference,
              "Demographic parity difference"=demo_parity_diff,
              "AOD"=AOD,
              "Treatment equality difference"=TE_diff))
  
}

fair_metrics(y_actual,y_pred_prob,y_pred_binary,X_test,protected_group,adv_val,disadv_val)  
