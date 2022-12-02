# Reweighting the data
# from page 74 RESPONSIBLE AI
# take the code from SPD and DI function

## 1
# define the favourable and unfavourable outcome

setwd("C:/Users/Davide Zulato/Desktop/Sim_Data")
data_bias <- read.csv("BIASED_data_4_11_22.csv")

n <- nrow(data_bias)

sa <- nrow(data_bias[which(data_bias$S==0),]) # total number of privileged
sd <- nrow(data_bias[which(data_bias$S==1),]) # total number of unprivileged
ypos <- nrow(data_bias[which(data_bias$Y==1),]) # total number of favourable
yneg <- nrow(data_bias[which(data_bias$Y==0),]) # total number of unfavourable
sa;sd;ypos;yneg

data_sa_ypos <- data_bias[which(data_bias$S==0 & data_bias$Y==1),] # privileged and favourable
data_sa_yneg <- data_bias[which(data_bias$S==0 & data_bias$Y==0),] # privileged and unfavourable
data_sd_ypos <- data_bias[which(data_bias$S==1 & data_bias$Y==1),] # unprivileged and favourable
data_sd_yneg <- data_bias[which(data_bias$S==1 & data_bias$Y==0),] # unprivileged and unfavourable

sa_ypos <- nrow(data_sa_ypos) # total number of privileged and favourable
sa_yneg <- nrow(data_sa_yneg) # total number of privileged and unfavourable
sd_ypos <- nrow(data_sd_ypos) # total number of unprivileged and favourable
sd_yneg <- nrow(data_sd_yneg) # total number of unprivileged and unfavourable
paste0("Total number of advantaged and favourable:",sa_ypos)
paste0("Total number of advantaged and unfavourable:",sa_yneg)
paste0("Total number of disadvantaged and favourable:",sd_ypos)
paste0("Total number of disadvantaged and unfavourable:",sd_yneg)

w_sa_ypos <- (ypos*sa)/(n*sa_ypos) # weight for privileged and favourable
w_sa_yneg <- (yneg*sa)/(n*sa_yneg) # weight for privileged and unfavourable
w_sd_ypos <- (ypos*sd)/(n*sd_ypos) # weight for unprivileged and favourable
w_sd_yneg <- (yneg*sd)/(n*sd_yneg) # weight for unprivileged and unfavaurable

datatest <- data_bias

discrimination_before <- (sa_ypos/sa)-(sd_ypos/sd)
round(discrimination_before,3)
discrimination_after <- (sa_ypos/sa * w_sa_ypos)-(sd_ypos/sd * w_sd_ypos)
round(discrimination_after,3)

# assign weights
datatest$weights <- vector(mode="numeric",length = nrow(datatest))
datatest$weights[which(datatest$S==0 & datatest$Y==1)] = w_sa_ypos
datatest$weights[which(datatest$S==0 & datatest$Y==0)] = w_sa_yneg
datatest$weights[which(datatest$S==1 & datatest$Y==1)] = w_sd_ypos
datatest$weights[which(datatest$S==1 & datatest$Y==0)] = w_sd_yneg 

# afrer the fairness_sim_data GBM you could run the following code
weights_custom <- datatest$weights

set.seed(876101)
gbm_model2     <- gbm(Y ~. ,
                     data = df,
                     weights = weights_custom, # weights Kamiran, Calders 2011
                     distribution = "bernoulli")
gbm_explainer_cw <- DALEX::explain(gbm_model2,
                                  data = df[,-13],
                                  y = df$Y,
                                  label = "gbm_custom_weight",
                                  verbose = FALSE)

fobject_custom <- fairness_check(gbm_explainer_cw,protected = factor(datatest$S),privileged = "0")
plot(fobject_custom)


fobject <- fairness_check(fobject, gbm_explainer_w,gbm_explainer_cw, verbose = FALSE)

plot(fobject)
# thei look very similar to the weights in the function "reweight"