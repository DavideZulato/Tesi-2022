## PROPENSITY SCORE MATCHING
# Tesi 2022 "Discrimination in HR analytics, a fair workflow"

# Is there a causal impact of the sensitive feature S on the probability of getting a positive outcome? 
# we will build a model to create a counterfactual group with observational data

library(readr)
library(dplyr)
library(ggplot2)

# We will use propensity score matching to create counterfactual group to assess 
# positive outcome denoted as Y=1 (or eventually flip it)
# The statistical quantity of interest is the causal effect of the treatment (S=1)
# on probability of getting a negative outcome (Y=0)

setwd("C:/Users/Davide Zulato/Desktop/Sim_Data")
Data = read_csv('BIASED_data_29_11_22.csv')[,-1]
head(Data)
Data <- Data %>% mutate_if(is.logical, as.numeric)
str(Data)

# packages MatchIt and optmatch
# install.packages("optmatch")
# install.packages("MatchIt")
library("MatchIt")

# Y is the dummy variable for the response in HRM context
# "S" the protected feature can be considered as the treatment variable (S_d),
# "Y" is the outcome, and the others are pre-treatment covariates.

# is the probability of getting negative outcome of the treated higher?
# is it a randomized experiment? May yes
# in HRM we could question the assumption of randomized experiment
# because subject may be self- selected, gender for instance obviously pre-exist.

# The regression should include relevalt variables
fit0 <- lm(Y ~ ., data = Data)
summary(fit0)
# Treatment "S=1" e.g. female 
prova <- glm(S ~ ., data=Data) # try to leverage out response
summary(prova)

# treat is a dummy variable, dependent is a dummy varable (logit/probit but ols is interpretable)
# Gender has an impact on the outcome => matching in order to combine similar observation

# matching, there are various matching mechanism

# variable that doesn't change over time or thet does't change the outcome (check theory)
# nearest neighbor propensity score matching.
# 1:1 NN PS matching w/o replacement
m.out1 <- matchit(S ~ interview + Github_account + proxy + proxy2 + age + X_score + score + simpson_score1 + simpson_score2 + Treatment, data=Data,
                  method = "nearest", distance = "glm")

summary(m.out1)
plot(m.out1, type = "jitter", interactive = FALSE) # distribution of the propensity score (similar treated and control)
plot(m.out1, type = "density", which.xs = c("age", "proxy"),
     subclass = 1)
plot(m.out1, type = "density", which.xs = c("interview", "X_score"),
     subclass = 1)
plot(m.out1, type = "density", which.xs = c("simpson_score1", "simpson_score2"),
     subclass = 1)

library(cobalt)
# Love plot of chosen balance statistics
love.plot(m.out1, binary = "std")
# Density plot for continuous variables
# displays distributional balance for a single covariate, similar to plot.matchit(). 
# Its default is to display kernel density plots for continuous variables and bar 
# graphs for categorical variables. It can also display eCDF plots and histograms
bal.plot(m.out1, var.name = "age", which = "both")
#Bar graph for categorical variables
bal.plot(m.out1, var.name = "score", which = "both")
bal.plot(m.out1, var.name = "X_score", which = "both")
bal.plot(m.out1, var.name = "simpson_score1", which = "both")
bal.plot(m.out1, var.name = "simpson_score2", which = "both")
bal.plot(m.out1, var.name = "Github_account", which = "both")
#Mirrored histogram
bal.plot(m.out1, var.name = "age", which = "both",
         type = "histogram", mirror = TRUE)


# unmatched have propensity score close to zero
plot(m.out1, type = "qq", interactive = FALSE,
     which.xs = c("proxy", "score","X_score")) # if perfect mach each observation has a twin (all on diagonal)
plot(summary(m.out1)) # absolute standardized mean difference, if the control is well done it should be about zero
# a good match reduces the difference, check the common support (very well if close to zero)
win.graph()
plot(m.out1, type = "hist") # expected to be common support, another way to assess the goodness of matching
# is looking to the distribution

## Estimating the Treatment Effect

m.data1 <- match.data(m.out1)
m.data1$S<-as.factor(m.data1$S)

library(viridis)
win.graph()
matched_data <- m.data1 %>%
  ggplot(aes(x=S, y=Y, fill=S)) +
  geom_boxplot() +
  scale_fill_viridis(discrete = TRUE, alpha=0.6) +
  geom_jitter(color="black", size=0.4, alpha=0.9)  +
  theme(
    legend.position="none",
    plot.title = element_text(size=11)
  ) +
  ggtitle("Treated vs. Matched controls") +
  xlab("")


original_data <- Data %>%
  ggplot( aes(x=S, y=Y, group=S)) +
  geom_boxplot() +
  scale_fill_viridis(discrete = TRUE, alpha=0.6) +
  geom_jitter(color="black", size=0.4, alpha=0.9)  +
  theme(
    legend.position="none",
    plot.title = element_text(size=11)
  ) +
  ggtitle("Treated vs. all controls") 

library(gridExtra)
gridExtra::grid.arrange(matched_data,original_data)

## Different matching
# full matching, which matches every treated unit to at least one control and every control 
# to at least one treated unit. We'll also try a 
# different link (probit) for the propensity score model.
m.out2 <- matchit(S ~ interview + Github_account + proxy + proxy2 + age + X_score + score + simpson_score1 + simpson_score2 + Treatment, data=Data,
                  method = "full", distance = "gbm", link = "probit", estimand = "ATT") # estimand=ATE for ATE
# The propensity scores are estimated using a generalized additive model (gam)
# The propensity scores are estimated using a generalized boosted model (gbm) resulted in better
# covariate balance

summary(m.out2)
plot(m.out2, type = "qq", interactive = FALSE,
     which.xs = c("proxy", "score","X_score")) # if perfect mach each observation has a twin (all on diagonal)

win.graph()
plot(summary(m.out2)) # the matched have a ASMD close to zero (cool)
title("matching","method=Full,distance=gbm,link=probit")
plot(m.out2, type = "hist")

plot(m.out2, type = "jitter", interactive = FALSE) # distribution of the propensity score (similar treated and control)
# Imbalances are represented by the differences between the black (treated) and gray (control) distributions
plot(m.out2, type = "density", which.xs = c("age", "proxy","proxy2"),subclass = 1)
plot(m.out2, type = "density", which.xs = c("interview", "X_score","score"),subclass = 1)
plot(m.out2, type = "density", which.xs = c("simpson_score1", "simpson_score2"),subclass = 1)

# balance in mean differences and on Kolmogorov-Smirnov statistics and 
# for both full matching and nearest neighbor matching simultaneously.
love.plot(m.out1, stats = c("m", "ks"), poly = 2, abs = TRUE,
          weights = list(nn = m.out2),
          drop.distance = TRUE, thresholds = c(m = .1),
          var.order = "unadjusted", binary = "std",
          shapes = c("circle filled", "triangle", "square"), 
          colors = c("red", "blue", "darkgreen"),
          sample.names = c("Original", "Full Matching", "NN Matching"),
          position = "bottom")


## Estimating the Treatment Effect

m.data2 <- match.data(m.out2)
str(m.data2)
m.data2$S<-as.factor(m.data2$S)
love.plot(m.out2, binary = "std")
love.plot(m.out2)

library("lmtest") #coeftest
library("sandwich") #vcovCL

library(viridis)
win.graph()
m.data2 %>%
  ggplot(aes(x=S, y=Y, fill=S)) +
  geom_boxplot() +
  scale_fill_viridis(discrete = TRUE, alpha=0.6) +
  geom_jitter(color="black", size=0.4, alpha=0.9)  +
  theme(
    legend.position="none",
    plot.title = element_text(size=11)
  ) +
  ggtitle("Treated vs. Matched controls") +
  xlab("")

#no covariates, effect is higher, if i run the same model only with treated (comparing the means basically)+
# see the effect without covariates
fit2 <- lm(Y ~ S, data = m.data2, weights = weights)
summary(fit2)

coeftest(fit2, vcov. = vcovCL, cluster = ~subclass)
# performing z and (quasi-)t Wald tests of estimated coefficients
# vcov is the specification for the covariance matrix


m.data2$S<-as.factor(m.data2$S)

# DIFFERENCE-IN-MEANS
# The means below indicate that we have attained a (low) degree of balance on 
# the covariates included in the model.
mean(m.data2$Y[which(m.data2$S==1)])-mean(m.data2$Y[which(m.data2$S==0)])
# what happened in the first specification (looks better from this standpoint)
mean(m.data1$Y[which(m.data1$S==1)])-mean(m.data1$Y[which(m.data1$S==0)])

# ESTIMATING TREATMENT EFFECTS
# Estimating the treatment effect on matched sample, t.test
with(m.data2, t.test(Y ~ S)) # reject null hypothesis
# gender has effect 

# see the effect
m.data2$S <- as.factor(m.data2$S)
m.data2$Y <- as.factor(m.data2$Y)
logit_S_match2 <- m.data2 %>%
  glm(formula = Y ~ S, family = "binomial")

# with the full matching 
jtools::summ(logit_S_match2, digits = 5, exp = T)
jtools::effect_plot(logit_S_match2, pred = S, plot.points = TRUE,
            jitter = c(0.1, 0.05), point.alpha = 0.1, colors = "firebrick4") +
  ylab("Pr(Y = 1)")+
  xlab("Exposure:S=1")+
  ggtitle("Effect plot with Logit")

library("marginaleffects")
# lm or glm
m.data2$Y <- as.numeric(m.data2$Y)-1
fit <- glm(Y ~ S * (age + interview + Github_account + proxy + proxy2 + 
                     X_score + score+simpson_score1+simpson_score2), data = m.data2, weights = weights)

comp <- comparisons(fit,
                    variables = "S",
                    vcov = ~subclass,
                    newdata = subset(m.data2, S == 1),
                    wts = "weights")
summary(comp)


# The estimated effect was 0.01436 (SE = 0.03624, Pr(>|z|) = 0.69192 ), indicating that the average 
# effect of the treatment for those who received it is not significantly different from zero.

## ESTIMATING TREATMENT EFFECT, ATT, ATE, RR (RISK RATIO)
#Extract matched data
md <- match.data(m.out2)

head(md)

# First, we fit a model for the outcome given the treatment and the covariates.
# include treatment-covariate interactions

# Linear model with covariates
fit1 <- lm(Y ~ S*(interview + Github_account + proxy + proxy2 + age + X_score + score + simpson_score1 + simpson_score2),
           data = md, weights = weights)
# marginaleffects::comparisons() to estimate the ATT.

comp1 <- comparisons(fit1,
                     variables = "S",
                     vcov = ~subclass,
                     newdata = subset(md, S == 1),
                     wts = "weights")
summary(comp1)

# average estimated potential outcomes

pred1 <- predictions(fit1,
                     variables = "S",
                     vcov = ~subclass,
                     newdata = subset(md, S == 1),
                     wts = "weights",
                     by = "S")

summary(pred1)

#Logistic regression model with covariates
fit2 <- glm(Y ~ S*(interview + Github_account + proxy + proxy2 + age + X_score + score + simpson_score1 + simpson_score2),
            data = md, weights = weights,
            family = quasibinomial())

#Compute effects
comp2 <- comparisons(fit2,
                     variables = "S",
                     vcov = ~subclass,
                     newdata = subset(md, S == 1),
                     wts = "weights",
                     transform_pre = "lnratioavg")

#Log RR, standard error, and Z value
summary(comp2)



# BOOTSTRAP CONFIDENCE INTERVAL
boot_fun <- function(data, i) {
  boot_data <- data[i,]
  
  #Do 1:1 PS matching with replacement
  m <- matchit(S ~.,
               data = boot_data,
               replace = TRUE)
  
  #Extract matched dataset
  md <- match.data(m, data = boot_data)
  
  #Fit outcome model
  fit <- glm(Y ~ S*(interview + Github_account + proxy + proxy2 + age + X_score + score + simpson_score1 + simpson_score2),
             data = md, weights = weights,
             family = quasibinomial())
  
  ## G-computation ##
  #Subset to treated units for ATT; skip for ATE
  md1 <- subset(md, S == 1)
  
  #Estimated potential outcomes under treatment
  p1 <- predict(fit, type = "response",
                newdata = transform(md1, S = 1))
  Ep1 <- weighted.mean(p1, md1$weights)
  
  #Estimated potential outcomes under control
  p0 <- predict(fit, type = "response",
                newdata = transform(md1, S = 0))
  Ep0 <- weighted.mean(p0, md1$weights)
  
  #Risk ratio
  return(Ep1 / Ep0)
}

library("boot")
set.seed(876101)
# original dataset supplied to perform the bootstrapping
boot_out <- boot(Data, boot_fun, R = 199)

boot_out

boot.ci(boot_out, type = "perc")
