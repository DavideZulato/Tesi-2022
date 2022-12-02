## PROPENSITY SCORE MATCHING

# Is there a causal impact of the sensitive feature S on the probability of getting a positive outcome? 
# we will build a model to create a counterfactual group with observational data

library(readr)
library(dplyr)
library(ggplot2)

# I will use propensity score matching to create counterfactual group to assess 
# positive outcome denoted as Y=1 (of flip it)
# The statistical quantity of interest is the causal effect of the treatment (Sensitive feature)
# on probability of getting a negative outcome (Y=0)

setwd("C:/Users/Davide Zulato/Desktop/Sim_Data")
# Data = read_csv('BIASED_data_11_11_22.csv')[,-1]
Data = read_csv('BIASED_data_4_11_22.csv')[,-1]
head(Data)
Data <- Data %>% mutate_if(is.logical, as.numeric)
str(Data)

Data$Y <- as.numeric(Data$Y)
Data$S <- as.numeric(Data$S) # 1 if male, 0 if Female


# packages MatchIt and optmatch
# install.packages("optmatch")
# install.packages("MatchIt")
library("MatchIt")


# Y is the dummy variable for the response in HRM context
# "S" the protected feature can be considered as the treatment variable,
# "Y" is the outcome, and the others are pre-treatment covariates.

## is the probability of getting negative outcome of the treated higher?
# is it a randomized experiment? May yes
# in HRM we could question the assumption of randomized experiment
# because subject doesn't self- selected, gender obviously pre-exist

# The regression should include relevalt variables
fit0 <- lm(Y ~ ., data = Data)
# fit0 <- lm(Churn ~ ., data = churn_data)
summary(fit0)
# Treatment "S=1" female ideally
prova <- glm(S ~ ., data=Data) # try to leverage out response
summary(prova)

# treat is a dummy variable, dependent is a dummy varable (logit/probit but ols is interpretable)
# Gender has an impact on the outcome => matching in order to combine similar observation

# matching, there are various matching mechanism

# variable that doesn't change over time or thet does't change the outcome (check theory)
# nearest neighbor propensity score matching.
# 1:1 NN PS matching w/o replacement
m.out1 <- matchit(S ~ ., data=Data,
                  method = "nearest", distance = "glm")

summary(m.out1)
plot(m.out1, type = "jitter", interactive = FALSE) # distribution of the propensity score (similar treated and control)
plot(m.out1, type = "density", which.xs = c("age", "proxy","Y"),
     subclass = 1)
plot(m.out1, type = "density", which.xs = c("interview", "X_score","Y"),
     subclass = 1)
plot(m.out1, type = "density", which.xs = c("simpson_score1", "simpson_score2","Y"),
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
# a good match reduces the difference, check the common support # # very well, close to zero
plot(m.out1, type = "hist") # expected to be common support, another way to assess the goodness of matching
# is looking to the distribution

## Estimating the Treatment Effect

m.data1 <- match.data(m.out1)
m.data1$S<-as.factor(m.data1$S)

library(viridis)
m.data1 %>%
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


Data %>%
  ggplot( aes(x=S, y=Y, group=S)) +
  geom_boxplot() +
  scale_fill_viridis(discrete = TRUE, alpha=0.6) +
  geom_jitter(color="black", size=0.4, alpha=0.9)  +
  theme(
    legend.position="none",
    plot.title = element_text(size=11)
  ) +
  ggtitle("Treated vs. all controls") 

## Different matching
# full matching, which matches every treated unit to at least one control and every control 
# to at least one treated unit. We'll also try a 
# different link (probit) for the propensity score model.
m.out2 <- matchit(S ~ ., data=Data,
                  method = "full", distance = "glm", link = "probit")

plot(summary(m.out2)) # the matched have a ASMD close to zero (cool)
title("matching","method=Full,distance=glm,link=probit")
plot(m.out2, type = "hist")

## Estimating the Treatment Effect

m.data2 <- match.data(m.out2)
str(m.data2)
m.data2$S<-as.factor(m.data2$S)
love.plot(m.out2, binary = "std")

library("lmtest") #coeftest
library("sandwich") #vcovCL


m.data1 <- match.data(m.out1) # not the original data
fit1 <- lm(Y ~ + ., data = m.data1, weights = weights)

coeftest(fit1, vcov. = vcovCL, cluster = ~subclass)
#  performing z and (quasi-)t Wald tests of estimated coefficients
# vcov is the specification for the covariance matrix

summary(fit1)
#no covariates, effect is higher, if i run the same model only with treated (comparing the means basically)+
# see the effect without covariates
fit2 <- lm(Y ~ S, data = m.data1, weights = weights)
summary(fit2)


m.data1$S<-as.factor(m.data1$S)

# DIFFERENCE-IN-MEANS
# The means below indicate that we have attained a high degree of balance on 
# the covariates included in the model.


# ESTIMATING TREATMENT EFFECTS
# Estimating the treatment effect on matched sample, t.test
with(m.data1, t.test(Y ~ S)) # reject null hypothesis
# gender has effect 

# see the effect
m.data1$S <- as.factor(m.data1$S)
m.data1$Y <- as.factor(m.data1$Y)
logit_Male_match <- m.data1 %>%
  glm(formula = Y ~ S, family = "binomial")
logit_Male_match2 <- m.data2 %>%
  glm(formula = Y ~ S, family = "binomial")
# first matching
library(jtools)
library(hrbrthemes)
jtools::summ(logit_Male_match, digits = 5, exp = T)
effect_plot(logit_Male_match, pred = S, plot.points = TRUE,
            jitter = c(0.1, 0.05), point.alpha = 0.1, colors = "firebrick4") +
  ylab("Pr(Y = 1)")+
  xlab("Exposure:S=1")+
  ggtitle("Effect plot with Logit")

# with the full matching 
jtools::summ(logit_Male_match2, digits = 5, exp = T)
effect_plot(logit_Male_match2, pred = S, plot.points = TRUE,
            jitter = c(0.1, 0.05), point.alpha = 0.1, colors = "firebrick4") +
  ylab("Pr(Y = 1)")+
  xlab("Exposure:S=1")+
  ggtitle("Effect plot with Logit")
