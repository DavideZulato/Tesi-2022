# R code for data generation
# check for SPD and DI

setwd("C:/Users/Davide Zulato/Desktop/Sim_Data")
# Libraries
library(ggplot2)
library(ggpubr)
library(hrbrthemes)
library(dplyr)
library(tidyr)
library(viridis)
library(simstudy)
library(MASS)
library(tidyverse)
library(data.table)

# numerosità campione
n <- 10000 
set.seed(876101)

# age, idealmente neolaureati triennali, distribuzione molto asimmetrica
age <- 22+rchisq(n,0.5) # sommo molti valori prossimi a zero
summary(age)
hist(age,breaks=50); boxplot(age)

# sensitive feature S (e.g. gender, AIDS etc...) dummy
S <- rbinom(n,1,0.2)
table(S)

# Poisson per una score in contesto HRM
lambda <- 1.5 + 0.2 * age + 1.5 * S + rnorm(1) # rumore white noise
interview <- rpois(n,lambda = lambda)
summary(interview)
dd <- as.data.frame(cbind(S,interview))

p1 <- ggplot(data=dd, aes(x=interview, group=S, fill=S),col=c(3,4)) +
  geom_density(adjust=1.5, alpha=.4) +
  ggtitle("Distribuzione punteggio per genere") 
p1


# Github account, associata negativamente con l'appartenenza alla classe protetta S
set.seed(876101)
z = 1-0.7*S+rnorm(1)        # linear combination with a bias
# pra = exp(z)/(1+exp(z))  
pr = 1/(1+exp(-z))         # pass through an inv-logit function
Github_account = rbinom(n,1,pr)      # bernoulli response variable

# relation with S e prop.table per riga
table(S, Github_account); round(prop.table(table(S, Github_account),1),2)
round(prop.table(table(S, Github_account),2),2)  
# tra i possessori di un account Github ~13% donne, sottorappresentate

#now feed it to glm:
df = data.frame(Github_account,S)
glm(Github_account ~S,data=df,family="binomial")

## Proxy feature
mu <- 3*S+rnorm(1) # linear combination + noise
proxy <- (rnorm(n,mu,2))
summary(proxy)
cor(proxy,S) # c.a. 0.5


## 2nd proxy debole correlazione lineare con S, dipende anche da proxy e age
mu <- (-1.5*S+3*(proxy)+2*age+rnorm(1))/50
proxy2 <- (rbeta(n,mu,8,ncp=0.9))
# distribution asimmetrica
hist(proxy2, breaks=30, xlim=c(0,3), col=rgb(1,0,0,0.5), xlab="proxy2", 
     ylab="nbr", main="distribution of proxy2" )
cor(proxy,proxy2) # troppo debole
cor(proxy2,S)
# take a look at the data frame
dd <- data.frame(S,interview,Github_account,proxy,proxy2,age)


# Score che non dipende dalla feature protetta rientra tra le X
M <- which(dd$S==0) # male
Fem <- which(dd$S==1) # Female
# score media 100 e sd=5
dd$X_score <- rnorm(nrow(dd),100,5)
dd$S<- as.factor(dd$S)
p1 <- ggplot(data=dd, aes(x=X_score, group=S, fill=S),col=c(3,4)) +
  geom_density(adjust=1.5, alpha=.4) +
  ggtitle("Distribuzione punteggio per genere") 
p1 # sovrapposizione
mean(dd$X_score[M]); mean(dd$X_score[Fem])
t.test(X_score ~ S, data = dd) # no evisenza vs H0
summary(round(dd$X_score))

# voti di uscita massimo 110
too_high <- which(dd$X_score>110)
dd$X_score[too_high] <- 110
summary(dd$X_score)
# X_score <- dd$X_score
# definibile anche in def

vectM <- rpois(length(M),lambda = 7)
vectF <- rpois(length(Fem),lambda = 5)
dd$score <- ifelse(dd$S==1,vectF,vectM)
# hist(dd$score)
# hist(dd$score, breaks = 50)
dd$S <- as.factor(dd$S)

# medie per valori di S
mean(dd$score[Fem]);mean(dd$score[M])
t.test(score ~ S, data = dd) # intervallo di confidenza non comprende lo zero
# differenza significativa

# With transparency (right)
p2 <- ggplot(data=dd, aes(x=score, group=S, fill=S),col=c(3,4)) +
  geom_density(adjust=1.5, alpha=.4) +
  ggtitle("Distribuzione punteggio per genere") 
p2 # similar to interview


## SIMPSONS SCORE (da Judea Pearl)
# in HRM an example is the relation between Neuroticism and salary
# we assume that the relation at global level between the two features (affecting the outcome)
# is the opposite (or merely different) if we look the same relation by group.

# start from bivariate normal conditioning to S_a, S=1 (e.g. female)
mu<-c(7,7)
sigma<-rbind(c(2,-0.1),c(0,2) )
Males<-as.data.frame(mvrnorm(n=length(M), mu=mu, Sigma=sigma))
# conditioning to and S_d, S=1
mu<-c(4,4)
sigma<-rbind(c(2,-0.7),c(-0.7,2) )
Females<-as.data.frame(mvrnorm(n=length(Fem), mu=mu, Sigma=sigma))

# Contenitori
dd$simpson_score1 <- vector(mode="numeric",length = nrow(dd))
dd$simpson_score2 <- vector(mode="numeric",length = nrow(dd))

dd$simpson_score1[which(dd$S==0)] <- Males[,1]
dd$simpson_score2[which(dd$S==0)] <- Males[,2]
dd$simpson_score1[which(dd$S==1)] <- Females[,1]
dd$simpson_score2[which(dd$S==1)] <- Females[,2]
min(dd$simpson_score1);max(dd$simpson_score1)
min(dd$simpson_score2);max(dd$simpson_score2)

# the combined data 
gg1 <- dd%>%ggplot(aes(x=simpson_score1, y=simpson_score2))+geom_point()+ 
  geom_smooth(method='lm')+
  ggtitle("Combined data for the two scores","Simpson's Paradox")

# by group
gg2 <- dd%>%ggplot(aes(x=simpson_score1, y=simpson_score2, group=S, col=S))+
  geom_point()+ geom_smooth(method='lm', col='black')+
  ggtitle("By group","Simpson's Paradox")

# Plot Simpson's paradox
ggarrange(gg1,gg2)

## outcome
# outcome binario
# dati centrati e scalati per avere i regressori sulla stessa scala 
str(dd)
dd$S <- as.numeric(dd$S)-1
ds <- scale(dd)
str(ds)
ds <- as.data.frame(ds)

z = 0.7*(ds$X_score)+1.9*ds$interview+0.7*ds$score+0.9*ds$simpson_score1+0.7*ds$simpson_score2-0.3*ds$simpson_score1*ds$simpson_score2 - 0.4*ds$age + 1.3*ds$Github_account + 1.7*(ds$proxy2^2) - 0.9*ds$proxy
pr = 1/(1+exp(-z))      # pass through an inv-logit function
Y = rbinom(n,1,pr)      # bernoulli response variable

#now feed it to glm:
df = data.frame(dd,Y)
# controlla i segni e che non ci siano troppi beta=0
glm(Y ~.,data=df,family="binomial")     

summary(df$Y)
table(df$Y)
table(df$Y[which(df$S==1)])
table(df$Y[which(df$S==0)])

# assign tratment 
# balanced, not related with the outcome
dt <- as.data.table(df)
dt <- trtAssign(dt, nTrt = 3, balanced = T,grpName = "Treatment")
# back to data.frame
df <- as.data.frame(dt)

# correlation
str(df)
# df$S <- as.numeric(df$S)-1
correlation <- cor(df)
corrplot::corrplot(correlation,method = "square",type="lower")

# round some features
df$age <- round(df$age)
df$X_score <- round(df$X_score)
str(df)

## STATISTICAL PARITY DIFFERENCE

Stat_parity_test <- function(data,protected_group,Sa_label,Sd_label,Y,fav_label){
  
  # data = dataset anche in formato "data.table" (from simstudy)
  # protected_group = "feature_protetta
  # Sa_label,Sd_label = Gruppo avvantagiato e scantaggiato della feature protetta
  # Y = outcome binario
  # fav_label = outcome preferibile/positivo
  
  data <- as.data.frame(data) # data frame
  # Privileged class
  Sa <- data[data[,protected_group]==Sa_label,]
  # Privileged class with positive outcome
  Fav_Sa <- Sa[Sa[,Y]==fav_label,]
  Fav_Sa_count <- nrow(Fav_Sa)
  
  # Disadvantaged group 
  Sd <- data[data[,protected_group]==Sd_label,]
  # Disadvantaged group with positive outcome
  Fav_Sd <- Sd[Sd[,Y]==fav_label,]
  Fav_Sd_count <- nrow(Fav_Sd)
  
  Advantageous <- nrow(Sa)
  dis_advantageous <- nrow(Sd)
  
  statistical_parity_difference = (Fav_Sd_count/dis_advantageous)-(Fav_Sa_count/Advantageous)
  Disparate_impact = (Fav_Sd_count/dis_advantageous)/(Fav_Sa_count/Advantageous)
  
  return(list("SPD"=statistical_parity_difference,"DI"=Disparate_impact))
  
}

Stat_parity_test(data=df,protected_group = "S",Sa_label=0,Sd_label=1,Y="Y",fav_label=1)

# now VIF and run classifier
#fit the regression model
model <- lm(Y ~ ., data = df)

#view the output of the regression model
summary(model)

# look at: R-squared, F-statistic, p-value

# VIF for each predictor variable in the model:

#load the car library
library(car)

#calculate the VIF for each predictor variable in the model
vif(model)

## Visualizing VIF values
#create vector of VIF values
vif_values <- vif(model)

#create horizontal bar chart to display each VIF value
barplot(vif_values, main = "VIF Values", horiz = TRUE, col = "steelblue",xlim = c(0,6),arg=names,las=1)

#add vertical line at 5
abline(v = 5, lwd = 1, lty = 2)

# look also at the correlations, gives similar informations to VIF
# df <- df %>% dplyr::select(-Y) # prova a togliere S e i vari G
correlation <- cor(df)
corrplot::corrplot(correlation,method = "square",type="lower")

# exploratory data analysis 
library("tableone")

table1 <- CreateTableOne(vars = colnames(df)[1:10],
                         data = df,
                         strata = "Y")
print(table1)

ggplot(df, aes(age)) +
  geom_histogram() +
  ggtitle("Histogram of age")


library("pheatmap")
pheatmap((df[,1:11] == "1") + 0)

# write csv
getwd()
write.csv(df, file = "BIASED_data_29_11_22.csv")
Data <- read.csv("BIASED_data_29_11_22.csv")
summary(Data)
