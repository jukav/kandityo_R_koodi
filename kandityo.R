# Kandityo: Ensemble learning -meethods
# Juha Kavka

library(tree)
library(ISLR)


### load data
summary(Carseats)
attach(Carseats)

High=ifelse(Sales <=8,"No","Yes")
Carseats=data.frame(Carseats,High)
#Carseats=data.frame(Carseats,H)

## tree model
tree.carseats=tree(High~.-Sales-High,Carseats)
tree(formula = High~.-Sales-High,data=Carseats)
summary(tree.carseats)

# test error for misclassification
set.seed (3)
train=sample (1: nrow(Carseats), 200)
Carseats.test=Carseats [-train,]
High.test=High[-train]
tree.carseats =tree(High~.-Sales-High,Carseats,subset =train )
tree.pred=predict (tree.carseats,Carseats.test ,type ="class")
table(tree.pred,High.test)
(23+44)/200 # misclassification rate



# cross validation for omptimal tree
set.seed (123)
cv.carseats=cv.tree(tree.carseats,FUN=prune.misclass)
names(cv.carseats)
cv.carseats

prune.carseats =prune.misclass (tree.carseats,best =4)
plot(prune.carseats)
text(prune.carseats ,pretty =0)
tree.pred=predict (prune.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
m1<-(24+39)/200 # misclassification rate 0.315

## random forest
library (randomForest)
set.seed (1)
bag.Carseats=randomForest(High~.-Sales-High,Carseats,subset=train,mtry=10,ntree=1000,importance =TRUE)
bag.Carseats

yhat.bag = predict(bag.Carseats,Carseats.test,type = "class")
table(yhat.bag,High.test)
m3<-(19+21)/200 # misclassification rate 0.2 (number of variables 3)
m2<-(24+22)/200 # misclassification rate 0.23 (number of varibles 10)

### bosting tree
library (gbm)
set.seed (1)

H <- ifelse((High == "Yes"),1,0)

t.data<-Carseats
H <- ifelse((High == "Yes"),1,0)
t.data <- data.frame(t.data,H)

t.data$Sales<-NULL
t.data$High<-NULL


train.data<-t.data[train,]


set.seed(123)
mcr=0


for (i in seq(500,1000,by=100)){
boost.Carseats=gbm(train.data$H~.,data=train.data, distribution= "bernoulli",n.trees=1000,interaction.depth=5)
summary(boost.Carseats)
yhat.boost=predict(boost.Carseats,newdata =t.data[-train ,],n.trees =1000)
yhat.boost

predict_class <- ifelse(yhat.boost > 0.55,1,0)
t<-table(predict_class,High.test)
mr<-((t[1,2]+t[2,1])/sum(t))
mcr[i]<-mr
t=0

m4<-(23+14)/200 #misclassification rate 0.185
}

sum(t)
t


plot(Carseats)

m=0
m[1]=m1
m[2]=m2
m[3]=m3
m[4]=m4

barplot(m,main = "misclassification rates for tree models",ylab="misclassification rate",col='grey',xlab = "1. pruned tree         2.bagged tree     3. random forest     4. boosted tree ")

#### Super leanrner

# Extract our outcome variable from the dataframe.
outcome = Carseats$H

# Create a dataframe to contain our explanatory variables.
data = subset(Carseats, select = -c(High,H,Sales))


# Check structure of our dataframe.
str(data)

# Set a seed for reproducibility in this random sampling.
set.seed(1)

# Reduce to a dataset of 200 observations to speed up model fitting.
train_obs = sample(nrow(data),200)

# X is our training sample.
X_train = data[train_obs, ]

# Create a holdout set for evaluating model performance.
# Note: cross-validation is even better than a single holdout sample.
X_holdout = data[-train_obs, ]

# Create a binary outcome variable: towns in which median home value is > 22,000.


Y_train = outcome[train_obs]
Y_holdout = outcome[-train_obs]

# Review the outcome variable distribution.
table(Y_train, useNA = "ifany")


library('SuperLearner')
library('arm')


listWrappers()
SL.glmnet

# Set the seed for reproducibility.
set.seed(1)

Y_train_bi<-ifelse(Y_train=="Yes",1,0)

# Fit lasso model.
sl_lasso = SuperLearner(Y = Y_train_bi, X = X_train, family = binomial(),
                        SL.library = "SL.glmnet")

# Fit random forest.
sl_rf = SuperLearner(Y = Y_train_bi, X = X_train, family = binomial(),
                     SL.library = "SL.randomForest")

set.seed(1)
sl = SuperLearner(Y = Y_train_bi, X = X_train, family = binomial(),
                  SL.library = c("SL.randomForest","SL.glm"))

summary(sl)



# Predict back on the holdout dataset.
# onlySL is set to TRUE so we don't fit algorithms that had weight = 0, saving computation.
pred = predict(sl, X_holdout, onlySL = T)

# Check the structure of this prediction object.
str(pred)
yhat<-pred$pred

predict_class <- ifelse(yhat > 0.50,1,0)
table(predict_class,Y_holdout)
(12+10)/200 # prediction error rate

# Histogram of our predicted values.

library(pillar)
library(ggplot2)
library(gam)

qplot(pred$pred[, 1]) + theme_minimal()

set.seed(1)

# Don't have timing info for the CV.SuperLearner unfortunately.
# So we need to time it manually.

# We use V = 3 to save computation time; for a real analysis use V = 10 or 20.
system.time({
  # This will take about 3x as long as the previous SuperLearner.
  cv_sl = CV.SuperLearner(Y = Y_train_bi, X = X_train, family = binomial(), V =10,
                          SL.library = c("SL.randomForest","SL.glm","SL.gam","SL.xgboost","SL.bayesglm"))
lili})
summary(cv_sl)


plot(cv_sl) + theme_bw()

listWrappers()



pr<-cv_sl$SL.predict
prclass<-ifelse(pred$pred>0.45,1,0)
table(prclass,Y_holdout)
(9+10)/200 # prediction error rate for test data

## Bayes mallit

# naive bayes classifier

library(e1071)

model <- naiveBayes(Carseats$High~.-Sales-High,data = Carseats, subset = train)
p <- predict(model,Carseats.test)
table (p,High.test)
pe=(20+20)/200 # prediction error
model

m=0

m[1]=m2
m[2]=m3
m[3]=pe
m[4]=m4


barplot(m,main = "misclassification rate for Naive Bayes",ylab="misclassification rate",col='grey',xlab = "1.bagged tree 2. random forest 3.Naive Bayes  4. boosted tree ")


library(caret)
library(klaR)
x<-Carseats[,c(-12,-1)]
y<-Carseats$High

modell<-train(x,y,'nb',trControl=trainControl(method='cv',number=10))
modell

table (predict(modell$finalModel,x)$class,y)

###

plot(Sales~Price)
plot(Sales~CompPrice)
D<-data.frame(Sales,Price,Age,ShelveLoc)
plot(D)
plot(Carseats)
boxplot(Carseats$Sales,main="Boxplot, Sales",xlab="mean value = 7.49  (X 1000 pcs)")
mean(Sales)
median(Sales)
quantile(Sales)

###


