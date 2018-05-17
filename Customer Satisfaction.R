library(ggplot2)
library(caret)
library(Matrix)
library(xgboost)
library(gmodels)

traindf <- read.csv("train.csv")

#371 Varibales

##### Data Understanding ######
str(traindf)

summary(traindf)

#Check if there is variables that are numeric
sapply(traindf,function(x)any(is.numeric(x)))

#ALL variables are numeric

Double_Vars <- sapply(traindf,function(x)any(is.double(x)))
Double_Vars <- names(traindf[Double_Vars])

####Data Cleaning#####

#Check if there is any NA
sapply(traindf,function(x)any(is.na(x)))

#There is no NA

##### Removing constant features
cat("\n## Removing constant features.\n")
for (f in names(traindf)) {
  if (length(unique(traindf[[f]])) == 1) {
    cat("-", f, "\n")
    traindf[[f]] <- NULL
  }
}

#337 Variables don't have unique values

#Change Double Variables to factors
for (f in Double_Vars) {
  traindf[[f]] <- as.factor(traindf[[f]]) 
}

str(traindf)


#Find High Correlated Variables with threshold 90%
cor.mat <- cor(traindf)

HighCorrCols <- findCorrelation(cor.mat,cutoff = 0.9, verbose = TRUE,names = TRUE)
traindf <- traindf[,-which(names(traindf) %in% HighCorrCols)]

##Now train dataframe has 175 variables

#Now Let's check each of the 175 variables and see their relations to the target

####Var 15####

summary(traindf$var15)
ggplot(data=traindf,aes(x=traindf$var15)) + geom_bar(alpha=0.75,fill="tomato",color="black") + xlim(c(5,105)) + ggtitle("Age Distribution") + 
  theme_bw() +theme(axis.title=element_text(size=24),plot.title=element_text(size=36),axis.text =element_text(size=16))

#Var 15 appears to be the Age

table(traindf$var15)

var15count <- table(traindf$TARGET, traindf$var15) 

barplot(var15count, ylab="#", main="var15count", legend.text=TRUE)

####Var 3####

summary(traindf$var3)
ggplot(data=traindf,aes(x=traindf$var3)) + geom_bar(alpha=0.75,fill="tomato",color="black") + xlim(c(-999999,238)) + ggtitle("Age Distribution") + 
  theme_bw() +theme(axis.title=element_text(size=24),plot.title=element_text(size=36),axis.text =element_text(size=16))

sum(traindf$var3!=2)

table(traindf$var3)

var3count <- table(traindf$TARGET, traindf$var3) 

barplot(var3count, ylab="#", main="var15count", legend.text=TRUE)


###imp_ent_var16_ult1####

ent16count <- table(traindf$TARGET, traindf$imp_ent_var16_ult1) 

barplot(ent16count, ylab="#", main="imp_ent_var16_ult1 count", legend.text=TRUE)


###imp_op_var40_comer_ult1####
op_40_comer_ult1count <- table(traindf$TARGET, traindf$imp_op_var40_comer_ult1) 

barplot(op_40_comer_ult1count, ylab="#", main="imp_ent_var16_ult1 count", legend.text=TRUE)

#### Data Visualization ####
par(mfrow=c(3,3))

for (f in names(traindf)) {
  barplot(table(traindf$TARGET,traindf[[f]]), ylab="Count", main=paste(f,"count",sep = "_"), legend.text=TRUE)
}




#### Model Building ####
###Logistic Regression####
train <- sparse.model.matrix(TARGET ~ ., data = traindf)

dtrain <- xgb.DMatrix(data=train, label=traindf$TARGET)

param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.02,
                max_depth           = 5,
                subsample           = 0.7,
                colsample_bytree    = 0.7
)

LogitModel <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 557, 
                    verbose             = 1,
                    #                    watchlist           = watchlist,
                    maximize            = FALSE
)


CV <- xgb.cv(params              = param,
              nfold               = 5,
              data                = dtrain, 
              nrounds             = 557, 
              verbose             = 1,
              maximize            = FALSE )


print(CV,verbose = TRUE)


### KNN ####
library(class)
cat("Training using KNN with cross-validation\n")

TrainDataLabels <- traindf[1:38010,175]
TestDataLabels <- traindf[38010:76020,175]

Train.Set <- traindf[1:38010,-175]
Test.Set <- traindf[38010:76020,-175]

KNN_CrossV <- knn.cv(train = Train.Set,cl = TrainDataLabels,k=5)

KNN_Test_Pred <- knn(train = Train.Set,test = TestData,cl = TrainDataLabels, k=2)

CrossTable(x=TrainDataLabels,y=KNN_CrossV,prop.chisq = FALSE)


