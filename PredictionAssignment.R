install.packages('rattle')
install.packages('rpart.plot')
install.packages('RColorBrewer')

# load require libraries
library(caret)
library(doParallel)
library(rattle)
# Download files
set.seed(0525)
download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv', 'pml-training.csv')
download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv','pml-test.csv' )

#Cleaning date
train   <- read.csv('pml-training.csv', na.strings=c("NA","#DIV/0!", ""))
test       <- read.csv('pml-test.csv' , na.strings=c("NA", "#DIV/0!", ""))

#Classe response variable had 5 distinct values and the distribution of the values are as following in percentage of the total values 
table(train$classe)/nrow(train) * 100

# remove columns having more than 60% N/A's value
goodVars    <- which((colSums(!is.na(train)) >= 0.6*nrow(train)))
train <- train[,goodVars]
test     <- test[,goodVars]
# remove Classe variable
test <- test[-ncol(test)]
# fix factor levels
test$new_window <- factor(test$new_window, levels=c("no","yes"))

# Remoce cpolumn 1, and 5
train <- train[,-c(1,5)]
test     <- test[,-c(1,5)]

# Data Splitting- Create data partition
partTrain  <- createDataPartition(train$classe, p = 0.6, list = FALSE)
train1<-train[partTrain,]
test1<-train[-partTrain,]


#Method 1 Base Model

set.seed(2)
bmOutput <- factor(rep("A", nrow(test1)), levels=c("A","B","C","D","E"))
bmTrain <- factor(rep("A", nrow(train1)), levels=c("A","B","C","D","E"))
bmTrainConfusion <- confusionMatrix(train1$classe, bmTrain)
bmTrainConfusion$overall[1]
#bmTrainConfusion
bmConfusion <- confusionMatrix(test1$classe, bmOutput)
bmConfusion$overall[1]

#method 2 Recursive Partitioning and Regression Trees Model
set.seed(4)
rPartModel <- train(classe ~ ., data=train1, method="rpart", trControl=trainControl(method="cv"))
fancyRpartPlot(rPartModel$finalModel)
rPartTrainPred <- predict(rPartModel, newdata=train1)
rPartTestPred <- predict(rPartModel, newdata=test1)
trainingConfusion <- confusionMatrix(train1$classe, rPartTrainPred)
trainingConfusion$overall[1]
testingConfusion <- confusionMatrix(test1$classe, rPartTestPred)
testingConfusion$overall[1]

#print(testingConfusion$table)

#method 3 random forest model

set.seed(40)
rfModel <- train(classe ~ ., data=train1, method="rf", trControl=trainControl(method="cv"))

plot(rfModel$finalModel, main="")
##from the plot above it is observed that random forest used accuracy to select the optimal solution of 28 trees.

rfTrainPred <- predict(rfModel, newdata=train1)
rfTestPred <- predict(rfModel, newdata=test1)
rfTrainingConfusion <- confusionMatrix(train1$classe, rfTrainPred)
rfTrainingConfusion$overall[1] #Accuracy : 1
rfTestingConfusion <- confusionMatrix(test1$classe, rfTestPred)
rfTrainingConfusion$overall[1] #Accuracy : 0.97


#method 4 Generalized Boosted Regression Modeling
set.seed(50)
gbmModel <- train(classe ~ ., data=train1, method="gbm", trControl=trainControl(method="cv"))

plot(gbmModel)

gbmTrainPred <- predict(gbmModel, newdata=train1)
gbmTestPred <- predict(gbmModel, newdata=test1)
gbmTrainingConfusion <- confusionMatrix(train1$classe, gbmTrainPred)
gbmTrainingConfusion$overall[1] #Accuracy
gbmTestingConfusion <- confusionMatrix(test1$classe, gbmTestPred)
gbmTestingConfusion$overall[1] #Accuracy
print(gbmTestingConfusion$table)

# Method 5 : Fitting Parallel random forest 
class <- train1$classe
data  <- train1[-ncol(train1)]
#Register multi-core backend
registerDoParallel()
rf <- train(data, class, method="parRF", tuneGrid=data.frame(mtry=3), trControl=trainControl(method="none"))
rf
plot(varImp(rf))

trainPredictions <- predict(rf, newdata=train1)
confMatrix1 <- confusionMatrix(trainPredictions,train1$classe)
confMatrix1$overall[1]
testPredictions <- predict(rf, newdata=test1)
confMatrix <- confusionMatrix(testPredictions,test1$classe)
confMatrix$overall[1]

# Testing of test data set with 20 records with parallel random forest algorithm.
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

answers <- predict(rf, test)
pml_write_files(answers)













