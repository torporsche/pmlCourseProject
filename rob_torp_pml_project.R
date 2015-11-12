getwd()
setwd("F:/Storage/Dropbox/Coursera/Practical Machine Learning/Data")
fitData <- read.csv("pml-training.csv")
fitTest <- read.csv("pml-testing.csv")



trCtrl <- trainControl(
  method = "repeatedcv"
  , number = 2
  , repeats = 5
  , allowParallel = TRUE
)


cl <- makePSOCKcluster(4)
clusterEvalQ(cl, library(foreach))
registerDoParallel(cl)

library(caret)
library(randomForest)
library(doParallel)
load("F:/Storage/Dropbox/Coursera/Practical Machine Learning/Data/pml_course_project.RData")

str(fitData$classe)
fitTest

utils::View(fitData)

fitData$classe <- as.factor(fitData$classe)

set.seed(33833)
fitTrain <- createDataPartition(y=fitData$classe,
                                p=0.7, list=FALSE)

training <- fitData[fitTrain,]
testing <- fitData[-fitTrain,]


modFDTrain <- train(classe ~ raw_timestamp_part_1 + raw_timestamp_part_2 + num_window + roll_belt + pitch_belt + yaw_belt + 
                      total_accel_belt + gyros_belt_x + gyros_belt_y + gyros_belt_z + accel_belt_x + accel_belt_y + accel_belt_z + 
                      magnet_belt_x + magnet_belt_y + magnet_belt_z + roll_arm + pitch_arm + yaw_arm + total_accel_arm + gyros_arm_x + 
                      gyros_arm_y + gyros_arm_z + accel_arm_x + accel_arm_y + accel_arm_z + magnet_arm_x + magnet_arm_y + magnet_arm_z + 
                      roll_dumbbell + pitch_dumbbell + yaw_dumbbell + total_accel_dumbbell + gyros_dumbbell_x + gyros_dumbbell_y + 
                      gyros_dumbbell_z + accel_dumbbell_x + accel_dumbbell_y + accel_dumbbell_z + magnet_dumbbell_x + 
                      magnet_dumbbell_y + magnet_dumbbell_z + roll_forearm + pitch_forearm + yaw_forearm + total_accel_forearm + 
                      gyros_forearm_x + gyros_forearm_y + gyros_forearm_z + accel_forearm_x + accel_forearm_y + accel_forearm_z + 
                      magnet_forearm_x + magnet_forearm_y + magnet_forearm_z
, data =training, method = "parRF", proximity=TRUE,trControl = trCtrl)
trainImp <- varImp(modFDTrain, scale = FALSE)
trainImp
summary(modFDTrain$finalModel)

# Creates a table that compares predicted classe vs actual classe
pred <- predict(modFDTrain,newdata=testing); testing$predRight <- pred == testing$classe
table(pred,testing$classe)

confusionMatrix(pred,testing$classe)


predTest <- predict(modFDTrain,newdata=fitTest)
predTest


answers = c("B", "A", "B", "A", "A", "E", "D", "B" ,"A", "A" ,"B", "C" ,"B" ,"A", "E" ,"E", "A", "B", "B" ,"B")


pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)


setwd("F:/Storage/Dropbox/Coursera/Practical Machine Learning/Data/Predictions")

save.image("pml_course_project.RData")
