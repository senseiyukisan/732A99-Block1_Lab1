setwd("C:/Users/Mounish/Desktop/ML/lab1")
dataset = read.csv("C:/Users/Mounish/Desktop/ML/Lab1/optdigits.csv")
dim(dataset)

set.seed(12345)
sample_train<- sample(seq_len(nrow(dataset)), size = floor(0.50*nrow(dataset)))
sample_valid<- sample(seq_len(nrow(dataset)), size = floor(0.25*nrow(dataset)))
sample_test <- sample(seq_len(nrow(dataset)), size = floor(0.25*nrow(dataset)))

train     <- dataset[sample_train, ]
validation<- dataset[sample_valid, ]
test      <- dataset[sample_test, ]

dim(train)
dim(test)
dim(validation)


#Model kknn

library(kknn)

knn.fit <- train.kknn(X0.26~., train, kmax = 30,
                  kernel = "rectangular")

pred.knn <- predict(knn.fit, train)
pred.knn_v <- predict(knn.fit, validation)
pred.knn_t <- predict(knn.fit, test)

plot(pred.knn)
table(pred.knn)

#Confusion Matrix

table(train$X0.26,as.factor(pred.knn)) # for train data

table(test$X0.26,as.factor(pred.knn_t)) # for test data

#Miss classification Rate

missclassrate=function(y,y_i)
  {
  n=length(y)
  v<-1-(sum(diag(table(y,y_i)))/n)
  return(v)
}


missclassrate(train$X0.26,as.factor(pred.knn)) # for training data


missclassrate(test$X0.26,as.factor(pred.knn_t)) # for test data

?diag


summary(pred.knn)

table(train$X0.26)


acc=function(x,y)
{
  n=length(x)
  ac=sum(diag(table(x,y)))/n
  return(ac)
}
acc(train$X0.26,as.factor(pred.knn))
 



####
#accuracy_check=function(x)
#{                    

k=1
k.optm=c() 
for (i in 1:30)
{ 
  knn.fit <- train.kknn(X0.26~., train, kmax = i,kernel = "rectangular")
  
  ypred = predict(knn.fit,newdata = train)
  
  k.optm[i] = 1-(sum(diag(as.matrix(table(Actual = train$X0.26, Predicted = ypred))))/nrow(train))
  
  
}
k.optm

#}

accuracy_check(1:30)
plot(n)
