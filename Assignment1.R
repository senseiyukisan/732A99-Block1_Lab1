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

## Data is splitted into training, tests and validation 

#Model kknn


library(kknn)

knn.fit <- kknn(as.factor(X0.26)~., train=train,test=train ,k = 30,
                  kernel = "rectangular")

knn.fit_v <- kknn(as.factor(X0.26)~., train=validation,test=validation,k = 30,
                  kernel = "rectangular")

knn.fit_t <- kknn(as.factor(X0.26)~., train=test,test=test,k = 30,
                kernel = "rectangular")

pred.knn <- fitted(knn.fit)
pred.knn_v <- fitted(knn.fit_v)
pred.knn_t <- fitted(knn.fit_t)


## Model is worked and trained by using the training data and also there is no miss-classification rate while training



#Accuracy  of the model while using k value=30
 
cm=as.matrix(table(actual=validation$X0.26, Predicted = pred.knn_v))

accuracy=sum(diag(cm))/length(validation$X0.26)

accuracy


#misscalculation
calc_class_err = function(actual, predicted) {
  mean(actual != predicted)
}
calc_class_err(validation$X0.26,pred.knn_v)




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

# Accuracy of the data
acc=function(x,y)
{
  n=length(x)
  ac=sum(diag(table(x,y)))/n
  return(ac)
}
acc(test$X0.26,as.factor(pred.knn_t))
 


# probability


#knn.fit$prob
v=data.frame(knn.fit$prob)
head(v)

estm_pb <- colnames(v)[apply(v, 1, which.max)]

v$y<-train$X0.26
head(v)
v$fit <- knn.fit$fitted

v$estm_pb <- estm_pb

###
y_8 <- v[v$y == 8,]

yhat_8 <- y_8[y_8$fit == 8,]

###
# Best
easy <- as.numeric(row.names(yhat_8[order(-yhat_8[,9]),][1:2,]))
easy

# Worse
tougher <- as.numeric(row.names(yhat_8[order(yhat_8[,9]),][1:3,]))
tougher
  #best
col=heat.colors(12)

heatmap(t(matrix(unlist(train[easy[1],-65]), nrow=8)),Colv = NA, Rowv = NA,col=rev(heat.colors(12)))

heatmap(t(matrix(unlist(train[easy[2],-65]), nrow=8)), Colv = NA, Rowv = NA,col=rev(heat.colors(12)))

#worst

heatmap(t(matrix(unlist(train[tougher[1],-65]), nrow=8)), Colv = NA, Rowv = NA,col=rev(heat.colors(12)))

heatmap(t(matrix(unlist(train[tougher[2],-65]), nrow=8)), Colv = NA, Rowv = NA,col=rev(heat.colors(12)))

heatmap(t(matrix(unlist(train[tougher[3],-65]), nrow=8)), Colv = NA, Rowv = NA,col=rev(heat.colors(12)))

#u=as.vector(train$X0.26)
#train$X0.26
#n<-as.data.frame(cbind(v,u))
#dim(n)
#train$X0.26



#Find the optimal K value for KNN model for this dataset.(k=1 to 30)                     

k=1
k.optm=c() 
y.optm=c()
for (i in 1:30){ 
  knn.fit <-kknn(as.factor(X0.26)~., train=train,test=train, k = i,kernel = "rectangular")
  
  knn.fit_v <-kknn(as.factor(X0.26)~., train=validation,test=validation, k = i,kernel = "rectangular")
  
  ypred = fitted(knn.fit)
  
  vpred = fitted(knn.fit_v)
  
  k.optm[i] = 1-(sum(diag(as.matrix(table(Actual = train$X0.26, Predicted = ypred))))/nrow(train))
  
  y.optm[i] = 1-(sum(diag(as.matrix(table(Actual = validation$X0.26, Predicted = vpred))))/nrow(validation))
  
}
k.optm

y.optm


my.df  <- data.frame(K_Value = c(1:30), Training= c(k.optm), Validation = c(y.optm))

plot3<-ggplot( ) +
  geom_line(aes(x=my.df$K_Value,y=my.df$Validation,colour="green")) +
  geom_line(aes(x=my.df$K_Value,y=my.df$Training,colour="red"))+
  ylab("Missclassification Rate ") +xlab("K_value")+
scale_color_manual(name = "Missclassification Rate", labels = c("Validation ", "Training "), 
                   values =c("green", "red"))


print(plot3)

optm_value_k <- which.min(y.optm - k.optm )

cat("calculated optimum value of K is: ", optm_value_k)



### Cross Entropy
rp <- function(i){
  n <- rep(0,10)
  n[i+1] <- 1
  return(I(n))
}
er<-c()
for (i in 1:30){ 
  knn.fit <-kknn(as.factor(X0.26)~., train=train,test=validation, k = i,kernel = "rectangular")
  
  x<- data.frame(knn.fit$prob)
  max_prob <- colnames(x)[apply(x ,1,which.max)]
  x$target <- validation$X0.26
  x$fit <- knn.fit$fitted
  x$max_prob <- max_prob
  x$binary <- (lapply(as.numeric(x$target)-1, rp))
  
  #cross entropy loss
  for (j in 1:nrow(x)){
    x[j, "cross_entropy"] <- -sum(log(x[j,1:10]+1e-15)* x[[j, "binary"]])
  }
  
  er[i] <- mean(x$cross_entropy)
}

er

df<-data.frame(cross_entropy=er,k_Value=c(1:30))

plot4<-ggplot(df,aes(x=k_Value,y=cross_entropy,col="red" ))  +
  geom_line()+ggtitle("Empirical Risk Of calculating K_value") 
print(plot4)

best_optm_k_value <-min(er)

cat("calculated optimum value of K by using the cross entropy function is: ", best_optm_k_value)
