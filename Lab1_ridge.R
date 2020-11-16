
### QUESTION 2


data <- parkinson[,-c(1:4,6)]


## Scaling the data set 60/40

set.seed(12)

ind <- floor(nrow(data)*0.6)

train_ind <- sample(1:(nrow(data)), size = ind)

train <- data[train_ind,]
test <- data[-train_ind,]


## Implementation of 4 functions

Loglikelihood <- function(i_w, input_data){
  
  beta <- as.matrix(i_w[1:16])
  sigma <- i_w[17]
  
  y <- as.matrix(input_data[,1])
  X <- as.matrix(input_data[,-1])
  n <- nrow(input_data) 
  
  RSS <- sum((y - X %*% beta)^2)
  
  l_Lhood <- -(n/2) * log(2*pi*sigma^2) - (1/(2*sigma^2)) * RSS 
  
  return(-l_Lhood)
  
}

Loglikelihood(i_w = rep(2,17), input_data = test)



Ridge <- function(i_w, input_data, lambda){
  
  beta <- as.matrix(i_w[1:16])
  sigma <- i_w[17]
  
  y <- as.matrix(input_data[1])
  X <- as.matrix(input_data[-1])
  n <- nrow(input_data) 
  
  
  RSS <- sum((y - X %*% beta)^2)
  penalty <- lambda  * sum(beta^2)
  
  l_Lhood <- -(n/2) * log(2*pi*sigma^2) - (1/(2*sigma^2)) * RSS + penalty 
  
  return(-l_Lhood)
  
}
Ridge(i_w = rep(1,17), input_data = test, lambda=10)


df <- function(lambda, input_data){
  
  X <- as.matrix(input_data[-1])
  d <- sum(diag( X %*% solve(t(X) %*% X + lambda * diag(ncol(X))) %*% t(X) ))
  return(d)
}
df(lambda = 5, input_data = test)


RidgeOpt <- function(lambda){
  
  opt_estimate <- optim(par=c(i_w = 0:17), fn = Ridge, lambda = lambda, input_data = test, method = 'BFGS')
  
  beta_coef <- opt_estimate$par[1:16]       #the 17th element is sigma
  
  return(beta_coef)
   
}

RidgeOpt(1)

l_1 <- RidgeOpt(1)

l_100 <- RidgeOpt(100)

l_1000 <- RidgeOpt(1000)


predicted_y <- function(coef_est, input_data){
  beta <- as.matrix(coef_est)
  X <- as.matrix(input_data[,-1])  
  y_hat <- X %*% beta
  
  return(y_hat)
} 

p1 <- predicted_y(coef_est = l_1, input_data = test)
p100 <- predicted_y(coef_est = l_100, input_data = test)
p1000 <- predicted_y(coef_est = l_1000, input_data = test)

MSE <- function(input_data, y_hat){
  n <- nrow(input_data)
  y <- input_data[,1]
  SSE <- sum( (y - y_hat)^2)
  
  M_S_E <- (1/n) * SSE
  return(M_S_E)
}

MSE(input_data = test, y_hat = p1000)


AIC <- function(input_data, lambda, i_w){
  
  beta <- as.matrix(i_w[1:16])
  sigma <- i_w[17]
  y <- as.matrix(input_data[1])
  n <- nrow(input_data) 
  X <- as.matrix(input_data[-1])
  
  RSS <- sum((y - X %*% beta)^2)
  penalty <- lambda * sum(beta^2)
  
  l_Lhood <- -(n/2) * log(2*pi*sigma^2) - (1/(2*sigma^2)) * RSS + penalty 
  df <- sum(diag( X %*% solve(t(X) %*% X + lambda * diag(ncol(X))) %*% t(X) ))
  
  A_I_C <- 2*df - 2*l_Lhood
  
  return(A_I_C)
}

AIC_1 <- AIC(input_data = train, lambda = 1, i_w = l_1)
AIC_100 <- AIC(input_data = train, lambda = 1, i_w = l_100)
AIC_1000 <- AIC(input_data = train, lambda = 1, i_w = l_1000)


