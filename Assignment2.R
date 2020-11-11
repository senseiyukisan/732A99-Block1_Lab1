# Load data 
data = read.csv("data/tecator.csv")

# Split data into train and test set
n = dim(data)[1]
set.seed(12345)
id = sample(1:n, floor(n*0.5))
train = data[id,]
test = data[-id,]

# 1)
# Assume that Fat can be modeled as a linear regression in which absorbance characteristics (Channels) are used as features. 
# Report the underlying probabilistic model, fit the linear regression to the training data and estimate the training and test errors. 

# TODO Probabilistic Model in RMarkdown y^ = ß_0 + ß_1*x + ß_2*x + ... + ß_100*x + €

# Fitting the linear regression model
predictors <- paste("Channel", 1:100, sep="")
fmla <- as.formula(paste("Fat ~ ", paste(predictors, collapse= "+")))

fit = lm(fmla, data=data)
summary(fit)

# Estimating training and test errors
yhat = predict(fit, train)
train_mse = mean((train$Fat-yhat)^2)
train_mae = mean(abs(train$Fat-yhat))
train_rmse = sqrt(mean((train$Fat-yhat)^2))

y0hat = predict(fit, test)
test_mse = mean((test$Fat-y0hat)^2)
test_mae = mean(abs(test$Fat-y0hat))
test_rmse = sqrt(mean((test$Fat-y0hat)^2))

cat("Train error\nMSE: ", train_mse, "\nMAE: ", train_mae, "\nRMSE: ", train_rmse, "\n\nTest error\nMSE: ", test_mse, "\nMAE: ", test_mae, "\nRMSE: ", test_rmse)

fitted <- predict(fit, interval = "confidence")

# Comment on the quality of fit and prediction and therefore on the quality of model.

# Our loss-function (MSE) is showing quite a difference between the prediction of train and test set values.
# The MSE for the train set is ~0.56 while the MSE for the test set returns a value of ~1.01. 
# Therefore, our model seems to be overfitting on the train data.

# 2)
# TODO Assume now that Fat can be modeled as a LASSO regression in which all Channels are used as features.
# Report the objective function that should be optimized in this scenario.
# https://en.wikipedia.org/wiki/Lasso_(statistics) y_i = Fat_i, x_i = Channel_i

# 3)

# import glmnet library for lasso and ridge regression
library(glmnet)
library(ggplot2)

# Fit the LASSO regression model to the training data. 
# [,-1] to remove the 'Sample' column
x = as.matrix(scale(model.matrix(fmla, train)[,-1]))
y = scale(train$Fat)

# alpha=1 to choose 'Lasso'
model_lasso = glmnet(x, y, alpha=1, family="gaussian")

# TODO Present a plot illustrating how the regression coefficients depend on the log of penalty factor (log 𝜆) and interpret this plot.
plot(model_lasso, xvar="lambda", label=T)

# TODO What value of the penalty factor can be chosen if we want to select a model with only three features?
# To solve this question we check the plot again and look for the lambda value that only has 3 nonzero coefficients. 
# The first column 'Df' shows the number of non-zero coefficients. Therefore, we look for Df=3 and detect a penalty factor of log(0.06773).

# 4) 
# TODO Present a plot of how degrees of freedom depend on the penalty parameter. Is the observed trend expected?
lambda = model_lasso$lambda
df = model_lasso$df
ggplot(data=NULL, aes(lambda, df)) + geom_point(color="red")
# The trend is as expected. 
# If lambda is large, the parameters are heavily constrained and the degrees of freedom will effectively be lower, tending to 0
# as lambda -> Inf.
# For lambda -> 0, we have *p* parameters. As in OLS all parameters stay the same, no penelazation to be found. Therefore degrees of freedom =
# There is a 1:1 mapping between lambda and the degrees of freedom, so in practice one may simply pick the effective degrees of freedom 
# that one would like associated with the fit, and solve for lambda
# TODO o.g. in eigene worte umwandeln!

# 5)
# Repeat step 3 but fit Ridge instead of the LASSO regression and compare the plots from steps 3 and 5. Conclusions?

# alpha=0 to choose 'Lasso'
model_ridge = glmnet(x, y, alpha=0, family="gaussian")
plot(model_ridge, xvar="lambda", label=T)

# 6)

# Use cross-validation to compute the optimal LASSO model. 
cv_lasso = cv.glmnet(x, y, alpha=1, family="gaussian", lambda=seq(0,1,0.001))

# TODO Present a plot showing the dependence of the CV score on log𝜆and comment how the CV score changes with log𝜆. 
plot(cv_lasso)
print(cv_lasso)
# 

# Report the optimal 𝜆and how many variables were chosen in this model. 
cat("The optimal 𝜆: ", cv_lasso$lambda.min, "\nNumber of variables choosen: ", length(coef(cv_lasso, s="lambda.min")))

# TODO Comment whether the selected 𝜆 value is statistically significantly better than log𝜆= −2. 

# TODO Finally, create a scatter plot of the original test versus predicted test values for the model corresponding to optimal lambda 
# and comment whether the model predictions are good.


# 7)
# Use the feature values from test data (the portion of test data with Channel columns) and the optimal LASSO model from step 6 to generate new target values. 
# (Hint: use rnorm() and compute 𝜎 as standard deviation of residuals from train data predictions). 
# Make a scatter plot of original Fat in test data versus newly generated ones. Comment on the quality of the data generation.



