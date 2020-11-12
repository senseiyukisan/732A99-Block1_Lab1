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

# Done in RMarkdown

# Fitting the linear regression model
predictors <- paste("Channel", 1:100, sep="")
fmla <- as.formula(paste("Fat ~ ", paste(predictors, collapse= "+")))

fit = lm(fmla, data=train)
summary(fit)

# Estimating training and test errors
train_predictions_lm = predict(fit, train)
train_mse = mean((train$Fat-train_predictions_lm)^2)
train_rmse = sqrt(mean((train$Fat-train_predictions_lm)^2))

test_predictions_lm = predict(fit, test)
test_mse = mean((test$Fat-test_predictions_lm)^2)
test_rmse = sqrt(mean((test$Fat-test_predictions_lm)^2))

cat("Train error\nMSE: ", train_mse, "\nRMSE: ", train_rmse, "\n\nTest error\nMSE: ", test_mse, "\nRMSE: ", test_rmse)

# Comment on the quality of fit and prediction and therefore on the quality of model.

# Our loss-function (MSE) is showing quite a difference between the prediction of train and test set values.
# The MSE for the train set is ~0.56 while the MSE for the test set returns a value of ~1.01. 
# Therefore, our model seems to be overfitting on the train data.

# 2)
# Assume now that Fat can be modeled as a LASSO regression in which all Channels are used as features.
# Report the objective function that should be optimized in this scenario.
# Done in RMarkdown

# 3)

# import glmnet library for lasso and ridge regression
library(glmnet)
library(ggplot2)

# Fit the LASSO regression model to the training data. 
# [,-1] to remove the 'Sample' column
covariates = scale(train[,2:101])
response = scale(train$Fat)

# alpha=1 to choose 'Lasso'
model_lasso = glmnet(as.matrix(covariates), response, alpha=1, family="gaussian")

# Present a plot illustrating how the regression coefficients depend on the log of penalty factor (log 𝜆) and interpret this plot.
plot(model_lasso, xvar="lambda", label=T)
# This plot shows the relationship between log(lambda) and value of coefficients in the lasso model. We can see that a bigger 
# log(lambda) evaluates to less non-zero coefficients. At log(lambda) = -8 most of the variables are non-zero. At this point
# we get the same coefficient values as for OLS. For each coefficient we can also see the influence (positive or negative)
# on our prediction depending on different log(lambda) values.


# What value of the penalty factor can be chosen if we want to select a model with only three features?
# To solve this question we check the plot again and look for the lambda value that only has 3 nonzero coefficients. 
# We see that the coefficient with the label '40' goes to 0 at around Log Lambda = -2.8. We only have 3 non-zero coefficients left at that point.

# 4) 
# Present a plot of how degrees of freedom depend on the penalty parameter. Is the observed trend expected?
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

# alpha=0 to choose 'Ridge'
model_ridge = glmnet(as.matrix(covariates), response, alpha=0, family="gaussian")
plot(model_ridge, xvar="lambda", label=T)

# Both Ridge and LASSO regression are shrinkage methods that try to reduce the complexity
# of a model by shrinking or eliminating (LASSO only) its coefficients. For the Ridge regression
# the number of coefficients stays the same regardless of how big log(lambda) gets.
# For LASSO regression the number of coefficients decreases with increasing value of log(lambda).
# Looking at the LASSO plot it is easier to determine which coefficients are the most
# significant compared to looking at the Ridge plot.

# 6)

# Use cross-validation to compute the optimal LASSO model. 
cv_lasso = cv.glmnet(as.matrix(covariates), response, alpha=1, family="gaussian", lambda=seq(0,1,0.001))
# Present a plot showing the dependence of the CV score on log𝜆and comment how the CV score changes with log𝜆. 
plot(cv_lasso)
# We can see a correlation between increasing log lambda and increasing MSE. Up until a log lambda value of around -4 we observe a 
# flat rise of MSE. Afterwards the MSE grows very steeply until it reaches a plateau at around -2.5. From that point until 
# we observe a slow rise again until we reach log(lambda) = -1. From this point onward the MSE stays at its maximum since 
# there are no non-zero coefficients left. 

# Report the optimal 𝜆and how many variables were chosen in this model. 
optimal_lambda = cv_lasso$lambda.min
cat("The optimal 𝜆: ", optimal_lambda, "\nNumber of variables choosen: ", length(coef(cv_lasso, s="lambda.min")))

# TODO Comment whether the selected 𝜆 value is statistically significantly better than log𝜆= −2. 

# We are comparing the cv-score for our optimal lambda given by lambda.min with the cv-score for log(lambda) = -2.

# transform lambda
lambda_to_compare = exp(-2)

# We check the cvm for lambda_to_compare and get a value of 0.81816120. Compared to our optimal lambda=0 with cv-score
# of 0.06060929 we calculate 

cv_lasso$cvm
cv_lasso$lambda
print(cv_lasso)

# TODO Finally, create a scatter plot of the original test versus predicted test values for the model corresponding to optimal lambda 
# and comment whether the model predictions are good.
test_x = as.matrix(model.matrix(fmla, test)[,-1])
test_y = scale(test$Fat)

lasso_model <- glmnet(as.matrix(covariates), response, alpha = 1, lambda=optimal_lambda)
test_predictions_lasso <- predict(lasso_model, s=optimal_lambda, newx=test_x, type="response")

test_mse_lasso = mean((test_y-test_predictions_lasso)^2)

# 7)
# Use the feature values from test data (the portion of test data with Channel columns) and the optimal LASSO model from step 6 to generate new target values. 
# (Hint: use rnorm() and compute 𝜎 as standard deviation of residuals from train data predictions). 
# Make a scatter plot of original Fat in test data versus newly generated ones. Comment on the quality of the data generation.



