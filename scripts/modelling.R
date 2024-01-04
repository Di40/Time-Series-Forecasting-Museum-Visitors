# ---------------------------------------------------------------------------- #
#                               Modelling
# ---------------------------------------------------------------------------- #

# Imports
library(ggplot2)
library(Metrics)
library(lubridate)
library(lmtest)
library(forecast)
library(gbm)
library(timetk)
library(caret)
library(xgboost)
library(DIMORA)

rm(list=ls())

# Change working directory
script_path <- rstudioapi::getSourceEditorContext()$path
script_dir <- dirname(script_path)
setwd(script_dir)

# To use English for the dates (instead of Macedonian/Italian)
Sys.setlocale("LC_TIME", "English")

set.seed(123)

# ToDos (At a later stage, not now):
# 1. Error analysis:
# - Analyze errors: absolute difference between prediction and ground truth
# - Analyze worst and best prediction
# - Get average error per month

# ---------------------------------------------------------------------------- #
# Read data

egizio_df <- readRDS("../data/egizio_final.rds")
print(head(egizio_df))
str(egizio_df)

# ---------------------------------------------------------------------------- #
# Dataframe with metrics for evaluating model performance

metrics_df <- data.frame(
  Model = character(),
  R2 = numeric(),
  Adj_R2 = numeric(),
  MSE = numeric(),
  RMSE = numeric(),
  MAE = numeric(),
  MAPE = numeric(),
  AIC = numeric(),
  stringsAsFactors = FALSE
)

features <- c('month', 'year' , 'trends', 'average_temperature', 'raining_days', 'school_holidays', 'arrivals')
target <- 'visitors'

# ---------------------------------------------------------------------------- #
# Train-test split

# ToDo (Grazina): Decide how to perform the train-test split. Due to the huge
# variation during COVID, we might need to avoid using it in the test set.

egizio_train_df <- subset(egizio_df, format(date, "%Y") != "2022")
egizio_test_df <- subset(egizio_df, format(date, "%Y") == "2022")

cat("Egizio train size:", nrow(egizio_train_df), "rows (months).")
cat("Egizio test size:", nrow(egizio_test_df), "rows (months).")

ratio_train <- nrow(egizio_train_df) / nrow(egizio_df)
ratio_test <- 1 - ratio_train
ratio_train <- ratio_train * 100
ratio_test <- ratio_test * 100
cat("Ratio of train set size to test set size:", ratio_train, ":", ratio_test)

# ---------------------------------------------------------------------------- #
# Standardization

# train_visitors_mean <- mean(egizio_train_df$visitors)
# train_visitors_sd <- sd(egizio_train_df$visitors)
# egizio_train_df$visitors <- (egizio_train_df$visitors - train_visitors_mean) / train_visitors_sd
# egizio_test_df$visitors <- (egizio_test_df$visitors - train_visitors_mean) / train_visitors_sd

# train_trends_mean <- mean(egizio_train_df$trends)
# train_trends_sd <- sd(egizio_train_df$trends)
# egizio_train_df$trends <- (egizio_train_df$trends - train_trends_mean) / train_trends_sd
# egizio_test_df$trends <- (egizio_test_df$trends - train_trends_mean) / train_trends_sd

#standardize_numeric_columns <- function(train_df, test_df) {
  # Extract numeric columns (excluding "date")
 # numeric_columns <- sapply(train_df, is.numeric) & names(train_df) != "date"
  
  # Calculate mean and standard deviation for each numeric column in the training set
  #means <- colMeans(train_df[, numeric_columns], na.rm = TRUE)
 # std_devs <- apply(train_df[, numeric_columns], 2, sd, na.rm = TRUE)
  
  # Standardize the columns in both train and test data frames using means and std_devs from the training set
 # for (col in names(train_df)[numeric_columns]) {
    # Impute missing values (if any) with mean in both train and test data frames
  #  mean_value <- means[col]
   # train_df[[col]][is.na(train_df[[col]])] <- mean_value
    #test_df[[col]][is.na(test_df[[col]])] <- mean_value
    
#    # Standardize the column in both train and test data frames using means and std_devs from the training set
  #  train_df[[col]] <- (train_df[[col]] - means[col]) / std_devs[col]
 #   test_df[[col]] <- (test_df[[col]] - means[col]) / std_devs[col]
  #}
  
  # Return the standardized data frames
#  return(list(train_df = train_df, test_df = test_df))
#}

#standardize <- standardize_numeric_columns(egizio_train_df, egizio_test_df)
#egizio_train_df <- standardize$train_df
#egizio_test_df <- standardize$test_df
print(head(egizio_train_df))
print(head(egizio_test_df))

# This dataframe will be used to store the predictions of all of the models, and make plotting easier.
egizio_predictions_df <- data.frame(date = egizio_test_df$date)
egizio_predictions_df$visitors_true <- egizio_test_df$visitors

# ToDo: Decide whether/how to improve the legend.
# ToDo: Also, maybe it's better to split it to two subplots: top - visitors, bottom - trends?
ggplot() +
  geom_line(data = egizio_train_df, aes(x = date, y = visitors, color = "Visitors", linetype = "Train"), linewidth = 1.5) +
  geom_line(data = egizio_test_df, aes(x = date, y = visitors, color = "Visitors", linetype = "Test"), linewidth = 1.5) +
  geom_line(data = egizio_train_df, aes(x = date, y = trends, color = "Trends", linetype = "Train"), linewidth = 1.5) +
  geom_line(data = egizio_test_df, aes(x = date, y = trends, color = "Trends", linetype = "Test"), linewidth = 1.5) +
  labs(title = "Visitors and Trends over time", x = "Date", y = "Values") +
  scale_color_manual(name = "Variable", values = c("Visitors"="red", "Trends"="blue")) +
  scale_linetype_manual(name = "Dataset", values = c("Train"="solid", "Test"="dashed")) +
  geom_vline(xintercept = as.numeric(min(egizio_test_df$date)), linetype = "dotted", color = "black") +
  theme_minimal()

# ---------------------------------------------------------------------------- #
# Metrics

# R-squared
RSQUARE <- function(y_actual, y_predict) {
  cor(y_actual, y_predict)^2
}

# Adjusted R-squared
# n - sample size (train/test)
# p - number of predictor variables
adjusted_R2 <- function(y_actual, y_predict, n, p) {
  R2 <- RSQUARE(y_actual, y_predict)
  adj_R2 <- 1 - ((1 - R2) * (n - 1) / (n - p - 1))
  return(adj_R2)
}

# For MAE, MAPE, MSE, RMSE use Metrics library methods: mae, mape, mse, rmse.

# ---------------------------------------------------------------------------- #
# Baseline model - mean of training

mean_train_visitors <- mean(egizio_train_df$visitors)
egizio_predictions_df$predicted_visitors_mean <- mean_train_visitors

# Calculate metrics

# We don't calculate R2 and Adj_R2 because this model is not capturing any
# variation in the target variable in the test set.

mse <- mse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_mean)
rmse <- rmse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_mean)
mae <- mae(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_mean)
mape <- mape(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_mean)

# For similar reasoning, we don't calculate AIC.

metrics_df <- rbind(metrics_df, list(Model = "Baseline - mean",
                                     R2 = NA, R2_adj = NA,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = NA))
print(metrics_df)


ggplot(egizio_predictions_df, aes(x = date)) +
  geom_line(aes(y = predicted_visitors_mean, color = "Predicted"),
            linetype = "dashed", size = 1) +
  geom_line(aes(y = visitors_true, color = "Actual"), size = 1) +
  labs(title = "Baseline - training mean",
       x = "Date",
       y = "Visitors") +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))

# ---------------------------------------------------------------------------- #
# Baseline model - same month last year

egizio_predictions_df$predicted_visitors_last_year <- NA
for (i in 1:nrow(egizio_test_df)) {
  # Find the corresponding date from the previous year
  last_year_date <- egizio_test_df$date[i] - months(12)
  
  # Check if the corresponding date exists in the training data
  if (last_year_date %in% egizio_train_df$date) {
    # Get the corresponding value from the training data
    corresponding_value <- egizio_train_df$visitors[egizio_train_df$date == last_year_date]
    
    # Assign the value to the predicted column
    egizio_predictions_df$predicted_visitors_last_year[i] <- corresponding_value
  }
}

# Calculate metrics
r_squared <- RSQUARE(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_last_year)
adj_r_squared <- adjusted_R2(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_last_year, nrow(egizio_train_df), 1)
mse <- mse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_last_year)
rmse <- rmse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_last_year)
mae <- mae(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_last_year)
mape <- mape(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_last_year)

metrics_df <- rbind(metrics_df, list(Model = "Baseline - last year",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = NA))
print(metrics_df)

ggplot(egizio_predictions_df, aes(x = date)) +
  geom_line(aes(y = predicted_visitors_last_year, color = "Predicted"),
            linetype = "dashed", size = 1) +
  geom_line(aes(y = visitors_true, color = "Actual"), size = 1) +
  labs(title = "Auto regressive Baseline - same month from previous year",
       x = "Date", y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))

# ---------------------------------------------------------------------------- #
# Model 1 - Multiple LR

# ToDo: Add other dependent variables
str(egizio_train_df)
all_features_regression <- lm(visitors ~ . - date - school_holidays, data = egizio_train_df)
# Month and quarter were perfectly collinear, so we have removed the quarter.
summary(all_features_regression)
# ToDo: The model works better if we include date. Investigate this.

egizio_predictions_df$predicted_multiple_lr <- predict(all_features_regression, newdata = egizio_test_df)

# Calculate metrics
r_squared <- summary(all_features_regression)$r.squared
adj_r_squared <- summary(all_features_regression)$adj.r.squared
aic <- AIC(all_features_regression)
mse <- mse(egizio_test_df$visitors, egizio_predictions_df$predicted_multiple_lr)
rmse <- rmse(egizio_test_df$visitors, egizio_predictions_df$predicted_multiple_lr)
mae <- mae(egizio_test_df$visitors, egizio_predictions_df$predicted_multiple_lr)
mape <- mape(egizio_test_df$visitors, egizio_predictions_df$predicted_multiple_lr)

metrics_df <- rbind(metrics_df, list(Model = "Multiple LR",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic))
print(metrics_df)

ggplot(egizio_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Visitors"), size = 1) +
  geom_line(aes(y = predicted_multiple_lr, color = "Predicted"),
            linetype = "dashed", size = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))

# Test of DW
dwtest(all_features_regression)
# The p-value is extremely small. => There is autocorrelation in the residuals.

# Check the residuals
res_lm <- residuals(all_features_regression)
plot(egizio_train_df$date, res_lm, xlab="date", ylab="Residuals", type= "b",  pch=16, lty=3, cex=0.6)

plot(Acf(res_lm), xlab = "Lag", main = "Autocorrelation of Residuals",
     col = "steelblue", lwd = 2.5, ci.col = "black", cex.lab = 1.2, cex.main = 1.5)
#######################
#Model with trend and seasonality
egizio_train.ts <- ts(egizio_train_df$visitors, frequency = 12)  
egizio_test.ts <- ts(egizio_test_df$visitors, frequency = 12)
ts.plot(egizio_train.ts, type="o")

# Fit a linear model with trend
fitts <- tslm(egizio_train.ts ~ trend+ season)
# Print the summary of the fitted model
summary(fitts)

# Perform the Durbin-Watson test
dwtest(fitts)

fcast <- forecast(fitts, newdata = egizio_test_df, h = nrow(egizio_test_df))

# Plot the forecast on the test data
plot(fcast)

r_squared <- summary(fitts)$r.squared
adj_r_squared <- summary(fitts)$adj.r.squared
aic <- AIC(fitts)
mse <- mean((fcast$mean - egizio_test_df$visitors)^2)
rmse <- sqrt(mse)
mae <- mean(abs(fcast$mean - egizio_test_df$visitors))
mape <- mean(abs((fcast$mean - egizio_test_df$visitors) / egizio_test_df$visitors)) * 100
metrics_df <- rbind(metrics_df, list(
  Model = "Trend and Seasonality",
  R2 = r_squared,
  R2_adj = adj_r_squared,
  MSE = mse,
  RMSE = rmse,
  MAE = mae,
  MAPE = mape,
  AIC = aic
))

# 8. Print and Append to Metrics DataFrame
print(metrics_df)

# --------------------------------------------------------------------- #

#########with the other feature

# Convert to time series
egizio_train <- ts(egizio_train_df$visitors, frequency = 12)
trends_train <- ts(egizio_train_df$trends, frequency = 12)
average_temperature_train <- ts(egizio_train_df$average_temperature, frequency = 12)
raining_days_train <- ts(egizio_train_df$raining_days, frequency = 12)
school_holidays_train <- ts(egizio_train_df$school_holidays, frequency = 12)
arrivals_train <- ts(egizio_train_df$arrivals, frequency = 12)
date_train <- ts(egizio_train_df$date, frequency = 12)  # Assuming 'date' is in a date format
covid_train <- ts(egizio_train_df$Covid_closures, frequency = 12)
renovation_train <- ts(egizio_train_df$renovation, frequency = 12)
# Combine variables into a data frame
train_data <- data.frame(
  visitors = egizio_train,
  trends = trends_train,
  average_temperature = average_temperature_train,
  raining_days = raining_days_train,
  school_holidays = school_holidays_train,
  arrivals = arrivals_train,
  date = date_train,
  covid=covid_train,
  renovation=renovation_train
)

# Fit the model on the training set
fit.vis <- tslm(visitors ~ date + trends + average_temperature + raining_days + school_holidays + arrivals + trend + season +covid+ renovation, data = train_data)

# Print summary
summary(fit.vis)
CV(fit.vis)      #Aikaike 

plot(egizio_train)

lines(fitted(fit.vis), col=2)   #make sens it didn't expect covid this bad and slow to restart

res<- residuals(fit.vis)
plot(res)
acf(res)   
dwtest(fit.vis)

# Cross-validation
cv_results <- CV(fit.vis)
print(cv_results)


# Forecasting on the test set
test_data <- data.frame(
  date = ts(egizio_test_df$date, frequency = 12),
  trends = ts(egizio_test_df$trends, frequency = 12),
  average_temperature = ts(egizio_test_df$average_temperature, frequency = 12),
  raining_days = ts(egizio_test_df$raining_days, frequency = 12),
  school_holidays = ts(egizio_test_df$school_holidays, frequency = 12),
  arrivals = ts(egizio_test_df$arrivals, frequency = 12),
  covid =ts(egizio_test_df$Covid_closures, frequency = 12),
  renovation= ts(egizio_test_df$renovation, frequency = 12)
)

fcast <- forecast(fit.vis, newdata = test_data, h = nrow(egizio_test_df))

plot(fcast)

# Calculate metrics on the test set
r_squared <- summary(fit.vis)$r.squared
adj_r_squared <- summary(fit.vis)$adj.r.squared
aic <- AIC(fit.vis)
mse <- mean((fcast$mean - egizio_test_df$visitors)^2)
rmse <- sqrt(mse)
mae <- mean(abs(fcast$mean - egizio_test_df$visitors))
mape <- mean(abs((fcast$mean - egizio_test_df$visitors) / egizio_test_df$visitors)) * 100

# Create metrics DataFrame
metrics_df <-  rbind(metrics_df, list(
  Model = "Tslm with features",
  R2 = r_squared,
  R2_adj = adj_r_squared,
  MSE = mse,
  RMSE = rmse,
  MAE = mae,
  MAPE = mape,
  AIC = aic
))

# Print metrics
print(metrics_df)


######LASSO########
library(glmnet)
egizio_train_df$date <- as.numeric(egizio_train_df$date)
egizio_test_df$date <- as.numeric(egizio_test_df$date)
y_train <- egizio_train_df[,4]
X_train <- egizio_train_df[,c(-4)]

# Convert data to matrix format
X_train <- as.matrix(X_train)
y_train <- as.matrix(y_train)

# Fit LASSO regression model
lasso_reg_model <- cv.glmnet(x = X_train, y = y_train, alpha = 1)


# Choose the optimal lambda based on cross-validated error
lambda_opt <- lasso_reg_model$lambda.min
lambda_opt

best_model <- glmnet(X_train, y_train, alpha = 1, lambda = lambda_opt)
coef(best_model)
# Predictions on the test set using the optimal lambda
X_test <- as.matrix(egizio_test_df[,-4])
predictions <- predict(lasso_reg_model, newx = X_test, s = lambda_opt)

# Calculate errors
errors <- mean((egizio_test_df[, 4] - predictions)^2)

# Print the optimal lambda and corresponding coefficients
cat("Optimal Lambda:", lambda_opt, "\n")
cat("Coefficients at Optimal Lambda:\n")
print(coef(lasso_reg_model, s = lambda_opt))

# Summary of the LASSO regression model
summary(lasso_reg_model)

mse_lasso <- mse(egizio_test_df[, 4], predictions)
rmse_lasso <- sqrt(mse_lasso)
mae_lasso <- mae(egizio_test_df[, 4], predictions)
mape_lasso <- mape(egizio_test_df[, 4], predictions)

# Print metrics for LASSO
cat("\nMetrics for LASSO:\n")
cat("MSE (LASSO):", mse_lasso, "\n")
cat("RMSE (LASSO):", rmse_lasso, "\n")
cat("MAE (LASSO):", mae_lasso, "\n")
cat("MAPE (LASSO):", mape_lasso, "\n")

metrics_df <-  rbind(metrics_df, list(
  Model = "Lasso",
  R2 =NA,
  R2_adj = NA,
  MSE = mse_lasso,
  RMSE = rmse_lasso,
  MAE = mae_lasso,
  MAPE = mape_lasso,
  AIC = NA
))

# Print metrics
print(metrics_df)
################################
##############GAM###############
################################
library(gam)

###################
##Stepwise GAM
#Start with a linear model (df=1)
g3 <- gam(visitors~., data=egizio_train_df)
summary(g3)
#Show the linear effects 
#par(mfrow=c(3,5))
#plot(g3, se=T) 
AIC(g3)

sc <- gam.scope(egizio_train_df[, -c(4, 11, 10)], arg = c("df=2", "df=3", "df=4"))
g4 <- step.Gam(g3, scope =sc, trace = TRUE)
summary(g4)

AIC(g4)

#par(mfrow=c(3,5))
#plot(g4, se=T)

#if we want to see better some plot
#par(mfrow=c(1,1))
#plot(g4, se=T, ask=T)


#Prediction
p.gam <- predict(g4,newdata=egizio_test_df)     
dev.gam <- sum((p.gam-egizio_test_df$visitors)^2)
dev.gam


g5 <- gam(visitors ~ s(date) + s(year) + s(month) + s(trends) + s(average_temperature) +
            s(raining_days) + s(school_holidays) + s(arrivals) + Covid_closures +
            renovation, 
          data = egizio_train_df)

summary(g5)
AIC(g5)



g5 <- gam(visitors ~s(date) + s(year) + s(month) + s(trends) + s(average_temperature) +
            s(raining_days) + s(school_holidays) + s(arrivals) + Covid_closures +
            renovation, 
          data = egizio_train_df)

summary(g5)
AIC(g5)
#################################################
###################Shocking######################
#################################################

bm_visitors <- BM(egizio_train_df$visitors, display = TRUE)
summary(bm_visitors)

# Predictions and instantaneous curve for BM
pred_bm_visitors <- predict(bm_visitors, newx = 1:216)
pred_inst_bm_visitors <- make.instantaneous(pred_bm_visitors)

# Plotting BM predictions
plot(egizio_train_df$visitors, type = "b", xlab = "Month", ylab = "Monthly Visitors", 
     pch = 16, lty = 3, cex = 0.6, xlim = c(1, 216))
lines(pred_inst_bm_visitors, lwd = 2, col = 2) 
scaled_visitors <- scale(egizio_train_df$visitors)

####try with shocking
gbm1<-GBM(egizio_df$visitors, shock = "exp", nshock = 1, alpha = 0.04,prelimestimates = c(1.878697e+07, 1.642189e-03, 9.073474e-03, 175
                  , -0.1, 0.5))    #this model the shock of 2015
summary(gbm1)
   

gbm2<-GBM(egizio_train_df$visitors, shock = "exp", nshock = 2, alpha = 0.04,prelimestimates = c(1.878697e+07, 1.642189e-03, 9.073474e-03, 135, -0.1, 0.5, 165 , -0.1, -0.5))
summary(gbm2)   #this model both the 2015 and covid shock 

gbm3 <- GBM(egizio_df$visitors, shock = "rett", nshock = 2,
                      prelimestimates = c(1.878697e+07, 1.642189e-03, 9.073474e-03, 135, 145, 0.1, 205, 210, -0.4),oos=10)
summary(gbm3)
gbm4<- GBM(egizio_df$visitors, shock = "exp", nshock = 3,
                      prelimestimates = c(1.878697e+07, 1.642189e-03, 9.073474e-03,  125, -0.1, 0.2, 160, 0.1, -0.4, 200, -0.1, +0-6))
summary(gbm4)
gbm5 <- GBM(egizio_df$visitors, shock = "exp", nshock = 2,
                      prelimestimates = c(1.878697e+07, 1.642189e-03, 9.073474e-03, 135, -0.4, 0.3, 195,-0.1, 0.2))

summary(gbm5)


gbm6 <- GBM(egizio_df$visitors, shock = "mixed", nshock = 2,
            prelimestimates = c(1.878697e+07, 1.642189e-03, 9.073474e-03, 125, -0.1, 0.2, 205, 210, -0.4),oos=10)
summary(gbm6)
pred_GBM11_visitors<- predict(gbm6, newx=c(1:230))
pred_GBM11_visitors.inst<- make.instantaneous(pred_GBM11_visitors)

# Plotting GBMe1 predictions
plot(egizio_df$visitors, type = "b", xlab = "Month", ylab = "Monthly Visitors", 
     pch = 16, lty = 3, cex = 0.6, xlim = c(1, 216))
lines(pred_GBM11_visitors.inst, lwd = 2, col = 2)
#############
GGM_tw<- GGM(egizio_df$visitors, prelimestimates=c(1.878697e+07, 0.001, 0.01,1.642189e-03, 9.073474e-03))
summary(GGM_tw)

pred_GGM_tw<- predict(GGM_tw, newx=c(1:230))
pred_GGM_tw.inst<- make.instantaneous(pred_GGM_tw)

plot(egizio_df$visitors, type= "b",xlab="Quarter", ylab="Monthly Visitors",  pch=16, lty=3, cex=0.6, xlim=c(1,60))
lines(pred_GGM_tw.inst, lwd=2, col=2)

###Analysis of residuals
res_GGMtw<- residuals(GGM_tw)
acf<- acf(residuals(GGM_tw))


fit_GGMtw<- fitted(GGM_tw)
fit_GGMtw_inst<- make.instantaneous(fit_GGMtw)


# --------------------------------------------------------------------- #
# Model - Boosting

# Modify graphical parameters
mai.old <- par()$mai
mai.new <- mai.old # new vector
mai.new[2] <- 2.5 #new space on the left
par(mai=mai.new)

# This can be used visitors ~ .- visitors - date + as.numeric(date)
boost_visitors <- gbm(visitors ~ . - date, data=egizio_train_df, 
                      distribution="gaussian", n.trees=5000, interaction.depth=1)

par(mai=mai.new)
summary(boost_visitors, las=1, cBar=10)
par(mai=mai.old)

yhat_boost <- predict(boost_visitors, newdata=egizio_test_df, n.trees=1:5000)
err <- apply(yhat_boost, 2, function(pred) mean((egizio_test_df$visitors - pred)^2))

# Error comparison (train and test)
plot(boost_visitors$train.error, type = "l", ylim = c(0, max(err)),
     ylab = "Train/Test Error", xlab = "n.trees")
lines(err, type = "l", col = 2)
best <- which.min(err)  # Minimum error in the test set
abline(v = best, lty = 2, col = 4)
min_error <- min(err)
title(main = sprintf("Min Error: %.4f", min_error))

# 2 Boosting - Deeper trees
boost_visitors <- gbm(visitors ~ . - date , data=egizio_train_df,
                      distribution="gaussian", n.trees=5000,
                      interaction.depth=4) # (with more than one variable)

par(mai=mai.new)
summary(boost_visitors, las=1, cBar=10)
par(mai=mai.old)

yhat_boost <- predict(boost_visitors, newdata=egizio_test_df, n.trees=1:5000)
err <- apply(yhat_boost, 2, function(pred) mean((egizio_test_df$visitors - pred)^2))

# Error comparison (train and test)
plot(boost_visitors$train.error, type = "l", ylim = c(0, max(err)),
     ylab = "Train/Test Error", xlab = "n.trees")
lines(err, type = "l", col = 2)
best <- which.min(err)  # Minimum error in the test set
abline(v = best, lty = 2, col = 4)
min_error <- min(err)
title(main = sprintf("Min Error: %.4f", min_error))

# 3 Boosting - Smaller learning rate 
boost_visitors <- gbm(visitors ~ . - date, data=egizio_train_df,
                      distribution="gaussian", n.trees=5000, interaction.depth=1,
                      shrinkage=0.01) # learning rate

par(mai=mai.new)
summary(boost_visitors, las=1, cBar=10)
par(mai=mai.old)

yhat_boost <- predict(boost_visitors, newdata=egizio_test_df, n.trees=1:5000)
err <- apply(yhat_boost, 2, function(pred) mean((egizio_test_df$visitors - pred)^2))

# Error comparison (train and test)
plot(boost_visitors$train.error, type = "l", ylim = c(0, max(err)),
     ylab = "Train/Test Error", xlab = "n.trees")
lines(err, type = "l", col = 2)
best <- which.min(err)  # Minimum error in the test set
abline(v = best, lty = 2, col = 4)
min_error <- min(err)
title(main = sprintf("Min Error: %.4f", min_error))

# 4 Boosting - combination of previous models
boost_visitors <- gbm(visitors ~ . - date, data=egizio_train_df,
                      distribution="gaussian", n.trees=5000,
                      interaction.depth=4, shrinkage=0.01)
par(mai=mai.new)
summary(boost_visitors, las=1, cBar=10)
par(mai=mai.old)

yhat_boost <- predict(boost_visitors, newdata=egizio_test_df, n.trees=1:5000)
err <- apply(yhat_boost, 2, function(pred) mean((egizio_test_df$visitors - pred)^2))

# Error comparison (train and test)
plot(boost_visitors$train.error, type = "l", ylim = c(0, max(err)),
     ylab = "Train/Test Error", xlab = "n.trees")
lines(err, type = "l", col = 2)
best <- which.min(err)  # Minimum error in the test set
abline(v = best, lty = 2, col = 4)
min_error <- min(err)
title(main = sprintf("Min Error: %.4f", min_error))

# partial dependence plots
plot(boost_visitors, i.var=1, n.trees = best, ylab = "visitors")
plot(boost_visitors, i.var=2, n.trees = best, ylab = "visitors")
plot(boost_visitors, i.var=3, n.trees = best, ylab = "visitors")
plot(boost_visitors, i.var=4, n.trees = best, ylab = "visitors")
plot(boost_visitors, i.var=5, n.trees = best, ylab = "visitors")
plot(boost_visitors, i.var=6, n.trees = best, ylab = "visitors")
# ToDo: Add another one after adding tourist data
plot(boost_visitors, i.var=c(3,4), n.trees = best)

# --------------------------------------------------------------------- #
# Time-series cross-validation

ts_cv_spec <- time_series_cv(data = egizio_train_df,
                             date_var = date,
                             initial = 48, # 4 years
                             assess = 12, # 1 year window for test
                             skip = 12, # data will operate on years
                             cumulative = TRUE)
print(ts_cv_spec %>% tk_time_series_cv_plan())

ts_cv_spec %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, visitors, .facet_ncol  = 2, .interactive = FALSE)

# Dejan:
# The above code is just for visualization purposes.
# I wasn't able to make it work with train controls, but the code below works.

# Another approach:

grid_boosting <- expand.grid(n.trees = 100, #c(500, 1000, 5000),
                    interaction.depth = 1:6, 
                    shrinkage = c(0.1, 0.025, 0.05, 0.075, 0.01),
                    n.minobsinnode = 2) #c(2, 5, 10))

train_controls <- trainControl(method = "timeslice",# time-series cross-validation
                               initialWindow = 48, # initial training window
                               horizon = 12, # forecast evaluation window
                               fixedWindow = TRUE, 
                               skip = 12,
                               allowParallel = TRUE) # allow parallel processing if available

gbm_grid <- train(visitors ~ . - date,
                  data = egizio_train_df,
                  method = "gbm",  
                  distribution = "gaussian",
                  trControl = train_controls,
                  tuneGrid = grid_boosting,
                  verbose = FALSE)

# View the results of the grid search
print(gbm_grid)

best_model_boosting <- gbm_grid$bestTune

final_model_boosting <- gbm(visitors ~ . - date,
                            data = egizio_train_df,
                            distribution = "gaussian",
                            n.trees = best_model_boosting$n.trees,
                            interaction.depth = best_model_boosting$interaction.depth,
                            shrinkage = best_model_boosting$shrinkage,
                            n.minobsinnode = best_model_boosting$n.minobsinnode)

par(mai=mai.new)
summary(final_model_boosting, las=1, cBar=10)
par(mai=mai.old)

egizio_predictions_df$predicted_visitors_boosting <- predict(final_model_boosting,
                                                             newdata = egizio_test_df,
                                                             n.trees = best_model_boosting$n.trees)
# Calculate metrics
r_squared <- RSQUARE(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_boosting)
adj_r_squared <- adjusted_R2(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_boosting, nrow(egizio_train_df), length(final_model_boosting$var.names))
mse <- mse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_boosting)
rmse <- rmse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_boosting)
mae <- mae(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_boosting)
mape <- mape(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_boosting)

metrics_df <- rbind(metrics_df, list(Model = "Boosting - TSCV",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = NA))
print(metrics_df)

# Plot predictions 
plot(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_boosting,
     ylab="Predictions", xlab="True")
abline(0,1)

ggplot(egizio_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Visitors"), size = 1) +
  geom_line(aes(y = predicted_visitors_boosting, color = "Predicted"),
            linetype = "dashed", size = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))

# --------------------------------------------------------------------- #
# XGBoost
training.x <- model.matrix(visitors ~ . - date, data = egizio_train_df)
testing.x <- model.matrix(visitors ~ . - date, data = egizio_test_df)

xgb_model <- xgboost(data=data.matrix(training.x[,-1]), # ignore intercept
                     label=as.numeric(as.character(egizio_train_df$visitors)),
                     eta=0.025, # default=0.3 - takes values in (0-1]
                     max_depth=6, # default=6 - takes values in (0,Inf), larger value => more complex => overfitting
                     nrounds=500, # default=100 - controls number of iterations (number of trees)
                     early_stopping_rounds=50,
                     print_every_n = 10,
                     objective="reg:squarederror") # for linear regression
# objective="reg:squarederror"
# eval_metric = "rmse"

importance_scores <- xgb.importance(model = xgb_model)
print(importance_scores)
xgb.plot.importance(importance_matrix = importance_scores)

egizio_predictions_df$predicted_visitors_xgboost <- predict(xgb_model, newdata = testing.x[,-1])

# Calculate metrics for XGBoost
r_squared <- RSQUARE(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_xgboost)
adj_r_squared <- adjusted_R2(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_xgboost, nrow(egizio_train_df), xgb_model$nfeatures)
mse <- mse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_xgboost)
rmse <- rmse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_xgboost)
mae <- mae(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_xgboost)
mape <- mape(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_xgboost)

# Update metrics
metrics_df <- rbind(metrics_df, list(Model = "XGBoost",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = NA)) # Note: AIC may not be applicable for XGBoost

print(metrics_df)

# Plot predictions 
plot(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_xgboost,
     ylab="Predictions", xlab="True")
abline(0,1)

ggplot(egizio_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Visitors"), size = 1) +
  geom_line(aes(y = predicted_visitors_xgboost, color = "Predicted"),
            linetype = "dashed", size = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))


# --------------------------------------------------------------------- #
# Cross-validation

train_controls <- trainControl(method = "timeslice",
                               initialWindow = 48,
                               horizon = 12,
                               fixedWindow = TRUE, 
                               skip = 12,
                               allowParallel = TRUE)

grid <- expand.grid(nrounds = c(100, 200), 
                    max_depth = c(5, 10),             
                    eta = c(0.01, 0.025, 0.05, 0.075, 0.1),           
                    gamma = 0,                        
                    colsample_bytree = 1,             
                    min_child_weight = 1,             
                    subsample = 1
                    #early_stopping_rounds = c(10, 20, 50, 100)
)

xgb_model <- train(x = data.matrix(training.x[, -1]), # Ignore intercept
                   y = as.numeric(as.character(egizio_train_df$visitors)),
                   method = "xgbTree", # XGBoost 
                   trControl = train_controls,
                   tuneGrid = grid,
                   verbose = FALSE)

# Print the best model
best_model_xgb <- xgb_model$bestTune
print(best_model_xgb)

final_model_xgb <- xgboost(data=data.matrix(training.x[,-1]),
                           label=egizio_train_df$visitors,
                           eta=best_model_xgb$eta, 
                           max_depth=best_model_xgb$max_depth, 
                           nrounds=best_model_xgb$nrounds,
                           colsample_bytree=best_model_xgb$colsample_bytree,
                           min_child_weight=best_model_xgb$min_child_weight,
                           subsample=best_model_xgb$subsample,
                           gamma=best_model_xgb$gamma,
                           objective="reg:squarederror")

importance_scores <- xgb.importance(model = final_model_xgb)
print(importance_scores)
xgb.plot.importance(importance_matrix = importance_scores)

# Perform predictions on the test set
egizio_predictions_df$predicted_visitors_xgboost_tscv <- predict(final_model_xgb,
                                                                 newdata = data.matrix(testing.x[, -1]))

# Calculate metrics for XGBoost
r_squared <- RSQUARE(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_xgboost_tscv)
adj_r_squared <- adjusted_R2(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_xgboost_tscv, nrow(egizio_train_df), final_model_xgb$nfeatures)
mse <- mse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_xgboost_tscv)
rmse <- rmse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_xgboost_tscv)
mae <- mae(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_xgboost_tscv)
mape <- mape(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_xgboost_tscv)

# Update metrics_df with XGBoost metrics
metrics_df <- rbind(metrics_df, list(Model = "XGBoost - TSCV",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = NA)) # Note: AIC may not be applicable for XGBoost

print(metrics_df)

# Plot predictions 
plot(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_xgboost_tscv,
     ylab="Predictions", xlab="True")
abline(0,1)

ggplot(egizio_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Visitors"), size = 1) +
  geom_line(aes(y = predicted_visitors_xgboost_tscv, color = "Predicted"),
            linetype = "dashed", size = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))
