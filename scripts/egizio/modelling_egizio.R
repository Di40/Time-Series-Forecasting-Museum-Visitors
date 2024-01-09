# ---------------------------------------------------------------------------- #
#                               Modelling
# ---------------------------------------------------------------------------- #

rm(list=ls())
graphics.off()

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
library(glmnet)
library(dplyr)
library(gam)
library(gridExtra)
library(MASS)
library(sm)
library(splines)
library(npreg)

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
# Load dataset

# Run preprocessing.R if the file doesn't exist
if (!file.exists("../../data/egizio_final.rds")) {
  source("preprocessing_egizio.R")
}

egizio_df <- readRDS("../../data/egizio_final.rds")
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

target <- "visitors"
features <- colnames(egizio_df)
features <- features[-grep(target, features)]

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

# This dataframe will be used to store the predictions of all of the models, and make plotting easier.
egizio_predictions_df <- data.frame(date = egizio_test_df$date)
egizio_predictions_df$visitors_true <- egizio_test_df$visitors

# ---------------------------------------------------------------------------- #
# Standardization

# Modify flag perform_standardization to TRUE/FALSE

standardize_numeric_columns <- function(train_df, test_df) {
  # Extract numeric columns (excluding "date")
  numeric_columns <- sapply(train_df, is.numeric) & names(train_df) != "date"
  
  # Calculate mean and standard deviation for each numeric column in the training set
  means <- colMeans(train_df[, numeric_columns], na.rm = TRUE)
  std_devs <- apply(train_df[, numeric_columns], 2, sd, na.rm = TRUE)
  
  # Standardize the columns in both train and test data frames using means and std_devs from the training set
  for (col in names(train_df)[numeric_columns]) {
    # Impute missing values (if any) with mean in both train and test data frames
    mean_value <- means[col]
    train_df[[col]][is.na(train_df[[col]])] <- mean_value
    test_df[[col]][is.na(test_df[[col]])] <- mean_value
    
    # Standardize the column in both train and test data frames using means and std_devs from the training set
    train_df[[col]] <- (train_df[[col]] - means[col]) / std_devs[col]
    test_df[[col]] <- (test_df[[col]] - means[col]) / std_devs[col]
  }
  
  # Return the standardized data frames
  return(list(train_df = train_df, test_df = test_df))
}

perform_standardization <- TRUE

# These two copies can be used to reset train/test
egizio_train_df_copy <- data.frame(egizio_train_df)
egizio_test_df_copy <- data.frame(egizio_test_df)

if (perform_standardization) {
  standardize <- standardize_numeric_columns(egizio_train_df, egizio_test_df)
  egizio_train_df <- standardize$train_df
  egizio_test_df <- standardize$test_df
  print(head(egizio_train_df))
  print(head(egizio_test_df))
  
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
}

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

# We don't calculate R2 and Adj_R2 because this model has 0 standard deviation.
# We don't capture any variation.

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
            linetype = "dashed", linewidth = 1) +
  geom_line(aes(y = visitors_true, color = "Actual"), linewidth = 1) +
  labs(title = "Baseline - training mean",
       x = "Date",
       y = "Visitors") +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))

# ---------------------------------------------------------------------------- #
# Baseline model - same month last year

egizio_train_df_old <- data.frame(egizio_train_df)
egizio_test_df_old <- data.frame(egizio_test_df)

egizio_test_df$visitors_same_month_last_year <- tail(egizio_train_df$visitors, 12)
egizio_train_df$visitors_same_month_last_year <- lag(egizio_train_df$visitors, 12)
egizio_train_df <- na.omit(egizio_train_df)

lm_last_year <- lm(visitors ~ visitors_same_month_last_year, data = egizio_train_df)
summary(lm_last_year)

egizio_predictions_df$predicted_visitors_last_year <- predict(lm_last_year, newdata = egizio_test_df)

# Calculate metrics
r_squared <- summary(lm_last_year)$r.squared
adj_r_squared <- summary(lm_last_year)$adj.r.squared
aic <- AIC(lm_last_year)
mse <- mse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_last_year)
rmse <- rmse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_last_year)
mae <- mae(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_last_year)
mape <- mape(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_last_year)

metrics_df <- rbind(metrics_df, list(Model = "Baseline - last year",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic))
print(metrics_df)

ggplot(egizio_predictions_df, aes(x = date)) +
  geom_line(aes(y = predicted_visitors_last_year, color = "Predicted"),
            linetype = "dashed", linewidth = 1) +
  geom_line(aes(y = visitors_true, color = "Actual"), linewidth = 1) +
  labs(title = "Auto regressive Baseline - same month from previous year",
       x = "Date", y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))

egizio_train_df <- data.frame(egizio_train_df_old)
egizio_test_df <- data.frame(egizio_test_df_old)

# ---------------------------------------------------------------------------- #
# Model 1 - Multiple LR

# Here we perform manual feature selection. We start with the best model, and
# we remove 1 by 1 the least significant features.

str(egizio_train_df)

multiple_linear_regression <- lm(visitors ~ . - date
                              - month
                              - lagged_arrivals
                              - lagged_raining_days
                              - raining_days
                              - Covid_closures
                              - lagged_trends
                              - lagged_renovation,
                              data = egizio_train_df)
# Month and quarter were perfectly collinear, so we have removed the quarter.
summary(multiple_linear_regression)
# Order of removal: month, lagged_arrivals, lagged_raining_days, raining_days,
# Covid_closures, lagged_trends, lagged_renovation
# After the modification: average_temperature and lagged_school_holidays **;
# school_holidays *. The rest are highly significant.

egizio_predictions_df$predicted_multiple_lr <- predict(multiple_linear_regression, newdata = egizio_test_df)

# Calculate metrics
r_squared <- summary(multiple_linear_regression)$r.squared
adj_r_squared <- summary(multiple_linear_regression)$adj.r.squared
aic <- AIC(multiple_linear_regression)
mse <- mse(egizio_test_df$visitors, egizio_predictions_df$predicted_multiple_lr)
rmse <- rmse(egizio_test_df$visitors, egizio_predictions_df$predicted_multiple_lr)
mae <- mae(egizio_test_df$visitors, egizio_predictions_df$predicted_multiple_lr)
mape <- mape(egizio_test_df$visitors, egizio_predictions_df$predicted_multiple_lr)

metrics_df <- rbind(metrics_df, list(Model = "Multiple LR Manual Features",
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

# DW test
dwtest(multiple_linear_regression)
# The p-value is extremely small. => There is autocorrelation in the residuals.

# Check the residuals
res_lm <- residuals(multiple_linear_regression)
plot(egizio_train_df$date, res_lm, xlab="date", ylab="Residuals", type= "b",  pch=16, lty=3, cex=0.6)

plot(Acf(res_lm), xlab = "Lag", main = "Autocorrelation of Residuals",
     col = "steelblue", lwd = 2.5, ci.col = "black", cex.lab = 1.2, cex.main = 1.5)

# ---------------------------------------------------------------------------- #
# Model 2 - Multiple LR - Stepwise

# Hybrid
all_features_regression <- lm(visitors ~ . - date, data = egizio_train_df)
stepwise_lr_selected_model <- stepAIC(all_features_regression, direction = "both")
summary(stepwise_lr_selected_model)

egizio_predictions_df$predicted_multiple_lr_stepAIC<- predict(stepwise_lr_selected_model, newdata = egizio_test_df)

# Calculate metrics
r_squared <- summary(stepwise_lr_selected_model)$r.squared
adj_r_squared <- summary(stepwise_lr_selected_model)$adj.r.squared
aic <- AIC(stepwise_lr_selected_model)
mse <- mse(egizio_test_df$visitors, egizio_predictions_df$predicted_multiple_lr_stepAIC)
rmse <- rmse(egizio_test_df$visitors, egizio_predictions_df$predicted_multiple_lr_stepAIC)
mae <- mae(egizio_test_df$visitors, egizio_predictions_df$predicted_multiple_lr_stepAIC)
mape <- mape(egizio_test_df$visitors, egizio_predictions_df$predicted_multiple_lr_stepAIC)

metrics_df <- rbind(metrics_df, list(Model = "Multiple LR Stepwise Both",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic))
print(metrics_df)

stepwise_lr_backward_selected_model <- stepAIC(all_features_regression, direction = "backward")
summary(stepwise_lr_backward_selected_model)
# We obtain the same model.

# DW test
dwtest(stepwise_lr_backward_selected_model)
# The p-value is extremely small. => There is autocorrelation in the residuals.

# Check the residuals
res_lm_stepaic <- residuals(stepwise_lr_backward_selected_model)
plot(egizio_train_df$date, res_lm_stepaic, xlab="date", ylab="Residuals", type= "b",  pch=16, lty=3, cex=0.6)

plot(Acf(res_lm_stepaic), xlab = "Lag", main = "Autocorrelation of Residuals",
     col = "steelblue", lwd = 2.5, ci.col = "black", cex.lab = 1.2, cex.main = 1.5)

# ---------------------------------------------------------------------------- #
# Model 3 - TSLM with trend and seasonality

egizio_visitors_train_ts <- ts(egizio_train_df$visitors, frequency = 12)
plot(egizio_train_df$date, egizio_visitors_train_ts, type="o")

# Fit a linear model with trend
tslm_basic <- tslm(egizio_visitors_train_ts ~ trend + season)
summary(tslm_basic)
# trend is highly significant

res <- residuals(tslm_basic)
plot(res)
Acf(res)
# There is a lot of information left in the residuals to be modeled.

# Perform the Durbin-Watson test
dwtest(tslm_basic)

# Forecast on the test data
fcast <- forecast(tslm_basic, newdata = egizio_test_df, h = nrow(egizio_test_df))
plot(fcast)

egizio_predictions_df$predicted_tslm <- fcast$mean

# Calculate metrics
r_squared <- summary(tslm_basic)$r.squared
adj_r_squared <- summary(tslm_basic)$adj.r.squared
aic <- AIC(tslm_basic)
mse <- mse(egizio_predictions_df$predicted_tslm, egizio_test_df$visitors)
rmse <- rmse(egizio_predictions_df$predicted_tslm, egizio_test_df$visitors)
mae <- mae(egizio_predictions_df$predicted_tslm, egizio_test_df$visitors)
mape <- mape(egizio_predictions_df$predicted_tslm, egizio_test_df$visitors)

metrics_df <- rbind(metrics_df, list(Model = "TSLM - Trend and Seasonality",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic))
print(metrics_df)

ggplot(egizio_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Visitors"), size = 1) +
  geom_line(aes(y = predicted_tslm, color = "Predicted"),
            linetype = "dashed", size = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))

# --------------------------------------------------------------------- #
# Model 4 - TSLM full (including the other features)

# Here we will perform manual feature selection, to keep just the
# significant features.

egizio_train_ts_df <- data.frame(
  date_numeric = ts(egizio_train_df$date_numeric, frequency = 12),
  visitors = ts(egizio_train_df$visitors, frequency = 12),
  trends = ts(egizio_train_df$trends, frequency = 12),
  average_temperature = ts(egizio_train_df$average_temperature, frequency = 12),
  raining_days = ts(egizio_train_df$raining_days, frequency = 12),
  school_holidays = ts(egizio_train_df$school_holidays, frequency = 12),
  arrivals = ts(egizio_train_df$arrivals, frequency = 12),
  covid = ts(egizio_train_df$Covid_closures, frequency = 12),
  renovation = ts(egizio_train_df$renovation, frequency = 12),
  # lagged,
  lagged_trends = ts(egizio_train_df$lagged_trends, frequency = 12),
  lagged_average_temperature = ts(egizio_train_df$lagged_average_temperature, frequency = 12),
  lagged_raining_days = ts(egizio_train_df$lagged_raining_days, frequency = 12),
  lagged_school_holidays = ts(egizio_train_df$lagged_school_holidays, frequency = 12),
  lagged_arrivals = ts(egizio_train_df$lagged_arrivals, frequency = 12),
  lagged_renovation = ts(egizio_train_df$lagged_renovation, frequency = 12)
)

# Fit the model on the training set
tslm_manual <- tslm(visitors ~ .
                    - covid
                    - raining_days
                    - lagged_raining_days
                    - lagged_trends
                    - lagged_school_holidays
                    - average_temperature
                    - date_numeric
                    - school_holidays
                    - lagged_average_temperature
                    - lagged_renovation,
                    data = egizio_train_ts_df)
summary(tslm_manual)

plot(egizio_train_ts_df$visitors)
lines(fitted(tslm_manual), col=2)

res <- residuals(tslm_manual)
plot(res)
Acf(res) 

dwtest(tslm_manual)

# Leave-one-out Cross-Validation Statistic
CV(tslm_manual)

# Forecasting on the test set
test_data <- data.frame(
  date_numeric = ts(egizio_test_df$date_numeric, frequency = 12),
  visitors = ts(egizio_test_df$visitors, frequency = 12),
  trends = ts(egizio_test_df$trends, frequency = 12),
  average_temperature = ts(egizio_test_df$average_temperature, frequency = 12),
  raining_days = ts(egizio_test_df$raining_days, frequency = 12),
  school_holidays = ts(egizio_test_df$school_holidays, frequency = 12),
  arrivals = ts(egizio_test_df$arrivals, frequency = 12),
  covid = ts(egizio_test_df$Covid_closures, frequency = 12),
  renovation = ts(egizio_test_df$renovation, frequency = 12),
  # lagged,
  lagged_trends = ts(egizio_test_df$lagged_trends, frequency = 12),
  lagged_average_temperature = ts(egizio_test_df$lagged_average_temperature, frequency = 12),
  lagged_raining_days = ts(egizio_test_df$lagged_raining_days, frequency = 12),
  lagged_school_holidays = ts(egizio_test_df$lagged_school_holidays, frequency = 12),
  lagged_arrivals = ts(egizio_test_df$lagged_arrivals, frequency = 12),
  lagged_renovation = ts(egizio_test_df$lagged_renovation, frequency = 12)
)

fcast <- forecast(tslm_manual, newdata = test_data, h = nrow(egizio_test_df))

plot(fcast)

egizio_predictions_df$predicted_tslm_manual <- fcast$mean

# Calculate metrics
r_squared <- summary(tslm_manual)$r.squared
adj_r_squared <- summary(tslm_manual)$adj.r.squared
aic <- AIC(tslm_manual)
mse <- mse(egizio_predictions_df$predicted_tslm_manual, egizio_test_df$visitors)
rmse <- rmse(egizio_predictions_df$predicted_tslm_manual, egizio_test_df$visitors)
mae <- mae(egizio_predictions_df$predicted_tslm_manual, egizio_test_df$visitors)
mape <- mape(egizio_predictions_df$predicted_tslm_manual, egizio_test_df$visitors)

metrics_df <- rbind(metrics_df, list(Model = "TSLM - Manual Features",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic))
print(metrics_df)

ggplot(egizio_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Visitors"), size = 1) +
  geom_line(aes(y = predicted_tslm_manual, color = "Predicted"),
            linetype = "dashed", size = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))

# --------------------------------------------------------------------- #
# Model 5 - Regularization using Ridge/Lasso

# We will utilize Time-series Cross-validation

y_train <- egizio_train_df$visitors
X_train <- egizio_train_df[, -which(names(egizio_train_df) %in% c("date", "visitors"))]
X_test <- egizio_test_df[, -which(names(egizio_test_df) %in% c("date", "visitors"))]
# Convert data to matrix format
X_train <- as.matrix(X_train)
y_train <- as.matrix(y_train)
X_test <- as.matrix(X_test)

train_controls <- trainControl(method = "timeslice",
                               initialWindow = 48,
                               horizon = 12,
                               fixedWindow = TRUE, 
                               skip = 12,
                               allowParallel = TRUE)

grid <- expand.grid(alpha = c(0, 1), # ridge or lasso
                    lambda = seq(0.001, 0.01, by = 0.001))

l1_or_l2_regularization_ts_cv <- train(x = X_train,
                              y = as.numeric(as.character(y_train)), # to use regression we have to convert like this
                              method = "glmnet", 
                              trControl = train_controls,
                              tuneGrid = grid, # expand.grid(alpha = 1), # Lasso
                              verbose = FALSE,
                              metric = "RMSE")


# Print the best model
best_model_l1_or_l2_regularization <- l1_or_l2_regularization_ts_cv$bestTune
print(best_model_l1_or_l2_regularization)
best_lambda <- l1_or_l2_regularization_ts_cv$bestTune$lambda

best_l1_or_l2_regularization_model <- glmnet(x = X_train,
                                             y = as.numeric(as.character(y_train)),
                                             alpha = best_model_l1_or_l2_regularization$alpha,
                                             lambda = best_lambda)

# Perform predictions on the test set
egizio_predictions_df$predicted_visitors_l1_or_l2_regularization_tscv <-
  predict(best_l1_or_l2_regularization_model, newx = X_test, s = best_lambda)

# Calculate metrics
mse <- mse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_l1_or_l2_regularization_tscv)
rmse <- rmse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_l1_or_l2_regularization_tscv)
mae <- mae(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_l1_or_l2_regularization_tscv)
mape <- mape(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_l1_or_l2_regularization_tscv)

metrics_df <- rbind(metrics_df, list(Model = "L1/L2 Regularization TS CV",
                                     R2 = NA, R2_adj = NA,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = NA))
print(metrics_df)

# Plot predictions 
plot(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_lasso_tscv,
     ylab="Predictions", xlab="True")
abline(0,1)

ggplot(egizio_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Visitors"), linewidth = 1) +
  geom_line(aes(y = predicted_visitors_l1_or_l2_regularization_tscv, color = "Predicted"),
            linetype = "dashed", linewidth = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))

# --------------------------------------------------------------------- #
# Model 6 - GAM

# Stepwise GAM

# Start with a linear model (df=1)
g3 <- gam(visitors ~. - date, data=egizio_train_df)
summary(g3)
AIC(g3)

sc <- gam.scope(egizio_train_df[, -which(names(egizio_train_df) %in% c("date", "visitors"))],
                arg = c("df=2", "df=3", "df=4"))
g4 <- step.Gam(g3, scope = sc, trace = TRUE)
summary(g4)

AIC(g4) # 177.3217

# par(mfrow=c(3,5))
# plot(g4, se=T)

# If we want to see better some plot
# par(mfrow=c(1,1))
# plot(g4, se=T, ask=T)

# Prediction
p.gam <- predict(g4, newdata=egizio_test_df)     
cat('Deviance:', sum((p.gam - egizio_test_df$visitors)^2))

colnames(egizio_train_df)
gam_visitors <- gam(visitors ~ s(date_numeric) + s(year) + s(month)
                    + s(trends) + s(average_temperature) + s(raining_days)
                    + s(school_holidays) + s(arrivals) + Covid_closures
                    + renovation + lagged_renovation + s(lagged_trends)
                    + s(lagged_average_temperature) + s(lagged_raining_days)
                    + s(lagged_school_holidays) + s(lagged_arrivals),
                    data = egizio_train_df)

summary(gam_visitors)
AIC(gam_visitors)

# ToDo (Anna): Can we perform stepwise GAM for the code above?
# ToDo (Anna): Compute predictions and calculate metrics.

# --------------------------------------------------------------------- #
# Model 7 - Generalized Bass Model (with shock)

egizio_train_unstandardized_df <- egizio_train_df_copy

# First we obtain the needed coefficients using BM:
bm_visitors <- BM(egizio_train_unstandardized_df$visitors, display = TRUE)
summary(bm_visitors)

m <- 1.878697e+07 
p <- 1.642189e-03
q <- 9.073474e-03

# Predictions and instantaneous curve for BM
pred_bm_visitors <- predict(bm_visitors, newx = 1:216)
pred_inst_bm_visitors <- make.instantaneous(pred_bm_visitors)

# Plotting BM predictions
plot(egizio_train_unstandardized_df$visitors, type = "b", xlab = "Month", ylab = "Monthly Visitors", 
     pch = 16, lty = 3, cex = 0.6, xlim = c(1, 216))
lines(pred_inst_bm_visitors, lwd = 2, col = 2)

# Try with shock

# One exponential shock - This models the shock of 2015
gbm_e1 <- GBM(egizio_train_unstandardized_df$visitors, shock = "exp", nshock = 1, alpha = 0.04,
            prelimestimates = c(m, p, q, 124, -0.1, 0.5))
summary(gbm_e1)

# Two exponential shocks - This models both the 2015 and Covid shock    
gbm_e2 <- GBM(egizio_train_unstandardized_df$visitors, shock = "exp", nshock = 2, alpha = 0.04,
            prelimestimates = c(m, p, q, 124, -0.1, 0.5, 183 , -0.1, -0.5))
summary(gbm_e2)   

# Two rectangular shocks
gbm_r2 <- GBM(egizio_train_unstandardized_df$visitors, shock = "rett", nshock = 2,
            prelimestimates = c(m, p, q, 124, 183, 0.1, 183, 196, -0.4), oos=10)
summary(gbm_r2)

# Three exponential shocks
gbm_e3 <- GBM(egizio_train_unstandardized_df$visitors, shock = "exp", nshock = 3,
                      prelimestimates = c(m, p, q,  124, -0.1, 0.2, 160, 0.1, -0.4, 196, -0.1, +0.6))
summary(gbm_e3)

# Exponential + rectangular shocks
gbm_er <- GBM(egizio_train_unstandardized_df$visitors, shock = "mixed", nshock = 2,
            prelimestimates = c(m, p, q, 124, -0.1, 0.2, 183, 196, -0.4),oos=10)
summary(gbm_er)

# The best candidates were: exp + rectangular and two rectangular.
# Decided to proceed with two rectangular shocks.
best_gbm <- gbm_r2

pred_GBM_visitors<- predict(best_gbm, newx=c(1:216))
pred_GBM_visitors.inst<- make.instantaneous(pred_GBM_visitors)

# Plotting GBM predictions
plot(egizio_train_unstandardized_df$visitors, type = "b",
     xlab = "Month", ylab = "Monthly Visitors", 
     pch = 16, lty = 3, cex = 0.6, xlim = c(1, 216))
lines(pred_GBM_visitors.inst, lwd = 2, col = 2)

# Calculate metrics
egizio_predictions_df$predicted_visitors_generalized_bass_model <- pred_GBM_visitors.inst[205:216]
# Standardize:
egizio_predictions_df$predicted_visitors_generalized_bass_model <- (egizio_predictions_df$predicted_visitors_generalized_bass_model - mean(egizio_train_unstandardized_df$visitors)) / sd(egizio_train_unstandardized_df$visitors)
adj_r_squared <- adjusted_R2(egizio_train_unstandardized_df$visitors, pred_GBM_visitors.inst[1:204], nrow(egizio_train_df), length(best_gbm$coefficients))
mse <- mse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_generalized_bass_model)
rmse <- rmse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_generalized_bass_model)
mae <- mae(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_generalized_bass_model)
mape <- mape(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_generalized_bass_model)

metrics_df <- rbind(metrics_df, list(Model = "Generalized Bass Model - 2R",
                                     R2 = best_gbm$Rsquared, R2_adj = NA,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = NA))
print(metrics_df)

# Plot predictions 
ggplot(egizio_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Visitors"), linewidth = 1) +
  geom_line(aes(y = predicted_visitors_generalized_bass_model, color = "Predicted"),
            linetype = "dashed", linewidth = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))

# This model is good just for modelling the trend.

plot(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_lasso_tscv,
     ylab="Predictions", xlab="True")
abline(0,1)

# GGM
GGM_model <- GGM(egizio_train_unstandardized_df$visitors, prelimestimates=c(m, 0.001, 0.01, p, q))
summary(GGM_model)

pred_GGM <- predict(GGM_model, newx=c(1:216))
pred_GGM.inst <- make.instantaneous(pred_GGM)

plot.ts(egizio_train_unstandardized_df$visitors)
lines(pred_GGM.inst, lwd=2, col=2)

# Analysis of residuals
res_GGM <- residuals(GGM_model)
acf <- acf(res_GGM)

# --------------------------------------------------------------------- #
# Model 8 - Auto ARIMA
egizio_visitors_train_ts <- ts(egizio_train_df$visitors, frequency = 12)

auto_arima <- auto.arima(egizio_visitors_train_ts)
summary(auto_arima) # AIC=364.99

predicted_visitors_auto_arima <- forecast(auto_arima, h=12)
egizio_predictions_df$predicted_visitors_auto_arima <- predicted_visitors_auto_arima$mean

# Calculate metrics for ARIMA
train_predictions <- fitted(auto_arima)
r_squared <- RSQUARE(egizio_train_df$visitors, train_predictions)
adj_r_squared <- adjusted_R2(egizio_train_df$visitors, train_predictions, length(egizio_train_df$visitors), length(coef(auto_arima)))
mse <- mse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_auto_arima)
rmse <- rmse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_auto_arima)
mae <- mae(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_auto_arima)
mape <- mape(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_auto_arima)
aic <- AIC(auto_arima)

# Update metrics_df with ARIMA metrics
metrics_df <- rbind(metrics_df, list(Model = "Auto ARIMA",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic))
print(metrics_df)

ggplot(egizio_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Visitors"), linewidth = 1) +
  geom_line(aes(y = predicted_visitors_auto_arima, color = "Predicted"),
            linetype = "dashed", linewidth = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))

plot(egizio_visitors_train_ts)

egizio_ts_df <- diff(egizio_visitors_train_ts)
p_ts_df <- autoplot(egizio_ts_df, xlab = "Time", ylab = "Visitors")

plot(p_ts_df)
# By differentiating the series, we can see that it
# seems to be more stationary (in term of mean).

# Next, we will check the residuals of the differentiated series in
# order to see if there are again some important behaviors:
p_acf_df <- ggAcf(egizio_ts_df)
p_pacf_df <- ggPacf(egizio_ts_df)

grid.arrange(p_acf_df, p_pacf_df, nrow = 2)
# We see that the lag at time 12 and 24 are relevant as always.
# So we need to model these particular characteristics.

# Next, we try to build our first custom Arima models.

# --------------------------------------------------------------------- #
# Model 9 - SARIMA
sarima <- Arima(egizio_visitors_train_ts, order = c(1,1,2), seasonal = c(1,1,2)) # Best model initially
summary(sarima) # AIC=323.37

# Arima (0,1,2) (0,1,2) : AIC=327.28
# Arima (0,1,0) (0,0,2) : AIC=396.58 
# Arima (2,1,0) (0,0,2) : AIC=380.23
# Arima (0,0,2) (0,1,2) : AIC=333.78
# --> with this model it's clear that there's a pattern that
#     the model is nor so able to understand

# Comparison between the original time series and values fitted by the ARIMA model.
ggplot(data = egizio_train_df,
       aes(x = date,
           y = as.numeric(egizio_visitors_train_ts))) +
  geom_line(color = "blue") +  
  geom_line(aes(y = fitted(sarima)), color = "red", linetype = "twodash") +
  ylab("Visitors") +
  labs(title = "True vs Fitted values by SARIMA")

ggplot(aes(date, y = as.numeric(residuals(sarima))), data = egizio_train_df) +
  geom_point(color = "blue") + xlab("Date") + ylab("Residuals") + ggtitle("Residuals of Arima")

# This model is improved bellow, so metrics aren't calculated here.

# Added by Dejan

# First difference
diff1 <- diff(egizio_visitors_train_ts) # default lag=1
tsdisplay(diff1)
# We removed the trend, but we see the seasonality better.

# Seasonal difference
diff12 <- diff(egizio_visitors_train_ts, lag=12) 
tsdisplay(diff12)
# The series no longer has seasonality.

# Arima (1,0,1) (0,1,1) : AIC=321.27 MSE=1.5944751 -> predictions follow the pattern, but are underestimated
# Arima (1,0,0) (0,1,1) : AIC=325.61 MSE=1.4843500 -> predictions follow the pattern, but are underestimated
# Arima (1,0,1) (1,1,1) : AIC=318.99 MSE=1.6191393 -> predictions follow the pattern, but are underestimated
# Arima (1,0,12) (0,1,2): AIC=315.3  MSE=0.3677237 -> best predictions (they follow the pattern and are close to the original ones)
# Arima (1,0,12) (0,1,1): AIC=313.65 MSE=0.3965948 -> good predictions (they follow the pattern and are close to the original ones, just the last point is bad)

egizio_visitors_train_ts <- ts(egizio_train_df$visitors, frequency = 12)

# Best model according to AIC, with low MSE:
sarima_improved <- Arima(egizio_visitors_train_ts, order = c(1,0,12),
                         seasonal = c(0,1,1), include.drift = TRUE)
summary(sarima_improved) 

train_predictions_sarima <- fitted(sarima_improved)

ggplot(data = egizio_train_df,
       aes(x = date,
           y = as.numeric(egizio_visitors_train_ts))) +
  geom_line(color = "blue") +  
  geom_line(aes(y = train_predictions_sarima),
            color = "red", linetype = "twodash") +
  xlab("Date") + ylab("Visitors") +
  labs(title = "True vs Fitted values by ARIMA")

# Residuals
ggplot(aes(date, y = as.numeric(residuals(sarima_improved))), data = egizio_train_df) +
  geom_point(color = "blue") + xlab("Date") + ylab("Residuals") + ggtitle("Residuals of Arima")
checkresiduals(sarima_improved)

pred_sarima_improved <- forecast(sarima_improved, h = 12)
egizio_predictions_df$predicted_visitors_sarima <- pred_sarima_improved$mean

# Calculate metrics
r_squared <- RSQUARE(egizio_train_df$visitors, train_predictions_sarima)
adj_r_squared <- adjusted_R2(egizio_train_df$visitors, train_predictions_sarima, length(egizio_train_df$visitors), length(coef(sarima_improved)))
mse <- mse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_sarima)
rmse <- rmse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_sarima)
mae <- mae(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_sarima)
mape <- mape(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_sarima)
aic <- AIC(sarima_improved)

# Update metrics_df
metrics_df <- rbind(metrics_df, list(Model = "SARIMA - Improved",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic))
print(metrics_df)

ggplot(egizio_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Visitors"), linewidth = 1) +
  geom_line(aes(y = predicted_visitors_sarima, color = "Predicted"),
            linetype = "dashed", linewidth = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))

# In the following, we see that for different windows of data the model needs
# a lot of training examples in order to fit well our series.
# So, we try now to represent how the forecasting gets better 
# giving to the model an "accumulated information" through time:

png("../../plots/arima_plots.png", width = 1000, height = 1500)
# mfrow.old <- par()$frow
# mar.old <- par()$mar
par(mfrow = c(7,2))
par(mar=c(2,2,2,2))
train_rows <- nrow(egizio_train_df) # 204
train_size <- 48 # initial
val_size <- 12
while (train_size <= train_rows) {
  # 48, 60, 72, ..., 204 -> total 14 windows will be created
  train_w <- egizio_visitors_train_ts[1:train_size] # subset the training set
  arima_w <- Arima(train_w, order = c(1,0,12), seasonal = c(0,1,1),
                   include.drift = TRUE) # try with different parameters
  pred_arima_w <- as.numeric(forecast(train_w, h = val_size)$mean)
  mse <- round(mse(train_w, pred_arima_w), 2)
  plot(x = 1:(train_size+val_size), egizio_visitors_train_ts[1:(train_size+val_size)],
       type = "l", xlab = "Time", ylab = "Visitors", lwd = 1.5,
       main = paste("MSE:", mse))
  #lines(x = (train_size-12):(train_size-1), y = tail(arima_w$fitted, 12), col = 2, lwd = 1.5)
  lines(x = 1:train_size, y = arima_w$fitted, col = 2, lwd = 1.5) # train predictions
  lines(x = train_size:(train_size+val_size-1), y = pred_arima_w, col = 4, lwd = 2.5)
  train_size <- train_size + 12
}
# par(frow = mfrow.old)
# par(mar = mar.old)
dev.off()

# ToDo: Dejan - Double check these plots, they seem incorrect.

# --------------------------------------------------------------------- #
# Model 10 - SARIMAX

# First, we try to obtain useful regressors.
# To do this, we will start by analyzing the coefficients obtained using
# stepwise regression and Lasso/Ridge Regularization.
summary(stepwise_lr_selected_model)
# year, month, trends, arrivals, lagged_arrivals, renovation1

# print(coef(best_l1_or_l2_regularization_model))
coefficients_l1_l2 <- coef(best_l1_or_l2_regularization_model)
print(coefficients_l1_l2[order(abs(coefficients_l1_l2), decreasing = TRUE), ])
# renovation, arrivals, trends, lagged_average_temperature, Covid_closures

# The coefficients in common are: renovation, arrivals, trends.
# Including month/year/date doesn't make sense for SARIMAX.

regressors_train <- egizio_train_df[, c("renovation", "arrivals", "trends")]
regressors_train <- as.matrix(regressors_train)
regressors_train <- apply(regressors_train, 2, as.numeric)
# sarimax <- auto.arima(egizio_visitors_train_ts, xreg = regressors_train)
sarimax <- Arima(egizio_visitors_train_ts, xreg = regressors_train,
                 order = c(1,0,12), seasonal = c(0,1,1), include.drift = TRUE)
summary(sarimax)
# ARIMA(1,0,0) (0,1,1):  AIC=206.17, RMSE=0.3803622 -> auto.arima
# ARIMA(1,0,12)(0,1,2): AIC=215.73, RMSE=0.3533746
# ARIMA(1,0,12)(0,1,1): AIC=213.73, RMSE=0.3532834 -> lowest RMSE
# ARIMA(1,0,1) (0,1,1):  AIC=209.91, RMSE=0.3804768

ggplot(data = egizio_train_df, aes(x = date, y = visitors)) +
  geom_line(color = "blue") +  
  geom_line(aes(y = fitted(sarimax)), color = "red", linetype = "twodash") +
  ylab("Visitors") +
  labs(title = "True vs Fitted values by SARIMAX")

ggplot(aes(date, y = as.numeric(residuals(sarimax))), data = egizio_train_df) +
  geom_point(color = "blue") + xlab("Date") + ylab("Residuals") +
  ggtitle("Residuals of SARIMAX")

# Forecasting
regressors_test <- egizio_test_df[, c("renovation", "arrivals", "trends")]
regressors_test <- as.matrix(regressors_test)
regressors_test <- apply(regressors_test, 2, as.numeric)

pred_sarimax <- forecast(sarimax, h = 12, xreg=regressors_test)
egizio_predictions_df$predicted_visitors_sarimax <- pred_sarimax$mean

train_predictions_sarimax <- fitted(sarimax)

# Calculate metrics
r_squared <- RSQUARE(egizio_train_df$visitors, train_predictions_sarimax)
adj_r_squared <- adjusted_R2(egizio_train_df$visitors, train_predictions_sarimax, length(egizio_train_df$visitors), length(coef(sarimax)))
mse <- mse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_sarimax)
rmse <- rmse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_sarimax)
mae <- mae(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_sarimax)
mape <- mape(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_sarimax)
aic <- AIC(sarima_improved)

# Update metrics_df
metrics_df <- rbind(metrics_df, list(Model = "SARIMAX",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic))
print(metrics_df)

ggplot(egizio_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Visitors"), linewidth = 1) +
  geom_line(aes(y = predicted_visitors_sarimax, color = "Predicted"),
            linetype = "dashed", linewidth = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))

# --------------------------------------------------------------------- #
# Model 11 - Exponential smoothing - Holt Winters
components_dfts <- decompose(egizio_visitors_train_ts)
plot(components_dfts)

HW1 <- HoltWinters(egizio_visitors_train_ts) # Smoothing parameters:alpha: 0.6785537, beta : 0.007338514, gamma: 1
HW2 <- HoltWinters(egizio_visitors_train_ts, alpha=0.2, beta=0.1, gamma=0.1)

# Visually evaluate the fits
plot(egizio_visitors_train_ts, ylab="Egizio visitors")
lines(HW1$fitted[,1], lty=2, col="blue")
lines(HW2$fitted[,1], lty=2, col="red")

# Forecasting
HW1_for <- forecast(HW1, h=12, level=c(80,95))
# Visualize our predictions:
plot(HW1_for)
lines(HW1_for$fitted, lty=2, col="purple")

HW1.pred <- predict(HW1, nrow(egizio_test_df), prediction.interval = TRUE, level=0.95)
# Visually evaluate the prediction
plot(egizio_visitors_train_ts, ylab="visitors")
lines(HW1$fitted[,1], lty=2, col="blue")
lines(HW1.pred[,1], col="red") # This looks good
lines(HW1.pred[,2], lty=2, col="orange")
lines(HW1.pred[,3], lty=2, col="purple")

egizio_predictions_df$predicted_HW1 <- HW1.pred[,1]

# Calculate metrics
# R2 and Adj R2 can't be calculated: length(HW1$fitted[,1]) = 192 != 204 (train)
mse <- mse(egizio_test_df$visitors, egizio_predictions_df$predicted_HW1)
rmse <- rmse(egizio_test_df$visitors, egizio_predictions_df$predicted_HW1)
mae <- mae(egizio_test_df$visitors, egizio_predictions_df$predicted_HW1)
mape <- mape(egizio_test_df$visitors, egizio_predictions_df$predicted_HW1)
sse1<- sse(egizio_test_df$visitors, egizio_predictions_df$predicted_HW1)
metrics_df <- rbind(metrics_df, list(Model = "Exp. smoothing HW-2",
                                     R2 = NA, R2_adj = NA,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = NA))
print(metrics_df)

# Let's check the residuals
acf(HW1_for$residuals, lag.max=20, na.action=na.pass)
Box.test(HW1_for$residuals, lag=20, type="Ljung-Box")
hist(HW1_for$residuals)

# For HW2
HW2_for <- forecast(HW2, h=12, level=c(80,95))
# Visualize our predictions:
plot(HW2_for)
lines(HW2_for$fitted, lty=2, col="green")

HW2.pred <- predict(HW2, 12, prediction.interval = TRUE, level=0.95)
# Visually evaluate the prediction
plot(egizio_visitors_train_ts, ylab="Egizio visitors")
lines(HW2$fitted[,1], lty=2, col="blue")
lines(HW2.pred[,1], col="red") # This looks good
lines(HW2.pred[,2], lty=2, col="orange")
lines(HW2.pred[,3], lty=2, col="purple")

egizio_predictions_df$predicted_HW2 <- HW2.pred[,1]

# Calculate metrics
mse <- mse(egizio_test_df$visitors, egizio_predictions_df$predicted_HW2)
rmse <- rmse(egizio_test_df$visitors, egizio_predictions_df$predicted_HW2)
mae <- mae(egizio_test_df$visitors, egizio_predictions_df$predicted_HW2)
mape <- mape(egizio_test_df$visitors, egizio_predictions_df$predicted_HW2)
sse2 <- sse(egizio_test_df$visitors, egizio_predictions_df$predicted_HW2)
metrics_df <- rbind(metrics_df, list(Model = "Exp. smoothing HW-2",
                                     R2 = NA, R2_adj = NA,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = NA))
print(metrics_df)

# Let's check the residuals
acf(HW2_for$residuals, lag.max=20, na.action=na.pass)
Box.test(HW2_for$residuals, lag=20, type="Ljung-Box")
hist(HW2_for$residuals)


# Multiplicative seasonality - Probably doesn't work!
HW3 <- HoltWinters(egizio_visitors_train_ts, seasonal = "multiplicative")
HW3.pred <- predict(HW3, 12, prediction.interval = TRUE, level=0.95)
plot(egizio_visitors_train_ts, ylab="Visitors")
lines(HW3$fitted[,1], lty=2, col="blue")
lines(HW3.pred[,1], col="red")
lines(HW3.pred[,2], lty=2, col="orange")
lines(HW3.pred[,3], lty=2, col="purple")

# --------------------------------------------------------------------- #
# Model 12 - Local Regression

# This function creates a nonparametric regression estimate from 
# data consisting of a single response variable and one or two covariates.

# x = vector/two-columns matrix of covariates
#     --> we consider the two best variables looking at the
#         previous results of the arima models for the residuals.
#         (they are already contained in the "regressors" matrix)

# y = vector of response

# ToDo (Anna): Fix the following lines with sm.regression
# The outputs seem wrong (or delete this part).

# Skip 1 - rennovation (binary variable)
x <- regressors_train[, c(2,3)] # arrivals and trends
y <- y_train

# Model with the Inflation covariate ??
plot.ts(egizio_visitors_train_ts)

sm.regression(x[,1], y, h = 100, add = T, col = 2)
sm.regression(x[,1], y, h = 10, add = T, ngrid=200, col=3)
sm.regression(x[,1], y, h = 30, ngrid=200, col=4)
sm.regression(x[,1], y, h = 50, add = T, ngrid=200, col=5)
sm.regression(x[,1], y, h = 5, add = T, ngrid=200, col=6)
sm.regression(x[,1], y, h = 1, add = T, col=7, ngrid=200)

# We add variability bands
sm.regression(x[,2], y, h = 30, ngrid=200, display="se")

# --------------------------------------------------------------------- #
# LOESS

# the x and y arguments provide the x and y coordinates for the plot. 

x = 1:204
y = egizio_train_df$visitors
# egizio_train_df$date<- as.numeric(egizio_train_df$date) # Commented out by Dejan
# If needed, there is a variable date_numeric.

# We use geom_smooth in order to graphically see the results

ggplot(data = egizio_train_df, 
       aes(x = date, y = visitors)) +
  geom_point() +
  xlab("Date") +
  ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_smooth(method = "loess", span = 2/3)

# We use the loess.smooth method for computing the values.
loess1_ecar <- loess.smooth(x, y) # span = 2/3

ggplot(data = egizio_train_df, 
       aes(x = date, y = visitors)) +
  geom_point() +
  xlab("Date") +
  ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_smooth(method = "loess", span = 0.9, color = "red")

loess2_ecar <- loess.smooth(x,y, span = 0.9) 

ggplot(data = egizio_train_df, 
       aes(x = date, y = visitors)) +
  geom_point() +
  xlab("Date") +
  ylab("ECarSales") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_smooth(method = "loess", span = 0.4, color = "green")

loess2_ecar <- loess.smooth(x,y, span = 0.4) 

# Complete comparison:

ggplot(data = egizio_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") + 
  ggtitle("Loess Model for Visitors") +
  geom_smooth(method = "loess", span = 2/3, se = F) +
  geom_smooth(method = "loess", span = 0.9, color = "red", se = F) +
  geom_smooth(method = "loess", span = 0.4, color = "green", se = F) +
  geom_smooth(method = "loess", span = 0.25, color = "yellow", se = F) +
  geom_smooth(method = "loess", span = 0.1, color = "orange", se = F)

# The orange one is the nearest to our model, but the best
# compromise is given by the yellow one.

# Smallest is the "span" value --> better is the interpolation

# --------------------------------------------------------------------- #
# CUBIC SPLINES

x <- 1:204
y <- egizio_train_df$visitors

# We may select the internal-knots by using the degrees of freedom: 

# (basic functions b-spline for a cubic spline (degree=3))

# --> df directly related to the number of knots
#     df = length(knots) + degree 
# The knots are selected by using the quantiles of 'x' 
# distribution 

ggplot(data = egizio_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Visitors")

# --------------------------------------------------------------------- #
# DEGREE 3

# Starting with 2 internal-knots
splines3_2 <- lm(y ~ bs(x, df = 5, degree = 3)) 
summary(splines3_2) # 0.2347
# 2-5 not significant
# 1-3-4 significant
ggplot(data = egizio_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_line(aes(y = splines3_2$fitted.values, color = "red"))

# Proceeding with 4 internal-knots
splines3_4 <- lm(y ~ bs(x, df = 7, degree = 3)) 
summary(splines3_4) # 0.3328
# 1-2-4 not significant
# 3-5-6-7 significant
ggplot(data = egizio_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_line(aes(y = splines3_4$fitted.values, color = "red"))

# Model with no internal-knots
splines3_0 <- lm(y ~ bs(x, df = 3, degree = 3)) 
summary(splines3_0) # 0.1787
# 2 significant
ggplot(data = egizio_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_line(aes(y = splines3_0$fitted.values, color = "red"))

# Model with 8 internal-knots
splines3_8 <- lm(y ~ bs(x, df = 11, degree = 3)) 
summary(splines3_8)
ggplot(data = egizio_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_line(aes(y = splines3_8$fitted.values, color = "red"))

# Best with degree = 3 --> 8 internal knots, df = 11

# --------------------------------------------------------------------- #
# DEGREE 4

# Starting with 2 internal-knots
splines4_2 <- lm(y ~ bs(x, df = 6, degree = 4)) 
summary(splines4_2) # 0.3056 
# 2-3-4 significant
# 1-5-6 not significant
ggplot(data = egizio_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_line(aes(y = splines4_2$fitted.values, color = "red"))

# Proceeding with 4 internal-knots
splines4_4 <- lm(y ~ bs(x, df = 8, degree = 4)) 
summary(splines4_4) # 0.3647 
# 1-2-3-4-5 not significant
# 6-7-8 significant
ggplot(data = egizio_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") + 
  ggtitle("Loess Model for Visitors") +
  geom_line(aes(y = splines4_4$fitted.values, color = "red"))

# Model with no internal-knots
splines4_0 <- lm(y ~ bs(x, df = 4, degree = 4)) 
summary(splines4_0) # 0.2224
# 1-2-3 significant
# 4 not significant
ggplot(data = egizio_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Visitors") +
  geom_line(aes(y = splines4_0$fitted.values, color = "red"))

# Model with 8 internal-knots
splines4_8 <- lm(y ~ bs(x, df = 12, degree = 4)) 
summary(splines4_8) # 0.3717
# 1-2-3-4-5-6-8-10-11 not significant
# 7-9-12 significant
ggplot(data = egizio_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_line(aes(y = splines4_8$fitted.values, color = "red"))

# With degree = 4 we are not able to find something better than
# the model we would choose with degree = 3.
# ToDo(Anna): Need to think.

# ToDo (Dejan): Use the result from here instead of GBM in Model Decomposition.

# --------------------------------------------------------------------- #
## SMOOTHING SPLINES

# Basic smooth splines

sm_spline <- ss(x,y, method = "AIC")
ggplot(data = egizio_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_line(aes(y = sm_spline$y, color = "red"))

# Model 1
sm_spline1 = ss(x,y, method = "AIC", lambda = 0.000001, all.knots = T)
ggplot(data = egizio_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_line(aes(y = sm_spline1$y, color = "red"))


# Model with Cross Validation Method for parameter selection
sm_spline_gcv <- ss(x,y, method = "GCV")
ggplot(data = egizio_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_line(aes(y = sm_spline_gcv$y, color = "red"))

sm_spline_gcv

# ToDo (Anna): Select the best model. Perform forecasting and calculate metrics.

# --------------------------------------------------------------------- #
# Model 13 - Boosting

# Modify graphical parameters
mai.old <- par()$mai
mai.new <- mai.old # new vector
mai.new[2] <- 2.5 # new space on the left

# This can be used visitors ~ .- visitors - date + as.numeric(date)
boost_visitors <- gbm(visitors ~ . - date - date_numeric, data=egizio_train_df, 
                      distribution="gaussian", n.trees=5000, interaction.depth=1)

par(mai=mai.new)
summary(boost_visitors, las=1, cBar=20)
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
boost_visitors <- gbm(visitors ~ . - date - date_numeric, data=egizio_train_df,
                      distribution="gaussian", n.trees=5000,
                      interaction.depth=4) # (with more than one variable)

par(mai=mai.new)
summary(boost_visitors, las=1, cBar=20)
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
boost_visitors <- gbm(visitors ~ . - date - date_numeric, data=egizio_train_df,
                      distribution="gaussian", n.trees=5000, interaction.depth=1,
                      shrinkage=0.01) # learning rate

par(mai=mai.new)
summary(boost_visitors, las=1, cBar=20)
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
boost_visitors <- gbm(visitors ~ . - date - date_numeric, data=egizio_train_df,
                      distribution="gaussian", n.trees=5000,
                      interaction.depth=4, shrinkage=0.01)
par(mai=mai.new)
summary(boost_visitors, las=1, cBar=20)
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

# MSE-=1.31 without using lagged variables

run_cross_validation <- FALSE

if (run_cross_validation) {
  # I started with this grid, and then modified it many times,
  # to expand the search (where it performs good.)
  grid_boosting <- expand.grid(n.trees = c(50, 100, 500, 1000, 5000),
                               interaction.depth = c(1,4,8), 
                               shrinkage = c(0.1, 0.05, 0.01, 0.001),
                               n.minobsinnode = c(2, 5, 10))
  
  train_controls <- trainControl(method = "timeslice",# time-series cross-validation
                                 initialWindow = 48, # initial training window
                                 horizon = 12, # forecast evaluation window
                                 fixedWindow = TRUE, 
                                 skip = 12,
                                 allowParallel = TRUE) # allow parallel processing if available
  
  gbm_grid <- train(visitors ~ . - date - date_numeric,
                    data = egizio_train_df,
                    method = "gbm",  
                    distribution = "gaussian",
                    trControl = train_controls,
                    tuneGrid = grid_boosting,
                    metric = "RMSE",
                    verbose = FALSE)
  
  # View the results of the grid search
  print(gbm_grid)
  
  best_model_boosting <- gbm_grid$bestTune
  
  final_model_boosting <- gbm(visitors ~ . - date + date_numeric,
                              data = egizio_train_df,
                              distribution = "gaussian",
                              n.trees = best_model_boosting$n.trees,
                              interaction.depth = best_model_boosting$interaction.depth,
                              shrinkage = best_model_boosting$shrinkage,
                              n.minobsinnode = best_model_boosting$n.minobsinnode)
} else {
  final_model_boosting <- gbm(visitors ~ . - date + date_numeric - year - month,
                              data = egizio_train_df,
                              distribution = "gaussian",
                              n.trees = 200,
                              interaction.depth = 7,
                              shrinkage = 0.05, # 0.001
                              n.minobsinnode = 4)
}

par(mai=mai.new)
summary(final_model_boosting, las=1, cBar=20)
par(mai=mai.old)

egizio_predictions_df$predicted_visitors_boosting <- predict(final_model_boosting,
                                                             newdata = egizio_test_df,
                                                             n.trees = final_model_boosting$n.trees)

egizio_training_preds <- predict(final_model_boosting,
                                 newdata = egizio_train_df,
                                 n.trees = final_model_boosting$n.trees)
# Calculate metrics
r_squared <- RSQUARE(egizio_train_df$visitors, egizio_training_preds)
adj_r_squared <- adjusted_R2(egizio_train_df$visitors, egizio_training_preds, nrow(egizio_train_df), length(final_model_boosting$var.names))
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

# partial dependence plots
# ToDo (Dejan): Move this after the cross-validated model.
plot(final_model_boosting, i.var=1, n.trees = final_model_boosting$n.trees, ylab = "visitors")
plot(final_model_boosting, i.var=2, n.trees = final_model_boosting$n.trees, ylab = "visitors")
plot(final_model_boosting, i.var=3, n.trees = final_model_boosting$n.trees, ylab = "visitors")
plot(final_model_boosting, i.var=4, n.trees = final_model_boosting$n.trees, ylab = "visitors")
plot(final_model_boosting, i.var=5, n.trees = final_model_boosting$n.trees, ylab = "visitors")
plot(final_model_boosting, i.var=6, n.trees = final_model_boosting$n.trees, ylab = "visitors")
plot(final_model_boosting, i.var=7, n.trees = final_model_boosting$n.trees, ylab = "visitors")
plot(final_model_boosting, i.var=8, n.trees = final_model_boosting$n.trees, ylab = "visitors")
plot(final_model_boosting, i.var=9, n.trees = final_model_boosting$n.trees, ylab = "visitors")
plot(final_model_boosting, i.var=c(2,4), n.trees = final_model_boosting$n.trees)

# --------------------------------------------------------------------- #
# Model 14 - XGBoost - no CV
training.x <- model.matrix(visitors ~ . - date - date_numeric,
                           data = egizio_train_df)
testing.x <- model.matrix(visitors ~ . - date - date_numeric,
                          data = egizio_test_df)

xgb_model <- xgboost(data=data.matrix(training.x[,-1]), # ignore intercept
                     label=as.numeric(as.character(egizio_train_df$visitors)),
                     eta=0.025, # default=0.3 - takes values in (0-1]
                     max_depth=6, # default=6 - takes values in (0,Inf), larger value => more complex => overfitting
                     nrounds=500, # default=100 - controls number of iterations (number of trees)
                     early_stopping_rounds=50,
                     print_every_n = 10,
                     objective="reg:squarederror", # for linear regression
                     verbose=FALSE) 
# objective="reg:squarederror"
# eval_metric = "rmse"

importance_scores <- xgb.importance(model = xgb_model)
print(importance_scores)
xgb.plot.importance(importance_matrix = importance_scores)

egizio_predictions_df$predicted_visitors_xgboost <- predict(xgb_model, newdata = testing.x[,-1])

# Calculate metrics for XGBoost
train_predicted_visitors_xgboost <- predict(xgb_model, newdata = training.x[,-1])
r_squared <- RSQUARE(egizio_train_df$visitors, train_predicted_visitors_xgboost)
adj_r_squared <- adjusted_R2(egizio_train_df$visitors, train_predicted_visitors_xgboost, nrow(egizio_train_df), xgb_model$nfeatures)
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
# Model 15 - XGboost with Cross-validation

# The following cross validation has been performed.
# To avoid running it again, I added it in an IF-ELSE statement.
# By default we will execute the ELSE part, which uses the results
# obtained during the extensive cross-validation.

run_cross_validation <- FALSE

if (run_cross_validation) {
  train_controls <- trainControl(method = "timeslice",
                                 initialWindow = 48,
                                 horizon = 12,
                                 fixedWindow = TRUE,
                                 skip = 12,
                                 allowParallel = TRUE)
  
  # Initial grid
  xgboost_grid <- expand.grid(nrounds = c(100, 200, 300),
                              max_depth = c(5, 7, 10),
                              eta = c(0.01, 0.025, 0.05, 0.075, 0.1),
                              gamma = c(0, 0.1, 0.2),
                              colsample_bytree = c(0.8, 1),
                              min_child_weight = c(1, 3, 5),
                              subsample = c(0.8, 1))
  # Result without lagged regressors:
  # The final values used for the model were nrounds = 300, max_depth = 7, eta = 0.05,
  # gamma = 0.1, colsample_bytree = 0.8, min_child_weight = 5 and subsample = 0.8.
  
  xgb_model <- train(x = data.matrix(training.x[, -1]), # Ignore intercept
                     y = as.numeric(as.character(egizio_train_df$visitors)),
                     method = "xgbTree", # XGBoost
                     trControl = train_controls,
                     tuneGrid = xgboost_grid,
                     verbose = FALSE,
                     metric = "RMSE")
  
  print(xgb_model)
  
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
                             objective="reg:squarederror",
                             verbose=FALSE)
  
} else {
  final_model_xgb <- xgboost(data=data.matrix(training.x[,-1]),
                             label=egizio_train_df$visitors,
                             eta=0.05,
                             max_depth=7,
                             nrounds=300,
                             colsample_bytree=0.8,
                             min_child_weight=5,
                             subsample=0.8,
                             gamma=0.1,
                             objective="reg:squarederror",
                             verbose=FALSE)
}

importance_scores <- xgb.importance(model = final_model_xgb)
print(importance_scores)
xgb.plot.importance(importance_matrix = importance_scores)

# Perform predictions on the test set
egizio_predictions_df$predicted_visitors_xgboost_tscv <- predict(final_model_xgb,
                                                                 newdata = data.matrix(testing.x[, -1]))

# Calculate metrics for XGBoost
egizio_training_xgboost_preds <- predict(final_model_xgb,
                                 newdata = data.matrix(training.x[, -1]))
# Calculate metrics
r_squared <- RSQUARE(egizio_train_df$visitors, egizio_training_xgboost_preds)
adj_r_squared <- adjusted_R2(egizio_train_df$visitors, egizio_training_xgboost_preds, nrow(egizio_train_df), final_model_xgb$nfeatures)
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

# --------------------------------------------------------------------- #
# Model 16 - Combined Model: GBM with Rectangular Shock + ARMAX for residuals

res_gbm <- make.instantaneous(best_gbm$residuals)
Acf(res_gbm)
Pacf(res_gbm)

# arima_res_gbm <- auto.arima(res_gbm, xreg = regressors_train)
arima_res_gbm <- Arima(res_gbm, xreg = regressors_train,
                 order = c(1,0,12), seasonal = c(0,1,1), include.drift = TRUE)
summary(arima_res_gbm)

ggplot(data = egizio_train_df, aes(x = date, y = res_gbm)) +
  geom_point(color = "black") +
  geom_line(aes(y = arima_res_gbm$fitted), color = "red") +
  xlab("Date") +
  ylab("Residuals") +
  ggtitle("Real residuals vs. Fitted values with ARMAX")

ggplot(data = egizio_train_df, aes(x = date, y = as.numeric(arima_res_gbm$residuals))) +
  geom_point(color = "blue") + xlab("Date") + ylab("Residuals") +
  ggtitle("Residuals Plot of ARMAX Model for Residuals")

vis_fitted_gbm <- make.instantaneous(best_gbm$fitted) + arima_res_gbm$fitted
vis_fitted_gbm <- (vis_fitted_gbm - mean(egizio_train_df_copy$visitors)) / sd(egizio_train_df_copy$visitors)

ggplot(data = egizio_train_df, aes(x = date, y = visitors)) +
  geom_point(color = "black") +
  geom_line(aes(y = vis_fitted_gbm), color = "red") +
  xlab("Date") +
  ggtitle("Visitors vs. Fitted values with Combined Model: GBM + ARMAX")

# The model above doesn't work well, therefore it won't be used for forecasting.

# --------------------------------------------------------------------- #
# Model 17 - Decomposition: GBM + TSLM + Boosting

egizio_visitors_unstandardized_train_ts <- ts(egizio_train_df_copy$visitors, frequency = 12)
egizio_visitors_comp <- decompose(egizio_visitors_unstandardized_train_ts)
plot(egizio_visitors_comp)

visitorSeasonAdj <- egizio_visitors_unstandardized_train_ts - egizio_visitors_comp$seasonal
plot.ts(visitorSeasonAdj)
# We obtain just the trend + random components

# For modelling the trend we will use the best GBM from above.

# We will use the egizio_train_unstandardized_df$visitors directly.
# To begin with, we just focus our attention to model the general trend.

# 1. Modelling the trend
plot.ts(egizio_visitors_comp$trend, ylab="Trend")

# Let's try with GBM
# The best candidates were: exp + rectangular and two rectangular.
# Decided to proceed with two rectangular shocks.
res_gbm <- make.instantaneous(best_gbm$residuals)
plot.ts(res_gbm) # Residuals when fitting a trend
lines(as.numeric(egizio_visitors_unstandardized_train_ts - egizio_visitors_comp$trend), col=2)

pred_gbm_visitors <- predict(best_gbm, newx = 1:216)
pred_inst_gbm_visitors <- make.instantaneous(pred_gbm_visitors)
plot.ts(pred_inst_gbm_visitors[1:204])
lines(as.numeric(egizio_visitors_comp$trend), col=2)

plot.ts(pred_inst_gbm_visitors[1:204] - as.numeric(egizio_visitors_comp$trend))
# Mostly the outliers are left and noise.

# 2. Model the seasonality
res_gbm_ts <- ts(res_gbm, frequency = 12)
tslm_res_gbm <- tslm(res_gbm_ts ~ season)
summary(tslm_res_gbm)

ggplot(egizio_train_unstandardized_df, aes(x = date)) +
  geom_line(aes(y = visitors, color = "Visitors"), size = 1) +
  geom_line(aes(y = tslm_res_gbm$fitted.values + pred_inst_gbm_visitors[1:204], 
                color = "Predicted"), linetype = "dashed", size = 1) +
  labs(title = "Visitors and Predicted Values Over Time", x = "Date", y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))


# 3. Random Noise
res_res <- tslm_res_gbm$residuals

# Standardization is not necessary
egizio_train_unstandardized_df$res <- as.numeric(res_res)

run_cross_validation <- FALSE

if (run_cross_validation) {
  grid_boosting_res <- expand.grid(n.trees = c(50, 100, 500, 1000, 5000),
                                   interaction.depth = c(1,4,8), 
                                   shrinkage = c(0.3, 0.1, 0.05, 0.01),
                                   n.minobsinnode = c(2, 5, 10))
  
  train_controls <- trainControl(method = "timeslice",
                                 initialWindow = 48,
                                 horizon = 12,
                                 fixedWindow = TRUE, 
                                 skip = 12,
                                 allowParallel = TRUE)
  
  gbm_grid_res <- train(res ~ . - date - date_numeric - visitors, # now we model the residuals using the other variables
                        data = egizio_train_unstandardized_df,
                        method = "gbm",  
                        distribution = "gaussian",
                        trControl = train_controls,
                        tuneGrid = grid_boosting_res,
                        verbose = FALSE)
  
  # View the results of the grid search
  print(gbm_grid_res)
  
  best_model_boosting_res <- gbm_grid_res$bestTune
  
  final_model_boosting_res <- gbm(res ~ . - date - date_numeric - visitors,
                                  data = egizio_train_unstandardized_df,
                                  distribution = "gaussian",
                                  n.trees = best_model_boosting_res$n.trees,
                                  interaction.depth = best_model_boosting_res$interaction.depth,
                                  shrinkage = best_model_boosting_res$shrinkage,
                                  n.minobsinnode = best_model_boosting_res$n.minobsinnode)
} else {
  final_model_boosting_res <- gbm(res ~ . - date - date_numeric - visitors,
                                  data = egizio_train_unstandardized_df,
                                  distribution = "gaussian",
                                  n.trees = 50,
                                  interaction.depth = 8,
                                  shrinkage = 0.01,
                                  n.minobsinnode = 2)
}

par(mai=mai.new)
summary(final_model_boosting_res, las=1, cBar=20)
par(mai=mai.old)

plot.ts(egizio_train_unstandardized_df$res)
lines(final_model_boosting_res$fit, col=2)
# We can see that the model doesn't overfit, which is good.
# But there is still some variability to be explained, due to outliers.

residuals <- egizio_train_unstandardized_df$visitors - final_model_boosting_res$fit
checkresiduals(remainder(decompose(ts(residuals, frequency = 12))))

# Can we use this model for forecasting? Yes.
trend_predictions <- pred_inst_gbm_visitors[205:216]
seasonality_prediction <- forecast(tslm_res_gbm, h=12)$mean
residual_predictions <- predict(final_model_boosting_res,
                                newdata=egizio_test_df_copy, # unstandardized
                                n.trees=final_model_boosting_res$n.trees)
final_predictions <- trend_predictions + seasonality_prediction + residual_predictions

# Calculate metrics
egizio_predictions_df$predicted_visitors_decomposed <- as.numeric(final_predictions)
egizio_predictions_df$predicted_visitors_decomposed <- (egizio_predictions_df$predicted_visitors_decomposed - mean(egizio_train_unstandardized_df$visitors)) / sd(egizio_train_unstandardized_df$visitors)

mse <- mse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_decomposed)
rmse <- rmse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_decomposed)
mae <- mae(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_decomposed)
mape <- mape(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_decomposed)

metrics_df <- rbind(metrics_df, list(Model = "Decomposed: GBM+TLSM+Boosting",
                                     R2 = NA, R2_adj = NA,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = NA))
print(metrics_df)

ggplot(egizio_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Visitors"), linewidth = 1) +
  geom_line(aes(y = predicted_visitors_decomposed, color = "Predicted"),
            linetype = "dashed", linewidth = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))

# --------------------------------------------------------------------- #
#                         COVID analysis
# --------------------------------------------------------------------- #
# The idea here is to see whether treating COVID months as outliers would help.
# We start by removing the COVID months and then fitting a model on this time
# series. Then we make forecasts for the removed months.
# Finally we create a new artificial training dataset, consisting of:
# original data pre COVID + forecasted data for COVID months.
# On this dataset a model is fit to forecast for 2022, i.e. our original test set.
# Then the forecasts are compared with the true data, but also the predictions
# from our best model from before, that uses the original training dataset.

# New data frame for storing the metrics of this analysis:
covid_metrics_df <- data.frame(
  Model = character(),
  R2 = numeric(),
  Adj_R2 = numeric(),
  MSE = numeric(),
  RMSE = numeric(),
  MAE = numeric(),
  MAPE = numeric(),
  AIC = numeric(),
  stringsAsFactors = FALSE)

min_mse_index <- which.min(metrics_df$MSE)
best_model_description <- metrics_df$Model[min_mse_index]
best_model_row <- subset(metrics_df, Model == best_model_description)
best_model_row$Model <- "Best model"
covid_metrics_df <- rbind(covid_metrics_df, best_model_row)
print(covid_metrics_df)

# To create a new artificial dataset, we first drop the COVID months.
egizio_train_no_covid_ds <- egizio_train_df[1:182,]

# Find a good model to predict the COVID months
egizio_train_no_covid_visitors_ts <- ts(egizio_train_no_covid_ds$visitors, frequency = 12)
# Different models were tried. Their metrics are shown below.
# The best model was the following:
sarima_no_covid <- Arima(egizio_train_no_covid_visitors_ts, order = c(1,0,12),
                         seasonal = c(0,1,2), include.drift=TRUE)
summary(sarima_no_covid) # AIC=210.52, RMSE=0.3710264
train_predictions_sarima_no_covid <- fitted(sarima_no_covid)

# This model seems to work better, but it's wrong.
# If we include the regressors, they also contain COVID data, so the end of
# our series will be underestimated, i.e., it will look similar to COVID.
sarimax_no_covid <- Arima(egizio_train_no_covid_visitors_ts, order = c(1,0,12),
                          seasonal = c(0,1,2), include.drift=TRUE,
                          xreg=regressors_train[1:182,])
summary(sarimax_no_covid) # AIC=125.18, RMSE=0.2856033
train_predictions_sarimax_no_covid <- fitted(sarimax_no_covid)

# Visualize the predictions for the COVID months:
plot.ts(egizio_train_no_covid_visitors_ts)
lines(train_predictions_sarima_no_covid,col=2)
lines(train_predictions_sarimax_no_covid, col=3) # this one is much better -> but it's wrong!

# We proceed with model: sarima_no_covid.

# Calculate metrics
train_r_squared <- RSQUARE(egizio_train_no_covid_visitors_ts, train_predictions_sarima_no_covid)
train_adj_r_squared <- adjusted_R2(egizio_train_no_covid_visitors_ts, train_predictions_sarima_no_covid, length(egizio_train_no_covid_visitors_ts), length(coef(sarima_no_covid)))
train_mse <- mse(egizio_train_no_covid_visitors_ts, train_predictions_sarima_no_covid)
train_rmse <- rmse(egizio_train_no_covid_visitors_ts, train_predictions_sarima_no_covid)
train_mae <- mae(egizio_train_no_covid_visitors_ts, train_predictions_sarima_no_covid)
train_mape <- mape(egizio_train_no_covid_visitors_ts, train_predictions_sarima_no_covid)
train_aic <- AIC(sarima_no_covid)
cat(train_r_squared, train_adj_r_squared, train_mse, train_rmse, train_mae, train_mape, train_aic)
# The following models were tried:
# ARIMA(1,0,0)(0,1,1)[12]  with drift  -> auto.arima
# ARIMA(1,0,0)(0,1,1)[12]
# ARIMA(0,1,0)(0,0,2)[12] -> worst
# ARIMA(2,1,0)(0,0,2)[12]
# ARIMA(0,0,2)(0,1,2)[12]
# ARIMA(1,0,1)(0,1,1)[12]
# ARIMA(1,0,1)(1,1,1)[12]
# ARIMA(1,0,1)(1,1,1)[12] with drift 
# ARIMA(1,0,12)(0,1,2)[12] with drift  -> best
# ARIMA(1,0,12)(0,1,2)[12] 
# ARIMA(1,0,12)(0,1,1)[12]
# ARIMA(1,0,12)(0,1,1)[12] with drift

# After finding the best model, use it for forecasting, and
# add the forecasts to the artificial training dataset.
predicted_visitors_sarima_no_covid <- forecast(sarima_no_covid, h=22)
plot(predicted_visitors_sarima_no_covid)

egizio_train_visitors_covid_replaced <- egizio_train_df
egizio_train_visitors_covid_replaced$visitors[183:nrow(egizio_train_df)] <- predicted_visitors_sarima_no_covid$mean

plot(egizio_train_visitors_covid_replaced$date,
     egizio_train_visitors_covid_replaced$visitors,
     type='l', xlab="Date", ylab="Visitors")
lines(egizio_train_df$date[182:204], egizio_train_df$visitors[182:204], col="red")

# Refit the same model from above (used for training) on the whole
# new artificial training time series egizio_train_visitors_covid_replaced:
egizio_train_visitors_covid_replaced_ts <- ts(egizio_train_visitors_covid_replaced$visitors, frequency=12)
sarimax_covid_replaced <- Arima(egizio_train_visitors_covid_replaced_ts, order = c(1,0,12),
                               seasonal = c(0,1,2), include.drift=TRUE,
                               xreg=regressors_train)
summary(sarimax_covid_replaced)

egizio_train_visitors_covid_replaced$visitors_predictions <- fitted(sarimax_covid_replaced)

# Visualize how it fits the new training data:
ggplot(data = egizio_train_visitors_covid_replaced, aes(x = date, y = visitors)) +
  geom_line(color = "blue") +  
  geom_line(aes(y = visitors_predictions), color = "red", linetype = "twodash") +
  labs(title = "True vs Fitted values by SARIMAX")

# Visualize the residuals:
ggplot(aes(date, y = as.numeric(residuals(sarimax_covid_replaced))),
       data = egizio_train_visitors_covid_replaced) +
  geom_point(color = "blue") + xlab("Date") + ylab("Residuals") + ggtitle("Residuals of SARIMAX")

checkresiduals(sarimax_covid_replaced)

# Finally, let's use it for forecasting:
pred_sarimax_covid_replaced <- forecast(sarimax_covid_replaced, h = 12, xreg=regressors_test)
egizio_predictions_df$predicted_visitors_sarimax_covid_replaced <- pred_sarimax_covid_replaced$mean

# Calculate metrics on the test set (2022):
r_squared <- RSQUARE(egizio_train_visitors_covid_replaced$visitors, egizio_train_visitors_covid_replaced$visitors_predictions)
adj_r_squared <- adjusted_R2(egizio_train_visitors_covid_replaced$visitors, egizio_train_visitors_covid_replaced$visitors_predictions, length(egizio_train_visitors_covid_replaced$visitors), length(coef(sarimax_covid_replaced)))
mse <- mse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_sarimax_covid_replaced)
rmse <- rmse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_sarimax_covid_replaced)
mae <- mae(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_sarimax_covid_replaced)
mape <- mape(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_sarimax_covid_replaced)
aic <- AIC(sarimax_covid_replaced)

covid_metrics_df <- rbind(covid_metrics_df, list(Model = "COVID interpolated - Training model",
                                                 R2 = r_squared, R2_adj = adj_r_squared,
                                                 MSE = mse, RMSE = rmse, MAE = mae,
                                                 MAPE = mape, AIC = aic))
print(covid_metrics_df)

ggplot(egizio_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "True"), linewidth = 1) +
  geom_line(aes(y = predicted_visitors_sarima, color = "Original predictions"),
            linetype = "dashed", linewidth = 1) +
  geom_line(aes(y = predicted_visitors_sarimax_covid_replaced, color = "COVID Interpolated predictions"),
            linetype = "dashed", linewidth = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("True"="red",
                                "Original predictions" = "green",
                                "COVID Interpolated predictions" = "blue"))

# This model performs worse. Let's try to improve it.

# --------------------------------------------------------------------- #
# Improved model

sarima_covid_replaced_improved <- auto.arima(egizio_train_visitors_covid_replaced_ts)
summary(sarima_covid_replaced_improved)

egizio_train_visitors_covid_replaced$visitors_predictions_improved <- fitted(sarima_covid_replaced_improved)

# Visualize how it fits the new training data
ggplot(data = egizio_train_visitors_covid_replaced, aes(x = date, y = visitors)) +
  geom_line(color = "blue") +  
  geom_line(aes(y = visitors_predictions_improved), color = "red", linetype = "twodash") +
  labs(title = "True vs Fitted values by ARIMA")

# Visualize the residuals
ggplot(aes(date, y = residuals(sarima_covid_replaced_improved)), data = egizio_train_visitors_covid_replaced) +
  geom_point(color = "blue") + xlab("Date") + ylab("Residuals") + ggtitle("Residuals of Arima")

checkresiduals(sarima_covid_replaced_improved)

# --------------------------------------------------------------------- #
# Let's use it for forecasting:
pred_sarima_covid_replaced_improved <- forecast(sarima_covid_replaced_improved, h = 12)
egizio_predictions_df$predicted_visitors_sarima_covid_replaced_improved <- pred_sarima_covid_replaced_improved$mean

# Calculate metrics
r_squared <- RSQUARE(egizio_train_visitors_covid_replaced$visitors, egizio_train_visitors_covid_replaced$visitors_predictions_improved)
adj_r_squared <- adjusted_R2(egizio_train_visitors_covid_replaced$visitors, egizio_train_visitors_covid_replaced$visitors_predictions_improved, length(egizio_train_visitors_covid_replaced$visitors), length(coef(sarima_covid_replaced_improved)))
mse <- mse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_sarima_covid_replaced_improved)
rmse <- rmse(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_sarima_covid_replaced_improved)
mae <- mae(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_sarima_covid_replaced_improved)
mape <- mape(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_sarima_covid_replaced_improved)
aic <- AIC(sarima_covid_replaced_improved)

covid_metrics_df <- rbind(covid_metrics_df, list(Model = "COVID Interpolated Improved",
                                                 R2 = r_squared, R2_adj = adj_r_squared,
                                                 MSE = mse, RMSE = rmse, MAE = mae,
                                                 MAPE = mape, AIC = aic))
print(covid_metrics_df)

ggplot(egizio_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "True"), linewidth = 1) +
  geom_line(aes(y = predicted_visitors_sarima, color = "Original predictions"),
            linetype = "dashed", linewidth = 1) +
  geom_line(aes(y = predicted_visitors_sarima_covid_replaced_improved, color = "COVID Interpolated predictions"),
            linetype = "dashed", linewidth = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("True"="red",
                                "Original predictions" = "green",
                                "COVID Interpolated predictions" = "blue"))
# Observation:
# Auto.arima performs better, but still the results are better if we use interpolated data for COVID.
# This is due to the fact that after COVID, the peaks aren't nearly as tall as before COVID.
# Any good model will predict tall peaks, so it will give a higher MSE.
# => No more experiments are needed.

