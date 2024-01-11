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
library(patchwork)

# Change working directory
script_path <- rstudioapi::getSourceEditorContext()$path
script_dir <- dirname(script_path)
setwd(script_dir)

# To use English for the dates (instead of Macedonian/Italian)
Sys.setlocale("LC_TIME", "English")

set.seed(123)

# ---------------------------------------------------------------------------- #
# Load dataset

# Run preprocessing.R if the file doesn't exist
if (!file.exists("../../data/cinema_final.rds")) {
  source("preprocessing_cinema.R")
}

cinema_df <- readRDS("../../data/cinema_final.rds")
print(head(cinema_df))
str(cinema_df)

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

# ---------------------------------------------------------------------------- #
# Train-test split

# Due to the huge variation during COVID, we need to avoid using it in the test set.

cinema_train_df <- subset(cinema_df, format(date, "%Y") != "2022")
cinema_test_df <- subset(cinema_df, format(date, "%Y") == "2022")

cat("Cinema train size:", nrow(cinema_train_df), "rows (months).")
cat("Cinema test size:", nrow(cinema_test_df), "rows (months).")

ratio_train <- nrow(cinema_train_df) / nrow(cinema_df)
ratio_test <- 1 - ratio_train
ratio_train <- ratio_train * 100
ratio_test <- ratio_test * 100
cat("Ratio of train set size to test set size:", ratio_train, ":", ratio_test)

# This dataframe will be used to store the predictions of all of the models, and make plotting easier.
cinema_predictions_df <- data.frame(date = cinema_test_df$date)
cinema_predictions_df$visitors_true <- cinema_test_df$visitors

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
cinema_train_df_copy <- data.frame(cinema_train_df)
cinema_test_df_copy <- data.frame(cinema_test_df)

if (perform_standardization) {
  standardize <- standardize_numeric_columns(cinema_train_df, cinema_test_df)
  cinema_train_df <- standardize$train_df
  cinema_test_df <- standardize$test_df
  print(head(cinema_train_df))
  print(head(cinema_test_df))
}

cinema_predictions_df$visitors_true <- cinema_test_df$visitors

# Plot visitors and trends on the same graph (and show separator between train and test)
ggplot() +
  geom_line(data = cinema_train_df, aes(x = date, y = visitors, color = "Visitors", linetype = "Train"), linewidth = 1) +
  geom_line(data = cinema_test_df, aes(x = date, y = visitors, color = "Visitors", linetype = "Test"), linewidth = 1) +
  geom_line(data = cinema_train_df, aes(x = date, y = trends, color = "Google Trends", linetype = "Train"), linewidth = 1) +
  geom_line(data = cinema_test_df, aes(x = date, y = trends, color = "Google Trends", linetype = "Test"), linewidth = 1) +
  labs(title = "Visitors and Google Trends over Time", x = "Date", y = "Values") +
  scale_color_manual(name = NULL, values = c("Visitors" = "red", "Google Trends" = "blue")) +
  scale_linetype_manual(name = "Dataset", values = c("Train" = "solid", "Test" = "solid")) +
  geom_vline(xintercept = as.numeric(min(cinema_test_df$date)), linetype = "dashed", color = "black", linewidth = 1.5) +  # Increase line thickness
  theme_minimal() + theme(legend.position = c(0.85, 0.95)) + 
  guides(linetype = "none")

# Show them on separate plots
plot_visitors <- ggplot() +
  geom_line(data = cinema_train_df, aes(x = date, y = visitors, color = "Train - Visitors", linetype = "Train"), linewidth = 1) +
  geom_line(data = cinema_test_df, aes(x = date, y = visitors, color = "Test - Visitors", linetype = "Test"), linewidth = 1) +
  labs(title = "Visitors over Time", x = "Date", y = "Visitors") +
  scale_color_manual(name = "Variable", values = c("Train - Visitors" = "red", "Test - Visitors" = "darkred")) +
  scale_linetype_manual(name = "Dataset", values = c("Train" = "solid", "Test" = "dashed")) +
  geom_vline(xintercept = as.numeric(min(cinema_test_df$date)), linetype = "dotted", color = "black") +
  theme_minimal() + guides(linetype = "none") +
  theme(legend.position = c(0.95,0.99), legend.title = element_blank())

plot_trends <- ggplot() +
  geom_line(data = cinema_train_df, aes(x = date, y = trends, color = "Train - Google Trends", linetype = "Train"), linewidth = 1) +
  geom_line(data = cinema_test_df, aes(x = date, y = trends, color = "Test - Google Trends", linetype = "Test"), linewidth = 1) +
  labs(title = "Google Trends over Time", x = "Date", y = "Google Trends") +
  scale_color_manual(name = "Variable", values = c("Train - Google Trends" = "blue", "Test - Google Trends" = "darkblue")) +
  scale_linetype_manual(name = "Dataset", values = c("Train" = "solid", "Test" = "dashed")) +
  geom_vline(xintercept = as.numeric(min(cinema_test_df$date)), linetype = "dotted", color = "black") +
  theme_minimal() + theme(legend.position = c(0.95, 0.99), legend.title = element_blank()) +
  guides(linetype = "none")

arranged_plots <- plot_visitors / plot_trends
arranged_plots

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

mean_train_visitors <- mean(cinema_train_df$visitors)
cinema_predictions_df$predicted_visitors_mean <- mean_train_visitors

# Calculate metrics

# We don't calculate R2 and Adj_R2 because this model has 0 standard deviation.
# We don't capture any variation.

mse <- mse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_mean)
rmse <- rmse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_mean)
mae <- mae(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_mean)
mape <- mape(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_mean)

# For similar reasoning, we don't calculate AIC.

metrics_df <- rbind(metrics_df, list(Model = "Baseline - mean",
                                     R2 = NA, R2_adj = NA,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = NA))
print(metrics_df)

ggplot(cinema_predictions_df, aes(x = date)) +
  geom_line(aes(y = predicted_visitors_mean, color = "Predicted"),
            linetype = "dashed", linewidth = 1) +
  geom_line(aes(y = visitors_true, color = "Actual"), linewidth = 1) +
  labs(title = "Baseline - training mean",
       x = "Date",
       y = "Visitors") +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))

# ---------------------------------------------------------------------------- #
# Baseline model - same month last year

cinema_train_df_old <- data.frame(cinema_train_df)
cinema_test_df_old <- data.frame(cinema_test_df)

cinema_test_df$visitors_same_month_last_year <- tail(cinema_train_df$visitors, 12)
cinema_train_df$visitors_same_month_last_year <- lag(cinema_train_df$visitors, 12)
cinema_train_df <- na.omit(cinema_train_df)

lm_last_year <- lm(visitors ~ visitors_same_month_last_year, data = cinema_train_df)
summary(lm_last_year)

cinema_predictions_df$predicted_visitors_last_year <- predict(lm_last_year, newdata = cinema_test_df)

# Calculate metrics
r_squared <- summary(lm_last_year)$r.squared
adj_r_squared <- summary(lm_last_year)$adj.r.squared
aic <- AIC(lm_last_year)
mse <- mse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_last_year)
rmse <- rmse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_last_year)
mae <- mae(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_last_year)
mape <- mape(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_last_year)

metrics_df <- rbind(metrics_df, list(Model = "Baseline - last year",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic))
print(metrics_df)

ggplot(cinema_predictions_df, aes(x = date)) +
  geom_line(aes(y = predicted_visitors_last_year, color = "Predicted"),
            linetype = "dashed", linewidth = 1) +
  geom_line(aes(y = visitors_true, color = "Actual"), linewidth = 1) +
  labs(title = "Auto regressive Baseline - same month from previous year",
       x = "Date", y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))

cinema_train_df <- data.frame(cinema_train_df_old)
cinema_test_df <- data.frame(cinema_test_df_old)

# ---------------------------------------------------------------------------- #
# Model 1 - Multiple LR

# Here we perform manual feature selection. We start with the best model, and
# we remove 1 by 1 the least significant features.

str(cinema_train_df)

multiple_linear_regression <- lm(visitors ~ . - date
                                 - lagged_raining_days - raining_days - month
                                 - lagged_arrivals - average_temperature
                                 - lagged_school_holidays - school_holidays
                                 - Covid_closures - lagged_covid_closures,
                                 data = cinema_train_df)
# Month and quarter were perfectly collinear, so we have removed the quarter.
summary(multiple_linear_regression)
# Order of removal: as shown above.
# After the modification: All coef. are highly significant.

cinema_predictions_df$predicted_multiple_lr <- predict(multiple_linear_regression, newdata = cinema_test_df)

# Calculate metrics
r_squared <- summary(multiple_linear_regression)$r.squared
adj_r_squared <- summary(multiple_linear_regression)$adj.r.squared
aic <- AIC(multiple_linear_regression)
mse <- mse(cinema_test_df$visitors, cinema_predictions_df$predicted_multiple_lr)
rmse <- rmse(cinema_test_df$visitors, cinema_predictions_df$predicted_multiple_lr)
mae <- mae(cinema_test_df$visitors, cinema_predictions_df$predicted_multiple_lr)
mape <- mape(cinema_test_df$visitors, cinema_predictions_df$predicted_multiple_lr)

metrics_df <- rbind(metrics_df, list(Model = "Multiple LR Manual Features",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic))
print(metrics_df)

ggplot(cinema_predictions_df, aes(x = date)) +
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
plot(cinema_train_df$date, res_lm, xlab="date", ylab="Residuals", type= "b",  pch=16, lty=3, cex=0.6)

plot(Acf(res_lm), xlab = "Lag", main = "Autocorrelation of Residuals",
     col = "steelblue", lwd = 2.5, ci.col = "black", cex.lab = 1.2, cex.main = 1.5)

# ---------------------------------------------------------------------------- #
# Model 2 - Multiple LR - Stepwise

# Hybrid
all_features_regression <- lm(visitors ~ . - date, data = cinema_train_df)
stepwise_lr_selected_model <- stepAIC(all_features_regression, direction = "both")
summary(stepwise_lr_selected_model)

cinema_predictions_df$predicted_multiple_lr_stepAIC<- predict(stepwise_lr_selected_model, newdata = cinema_test_df)

# Calculate metrics
r_squared <- summary(stepwise_lr_selected_model)$r.squared
adj_r_squared <- summary(stepwise_lr_selected_model)$adj.r.squared
aic <- AIC(stepwise_lr_selected_model)
mse <- mse(cinema_test_df$visitors, cinema_predictions_df$predicted_multiple_lr_stepAIC)
rmse <- rmse(cinema_test_df$visitors, cinema_predictions_df$predicted_multiple_lr_stepAIC)
mae <- mae(cinema_test_df$visitors, cinema_predictions_df$predicted_multiple_lr_stepAIC)
mape <- mape(cinema_test_df$visitors, cinema_predictions_df$predicted_multiple_lr_stepAIC)

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
plot(cinema_train_df$date, res_lm_stepaic, xlab="date", ylab="Residuals", type= "b",  pch=16, lty=3, cex=0.6)

plot(Acf(res_lm_stepaic), xlab = "Lag", main = "Autocorrelation of Residuals",
     col = "steelblue", lwd = 2.5, ci.col = "black", cex.lab = 1.2, cex.main = 1.5)

# ---------------------------------------------------------------------------- #
# Model 3 - TSLM with trend and seasonality

cinema_visitors_train_ts <- ts(cinema_train_df$visitors, frequency = 12)
plot(cinema_train_df$date, cinema_visitors_train_ts, type="o")

# Fit a linear model with trend
tslm_basic <- tslm(cinema_visitors_train_ts ~ trend + season)
summary(tslm_basic)
# trend is not significant

res <- residuals(tslm_basic)
plot(res)
Acf(res)
# There is a lot of information left in the residuals to be modeled.

# Perform the Durbin-Watson test
dwtest(tslm_basic)

# Forecast on the test data
fcast <- forecast(tslm_basic, newdata = cinema_test_df, h = nrow(cinema_test_df))
plot(fcast)

cinema_predictions_df$predicted_tslm <- fcast$mean

# Calculate metrics
r_squared <- summary(tslm_basic)$r.squared
adj_r_squared <- summary(tslm_basic)$adj.r.squared
aic <- AIC(tslm_basic)
mse <- mse(cinema_predictions_df$predicted_tslm, cinema_test_df$visitors)
rmse <- rmse(cinema_predictions_df$predicted_tslm, cinema_test_df$visitors)
mae <- mae(cinema_predictions_df$predicted_tslm, cinema_test_df$visitors)
mape <- mape(cinema_predictions_df$predicted_tslm, cinema_test_df$visitors)

metrics_df <- rbind(metrics_df, list(Model = "TSLM - Trend and Seasonality",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic))
print(metrics_df)

ggplot(cinema_predictions_df, aes(x = date)) +
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

cinema_train_ts_df <- data.frame(
  date_numeric = ts(cinema_train_df$date_numeric, frequency = 12),
  visitors = ts(cinema_train_df$visitors, frequency = 12),
  google_trends = ts(cinema_train_df$trends, frequency = 12),
  average_temperature = ts(cinema_train_df$average_temperature, frequency = 12),
  raining_days = ts(cinema_train_df$raining_days, frequency = 12),
  school_holidays = ts(cinema_train_df$school_holidays, frequency = 12),
  arrivals = ts(cinema_train_df$arrivals, frequency = 12),
  covid = ts(cinema_train_df$Covid_closures, frequency = 12),
  # lagged:
  lagged_google_trends = ts(cinema_train_df$lagged_trends, frequency = 12),
  lagged_average_temperature = ts(cinema_train_df$lagged_average_temperature, frequency = 12),
  lagged_raining_days = ts(cinema_train_df$lagged_raining_days, frequency = 12),
  lagged_school_holidays = ts(cinema_train_df$lagged_school_holidays, frequency = 12),
  lagged_arrivals = ts(cinema_train_df$lagged_arrivals, frequency = 12),
  lagged_covid = ts(cinema_train_df$lagged_covid_closures, frequency = 12)
)

# Fit the model on the training set
tslm_manual <- tslm(visitors ~ . - date_numeric - lagged_average_temperature
                    - lagged_school_holidays - average_temperature
                    - raining_days - lagged_raining_days - lagged_arrivals
                    - school_holidays - covid - lagged_covid - lagged_google_trends,
                    data = cinema_train_ts_df)
summary(tslm_manual)

plot(cinema_train_ts_df$visitors)
lines(fitted(tslm_manual), col=2)

res <- residuals(tslm_manual)
plot(res)
Acf(res) 

dwtest(tslm_manual)

# Leave-one-out Cross-Validation Statistic
CV(tslm_manual)

# Forecasting on the test set
test_data <- data.frame(
  date_numeric = ts(cinema_test_df$date_numeric, frequency = 12),
  visitors = ts(cinema_test_df$visitors, frequency = 12),
  google_trends = ts(cinema_test_df$trends, frequency = 12),
  average_temperature = ts(cinema_test_df$average_temperature, frequency = 12),
  raining_days = ts(cinema_test_df$raining_days, frequency = 12),
  school_holidays = ts(cinema_test_df$school_holidays, frequency = 12),
  arrivals = ts(cinema_test_df$arrivals, frequency = 12),
  covid = ts(cinema_test_df$Covid_closures, frequency = 12),
  # lagged:
  lagged_google_trends = ts(cinema_test_df$lagged_trends, frequency = 12),
  lagged_average_temperature = ts(cinema_test_df$lagged_average_temperature, frequency = 12),
  lagged_raining_days = ts(cinema_test_df$lagged_raining_days, frequency = 12),
  lagged_school_holidays = ts(cinema_test_df$lagged_school_holidays, frequency = 12),
  lagged_arrivals = ts(cinema_test_df$lagged_arrivals, frequency = 12),
  lagged_covid = ts(cinema_test_df$lagged_covid_closures, frequency = 12)
)

fcast <- forecast(tslm_manual, newdata = test_data, h = nrow(cinema_test_df))

plot(fcast)

cinema_predictions_df$predicted_tslm_manual <- fcast$mean

# Calculate metrics
r_squared <- summary(tslm_manual)$r.squared
adj_r_squared <- summary(tslm_manual)$adj.r.squared
aic <- AIC(tslm_manual)
mse <- mse(cinema_predictions_df$predicted_tslm_manual, cinema_test_df$visitors)
rmse <- rmse(cinema_predictions_df$predicted_tslm_manual, cinema_test_df$visitors)
mae <- mae(cinema_predictions_df$predicted_tslm_manual, cinema_test_df$visitors)
mape <- mape(cinema_predictions_df$predicted_tslm_manual, cinema_test_df$visitors)

metrics_df <- rbind(metrics_df, list(Model = "TSLM - Manual Features",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic))
print(metrics_df)

ggplot(cinema_predictions_df, aes(x = date)) +
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

y_train <- cinema_train_df$visitors
X_train <- cinema_train_df[, -which(names(cinema_train_df) %in% c("date", "visitors"))]
X_train <- X_train[, !startsWith(names(X_train), "lagged_")]
# Performs better without the lagged variables, so we exclude them.

X_test <- cinema_test_df[, -which(names(cinema_test_df) %in% c("date", "visitors"))]
X_test <- X_test[, !startsWith(names(X_test), "lagged_")]

# Convert data to matrix format
X_train <- as.matrix(X_train)
X_train <- apply(X_train, 2, as.numeric)
y_train <- as.matrix(y_train)
X_test <- as.matrix(X_test)
X_test <- apply(X_test, 2, as.numeric)

train_controls <- trainControl(method = "timeslice",
                               initialWindow = 48,
                               horizon = 12,
                               fixedWindow = TRUE, 
                               skip = 12,
                               allowParallel = TRUE)

grid <- expand.grid(alpha = c(0, 1), # ridge or lasso
                    lambda =  seq(0.001, 0.01, by = 0.001))

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
cat('dev.ratio:', best_l1_or_l2_regularization_model$dev.ratio)
# Dev ratio without lagged regressors: 0.6616036.

plot.ts(predict(best_l1_or_l2_regularization_model, newx=X_train, s=best_lambda))
lines(cinema_train_df$visitors,col=2)

# Perform predictions on the test set
cinema_predictions_df$predicted_visitors_l1_or_l2_regularization_tscv <-
  predict(best_l1_or_l2_regularization_model, newx = X_test, s = best_lambda)

# Calculate metrics
mse <- mse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_l1_or_l2_regularization_tscv)
rmse <- rmse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_l1_or_l2_regularization_tscv)
mae <- mae(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_l1_or_l2_regularization_tscv)
mape <- mape(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_l1_or_l2_regularization_tscv)

metrics_df <- rbind(metrics_df, list(Model = "L1/L2 Regularization TS CV",
                                     R2 = NA, R2_adj = NA,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = NA))
print(metrics_df)

# Plot predictions 
plot(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_lasso_tscv,
     ylab="Predictions", xlab="True")
abline(0,1)

ggplot(cinema_predictions_df, aes(x = date)) +
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
gam1 <- gam(visitors ~. - date - date_numeric, data=cinema_train_df)
summary(gam1)
AIC(gam1) # 344.7018

sc <- gam.scope(cinema_train_df[, -which(names(cinema_train_df) %in% c("date", "visitors"))],
                arg = c("df=2", "df=3", "df=4"))
step_gam <- step.Gam(gam1, scope = sc, trace = TRUE)
summary(step_gam)
AIC(step_gam) # 222.8829

# Prediction
p.gam1 <- predict(gam1, newdata=cinema_test_df)     # Bigger AIC
p.gam <- predict(step_gam, newdata=cinema_test_df)  # Smaller AIC   
cat('Deviance:', sum((p.gam - cinema_test_df$visitors)^2))
cat('Deviance gam1 :', sum((p.gam1 - cinema_test_df$visitors)^2)) # gam 1 has smaller deviance

cinema_predictions_df$predicted_gam <- predict(step_gam, newdata=cinema_test_df)  

# Calculate metrics
aic <- AIC(step_gam)
mse <- mse(cinema_test_df$visitors, cinema_predictions_df$predicted_gam)
rmse <- rmse(cinema_test_df$visitors, cinema_predictions_df$predicted_gam)
mae <- mae(cinema_test_df$visitors, cinema_predictions_df$predicted_gam)
mape <- mape(cinema_test_df$visitors, cinema_predictions_df$predicted_gam)

metrics_df <- rbind(metrics_df, list(Model = "GAM Stepwise",
                                     R2 = NA, R2_adj = NA,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic))
print(metrics_df)

# Plot predictions 
plot(cinema_test_df$visitors, cinema_predictions_df$predicted_gam,
     ylab="Predictions", xlab="True")
abline(0,1)

ggplot(cinema_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Visitors"), linewidth = 1) +
  geom_line(aes(y = p.gam, color = "Predicted"),
            linetype = "dashed", linewidth = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))
# --------------------------------------------------------------------- #
# Model 7 - Generalized Bass Model (with shock)

cinema_train_unstandardized_df <- cinema_train_df_copy

# First we obtain the needed coefficients using BM:
bm_visitors <- BM(cinema_train_unstandardized_df$visitors, display = TRUE)
summary(bm_visitors)

m <- 1.365561e+07 
p <- 2.483577e-03
q <- 9.816231e-03

# Predictions and instantaneous curve for BM
pred_bm_visitors <- predict(bm_visitors, newx = 1:216)
pred_inst_bm_visitors <- make.instantaneous(pred_bm_visitors)

# Plotting BM predictions
plot(cinema_train_unstandardized_df$visitors, type = "b", xlab = "Month", ylab = "Monthly Visitors", 
     pch = 16, lty = 3, cex = 0.6, xlim = c(1, 216))
lines(pred_inst_bm_visitors, lwd = 2, col = 2)

# Try with shock

# One exponential shock - This models the shock of COVID
gbm_e1 <- GBM(cinema_train_unstandardized_df$visitors, shock = "exp", nshock = 1, alpha = 0.04,
              prelimestimates = c(m, p, q, 183, -0.1, -0.5))
summary(gbm_e1)

# One rectangular shock
gbm_r1 <- GBM(cinema_train_unstandardized_df$visitors, shock = "rett", nshock = 1,
              prelimestimates = c(m, p, q, 183, 196, -0.4), oos=10)
summary(gbm_r1)

# Decided to proceed with the rectangular shocks.
best_gbm <- gbm_r1

pred_GBM_visitors<- predict(best_gbm, newx=c(1:216))
pred_GBM_visitors.inst<- make.instantaneous(pred_GBM_visitors)

# Plotting GBM predictions
plot(cinema_train_unstandardized_df$visitors, type = "b",
     xlab = "Month", ylab = "Monthly Visitors", 
     pch = 16, lty = 3, cex = 0.6, xlim = c(1, 216))
lines(pred_GBM_visitors.inst, lwd = 2, col = 2)

# Calculate metrics
cinema_predictions_df$predicted_visitors_generalized_bass_model <- pred_GBM_visitors.inst[205:216]
# Standardize:
cinema_predictions_df$predicted_visitors_generalized_bass_model <- (cinema_predictions_df$predicted_visitors_generalized_bass_model - mean(cinema_train_unstandardized_df$visitors)) / sd(cinema_train_unstandardized_df$visitors)
adj_r_squared <- adjusted_R2(cinema_train_unstandardized_df$visitors, pred_GBM_visitors.inst[1:204], nrow(cinema_train_df), length(best_gbm$coefficients))
mse <- mse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_generalized_bass_model)
rmse <- rmse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_generalized_bass_model)
mae <- mae(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_generalized_bass_model)
mape <- mape(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_generalized_bass_model)

metrics_df <- rbind(metrics_df, list(Model = "Generalized Bass Model - 2R",
                                     R2 = best_gbm$Rsquared, R2_adj = NA,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = NA))
print(metrics_df)

# Plot predictions 
ggplot(cinema_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Visitors"), linewidth = 1) +
  geom_line(aes(y = predicted_visitors_generalized_bass_model, color = "Predicted"),
            linetype = "dashed", linewidth = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))

# This model is good just for modelling the trend.

plot(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_lasso_tscv,
     ylab="Predictions", xlab="True")
abline(0,1)

# GGM
GGM_model <- GGM(cinema_train_unstandardized_df$visitors, prelimestimates=c(m, 0.001, 0.01, p, q))
summary(GGM_model)

pred_GGM <- predict(GGM_model, newx=c(1:216))
pred_GGM.inst <- make.instantaneous(pred_GGM)

plot.ts(cinema_train_unstandardized_df$visitors)
lines(pred_GGM.inst, lwd=2, col=2)

# Analysis of residuals
res_GGM <- residuals(GGM_model)
acf <- acf(res_GGM)

# --------------------------------------------------------------------- #
# Model 8 - Auto ARIMA
cinema_visitors_train_ts <- ts(cinema_train_df$visitors, frequency = 12)

auto_arima <- auto.arima(cinema_visitors_train_ts)
summary(auto_arima) # AIC=298.95

predicted_visitors_auto_arima <- forecast(auto_arima, h=12)
cinema_predictions_df$predicted_visitors_auto_arima <- predicted_visitors_auto_arima$mean

# Calculate metrics for ARIMA
train_predictions <- fitted(auto_arima)
r_squared <- RSQUARE(cinema_train_df$visitors, train_predictions)
adj_r_squared <- adjusted_R2(cinema_train_df$visitors, train_predictions, length(cinema_train_df$visitors), length(coef(auto_arima)))
mse <- mse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_auto_arima)
rmse <- rmse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_auto_arima)
mae <- mae(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_auto_arima)
mape <- mape(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_auto_arima)
aic <- AIC(auto_arima)

# Update metrics_df with ARIMA metrics
metrics_df <- rbind(metrics_df, list(Model = "Auto ARIMA",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic))
print(metrics_df)

ggplot(cinema_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Visitors"), linewidth = 1) +
  geom_line(aes(y = predicted_visitors_auto_arima, color = "Predicted"),
            linetype = "dashed", linewidth = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))

plot(cinema_visitors_train_ts)

cinema_ts_df <- diff(cinema_visitors_train_ts)
p_ts_df <- autoplot(cinema_ts_df, xlab = "Time", ylab = "Visitors")

plot(p_ts_df)
# By differentiating the series, we can see that it
# seems to be more stationary (in term of mean).

# Next, we will check the residuals of the differentiated series in
# order to see if there are again some important behaviors:
p_acf_df <- ggAcf(cinema_ts_df)
p_pacf_df <- ggPacf(cinema_ts_df)

grid.arrange(p_acf_df, p_pacf_df, nrow = 2)
# We see that the lag at time 12 and 24 are relevant as always.
# So we need to model these particular characteristics.

# Next, we try to build our first custom Arima models.

# --------------------------------------------------------------------- #
# Model 9 - SARIMA
sarima_improved <- Arima(cinema_visitors_train_ts, order = c(1,0,1), seasonal = c(0,1,2)) # Best model initially
summary(sarima_improved)  # include.drift = TRUE
# Arima (1,1,2) (1,1,2) : AIC=301.65 RMSE=0.4676794
# Arima (0,1,2) (0,1,2) : AIC=299.99 RMSE=0.4817189
# Arima (0,1,0) (0,0,2) : AIC=380.86 RMSE=0.5976933
# Arima (2,1,0) (0,0,2) : AIC=367.39 RMSE=0.5724812
# Arima (0,0,2) (0,1,2) : AIC=338.34 RMSE=0.5100456
# Arima (1,0,1) (0,1,1) : AIC=302.05 RMSE=0.494153 
# Arima (1,0,0) (0,1,1) : AIC=300.19 RMSE=0.4947074
# Arima (1,0,1) (1,1,1) : AIC=299.58 RMSE=0.462234  <- LOWEST AIC # 0.4361038 MSE
# Arima (1,0,12) (0,1,2): AIC=315.3  RMSE=0.3677237 <- LOWEST RMSE
# Arima (1,0,12) (0,1,1): AIC=298.14 RMSE=0.4569909 <- best for the residuals

cinema_visitors_train_ts <- ts(cinema_train_df$visitors, frequency = 12)

train_predictions_sarima <- fitted(sarima_improved)

ggplot(data = cinema_train_df,
       aes(x = date,
           y = as.numeric(cinema_visitors_train_ts))) +
  geom_line(color = "blue") +  
  geom_line(aes(y = train_predictions_sarima),
            color = "red", linetype = "twodash") +
  xlab("Date") + ylab("Visitors") +
  labs(title = "True vs Fitted values by ARIMA")

# Residuals
ggplot(aes(date, y = as.numeric(residuals(sarima_improved))), data = cinema_train_df) +
  geom_point(color = "blue") + xlab("Date") + ylab("Residuals") + ggtitle("Residuals of Arima")
checkresiduals(sarima_improved)

pred_sarima_improved <- forecast(sarima_improved, h = 12)
cinema_predictions_df$predicted_visitors_sarima <- pred_sarima_improved$mean

# Calculate metrics
r_squared <- RSQUARE(cinema_train_df$visitors, train_predictions_sarima)
adj_r_squared <- adjusted_R2(cinema_train_df$visitors, train_predictions_sarima, length(cinema_train_df$visitors), length(coef(sarima_improved)))
mse <- mse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_sarima)
rmse <- rmse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_sarima)
mae <- mae(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_sarima)
mape <- mape(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_sarima)
aic <- AIC(sarima_improved)

# Update metrics_df
metrics_df <- rbind(metrics_df, list(Model = "SARIMA - Improved",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic))
print(metrics_df)

ggplot(cinema_predictions_df, aes(x = date)) +
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
train_rows <- nrow(cinema_train_df) # 204
train_size <- 48 # initial
val_size <- 12
while (train_size <= train_rows) {
  # 48, 60, 72, ..., 204 -> total 14 windows will be created
  train_w <- cinema_visitors_train_ts[1:train_size] # subset the training set
  arima_w <- Arima(train_w, order = c(1,0,12), seasonal = c(0,1,1),
                   include.drift = TRUE) # try with different parameters
  pred_arima_w <- as.numeric(forecast(train_w, h = val_size)$mean)
  mse <- round(mse(train_w, pred_arima_w), 2)
  plot(x = 1:(train_size+val_size), cinema_visitors_train_ts[1:(train_size+val_size)],
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
# Double check these plots, they seem incorrect.

# --------------------------------------------------------------------- #
# Model 10 - SARIMAX

# First, we try to obtain useful regressors.
# To do this, we will start by analyzing the coefficients obtained using
# stepwise regression and Lasso/Ridge Regularization.
summary(stepwise_lr_selected_model)
# trends, school_holidays, arrivals, Covid_closures1,
# lagged_trends, lagged_arrivals, lagged_covid_closures1

# print(coef(best_l1_or_l2_regularization_model))
coefficients_l1_l2 <- coef(best_l1_or_l2_regularization_model)
print(coefficients_l1_l2[order(abs(coefficients_l1_l2), decreasing = TRUE), ])
# arrivals, trends, Covid_closures, raining_days,
# school_holidays, average_temperature

# The coefficients in common are: arrivals, trends, Covid_closures, school_holidays.
# Including month/year/date doesn't make sense for SARIMAX.

regressors_train <- cinema_train_df[, c("school_holidays", "Covid_closures", "arrivals", "trends")]
regressors_train <- as.matrix(regressors_train)
regressors_train <- apply(regressors_train, 2, as.numeric)

regressors_test <- cinema_test_df[, c("school_holidays", "Covid_closures", "arrivals", "trends")]
regressors_test <- as.matrix(regressors_test)
regressors_test <- apply(regressors_test, 2, as.numeric)

# sarimax <- auto.arima(cinema_visitors_train_ts, xreg = regressors_train) # 
sarimax <- Arima(cinema_visitors_train_ts, xreg = regressors_train,
                 order = c(1,0,12), seasonal = c(0,1,1), include.drift=TRUE)
summary(sarimax)
# ARIMA(1,1,0)(1,1,0):  AIC=253.86, RMSE=0.4346919 -> auto.arima
# ARIMA(1,0,12)(0,1,2) with drift: AIC=219.39, RMSE=0.3463853 -> better
# ARIMA(1,0,12)(0,1,1) with drift: AIC=219.4, RMSE=0.3427954

ggplot(data = cinema_train_df, aes(x = date, y = visitors)) +
  geom_line(color = "blue") +  
  geom_line(aes(y = fitted(sarimax)), color = "red", linetype = "twodash") +
  ylab("Visitors") +
  labs(title = "True vs Fitted values by SARIMAX")

ggplot(aes(date, y = as.numeric(residuals(sarimax))), data = cinema_train_df) +
  geom_point(color = "blue") + xlab("Date") + ylab("Residuals") +
  ggtitle("Residuals of SARIMAX")

# Forecasting
pred_sarimax <- forecast(sarimax, h = 12, xreg=regressors_test)
cinema_predictions_df$predicted_visitors_sarimax <- pred_sarimax$mean

train_predictions_sarimax <- fitted(sarimax)

# Calculate metrics
r_squared <- RSQUARE(cinema_train_df$visitors, train_predictions_sarimax)
adj_r_squared <- adjusted_R2(cinema_train_df$visitors, train_predictions_sarimax, length(cinema_train_df$visitors), length(coef(sarimax)))
mse <- mse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_sarimax)
rmse <- rmse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_sarimax)
mae <- mae(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_sarimax)
mape <- mape(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_sarimax)
aic <- AIC(sarimax)

# Update metrics_df
metrics_df <- rbind(metrics_df, list(Model = "SARIMAX",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic))
print(metrics_df)

ggplot(cinema_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Visitors"), linewidth = 1) +
  geom_line(aes(y = predicted_visitors_sarimax, color = "Predicted"),
            linetype = "dashed", linewidth = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))

# --------------------------------------------------------------------- #
# Model 11 - Exponential smoothing - Holt Winters
components_dfts <- decompose(cinema_visitors_train_ts)
plot(components_dfts)

HW_initial <- HoltWinters(cinema_visitors_train_ts) # Smoothing parameters: alpha: 0.5617419, beta: 0.01097067, gamma: 0.9179943
Exp_Smooth_HW <- HoltWinters(cinema_visitors_train_ts, alpha=0.21, beta=0.13, gamma=0.1)
#best for now alpha=0.21, beta=0.13, gamma=0.1
# Visually evaluate the fits
plot(cinema_visitors_train_ts, ylab="Egizio visitors")
lines(HW_initial$fitted[,1], lty=2, col="blue")
lines(Exp_Smooth_HW$fitted[,1], lty=2, col="red")

# Forecasting
HW_initial_for <- forecast(HW_initial, h=12, level=c(80,95))
# Visualize our predictions:
plot(HW_initial_for)
lines(HW_initial_for$fitted, lty=2, col="purple")

HW_initial.pred <- predict(HW_initial, nrow(cinema_test_df), prediction.interval = TRUE, level=0.95)
# Visually evaluate the prediction
plot(cinema_visitors_train_ts, ylab="visitors")
lines(HW_initial$fitted[,1], lty=2, col="blue")
lines(HW_initial.pred[,1], col="red") # This looks good
lines(HW_initial.pred[,2], lty=2, col="orange")
lines(HW_initial.pred[,3], lty=2, col="purple")

# Let's check the residuals
acf(HW_initial_for$residuals, lag.max=20, na.action=na.pass)
Box.test(HW_initial_for$residuals, lag=20, type="Ljung-Box")
hist(HW_initial_for$residuals)

# Let's analyze Exp_Smooth_HW
HW_for <- forecast(Exp_Smooth_HW, h=12, level=c(80,95))
# Visualize our predictions:
plot(HW_for)
lines(HW_for$fitted, lty=2, col="green")

HW.pred <- predict(Exp_Smooth_HW, 12, prediction.interval = TRUE, level=0.95)
# Visually evaluate the prediction
plot(cinema_visitors_train_ts, ylab="Egizio visitors")
lines(Exp_Smooth_HW$fitted[,1], lty=2, col="blue")
lines(HW.pred[,1], col="red") # This looks good
lines(HW.pred[,2], lty=2, col="orange")
lines(HW.pred[,3], lty=2, col="purple")

# Exp_Smooth_HW works better. We will use it for forecasting.

cinema_predictions_df$predicted_ESHW <- HW.pred[,1]

# Calculate metrics
mse <- mse(cinema_test_df$visitors, cinema_predictions_df$predicted_ESHW)
rmse <- rmse(cinema_test_df$visitors, cinema_predictions_df$predicted_ESHW)
mae <- mae(cinema_test_df$visitors, cinema_predictions_df$predicted_ESHW)
mape <- mape(cinema_test_df$visitors, cinema_predictions_df$predicted_ESHW)
sse2 <- sse(cinema_test_df$visitors, cinema_predictions_df$predicted_ESHW )
metrics_df <- rbind(metrics_df, list(Model = "Exp. smoothing Holt Winters",
                                     R2 = NA, R2_adj = NA,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = NA))
print(metrics_df)

# Let's check the residuals
acf(HW_for$residuals, lag.max=20, na.action=na.pass)
Box.test(HW_for$residuals, lag=20, type="Ljung-Box")
hist(HW_for$residuals)

# Multiplicative seasonality - Probably doesn't work!
HW3 <- HoltWinters(cinema_visitors_train_ts, seasonal = "multiplicative")
HW3.pred <- predict(HW3, 12, prediction.interval = TRUE, level=0.95)
plot(cinema_visitors_train_ts, ylab="Visitors")
lines(HW3$fitted[,1], lty=2, col="blue")
lines(HW3.pred[,1], col="red")
lines(HW3.pred[,2], lty=2, col="orange")
lines(HW3.pred[,3], lty=2, col="purple")

# --------------------------------------------------------------------- #
# Loess Regression

# the x and y arguments provide the x and y coordinates for the plot. 

x <- cinema_train_df$date
y <- cinema_train_df$visitors

# We use geom_smooth in order to graphically see the results

ggplot(data = cinema_train_df, 
       aes(x = date, y = visitors)) +
  geom_point() +
  xlab("Date") +
  ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_smooth(method = "loess", span = 2/3)

# We use the loess.smooth method for computing the values.
loess1_vis <- loess.smooth(x, y) # span = 2/3

ggplot(data = cinema_train_df, 
       aes(x = date, y = visitors)) +
  geom_point() +
  xlab("Date") +
  ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_smooth(method = "loess", span = 0.9, color = "red")

loess2_vis <- loess.smooth(x,y, span = 0.9) 

ggplot(data = cinema_train_df, 
       aes(x = date, y = visitors)) +
  geom_point() +
  xlab("Date") +
  ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_smooth(method = "loess", span = 0.4, color = "green")

loess2_vis <- loess.smooth(x,y, span = 0.4) 

# Complete comparison:

ggplot(data = cinema_train_df, aes(x = date, y = visitors)) +
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
#loess_pred_1 <- predict(loess1_vis, newdata = cinema_test_df$date)
#loess_pred_2 <- predict(loess2_vis, newdata = data.frame(x = cinema_test_df$date_numeric))

# --------------------------------------------------------------------- #
# CUBIC SPLINES

x <- 1:204
y <- cinema_train_df$visitors

# We may select the internal-knots by using the degrees of freedom: 

# (basic functions b-spline for a cubic spline (degree=3))

# --> df directly related to the number of knots
#     df = length(knots) + degree 
# The knots are selected by using the quantiles of 'x' 
# distribution 

ggplot(data = cinema_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Visitors")

# --------------------------------------------------------------------- #
# DEGREE 3

# Starting with 2 internal-knots
splines3_2 <- lm(y ~ bs(x, df = 5, degree = 3)) 
summary(splines3_2) # 0.2347
# 2-5 not significant
# 1-3-4 significant
ggplot(data = cinema_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_line(aes(y = splines3_2$fitted.values, color = "red"))

# Proceeding with 4 internal-knots
splines3_4 <- lm(y ~ bs(x, df = 7, degree = 3)) 
summary(splines3_4) # 0.3328
# 1-2-4 not significant
# 3-5-6-7 significant
ggplot(data = cinema_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_line(aes(y = splines3_4$fitted.values, color = "red"))

# Model with no internal-knots
splines3_0 <- lm(y ~ bs(x, df = 3, degree = 3)) 
summary(splines3_0) # 0.1787
# 2 significant
ggplot(data = cinema_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_line(aes(y = splines3_0$fitted.values, color = "red"))

# Model with 8 internal-knots
splines3_8 <- lm(y ~ bs(x, df = 11, degree = 3)) 
summary(splines3_8)
ggplot(data = cinema_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_line(aes(y = splines3_8$fitted.values, color = "red"))


#cinema_test_df$predicted_visitors <- predict(splines3_8, newdata =cinema_test_df$date)
# Best with degree = 3 --> 8 internal knots, df = 11

# --------------------------------------------------------------------- #
# DEGREE 4

# Starting with 2 internal-knots
splines4_2 <- lm(y ~ bs(x, df = 6, degree = 4)) 
summary(splines4_2) # 0.3056 
# 2-3-4 significant
# 1-5-6 not significant
ggplot(data = cinema_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_line(aes(y = splines4_2$fitted.values, color = "red"))

# Proceeding with 4 internal-knots
splines4_4 <- lm(y ~ bs(x, df = 8, degree = 4)) 
summary(splines4_4) # 0.3647 
# 1-2-3-4-5 not significant
# 6-7-8 significant
ggplot(data = cinema_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") + 
  ggtitle("Loess Model for Visitors") +
  geom_line(aes(y = splines4_4$fitted.values, color = "red"))

# Model with no internal-knots
splines4_0 <- lm(y ~ bs(x, df = 4, degree = 4)) 
summary(splines4_0) # 0.2224
# 1-2-3 significant
# 4 not significant
ggplot(data = cinema_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Visitors") +
  geom_line(aes(y = splines4_0$fitted.values, color = "red"))

# Model with 8 internal-knots
splines4_8 <- lm(y ~ bs(x, df = 12, degree = 4)) 
summary(splines4_8) # 0.3717
# 1-2-3-4-5-6-8-10-11 not significant
# 7-9-12 significant
ggplot(data = cinema_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_line(aes(y = splines4_8$fitted.values, color = "red"))

# With degree = 4 we are not able to find something better than
# the model we would choose with degree = 3.

# The result from here can be used instead of GBM in Model 15 (Decomposition).

# --------------------------------------------------------------------- #
## SMOOTHING SPLINES

# Basic smooth splines

sm_spline <- ss(x,y, method = "AIC")
ggplot(data = cinema_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_line(aes(y = sm_spline$y, color = "red"))

# Model 1
sm_spline1 = ss(x,y, method = "AIC", lambda = 0.000001, all.knots = T)
ggplot(data = cinema_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_line(aes(y = sm_spline1$y, color = "red"))


# Model with Cross Validation Method for parameter selection
sm_spline_gcv <- ss(x,y, method = "GCV")
ggplot(data = cinema_train_df, aes(x = date, y = visitors)) +
  geom_point() + xlab("Date") + ylab("Visitors") +
  ggtitle("Loess Model for Egizio Visitors") +
  geom_line(aes(y = sm_spline_gcv$y, color = "red"))

sm_spline_gcv

# Forecasting won't be done here, as this just models the trend.

# --------------------------------------------------------------------- #
# Model 12 - Boosting

mai.old <- par()$mai
mai.new <- mai.old # new vector
mai.new[2] <- 2.5 # new space on the left

# Time-series cross-validation

ts_cv_spec <- time_series_cv(data = cinema_train_df,
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

run_cross_validation <- FALSE

cinema_train_no_lagged_df <- cinema_train_df_copy[, -which(names(cinema_train_df) %in% c("date", "month", "year"))]
cinema_train_no_lagged_df <- cinema_train_no_lagged_df[, !startsWith(names(cinema_train_no_lagged_df), "lagged_")]
# Too difficult to optimize with a lot of variables.
# Also, date_numeric outperforms month+year.
cinema_test_no_lagged_df <- cinema_test_df_copy[, -which(names(cinema_test_df) %in% c("date", "month", "year"))]
cinema_test_no_lagged_df <- cinema_test_no_lagged_df[, !startsWith(names(cinema_test_no_lagged_df), "lagged_")]

if (run_cross_validation) {
  grid_boosting <- expand.grid(n.trees = c(50, 100, 500, 1000, 5000),
                               interaction.depth = c(1,4,7), 
                               shrinkage = c(0.1, 0.05, 0.01, 0.001),
                               n.minobsinnode = c(2, 4, 10))
  
  train_controls <- trainControl(method = "timeslice",# time-series cross-validation
                                 initialWindow = 48, # initial training window
                                 horizon = 12, # forecast evaluation window
                                 fixedWindow = TRUE, 
                                 skip = 12,
                                 allowParallel = TRUE) # allow parallel processing if available
  
  gbm_grid <- train(visitors ~ .,
                    data = cinema_train_no_lagged_df,
                    method = "gbm",  
                    distribution = "gaussian",
                    trControl = train_controls,
                    tuneGrid = grid_boosting,
                    metric = "RMSE",
                    verbose = FALSE)
  
  print(gbm_grid)
  
  best_model_boosting <- gbm_grid$bestTune
  
  final_model_boosting <- gbm(visitors ~ .,
                              data = cinema_train_no_lagged_df,
                              distribution = "gaussian",
                              n.trees = best_model_boosting$n.trees,
                              interaction.depth = best_model_boosting$interaction.depth,
                              shrinkage = best_model_boosting$shrinkage,
                              n.minobsinnode = best_model_boosting$n.minobsinnode)
} else {
  final_model_boosting <- gbm(visitors ~ .,
                              data = cinema_train_no_lagged_df,
                              distribution = "gaussian",
                              n.trees = 100,
                              interaction.depth = 5,
                              shrinkage = 0.001,
                              n.minobsinnode = 2)
}

par(mai=mai.new)
summary(final_model_boosting, las=1, cBar=20)
par(mai=mai.old)

cinema_predictions_df$predicted_visitors_boosting <- predict(final_model_boosting,
                                                             newdata = cinema_test_no_lagged_df,
                                                             n.trees = final_model_boosting$n.trees)

cinema_training_preds <- predict(final_model_boosting,
                                 newdata = cinema_train_no_lagged_df,
                                 n.trees = final_model_boosting$n.trees)
# Calculate metrics
r_squared <- RSQUARE(cinema_train_df$visitors, cinema_training_preds)
adj_r_squared <- adjusted_R2(cinema_train_df$visitors, cinema_training_preds, nrow(cinema_train_df), length(final_model_boosting$var.names))
mse <- mse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_boosting)
rmse <- rmse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_boosting)
mae <- mae(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_boosting)
mape <- mape(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_boosting)

metrics_df <- rbind(metrics_df, list(Model = "Boosting - TSCV",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = NA))
print(metrics_df)

# Plot predictions 
plot(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_boosting,
     ylab="Predictions", xlab="True")
abline(0,1)

ggplot(cinema_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Visitors"), size = 1) +
  geom_line(aes(y = predicted_visitors_boosting, color = "Predicted"),
            linetype = "dashed", size = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))

# Partial dependence plots
plot(final_model_boosting, i.var=1, n.trees = final_model_boosting$n.trees, ylab = "visitors", lwd=4)
plot(final_model_boosting, i.var=2, n.trees = final_model_boosting$n.trees, ylab = "visitors", lwd=4)
plot(final_model_boosting, i.var=3, n.trees = final_model_boosting$n.trees, ylab = "visitors", lwd=4)
plot(final_model_boosting, i.var=4, n.trees = final_model_boosting$n.trees, ylab = "visitors", lwd=4)
plot(final_model_boosting, i.var=5, n.trees = final_model_boosting$n.trees, ylab = "visitors", lwd=4)
plot(final_model_boosting, i.var=6, n.trees = final_model_boosting$n.trees, ylab = "visitors", lwd=4)
plot(final_model_boosting, i.var=7, n.trees = final_model_boosting$n.trees, ylab = "visitors", lwd=4)
plot(final_model_boosting, i.var=c(2,4), n.trees = final_model_boosting$n.trees)

# --------------------------------------------------------------------- #
# Model 13 - XGboost with Cross-validation

training.x <- model.matrix(visitors ~ . - date - month - year,
                           data = cinema_train_df)
training.x <- training.x[, !startsWith(colnames(training.x), "lagged_")]
testing.x <- model.matrix(visitors ~ . - date - month - year,
                          data = cinema_test_df)
testing.x <- testing.x[, !startsWith(colnames(testing.x), "lagged_")]

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
                              max_depth = c(5, 6, 7),
                              eta = c(0.01, 0.025, 0.05, 0.075, 0.1),
                              gamma = c(0, 0.1, 0.2),
                              colsample_bytree = c(0.8, 1),
                              min_child_weight = c(1, 3, 5),
                              subsample = c(0.8, 1))
  # Result without lagged regressors:
  # The final values used for the model were nrounds = 300, max_depth = 7, eta = 0.05,
  # gamma = 0.1, colsample_bytree = 0.8, min_child_weight = 5 and subsample = 0.8.
  
  xgb_model <- train(x = data.matrix(training.x[, -1]), # Ignore intercept
                     y = as.numeric(as.character(cinema_train_df$visitors)),
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
                             label=cinema_train_df$visitors,
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
  final_model_xgb <- xgboost(data=data.matrix(training.x[,-1]), # ignore intercept
                             label=as.numeric(as.character(cinema_train_df$visitors)),
                             eta=0.001, # default=0.3 - takes values in (0-1]
                             max_depth=6, # default=6 - takes values in (0,Inf), larger value => more complex => overfitting
                             nrounds=500, # default=100 - controls number of iterations (number of trees)
                             early_stopping_rounds=50,
                             objective="reg:squarederror", # for linear regression
                             verbose=FALSE) 
}

importance_scores <- xgb.importance(model = final_model_xgb)
print(importance_scores)
xgb.plot.importance(importance_matrix = importance_scores)

# Perform predictions on the test set
cinema_predictions_df$predicted_visitors_xgboost_tscv <- predict(final_model_xgb,
                                                                 newdata = data.matrix(testing.x[, -1]))

# Calculate metrics for XGBoost
cinema_training_xgboost_preds <- predict(final_model_xgb,
                                         newdata = data.matrix(training.x[, -1]))
# Calculate metrics
r_squared <- RSQUARE(cinema_train_df$visitors, cinema_training_xgboost_preds)
adj_r_squared <- adjusted_R2(cinema_train_df$visitors, cinema_training_xgboost_preds, nrow(cinema_train_df), final_model_xgb$nfeatures)
mse <- mse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_xgboost_tscv)
rmse <- rmse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_xgboost_tscv)
mae <- mae(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_xgboost_tscv)
mape <- mape(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_xgboost_tscv)

# Update metrics_df with XGBoost metrics
metrics_df <- rbind(metrics_df, list(Model = "XGBoost - TSCV",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = NA)) # Note: AIC may not be applicable for XGBoost

print(metrics_df)

# Plot predictions 
plot(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_xgboost_tscv,
     ylab="Predictions", xlab="True")
abline(0,1)

ggplot(cinema_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Visitors"), size = 1) +
  geom_line(aes(y = predicted_visitors_xgboost_tscv, color = "Predicted"),
            linetype = "dashed", size = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))

# --------------------------------------------------------------------- #
# Model 14 - Combined Model: GBM with Rectangular Shock + ARMAX for residuals

res_gbm <- make.instantaneous(best_gbm$residuals)
Acf(res_gbm)
Pacf(res_gbm)

# arima_res_gbm <- auto.arima(res_gbm, xreg = regressors_train)
arima_res_gbm <- Arima(res_gbm, xreg = regressors_train,
                       order = c(1,0,12), seasonal = c(0,1,1), include.drift = TRUE)
summary(arima_res_gbm)

ggplot(data = cinema_train_df, aes(x = date, y = res_gbm)) +
  geom_point(color = "black") +
  geom_line(aes(y = arima_res_gbm$fitted), color = "red") +
  xlab("Date") +
  ylab("Residuals") +
  ggtitle("Real residuals vs. Fitted values with ARMAX")

ggplot(data = cinema_train_df, aes(x = date, y = as.numeric(arima_res_gbm$residuals))) +
  geom_point(color = "blue") + xlab("Date") + ylab("Residuals") +
  ggtitle("Residuals Plot of ARMAX Model for Residuals")

vis_fitted_gbm <- make.instantaneous(best_gbm$fitted) + arima_res_gbm$fitted
vis_fitted_gbm <- (vis_fitted_gbm - mean(cinema_train_df_copy$visitors)) / sd(cinema_train_df_copy$visitors)

ggplot(data = cinema_train_df, aes(x = date, y = visitors)) +
  geom_point(color = "black") +
  geom_line(aes(y = vis_fitted_gbm), color = "red") +
  xlab("Date") +
  ggtitle("Visitors vs. Fitted values with Combined Model: GBM + ARMAX")

# The model above doesn't work well, therefore it won't be used for forecasting.

# --------------------------------------------------------------------- #
# Model 15 - Decomposition: GBM + TSLM + Boosting

cinema_visitors_unstandardized_train_ts <- ts(cinema_train_df_copy$visitors, frequency = 12)
cinema_visitors_comp <- decompose(cinema_visitors_unstandardized_train_ts)
plot(cinema_visitors_comp)

visitorSeasonAdj <- cinema_visitors_unstandardized_train_ts - cinema_visitors_comp$seasonal
plot.ts(visitorSeasonAdj)
# We obtain just the trend + random components

# For modelling the trend we will use the best GBM from above.

# We will use the cinema_train_unstandardized_df$visitors directly.
# To begin with, we just focus our attention to model the general trend.

# 1. Modelling the trend
plot.ts(cinema_visitors_comp$trend, ylab="Trend")

# Let's try with GBM
# The best candidates were: exp + rectangular and two rectangular.
# Decided to proceed with two rectangular shocks.
res_gbm <- make.instantaneous(best_gbm$residuals)
plot.ts(res_gbm) # Residuals when fitting a trend
lines(as.numeric(cinema_visitors_unstandardized_train_ts - cinema_visitors_comp$trend), col=2)

pred_gbm_visitors <- predict(best_gbm, newx = 1:216)
pred_inst_gbm_visitors <- make.instantaneous(pred_gbm_visitors)
plot.ts(pred_inst_gbm_visitors[1:204])
lines(as.numeric(cinema_visitors_comp$trend), col=2)

plot.ts(pred_inst_gbm_visitors[1:204] - as.numeric(cinema_visitors_comp$trend))
# Mostly the outliers are left and noise.

# 2. Model the seasonality
res_gbm_ts <- ts(res_gbm, frequency = 12)
tslm_res_gbm <- tslm(res_gbm_ts ~ season)
summary(tslm_res_gbm)

ggplot(cinema_train_unstandardized_df, aes(x = date)) +
  geom_line(aes(y = visitors, color = "Visitors"), size = 1) +
  geom_line(aes(y = tslm_res_gbm$fitted.values + pred_inst_gbm_visitors[1:204], 
                color = "Predicted"), linetype = "dashed", size = 1) +
  labs(title = "Visitors and Predicted Values Over Time", x = "Date", y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))


# 3. Random Noise
res_res <- tslm_res_gbm$residuals

# Standardization is not necessary
cinema_train_unstandardized_df$res <- as.numeric(res_res)

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
                        data = cinema_train_unstandardized_df,
                        method = "gbm",  
                        distribution = "gaussian",
                        trControl = train_controls,
                        tuneGrid = grid_boosting_res,
                        verbose = FALSE)
  
  # View the results of the grid search
  print(gbm_grid_res)
  
  best_model_boosting_res <- gbm_grid_res$bestTune
  
  final_model_boosting_res <- gbm(res ~ . - date - date_numeric - visitors,
                                  data = cinema_train_unstandardized_df,
                                  distribution = "gaussian",
                                  n.trees = best_model_boosting_res$n.trees,
                                  interaction.depth = best_model_boosting_res$interaction.depth,
                                  shrinkage = best_model_boosting_res$shrinkage,
                                  n.minobsinnode = best_model_boosting_res$n.minobsinnode)
} else {
  final_model_boosting_res <- gbm(res ~ . - date - date_numeric - visitors,
                                  data = cinema_train_unstandardized_df,
                                  distribution = "gaussian",
                                  n.trees = 50,
                                  interaction.depth = 8,
                                  shrinkage = 0.01,
                                  n.minobsinnode = 2)
}

par(mai=mai.new)
summary(final_model_boosting_res, las=1, cBar=20)
par(mai=mai.old)

plot.ts(cinema_train_unstandardized_df$res)
lines(final_model_boosting_res$fit, col=2)
# We can see that the model doesn't overfit, which is good.
# But there is still some variability to be explained, due to outliers.

residuals <- cinema_train_unstandardized_df$visitors - final_model_boosting_res$fit
checkresiduals(remainder(decompose(ts(residuals, frequency = 12))))

# Can we use this model for forecasting? Yes.
trend_predictions <- pred_inst_gbm_visitors[205:216]
seasonality_prediction <- forecast(tslm_res_gbm, h=12)$mean
residual_predictions <- predict(final_model_boosting_res,
                                newdata=cinema_test_df_copy, # unstandardized
                                n.trees=final_model_boosting_res$n.trees)
final_predictions <- trend_predictions + seasonality_prediction + residual_predictions

# Calculate metrics
cinema_predictions_df$predicted_visitors_decomposed <- as.numeric(final_predictions)
cinema_predictions_df$predicted_visitors_decomposed <- (cinema_predictions_df$predicted_visitors_decomposed - mean(cinema_train_unstandardized_df$visitors)) / sd(cinema_train_unstandardized_df$visitors)

mse <- mse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_decomposed)
rmse <- rmse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_decomposed)
mae <- mae(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_decomposed)
mape <- mape(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_decomposed)

metrics_df <- rbind(metrics_df, list(Model = "Decomposed: GBM+TLSM+Boosting",
                                     R2 = NA, R2_adj = NA,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = NA))
print(metrics_df)

ggplot(cinema_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Visitors"), linewidth = 1) +
  geom_line(aes(y = predicted_visitors_decomposed, color = "Predicted"),
            linetype = "dashed", linewidth = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))

# --------------------------------------------------------------------- #
#                    Best model and error analysis
# --------------------------------------------------------------------- #

rounded_metrics <- round(metrics_df[, -1], 3)
rounded_metrics_df <- cbind(metrics_df[, 1, drop = FALSE], rounded_metrics)
print(rounded_metrics_df)

# Sort based on RMSE, AIC
sorted_metrics_df <- rounded_metrics_df[order(rounded_metrics_df$RMSE, rounded_metrics_df$MAPE),]
print(sorted_metrics_df)

# Print the top 5 models
cat("Top 5 models:\n")
print(head(sorted_metrics_df, 5))

ggplot(cinema_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "True"), linewidth = 1.5) +
  
  geom_line(aes(y = predicted_ESHW, color = "ESHW"),
            linetype = "dashed", linewidth = 1.5) +
  geom_line(aes(y = predicted_tslm, color = "TSLM"),
            linetype = "dashed", linewidth = 1.5) +
  geom_line(aes(y = predicted_visitors_sarima, color = "SARIMA"),
            linetype = "dashed", linewidth = 1.5) +
  labs(title = "True vs Predicted Visitors Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("True" = "#572F43",
                                "ESHW" = "#D5A021",
                                "TSLM" = "#00a6fb",
                                "SARIMA" = "#86BA90")) +
  labs(color = NULL) +
  theme(legend.position = c(1, 0), 
        legend.justification = c(1, 0),
        legend.key.size = unit(2, "lines"),  # Adjust the legend key size
        legend.background = element_rect(fill = "transparent"))  # Make the legend transparent


# Error analysis: # ToDo: Replace predicted_visitors_sarima with the best model column.
best_model <- "SARIMA - Improved"
sorted_cinema_predictions_df <- cinema_predictions_df[, c("visitors_true", "predicted_visitors_sarima")]

# Unstandardize
sorted_cinema_predictions_df$visitors_true <- (sorted_cinema_predictions_df$visitors_true * sd(cinema_train_unstandardized_df$visitors)) + mean(cinema_train_unstandardized_df$visitors)
sorted_cinema_predictions_df$predicted_visitors_sarima <- (sorted_cinema_predictions_df$predicted_visitors_sarima * sd(cinema_train_unstandardized_df$visitors)) + mean(cinema_train_unstandardized_df$visitors)

sorted_cinema_predictions_df$error <- abs(sorted_cinema_predictions_df$visitors_true - sorted_cinema_predictions_df$predicted_visitors_sarima)
sorted_cinema_predictions_df$error_percentage <- (sorted_cinema_predictions_df$error / sorted_cinema_predictions_df$visitors_true) * 100
sorted_cinema_predictions_df$error_percentage <- round(sorted_cinema_predictions_df$error_percentage, 2)
sorted_cinema_predictions_df <- sorted_cinema_predictions_df[order(sorted_cinema_predictions_df$error_percentage), ]

colnames(sorted_cinema_predictions_df) <- c("Visitors", "Predicted", "Error", "Error (%)")
sorted_cinema_predictions_df$Predicted <- round(sorted_cinema_predictions_df$Predicted, digits=0)
# Total Error (%) for SARIMA: 155.15
# Total Error (%) for Holt-Winters: 154.63
# Worst predictions for Holt-Winters: 19.81, 25.63, 33.10
# Worst predictions for SARIMA:       19.17, 22.99, 23.98
# A better model here is SARIMA.

# Print the best 3 predictions
best_3_predictions <- head(sorted_cinema_predictions_df, 3)
cat("Best 3 predictions (True vs Predicted value):\n")
print(best_3_predictions[, c(-3)])

# Print the worst 3 predictions
worst_3_predictions <- tail(sorted_cinema_predictions_df, 3)
cat("Worst 3 predictions (True vs Predicted value):\n")
print(worst_3_predictions[, c(-3)])

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

# min_rmse_index <- which.min(metrics_df$RMSE)
best_model_description <- metrics_df$Model[metrics_df$Model == best_model]
print(best_model_description)
best_model_row <- subset(metrics_df, Model == best_model_description)
best_model_row$Model <- "Best model"
covid_metrics_df <- rbind(covid_metrics_df, best_model_row)
print(covid_metrics_df)

# To create a new artificial dataset, we first drop the COVID months.
cinema_train_no_covid_ds <- cinema_train_df[1:182,]

# Find a good model to predict the COVID months
cinema_train_no_covid_visitors_ts <- ts(cinema_train_no_covid_ds$visitors, frequency = 12)
# Different models were tried (shown below).
# The best model was the following:
sarima_no_covid <- auto.arima(cinema_train_no_covid_visitors_ts)
summary(sarima_no_covid) # AIC=210.52, RMSE=0.3710264
train_predictions_sarima_no_covid <- fitted(sarima_no_covid)

# Visualize the predictions for the COVID months:
plot.ts(cinema_train_no_covid_visitors_ts)
lines(train_predictions_sarima_no_covid,col=2)

# Calculate metrics
train_r_squared <- RSQUARE(cinema_train_no_covid_visitors_ts, train_predictions_sarima_no_covid)
train_adj_r_squared <- adjusted_R2(cinema_train_no_covid_visitors_ts, train_predictions_sarima_no_covid, length(cinema_train_no_covid_visitors_ts), length(coef(sarima_no_covid)))
train_mse <- mse(cinema_train_no_covid_visitors_ts, train_predictions_sarima_no_covid)
train_rmse <- rmse(cinema_train_no_covid_visitors_ts, train_predictions_sarima_no_covid)
train_mae <- mae(cinema_train_no_covid_visitors_ts, train_predictions_sarima_no_covid)
train_mape <- mape(cinema_train_no_covid_visitors_ts, train_predictions_sarima_no_covid)
train_aic <- AIC(sarima_no_covid)
cat(train_r_squared, train_adj_r_squared, train_mse, train_rmse, train_mae, train_mape, train_aic)
# The following models were tried:
# ARIMA(1,0,1)(2,1,2)[12] -> auto.arima
# 0.8173506 0.8100026 0.1107434 0.3327814 0.2471304 1.612489 148.0625

# After finding the best model, use it for forecasting, and
# add the forecasts to the artificial training dataset.
predicted_visitors_sarima_no_covid <- forecast(sarima_no_covid, h=22)
plot(predicted_visitors_sarima_no_covid)

cinema_train_visitors_covid_replaced <- cinema_train_df
cinema_train_visitors_covid_replaced$visitors[183:nrow(cinema_train_df)] <- predicted_visitors_sarima_no_covid$mean

# Visualize the whole time series
plot(cinema_train_visitors_covid_replaced$date,
     ((cinema_train_visitors_covid_replaced$visitors * sd(cinema_train_df_copy$visitors)) + mean(cinema_train_df_copy$visitors)),
     type='l', xlab="Date", ylab="Visitors",
     main="Egizio Visitors with Forecasting Interpolation for COVID months", col="#015047", lwd=2.5, ylim=c(0,120000))
lines(cinema_train_df$date[182:204], ((cinema_train_df$visitors[182:204] * sd(cinema_train_df_copy$visitors)) + mean(cinema_train_df_copy$visitors)), col="#D5A021", lwd=4.5)
abline(v = cinema_train_df$date[182], col = "#775761", lty = 2, lwd = 2.5)

# Visualize just the COVID interpolation
plot(cinema_train_visitors_covid_replaced$date[182:204],
     ((cinema_train_visitors_covid_replaced$visitors[182:204] * sd(cinema_train_df_copy$visitors)) + mean(cinema_train_df_copy$visitors)),
     type='l', xlab="Date", ylab="Visitors",
     main="Interpolation using Forecasting", col="#015047", lwd=3.5, ylim=c(0,85000))
lines(cinema_train_df$date[182:204], ((cinema_train_df$visitors[182:204] * sd(cinema_train_df_copy$visitors)) + mean(cinema_train_df_copy$visitors)), col="#D5A021", lwd=3.5)
abline(h = max(((cinema_train_visitors_covid_replaced$visitors[182:204] * sd(cinema_train_df_copy$visitors)) + mean(cinema_train_df_copy$visitors))), col = "#775761", lty = 2, lwd = 2.5)

# Refit the same model from above (used for training) on the whole
# new artificial training time series cinema_train_visitors_covid_replaced:
cinema_train_visitors_covid_replaced_ts <- ts(cinema_train_visitors_covid_replaced$visitors, frequency=12)

# sarima_covid_replaced <- Arima(cinema_train_visitors_covid_replaced_ts, order = c(1,0,12),
#                                seasonal = c(0,1,2), include.drift=TRUE)
sarima_covid_replaced <- auto.arima(cinema_train_visitors_covid_replaced_ts)

summary(sarima_covid_replaced)

cinema_train_visitors_covid_replaced$visitors_predictions <- fitted(sarima_covid_replaced)

# Visualize how it fits the new training data:
ggplot(data = cinema_train_visitors_covid_replaced, aes(x = date, y = visitors)) +
  geom_line(color = "blue") +  
  geom_line(aes(y = visitors_predictions), color = "red", linetype = "twodash") +
  labs(title = "True vs Fitted values by SARIMA")

# Visualize the residuals:
ggplot(aes(date, y = as.numeric(residuals(sarima_covid_replaced))),
       data = cinema_train_visitors_covid_replaced) +
  geom_point(color = "blue") + xlab("Date") + ylab("Residuals") + ggtitle("Residuals of SARIMA")

checkresiduals(sarima_covid_replaced)

# Finally, let's use it for forecasting:
pred_sarima_covid_replaced <- forecast(sarima_covid_replaced, h = 12)
cinema_predictions_df$predicted_visitors_sarima_covid_replaced_forecast <- pred_sarima_covid_replaced$mean

# Calculate metrics on the test set (2022):
r_squared <- RSQUARE(cinema_train_visitors_covid_replaced$visitors, cinema_train_visitors_covid_replaced$visitors_predictions)
adj_r_squared <- adjusted_R2(cinema_train_visitors_covid_replaced$visitors, cinema_train_visitors_covid_replaced$visitors_predictions, length(cinema_train_visitors_covid_replaced$visitors), length(coef(sarima_covid_replaced)))
mse <- mse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_sarima_covid_replaced_forecast)
rmse <- rmse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_sarima_covid_replaced_forecast)
mae <- mae(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_sarima_covid_replaced_forecast)
mape <- mape(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_sarima_covid_replaced_forecast)
aic <- AIC(sarima_covid_replaced)

covid_metrics_df <- rbind(covid_metrics_df, list(Model = "COVID interpolated with forecasting",
                                                 R2 = r_squared, R2_adj = adj_r_squared,
                                                 MSE = mse, RMSE = rmse, MAE = mae,
                                                 MAPE = mape, AIC = aic))
print(covid_metrics_df)

ggplot(cinema_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "True"), linewidth = 1) +
  geom_line(aes(y = predicted_visitors_sarima, color = "Original predictions"),
            linetype = "dashed", linewidth = 1) +
  geom_line(aes(y = predicted_visitors_sarima_covid_replaced_forecast, color = "COVID Interpolated predictions"),
            linetype = "dashed", linewidth = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("True"="red",
                                "Original predictions" = "green",
                                "COVID Interpolated predictions" = "blue"))

# This model performs worse than the model with no interpolation.
# Observation:
# The results are better if we don't use interpolated data for COVID. This is
# due to the fact that after COVID, the peaks aren't nearly as tall as before
# COVID. Any good model will predict tall peaks, so it will give a higher MSE.
# => No more experiments are needed.

# --------------------------------------------------------------------- #
# Interpolation of COVID months using the mean of the 
# corresponding month of the pre-COVID training dataset.

cinema_train_covid_interpolated_mean_ds <- cinema_train_df[1:180, c("date", "visitors")]

# Calculate mean values for each month
monthly_means <- sapply(1:12, function(month) {
  mean(cinema_train_covid_interpolated_mean_ds[seq(month, nrow(cinema_train_covid_interpolated_mean_ds), by = 12), ]$visitors)
})

# Add mean values to cinema_train_covid_interpolated_mean_ds
month_idx <- 1
for (i in seq(181, 204)) {
  visitors_i <- monthly_means[month_idx]
  last_date <- tail(cinema_train_covid_interpolated_mean_ds$date, 1)
  date_i <- last_date %m+% months(1)
  new_row <- data.frame(date=date_i, visitors=visitors_i)
  cinema_train_covid_interpolated_mean_ds <- rbind(cinema_train_covid_interpolated_mean_ds, new_row)
  month_idx <- month_idx + 1
  if (month_idx == 13) {
    month_idx <- 1
  }
}

# Visualize the whole time series
plot(cinema_train_covid_interpolated_mean_ds$date,
     ((cinema_train_covid_interpolated_mean_ds$visitors * sd(cinema_train_df_copy$visitors)) + mean(cinema_train_df_copy$visitors)),
     type='l', xlab="Date", ylab="Visitors",
     main="Egizio Visitors with Forecasting Interpolation for COVID months", col="#015047", lwd=2.5, ylim=c(0,120000))
lines(cinema_train_df$date[182:204], ((cinema_train_df$visitors[182:204] * sd(cinema_train_df_copy$visitors)) + mean(cinema_train_df_copy$visitors)), col="#D5A021", lwd=4.5)
abline(v = cinema_train_df$date[182], col = "#775761", lty = 2, lwd = 2.5)

# Visualize just the COVID interpolation
plot(cinema_train_covid_interpolated_mean_ds$date[182:204],
     ((cinema_train_covid_interpolated_mean_ds$visitors[182:204] * sd(cinema_train_df_copy$visitors)) + mean(cinema_train_df_copy$visitors)),
     type='l', xlab="Date", ylab="Visitors",
     main="Interpolation using monthly mean", col="#015047", lwd=3.5, ylim=c(0,78000))
lines(cinema_train_df$date[182:204], ((cinema_train_df$visitors[182:204] * sd(cinema_train_df_copy$visitors)) + mean(cinema_train_df_copy$visitors)), col="#D5A021", lwd=3.5)
abline(v = cinema_train_df$date[182], col = "#775761", lty = 2, lwd = 2.5)

# Fit on the whole new artificial training time series cinema_train_visitors_covid_replaced:
cinema_train_visitors_covid_interpolated_mean_ts <- ts(cinema_train_covid_interpolated_mean_ds$visitors, frequency=12)

sarima_covid_replaced_mean <- auto.arima(cinema_train_visitors_covid_interpolated_mean_ts)
summary(sarima_covid_replaced_mean)

cinema_train_visitors_covid_replaced$visitors_predictions <- fitted(sarima_covid_replaced_mean)

# Visualize how it fits the new training data:
ggplot(data = cinema_train_visitors_covid_replaced, aes(x = date, y = visitors)) +
  geom_line(color = "blue") +  
  geom_line(aes(y = visitors_predictions), color = "red", linetype = "twodash") +
  labs(title = "True vs Fitted values by SARIMA")

# Visualize the residuals:
ggplot(aes(date, y = as.numeric(residuals(sarima_covid_replaced_mean))),
       data = cinema_train_visitors_covid_replaced) +
  geom_point(color = "blue") + xlab("Date") + ylab("Residuals") + ggtitle("Residuals of SARIMA")

checkresiduals(sarima_covid_replaced_mean)

# Finally, let's use it for forecasting:
pred_sarima_covid_replaced_mean <- forecast(sarima_covid_replaced_mean, h = 12)
cinema_predictions_df$predicted_visitors_sarima_covid_replaced_mean <- pred_sarima_covid_replaced_mean$mean

# Calculate metrics on the test set (2022):
r_squared <- RSQUARE(cinema_train_visitors_covid_replaced$visitors, cinema_train_visitors_covid_replaced$visitors_predictions)
adj_r_squared <- adjusted_R2(cinema_train_visitors_covid_replaced$visitors, cinema_train_visitors_covid_replaced$visitors_predictions, length(cinema_train_visitors_covid_replaced$visitors), length(coef(sarima_covid_replaced_mean)))
mse <- mse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_sarima_covid_replaced_mean)
rmse <- rmse(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_sarima_covid_replaced_mean)
mae <- mae(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_sarima_covid_replaced_mean)
mape <- mape(cinema_test_df$visitors, cinema_predictions_df$predicted_visitors_sarima_covid_replaced_mean)
aic <- AIC(sarima_covid_replaced_mean)

covid_metrics_df <- rbind(covid_metrics_df, list(Model = "COVID interpolated - mean",
                                                 R2 = r_squared, R2_adj = adj_r_squared,
                                                 MSE = mse, RMSE = rmse, MAE = mae,
                                                 MAPE = mape, AIC = aic))
print(covid_metrics_df)

ggplot(cinema_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "True"), linewidth = 1) +
  geom_line(aes(y = predicted_visitors_sarima, color = "Original predictions"),
            linetype = "dashed", linewidth = 1) +
  geom_line(aes(y = predicted_visitors_sarima_covid_replaced_mean, color = "COVID Interpolated predictions"),
            linetype = "dashed", linewidth = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("True"="red",
                                "Original predictions" = "green",
                                "COVID Interpolated predictions" = "blue"))

# Compare all predictions:
ggplot(cinema_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "True"), linewidth = 1.5) +
  geom_line(aes(y = predicted_visitors_sarima, color = "No interpolation"),
            linetype = "dashed", linewidth = 1.5) +
  geom_line(aes(y = predicted_visitors_sarima_covid_replaced_mean, color = "COVID Interpolated - Mean"),
            linetype = "dashed", linewidth = 1.5) +
  geom_line(aes(y = predicted_visitors_sarima_covid_replaced_forecast, color = "COVID Interpolated - Forecast"),
            linetype = "dashed", linewidth = 1.5) +
  labs(title = "True vs Predicted Visitors Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("True"="#572F43",
                                "No interpolation" = "#D5A021",
                                "COVID Interpolated - Mean" = "#00a6fb",
                                "COVID Interpolated - Forecast" = "#86BA90")) +
  labs(color = NULL) +
  theme(legend.position = c(1, 1),  # Place legend on top right
        legend.justification = c(1, 1),  # Justify legend to the top right
        legend.key.size = unit(2, "lines"),  # Adjust the legend key size
        legend.text = element_text(size = 15),  # Set the legend text size
        legend.background = element_rect(fill = "transparent"))  # Make the legend transparent

# Show both interpolations and the true values on the same plot
plot(cinema_train_df$date, ((cinema_train_df$visitors * sd(cinema_train_df_copy$visitors)) + mean(cinema_train_df_copy$visitors)), type='l', xlab="Date", ylab="Visitors", main="Interpolation of COVID months using Forecasting/Monthly mean", col="#015047", lwd=2.5, ylim=c(0,85000))
lines(cinema_train_df$date[182:204], ((cinema_train_visitors_covid_replaced$visitors[182:204] * sd(cinema_train_df_copy$visitors)) + mean(cinema_train_df_copy$visitors)), col="#D5A021", lwd=3.5)
lines(cinema_train_df$date[182:204], ((cinema_train_covid_interpolated_mean_ds$visitors[182:204] * sd(cinema_train_df_copy$visitors)) + mean(cinema_train_df_copy$visitors)), col="#572F43", lwd=3.5)
abline(v = cinema_train_df$date[182], col = "#DCD6F7", lty = 2, lwd = 2.5)
legend("bottomleft", 
       legend = c("Original Data", "COVID Interpolation using Forecasting", "COVID Interpolation using Monthly Mean"),
       col = c("#015047", "#D5A021", "#572F43"),
       lty = c(1, 1, 1),  # Line types for each line
       lwd = c(3.5, 3.5, 3.5),  # Line widths for each line
       cex = 0.8,
       box.lty = 0)

# Round and print metrics for COVID
rounded_covid_metrics <- round(covid_metrics_df[, -1], 3)
rounded_covid_metrics_df <- cbind(covid_metrics_df[, 1, drop = FALSE], rounded_covid_metrics)
print(rounded_covid_metrics_df)
