# ---------------------------------------------------------------------------- #
# Imports
library(zoo)
library(ggplot2)
library(ggpubr)
library(lubridate)
library(lmtest)
library (gbm)
library(viridis)
library(caret)
library(xgboost)
library(timetk)
library(forecast)
library(car)

# ---------------------------------------------------------------------------- #

# ToDos (At a later stage, not now):
# 1. Restructure the code into multiple files:
# - preprocessing.R
# - EDA.R
# - modelling.R

# 2. Error analysis:
# - Analyze errors: absolute difference between prediction and ground truth
# - Analyze worst and best prediction
# - Get average error per month

# ---------------------------------------------------------------------------- #

# Change working directory
script_path <- rstudioapi::getSourceEditorContext()$path
script_dir <- dirname(script_path)
setwd(script_dir)

# To use English for the dates (instead of Macedonian/Italian)
Sys.setlocale("LC_TIME", "English")

set.seed(123)

# ---------------------------------------------------------------------------- #
#                              Preprocessing
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Egizio visitors

egizio_visitors_df <- read.csv("../data/museo_egizio_visitors.csv")

egizio_visitors_df$date <- as.Date(as.yearmon(paste(egizio_visitors_df$year,
                                                    egizio_visitors_df$month,
                                                    sep = "-")))

egizio_visitors_df$quarter <- as.ordered(quarter(egizio_visitors_df$date))
egizio_visitors_df$month <- as.ordered(egizio_visitors_df$month)

# We can't use the year as a categorical variable, because if in the training
# set we have years 2006-2021, and we want to predict for the year 2022, this
# year will have no variable associated to it in the model, and predictions
# won't be possible.
# Therefore, we will keep year as integer.

# ToDo: Create lagged variables.

# Reorder columns
egizio_visitors_df <- egizio_visitors_df[, c('date', 'month', 'quarter', 'year', 'visitors')]

str(egizio_visitors_df)

print("Count of NAs in each column:")
print(colSums(is.na(egizio_visitors_df)))

egizio_visitors_df$visitors <- as.integer(egizio_visitors_df$visitors)
# There are NAs, because visitors was a string column, and there were empty strings.

na_egizio_count <- colSums(is.na(egizio_visitors_df))
rows_egizio_with_na <- which(rowSums(is.na(egizio_visitors_df)) > 0)
rows_with_na_data <- egizio_visitors_df[rows_egizio_with_na, ]
egizio_visitors_df[is.na(egizio_visitors_df)] <- 0

print("Count of NAs in each column:")
print(na_egizio_count)

print("Rows with NAs:")
print(rows_with_na_data)

print("Count of NAs in each column after modification:")
print(colSums(is.na(egizio_visitors_df)))

print("Count of NAs in each column:")
print(colSums(is.na(egizio_visitors_df)))

egizio_visitors_df$visitors <- as.integer(egizio_visitors_df$visitors)
str(egizio_visitors_df)

# ToDo: Check if for the other museums the closing months for COVID are pretty much the same.

ggplot(egizio_visitors_df, aes(x = date, y = visitors)) +
  geom_line() +
  labs(title = "Egyptian Museum Visitors", x = "Date", y = "Visitors") +
  theme_minimal()

# ---------------------------------------------------------------------------- #
# Read Google Trends data
egizio_googletrends_df <- read.csv("../data/museo_egizio_google_trends.csv")

egizio_googletrends_df$date <- as.Date(as.yearmon(egizio_googletrends_df$date))
egizio_googletrends_df$trends <- as.integer(egizio_googletrends_df$trends)

print("Count of NAs in each column:")
print(colSums(is.na(egizio_googletrends_df)))

# ---------------------------------------------------------------------------- #
# Read Turin weather data
turin_weather_df <- read.csv("../data/turin_weather.csv")

turin_weather_df$date <- as.Date(as.yearmon(paste(turin_weather_df$year,
                                                  turin_weather_df$month,
                                                  sep = "-")))
# Drop 'year' and 'month' and reorganize columns
turin_weather_df <- turin_weather_df[, c('date', 'average_temperature', 'raining_days')]

str(turin_weather_df)
turin_weather_df$average_temperature <- as.integer((turin_weather_df$average_temperature))

print("Count of NAs in each column:")
print(colSums(is.na(turin_weather_df)))

# ---------------------------------------------------------------------------- #
# Read Italy holiday data
italy_holidays_df <- read.csv("../data/italy_holidays.csv")

italy_holidays_df$date <- as.Date(as.yearmon(paste(italy_holidays_df$year,
                                                   italy_holidays_df$month,
                                                   sep = "-")))
# Drop 'year' and 'month' and reorganize columns
italy_holidays_df <- italy_holidays_df[, c('date', 'hl_sch')]
names(italy_holidays_df)[names(italy_holidays_df) == "hl_sch"] <- "school_holidays" # Rename

str(italy_holidays_df)

print("Count of NAs in each column:")
print(colSums(is.na(italy_holidays_df)))

# ToDo: Find and load number of tourists data.

# ---------------------------------------------------------------------------- #
# Concatenate dataframes

egizio_df <- merge(egizio_visitors_df, egizio_googletrends_df, by = "date", all.x = TRUE)
egizio_df <- merge(egizio_df, turin_weather_df, by = "date", all.x = TRUE)
egizio_df <- merge(egizio_df, italy_holidays_df, by = "date", all.x = TRUE)

egizio_df$school_holidays <- factor(egizio_df$school_holidays, ordered = TRUE, levels = 0:31)

str(egizio_df)

write.csv(egizio_df, "../data/egizio_final.csv", row.names = FALSE)

# ToDo: The code up to here shall be moved to preprocessing.R. -> at a later stage

# ---------------------------------------------------------------------------- #
#                       Exploratory Data Analysis
# ---------------------------------------------------------------------------- #
# Plotting

egizio_visitors_df <- read.csv("../data/egizio_final.csv")

str(egizio_visitors_df)

# ToDos: Check whether better visualization techniques can be used.

# Old plots

# plot_visitors <- ggplot(egizio_df, aes(x = date, y = visitors)) +
#   geom_line(color = "red") +
#   labs(title = "Visitors over time", x = "Date", y = "Visitors")
# 
# plot_trends <- ggplot(egizio_df, aes(x = date, y = trends)) +
#   geom_line(color = "blue") +
#   labs(title = "Trends over time", x = "Date", y = "Trends")
# 
# plot_school_holidays <- ggplot(egizio_df, aes(x = date, y = school_holidays)) +
#   geom_line(color = "green") +
#   labs(title = "School holidays over time", x = "Date", y = "School holidays")
# 
# plot_avg_temp <- ggplot(egizio_df, aes(x = date, y = average_temperature)) +
#   geom_line(color = "purple") +
#   labs(title = "Average temperature over time", x = "Date", y = "Avg. Temp.")
# 
# plot_rain <- ggplot(egizio_df, aes(x = date, y = raining_days)) +
#   geom_line(color = "cyan") +
#   labs(title = "Raining days over time", x = "Date", y = "Raining days")

# combined_plot <- ggarrange(plot_trends, plot_visitors, plot_school_holidays, plot_avg_temp, plot_rain, ncol = 1, nrow = 5)

# Print the combined plot
# print(combined_plot)
# Print the two separate plots
# print(plot_trends)
# print(plot_visitors)

# ---------------------------------------------------------------------------- #
# Smarter plotting

# Melt the dataframe for easier plotting
exclude_vars <- c("month", "quarter", "year")
egizio_filtered_df <- egizio_df[, !(names(egizio_df) %in% exclude_vars)]
egizio_melted_df <- reshape2::melt(egizio_filtered_df, id.vars = "date")

# Create a single plot for all variables
ggplot(egizio_melted_df, aes(x = date, y = value, color = variable)) +
  geom_line() +
  labs(title = "Egizio", x = "Date", y = "Values") +
  facet_wrap(~ variable, scales = "free_y", ncol = 1)

# ---------------------------------------------------------------------------- #
# Boxplots

ggplot(egizio_melted_df, aes(x = date, y = value, fill = variable)) +
  geom_boxplot() +
  labs(title = "Egizio - Boxplots", x = "Date", y = "Values") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +  # Rotate x-axis labels
  theme(axis.title.x = element_blank(), axis.text.x = element_blank()) +
  facet_wrap(~ variable, scales = "free_y", ncol = length(unique(egizio_melted_df$variable)))

# ---------------------------------------------------------------------------- #
# Violin plots

ggplot(egizio_melted_df, aes(x = date, y = value, fill = variable)) +
  geom_violin() +
  labs(title = "Egizio - Violinplots", x = "Date", y = "Values") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +  # Rotate x-axis labels
  theme(axis.title.x = element_blank(), axis.text.x = element_blank()) +
  facet_wrap(~ variable, scales = "free_y", ncol = length(unique(egizio_melted_df$variable)))

# ---------------------------------------------------------------------------- #
# Plot just for one year
select_year <- "2005"
egizio_one_year_df <- subset(egizio_df, year == select_year)
egizio_one_year_filtered_df <- egizio_one_year_df[, !(names(egizio_one_year_df) %in% exclude_vars)]
egizio_one_year_melted_df <- reshape2::melt(egizio_one_year_filtered_df, id.vars = "date")

ggplot(egizio_one_year_melted_df, aes(x = date, y = value, color = variable)) +
  geom_line() +
  labs(title = paste("Egizio -", select_year), x = "Date", y = "Values") +
  facet_wrap(~ variable, scales = "free_y", ncol = 1)

# ---------------------------------------------------------------------------- #
# Monthly and yearly boxplots

egizio_melted_df <- egizio_df
egizio_melted_df$month <- as.integer(egizio_melted_df$month)
exclude_vars <- c("year", "quarter")
egizio_melted_df <- egizio_melted_df[, !(names(egizio_melted_df) %in% exclude_vars)]

egizio_melted_df <- reshape2::melt(egizio_melted_df, id.vars = c("date", "month"))
egizio_df_melted_visitors <- subset(egizio_melted_df, variable == "visitors")

ggplot(egizio_df_melted_visitors, aes(x = factor(month), y = value, fill = variable)) +
  geom_boxplot() +
  labs(title = "Egizio - Monthly Boxplots for Visitors", x = "Month", y = "Visitors") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +  # Rotate x-axis labels
  theme(axis.title.x = element_blank(), axis.text.x = element_blank()) +
  facet_wrap(~ month, scales = "free_x", ncol = 12)

# Create boxplots for each year

egizio_melted_df <- egizio_df
egizio_melted_df$month <- as.integer(egizio_melted_df$month)
exclude_vars <- c("month", "quarter")
egizio_melted_df <- egizio_melted_df[, !(names(egizio_melted_df) %in% exclude_vars)]

egizio_melted_df <- reshape2::melt(egizio_melted_df, id.vars = c("date", "year"))
egizio_df_melted_visitors <- subset(egizio_melted_df, variable == "visitors")

ggplot(egizio_df_melted_visitors, aes(x = factor(year), y = value, fill = variable)) +
  geom_boxplot() +
  labs(title = "Egizio - Boxplots for Visitors", x = "Year", y = "Visitors") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +  # Rotate x-axis labels
  theme(axis.title.x = element_blank(), axis.text.x = element_blank()) +
  facet_wrap(~ year, scales = "free_x", ncol = length(unique(egizio_df_melted_visitors$year)))

# ---------------------------------------------------------------------------- #
#                               Modelling
# ---------------------------------------------------------------------------- #

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

features <- c('month', 'quarter', 'year' , 'trends', 'average_temperature', 'raining_days', 'school_holidays')
target <- 'visitors'

# ---------------------------------------------------------------------------- #
# Train-test split

# ToDo (Grazina): Think how to do the train-test splitting (and what to do with COVID years).

egizio_train_df <- subset(egizio_df, format(date, "%Y") != "2022")
egizio_test_df <- subset(egizio_df, format(date, "%Y") == "2022")

cat("Egizio train size:", nrow(egizio_train_df), "rows (months).")
cat("Egizio test size:", nrow(egizio_test_df), "rows (months).")

ratio_train <- nrow(egizio_train_df) / nrow(egizio_df)
ratio_test <- 1 - ratio_train
ratio_train <- ratio_train * 100
ratio_test <- ratio_test * 100
cat("Ratio of train set size to test set size:", ratio_train, ":", ratio_test)

# ToDo: Decide how to perform the train-test split. Due to the huge variation
# during COVID, we might need to avoid using it in the test set.

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

# Example usage
standardize <- standardize_numeric_columns(egizio_train_df, egizio_test_df)
egizio_train_df <- standardize$train_df
egizio_test_df <- standardize$test_df


# ToDo: Decide how to improve the legend.
# ToDo: Also, maybe it's better to split it to two subplots: top - visitors, bottom - trends?
ggplot() +
  geom_line(data = egizio_train_df, aes(x = date, y = visitors, color = "Visitors", linetype = "Train"), size = 1.5) +
  geom_line(data = egizio_test_df, aes(x = date, y = visitors, color = "Visitors", linetype = "Test"), size = 1.5) +
  geom_line(data = egizio_train_df, aes(x = date, y = trends, color = "Trends", linetype = "Train"), size = 1.5) +
  geom_line(data = egizio_test_df, aes(x = date, y = trends, color = "Trends", linetype = "Test"), size = 1.5) +
  labs(title = "Visitors and Trends over time", x = "Date", y = "Values") +
  scale_color_manual(name = "Variable", values = c("Visitors"="red", "Trends"="blue")) +
  scale_linetype_manual(name = "Dataset", values = c("Train"="solid", "Test"="dashed")) +
  geom_vline(xintercept = as.numeric(min(egizio_test_df$date)), linetype = "dotted", color = "black") +
  theme_minimal()

# This dataframe will be used to store the predictions of all of the models, and make plotting easier.
egizio_predictions_df <- data.frame(date = egizio_test_df$date)
egizio_predictions_df$visitors_true <- egizio_test_df$visitors

# ---------------------------------------------------------------------------- #
# Baseline model - mean of training

mean_train_visitors <- mean(egizio_train_df$visitors)
egizio_predictions_df$predicted_visitors_mean <- mean_train_visitors

# Calculate metrics
# ToDo: Check the R_squared calculation.
# residual sum of squares
rss <- sum((egizio_predictions_df$predicted_visitors_mean - egizio_test_df$visitors) ^ 2)
# total sum of squares
tss <- sum((egizio_test_df$visitors - mean(egizio_test_df$visitors)) ^ 2)
r_squared <- 1 - rss / tss

mse <- mean((egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_mean) ^ 2)
rmse <- sqrt(mse)
mae <- mean(abs(egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_mean))
mape <- mean(abs((egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_mean) / egizio_test_df$visitors)) * 100

# ToDo: Check whether the AIC calculation makes sense.
n <- nrow(egizio_test_df)
sse <- sum((egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_mean) ^ 2)
k <- 1
aic <- n * log(sse / n) + 2 * k
adj_r_squared <- 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

metrics_df <- rbind(metrics_df, list(Model = "Baseline - mean",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic))
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

# ToDo: Check the R_squared calculation.
rss <- sum((egizio_predictions_df$predicted_visitors_last_year - egizio_test_df$visitors) ^ 2)
tss <- sum((egizio_test_df$visitors - mean(egizio_test_df$visitors)) ^ 2)
r_squared <- 1 - rss / tss

mse <- mean((egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_last_year) ^ 2)
rmse <- sqrt(mse)
mae <- mean(abs(egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_last_year))
mape <- mean(abs((egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_last_year) / egizio_test_df$visitors)) * 100

n <- nrow(egizio_test_df)
sse <- sum((egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_last_year) ^ 2)
k <- 1
aic <- n * log(sse / n) + 2 * k
adj_r_squared <- 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

metrics_df <- rbind(metrics_df, list(Model = "Baseline - last year",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic))
print(metrics_df)

ggplot(egizio_predictions_df, aes(x = date)) +
  geom_line(aes(y = predicted_visitors_last_year, color = "Predicted"),
    linetype = "dashed", size = 1) +
  geom_line(aes(y = visitors_true, color = "Actual"), size = 1) +
  labs(title = "Auto regressive Baseline - same month from previous year",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))

# ---------------------------------------------------------------------------- #
# Baseline model - auto regressive (tomorrow=today)

# Create a lagged variable for 'dependent_variable'
egizio_test_df$visitors_yesterday_value <- c(NA, egizio_test_df$visitors[-nrow(egizio_test_df)])
# Filter out the first row with NA value
baseline_data <- egizio_test_df[complete.cases(egizio_test_df$visitors_yesterday_value),]
# Fit a baseline model
baseline_model <- lm(visitors ~ visitors_yesterday_value, data = baseline_data)
# Make predictions using the baseline model
baseline_data$predicted_visitors_autoregressive <- predict(baseline_model, newdata = baseline_data)
egizio_predictions_df$predicted_visitors_autoregressive[2:nrow(egizio_predictions_df)] <- baseline_data$predicted_visitors_autoregressive

# Calculate metrics
r_squared <- summary(baseline_model)$r.squared
mse <- mean((baseline_data$visitors - baseline_data$predicted_visitors_autoregressive) ^ 2)
rmse <- sqrt(mean((baseline_data$visitors - baseline_data$predicted_visitors_autoregressive) ^ 2))
mae <- mean(abs(baseline_data$visitors - baseline_data$predicted_visitors_autoregressive))
mape <- mean(abs((baseline_data$visitors - baseline_data$predicted_visitors_autoregressive) / baseline_data$visitors)) * 100
aic <- AIC(baseline_model)
adj_r_squared <- 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

metrics_df <- rbind(metrics_df, list(Model = "Baseline - auto regressive",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic))
print(metrics_df)

ggplot(egizio_predictions_df[-1, ], aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Actual"), size = 1) +
  geom_line(aes(y = predicted_visitors_autoregressive, color = "Predicted"),
    linetype = "dashed", size = 1) +
  labs(title = "Auto regressive Baseline",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))

egizio_test_df <- egizio_test_df[, !colnames(egizio_test_df) %in% "visitors_yesterday_value"]

# ---------------------------------------------------------------------------- #
# Model 1 - Multiple LR

# ToDo: Add other dependent variables
str(egizio_train_df)
all_features_regression <- lm(visitors ~ .-visitors - date - quarter, data = egizio_train_df)
# Month and quarter are correlated, so we have to remove the quarter.
summary(all_features_regression)
# ToDo: The model works better if we include date. Investigate this.

egizio_predictions_df$predicted_multiple_lr <- predict(all_features_regression, newdata = egizio_test_df)

# Calculate metrics
r_squared <- summary(all_features_regression)$r.squared
adj_r_squared <- summary(all_features_regression)$adj.r.squared
mse <- mean((egizio_test_df$visitors - egizio_predictions_df$predicted_multiple_lr) ^ 2)
rmse <- sqrt(mean((egizio_test_df$visitors - egizio_predictions_df$predicted_multiple_lr) ^ 2))
mae <- mean(abs(egizio_test_df$visitors - egizio_predictions_df$predicted_multiple_lr))
mape <- mean(abs((egizio_test_df$visitors - egizio_predictions_df$predicted_multiple_lr) / egizio_test_df$visitors)) * 100
aic <- AIC(all_features_regression)

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

# --------------------------------------------------------------------- #
# Model - Boosting

# Modify graphical parameters
mai.old <- par()$mai
mai.new <- mai.old # new vector
mai.new[2] <- 2.5 #new space on the left

# This can be used visitors ~ .- visitors - date + as.numeric(date)
boost_visitors <- gbm(visitors ~ .-visitors - date - quarter, data=egizio_train_df, 
                      distribution="gaussian", n.trees=5000, interaction.depth=1)

par(mai=mai.new)
summary(boost_visitors, las=1, cBar=10)
par(mai=mai.old)

yhat_boost <- predict(boost_visitors, newdata=egizio_test_df, n.trees=1:5000)
err <- apply(yhat_boost, 2, function(pred) mean((egizio_test_df$visitors - pred)^2))

# Error comparison (train and test)
plot(boost_visitors$train.error, type="l", ylim=c(0, max(err)))
lines(err, type="l", col=2)
best <- which.min(err) # Minimum error in test set
abline(v=best, lty=2, col=4)
print(min(err)) # minimum error

# 2 Boosting - Deeper trees
boost_visitors <- gbm(visitors ~ . - visitors - date - quarter,
                      data=egizio_train_df, distribution="gaussian", n.trees=5000,
                      interaction.depth=4) # (with more than one variable)

par(mai=mai.new)
summary(boost_visitors, las=1, cBar=10)
par(mai=mai.old)

yhat_boost <- predict(boost_visitors, newdata=egizio_test_df, n.trees=1:5000)
err <- apply(yhat_boost, 2, function(pred) mean((egizio_test_df$visitors - pred)^2))

plot(boost_visitors$train.error, type="l", ylim=c(0, max(err)))
lines(err, type="l", col=2)
best <- which.min(err)
abline(v=best, lty=2, col=4)
print(min(err))

# 3 Boosting - Smaller learning rate 
boost_visitors <- gbm(visitors ~ .- visitors - date - quarter, data=egizio_train_df,
                      distribution="gaussian", n.trees=5000, interaction.depth=1,
                      shrinkage=0.01) # learning rate

par(mai=mai.new)
summary(boost_visitors, las=1, cBar=10)
par(mai=mai.old)

yhat_boost <- predict(boost_visitors, newdata=egizio_test_df, n.trees=1:5000)
err <- apply(yhat_boost, 2, function(pred) mean((egizio_test_df$visitors - pred)^2))

plot(boost_visitors$train.error, type="l", ylim=c(0, max(err)))
lines(err, type="l", col=2)
best <- which.min(err)
abline(v=best, lty=2, col=4)
print(min(err))

# 4 Boosting - combination of previous models
boost_visitors <- gbm(visitors ~ .- visitors - date - quarter, data=egizio_train_df,
                      distribution="gaussian", n.trees=5000,
                      interaction.depth=4, shrinkage=0.01)
# interaction.depth=4 we consider a deeper three, interactions inside the tree

par(mai=mai.new)
summary(boost_visitors, las=1, cBar=10)
par(mai=mai.old)

yhat_boost <- predict(boost_visitors, newdata=egizio_test_df, n.trees=1:5000)
err <- apply(yhat_boost, 2, function(pred) mean((egizio_test_df$visitors - pred)^2))

plot(boost_visitors$train.error, type="l", ylim=c(0, max(err)))
lines(err, type="l", col=2)
best <- which.min(err)
abline(v=best, lty=2, col=4)
print(min(err))

# partial dependence plots
plot(boost_visitors, i.var=1, n.trees = best)
plot(boost_visitors, i.var=2, n.trees = best)
plot(boost_visitors, i.var=3, n.trees = best)
plot(boost_visitors, i.var=4, n.trees = best)
plot(boost_visitors, i.var=5, n.trees = best)
plot(boost_visitors, i.var=6, n.trees = best)
plot(boost_visitors, i.var=7, n.trees = best)
# ToDo: Add another one after adding tourist data
plot(boost_visitors, i.var=c(4,6), n.trees = best)

# --------------------------------------------------------------------- #
# Cross-validation for boosting - not the time series method => wrong

grid <- expand.grid(n.trees = c(100, 500, 1000, 5000),
                    interaction.depth = 1:5, 
                    shrinkage = c(0.1, 0.025, 0.05, 0.075, 0.01),
                    n.minobsinnode = c(3, 5, 10, 20))

train_control <- trainControl(method="cv", number=5)  

gbm_grid <- train(visitors ~ . - visitors - date, data=egizio_train_df, 
                  method = "gbm",
                  trControl = train_control,
                  tuneGrid = grid,
                  verbose = FALSE)

best_params <- gbm_grid$bestTune 
# gbm_grid$results

# Train final model on full training set
best_model <- gbm(visitors ~ . - visitors - date, data = egizio_train_df, 
                  n.trees = best_params$n.trees,
                  interaction.depth = best_params$interaction.depth,
                  shrinkage = best_params$shrinkage,
                  n.minobsinnode = best_params$n.minobsinnode)

# Make predictions on test set
yhat_boost <- predict(best_model, newdata = egizio_test_df)

# Evaluate test MSE
mean((egizio_test_df$visitors - yhat_boost)^2)

# Plot predictions 
plot(egizio_test_df$visitors, yhat_boost)
abline(0,1)

# --------------------------------------------------------------------- #
# Time-series cross-validation

ts_cv_spec <- time_series_cv(data = egizio_train_df,
                             date_var = date,
                             initial = 48,
                             assess = 12,
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

grid <- expand.grid(n.trees = 5000,
                    interaction.depth = 1:5, 
                    shrinkage = c(0.1, 0.025, 0.05, 0.075, 0.01),
                    n.minobsinnode = 10)

train_controls <- trainControl(method = "timeslice",# time-series cross-validation
                               initialWindow = 48, # initial training window
                               horizon = 12, # forecast evaluation window
                               fixedWindow = TRUE, 
                               skip = 12,
                               allowParallel = TRUE) # allow parallel processing if available

gbm_grid <- train(visitors ~ . - visitors - date,
                  data = egizio_train_df,
                  method = "gbm",  
                  distribution = "gaussian",
                  trControl = train_controls,
                  tuneGrid = grid,
                  verbose = FALSE)

# View the results of the grid search
print(gbm_grid)

best_model <- gbm_grid$bestTune

final_model_boosting <- gbm(visitors ~ . - visitors - date, data = egizio_train_df,
                        distribution = "gaussian",
                        n.trees = best_model$n.trees,
                        interaction.depth = best_model$interaction.depth,
                        shrinkage = best_model$shrinkage,
                        n.minobsinnode = best_model$n.minobsinnode)

par(mai=mai.new)
summary(final_model_boosting, las=1, cBar=10)
par(mai=mai.old)

egizio_predictions_df$predicted_visitors_boosting <- predict(final_model_boosting,
                                                             newdata = egizio_test_df,
                                                             n.trees = best_model$n.trees)

r_squared <- cor(egizio_test_df$visitors, egizio_predictions_df$predicted_visitors_boosting)^2
mse <- mean((egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_boosting) ^ 2)
rmse <- sqrt(mean((egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_boosting) ^ 2))
mae <- mean(abs(egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_boosting))
mape <- mean(abs((egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_boosting) / egizio_test_df$visitors)) * 100

metrics_df <- rbind(metrics_df, list(Model = "Boosting - TSCV",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = NA))
print(metrics_df)

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
training.x <- model.matrix(visitors ~ . - visitors - date, data = egizio_train_df)
testing.x <- model.matrix(visitors ~ . - visitors - date, data = egizio_test_df)

model.xgb <- xgboost(data=data.matrix(training.x[,-1]), # ignore intercept
                     label=as.numeric(as.character(egizio_train_df$visitors)),
                     eta=0.01, # default=0.3 (takes values in (0-1])
                     max_depth=10, # default=6 (takes values in (0,Inf)), larger value => more complex => overfitting
                     nrounds=200, # default=100 - controls number of iterations (number of trees)
                     early_stopping_rounds=50,
                     print_every_n = 10,
                     objective="reg:squarederror") # for linear regression
# objective="reg:squarederror"
# eval_metric = "rmse"

egizio_predictions_df$predicted_visitors_xgboost <- predict(model.xgb, newdata = testing.x[,-1])

# Calculate metrics for XGBoost
r_squared_xgb <- 1 - mean((egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_xgboost) ^ 2) / mean((egizio_test_df$visitors - mean(egizio_test_df$visitors)) ^ 2)
mse_xgb <- mean((egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_xgboost) ^ 2)
rmse_xgb <- sqrt(mean((egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_xgboost) ^ 2))
mae_xgb <- mean(abs(egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_xgboost))
mape_xgb <- mean(abs((egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_xgboost) / egizio_test_df$visitors)) * 100

# Update metrics_df with XGBoost metrics
metrics_df <- rbind(metrics_df, list(Model = "XGBoost",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = NA)) # Note: AIC may not be applicable for XGBoost

print(metrics_df)

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

model.xgb <- train(x = data.matrix(training.x[, -1]), # Ignore intercept
                   y = as.numeric(as.character(egizio_train_df$visitors)),
                   method = "xgbTree", # XGBoost 
                   trControl = train_controls,
                   tuneGrid = grid,
                   verbose = FALSE)

# Print the best model
best_model <- model.xgb$bestTune
print(best_model)

final_model_xgb <- xgboost(data=data.matrix(training.x[,-1]),
                           label=as.numeric(as.character(egizio_train_df$visitors)),
                           eta=best_model$eta, 
                           max_depth=best_model$max_depth, 
                           nrounds=best_model$nrounds,
                           colsample_bytree=best_model$colsample_bytree,
                           min_child_weight=best_model$min_child_weight,
                           subsample=best_model$subsample,
                           gamma=best_model$gamma,
                           objective="reg:squarederror")

importance_scores <- xgb.importance(model = final_model_xgb)
print(importance_scores)
xgb.plot.importance(importance_matrix = importance_scores)

# Perform predictions on the test set
egizio_predictions_df$predicted_visitors_xgboost_tscv <- predict(final_model_xgb,
                                                            newdata = data.matrix(testing.x[, -1]))

# Calculate metrics for XGBoost
r_squared <- 1 - mean((egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_xgboost_tscv) ^ 2) / mean((egizio_test_df$visitors - mean(egizio_test_df$visitors)) ^ 2)
mse <- mean((egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_xgboost_tscv) ^ 2)
rmse <- sqrt(mean((egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_xgboost_tscv) ^ 2))
mae <- mean(abs(egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_xgboost_tscv))
mape <- mean(abs((egizio_test_df$visitors - egizio_predictions_df$predicted_visitors_xgboost_tscv) / egizio_test_df$visitors)) * 100

# Update metrics_df with XGBoost metrics
metrics_df <- rbind(metrics_df, list(Model = "XGBoost - TSCV",
                                     R2 = r_squared, R2_adj = adj_r_squared,
                                     MSE = mse, RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = NA)) # Note: AIC may not be applicable for XGBoost

print(metrics_df)

ggplot(egizio_predictions_df, aes(x = date)) +
  geom_line(aes(y = visitors_true, color = "Visitors"), size = 1) +
  geom_line(aes(y = predicted_visitors_xgboost_tscv, color = "Predicted"),
            linetype = "dashed", size = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))

