# ---------------------------------------------------------------------------- #
# Imports
library(zoo)
library(ggplot2)
library(ggpubr)

# ---------------------------------------------------------------------------- #

# ToDo (At a later stage, not now): Restructure the code into multiple files:
# preprocessing.R
# EDA.R
# modelling.R

# ToDo: Load weather data.
# ToDo: Load holiday data.
# ToDo: Find and load number of tourists data.

# ---------------------------------------------------------------------------- #

script_path <- rstudioapi::getSourceEditorContext()$path
script_dir <- dirname(script_path)
setwd(script_dir)

# ---------------------------------------------------------------------------- #
# Egizio

egizio_visitors_df <- read.csv("../data/museo_egizio_visitors.csv")

egizio_visitors_df$date <- as.yearmon(paste(egizio_visitors_df$year,
                                            egizio_visitors_df$month,
                                            sep = "-"))
egizio_visitors_df$date <- as.Date(egizio_visitors_df$date)

str(egizio_visitors_df)

egizio_visitors_df$visitors <- as.integer(egizio_visitors_df$visitors)
# There are NAs

na_egizio_count <- colSums(is.na(egizio_visitors_df))
rows_egizio_with_na <- which(rowSums(is.na(egizio_visitors_df)) > 0)
rows_with_na_data <- egizio_visitors_df[rows_egizio_with_na,]
egizio_visitors_df[is.na(egizio_visitors_df)] <- 0

print("Count of NAs in each column:")
print(na_egizio_count)

print("Rows with NAs:")
print(rows_with_na_data)

print("Modified data frame with NAs replaced by 0:")
print(egizio_visitors_df)
# ToDo: Check if for the other museums the closing months
# for COVID are pretty much the same

ggplot(egizio_visitors_df, aes(x = date, y = visitors)) +
  geom_line() +
  labs(title = "Egyptian Museum Visitors", x = "Date", y = "Visitors") +
  theme_minimal()


egizio_googletrends_df <- read.csv("../data/museo_egizio_google_trends.csv")

egizio_googletrends_df$trends <- as.integer(egizio_googletrends_df$trends)

egizio_googletrends_df$date <- as.Date(as.yearmon(egizio_googletrends_df$date))

print("Count of NAs in each column:")
print(colSums(is.na(egizio_googletrends_df)))

# Concatenate dataframes

egizio_df <- merge(egizio_visitors_df, egizio_googletrends_df,
                   by = "date", all.x = TRUE)
egizio_df <- egizio_df[,!(colnames(egizio_df) %in% c("year", "month"))]

plot_visitors <- ggplot(egizio_df, aes(x = date, y = visitors)) +
  geom_line(color = "red") +
  labs(title = "Visitors over time", x = "Date", y = "Visitors")

plot_trends <- ggplot(egizio_df, aes(x = date, y = trends)) +
  geom_line(color = "blue") +
  labs(title = "Trends over time", x = "Date", y = "Trends")

combined_plot <- ggarrange(plot_trends, plot_visitors, ncol = 1, nrow = 2)

# Print the combined plot
print(combined_plot)
# Print the two separate plots
print(plot_trends)
print(plot_visitors)

# ---------------------------------------------------------------------------- #
# Modelling

metrics_df <- data.frame(
  Model = character(),
  R2 = numeric(),
  MSE = numeric(),
  RMSE = numeric(),
  MAE = numeric(),
  MAPE = numeric(),
  AIC = numeric(),
  stringsAsFactors = FALSE
)

egizio_train_df <- subset(egizio_df, format(date, "%Y") != "2022")
egizio_test_df <- subset(egizio_df, format(date, "%Y") == "2022")

cat("Egizio train size:", nrow(egizio_train_df), "rows\n")
cat("Egizio test size:", nrow(egizio_test_df), "rows\n")

ratio_train <- nrow(egizio_train_df) / nrow(egizio_df)
ratio_test <- 1 - ratio_train
ratio_train <- ratio_train * 100
ratio_test <- ratio_test * 100
cat("Ratio of train set size to test set size:", ratio_train, ":", ratio_test)

# ToDo: Decide how to perform the train-test split. Due to the huge variation
# during COVID, we might need to avoid using it in the test set.

# ---------------------------------------------------------------------------- #
# Baseline model - mean of training

mean_train_visitors <- mean(egizio_train_df$visitors)
egizio_test_df$predicted_visitors <- mean_train_visitors

# Calculate metrics
# residual sum of squares
rss <- sum((egizio_test_df$predicted_visitors - egizio_test_df$visitors) ^ 2)
# total sum of squares
tss <- sum((egizio_test_df$visitors - mean(egizio_test_df$visitors)) ^ 2)
r_squared <- 1 - rss / tss

mse <- mean((egizio_test_df$visitors - egizio_test_df$predicted_visitors) ^ 2)
rmse <- sqrt(mse)
mae <- mean(abs(egizio_test_df$visitors - egizio_test_df$predicted_visitors))
mape <- mean(abs((egizio_test_df$visitors - egizio_test_df$predicted_visitors) / egizio_test_df$visitors)) * 100

n <- nrow(egizio_test_df)
sse <- sum((egizio_test_df$visitors - egizio_test_df$predicted_visitors) ^ 2)
k <- 1
aic <- n * log(sse / n) + 2 * k

metrics_df <- rbind(metrics_df, list(Model = "Baseline - mean",
                                     R2 = r_squared, MSE = mse,
                                     RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic)
                    )
View(metrics_df)

ggplot(egizio_test_df, aes(x = date)) +
  geom_line(aes(y = predicted_visitors, color = "Predicted"),
            linetype = "dashed", size = 1) +
  geom_line(aes(y = visitors, color = "Actual"), size = 1) +
  labs(title = "Baseline - training mean",
       x = "Date",
       y = "Visitors") +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))

# ---------------------------------------------------------------------------- #
# Baseline model - same month last year

egizio_test_df$predicted_visitors_last_year <- NA
for (i in 1:nrow(egizio_test_df)) {
  # Find the corresponding date from the previous year
  last_year_date <- egizio_test_df$date[i] - months(12)
  
  # Check if the corresponding date exists in the training data
  if (last_year_date %in% egizio_train_df$date) {
    # Get the corresponding value from the training data
    corresponding_value <- egizio_train_df$visitors[egizio_train_df$date == last_year_date]
    
    # Assign the value to the predicted column
    egizio_test_df$predicted_visitors_last_year[i] <- corresponding_value
  }
}

rss <- sum((egizio_test_df$predicted_visitors_last_year - egizio_test_df$visitors) ^ 2)
tss <- sum((egizio_test_df$visitors - mean(egizio_test_df$visitors)) ^ 2)
r_squared <- 1 - rss / tss

mse <- mean((egizio_test_df$visitors - egizio_test_df$predicted_visitors_last_year) ^ 2)
rmse <- sqrt(mse)
mae <- mean(abs(egizio_test_df$visitors - egizio_test_df$predicted_visitors_last_year))
mape <- mean(abs((egizio_test_df$visitors - egizio_test_df$predicted_visitors_last_year) / egizio_test_df$visitors)) * 100

n <- nrow(egizio_test_df)
sse <- sum((egizio_test_df$visitors - egizio_test_df$predicted_visitors_last_year) ^ 2)
k <- 1
aic <- n * log(sse / n) + 2 * k


metrics_df <- rbind(metrics_df, list(Model = "Baseline - last year",
                                     R2 = r_squared, MSE = mse,
                                     RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic)
)

View(metrics_df)

ggplot(egizio_test_df, aes(x = date)) +
  geom_line(aes(y = predicted_visitors_last_year, color = "Predicted"),
    linetype = "dashed", size = 1) +
  geom_line(aes(y = visitors, color = "Actual"), size = 1) +
  labs(title = "Auto regressive Baseline - same month from previous year",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))

# ---------------------------------------------------------------------------- #
# Baseline model - auto regressive (tomorrow=today)

# Create a lagged variable for 'dependent_variable'
egizio_test_df$visitors_yesterday_value <-
  c(NA, egizio_test_df$visitors[-nrow(egizio_test_df)])
# Filter out the first row with NA value
baseline_data <-
  egizio_test_df[complete.cases(egizio_test_df$visitors_yesterday_value),]
# Fit a baseline model
baseline_model <-
  lm(visitors ~ visitors_yesterday_value, data = baseline_data)
# Make predictions using the baseline model
baseline_data$predicted_baseline <-
  predict(baseline_model, newdata = baseline_data)

# Calculate metrics
r_squared <- summary(baseline_model)$r.squared
mse <- mean((baseline_data$visitors - baseline_data$predicted_baseline) ^ 2)
rmse <- sqrt(mean((baseline_data$visitors - baseline_data$predicted_baseline) ^ 2))
mae <- mean(abs(baseline_data$visitors - baseline_data$predicted_baseline))
mape <- mean(abs((baseline_data$visitors - baseline_data$predicted_baseline) / baseline_data$visitors)) * 100
aic <- AIC(baseline_model)

metrics_df <- rbind(metrics_df, list(Model = "Baseline - auto regressive",
                                     R2 = r_squared, MSE = mse,
                                     RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic)
)

View(metrics_df)

ggplot(baseline_data, aes(x = date)) +
  geom_line(aes(y = visitors, color = "Actual"), size = 1) +
  geom_line(aes(y = predicted_baseline, color = "Predicted"),
    linetype = "dashed", size = 1) +
  labs(title = "Auto regressive Baseline",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))

# ---------------------------------------------------------------------------- #
# Model 1 - Multiple LR

# ToDo: Add other dependent variables
lm_model <- lm(visitors ~ trends, data = egizio_train_df)
summary(lm_model)

egizio_test_df$predicted_lm <- predict(lm_model, newdata = egizio_test_df)

# Calculate metrics
r_squared <- summary(lm_model)$r.squared
mse <- mean((egizio_test_df$visitors - egizio_test_df$predicted_lm) ^ 2)
rmse <- sqrt(mean((egizio_test_df$visitors - egizio_test_df$predicted_lm) ^ 2))
mae <- mean(abs(egizio_test_df$visitors - egizio_test_df$predicted_lm))
mape <- mean(abs((egizio_test_df$visitors - egizio_test_df$predicted_lm) / egizio_test_df$visitors)) * 100
aic <- AIC(lm_model)

metrics_df <- rbind(metrics_df, list(Model = "Multiple LR",
                                     R2 = r_squared, MSE = mse,
                                     RMSE = rmse, MAE = mae,
                                     MAPE = mape, AIC = aic)
                    )

View(metrics_df)

ggplot(egizio_test_df, aes(x = date)) +
  geom_line(aes(y = visitors, color = "Visitors"), size = 1) +
  geom_line(aes(y = predicted_lm, color = "Predicted"),
            linetype = "dashed", size = 1) +
  labs(title = "Visitors and Predicted Values Over Time",
       x = "Date",
       y = "Values") +
  scale_color_manual(values = c("Visitors" = "red", "Predicted" = "blue"))

# --------------------------------------------------------------------- #