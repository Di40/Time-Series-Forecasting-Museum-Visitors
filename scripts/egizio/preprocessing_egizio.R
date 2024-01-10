# ---------------------------------------------------------------------------- #
#                              Preprocessing
# ---------------------------------------------------------------------------- #

# Imports
library(zoo)
library(lubridate)
library(ggplot2)
library(stats)

rm(list=ls())

# Change working directory
script_path <- rstudioapi::getSourceEditorContext()$path
script_dir <- dirname(script_path)
setwd(script_dir)

# To use English for the dates (instead of Macedonian/Italian)
Sys.setlocale("LC_TIME", "English")

# ---------------------------------------------------------------------------- #
# Egizio visitors

egizio_visitors_df <- read.csv("../../data/museo_egizio_visitors.csv")

egizio_visitors_df$date <- as.Date(as.yearmon(paste(egizio_visitors_df$year,
                                                    egizio_visitors_df$month,
                                                    sep = "-")))

# egizio_visitors_df$quarter <- as.ordered(quarter(egizio_visitors_df$date))
# Decided to drop quarter due to high correlation with month.
egizio_visitors_df$month <- as.ordered(egizio_visitors_df$month)

# We can't use the year as a categorical variable, because if in the training
# set we have years 2006-2021, and we want to predict for the year 2022, this
# year will have no variable associated to it in the model, and predictions
# won't be possible.
# Therefore, we will keep year as integer.

# ToDo: Create lagged variables.

# Reorder columns
egizio_visitors_df <- egizio_visitors_df[, c('date', 'year', 'month', 'visitors')]

egizio_visitors_df$date_numeric <- as.numeric(egizio_visitors_df$date)

str(egizio_visitors_df)

print("Count of NAs in each column:")
print(colSums(is.na(egizio_visitors_df)))

egizio_visitors_df$visitors <- as.integer(egizio_visitors_df$visitors)
# NAs are created, because visitors was a string column, and there were empty strings.

print("Count of NAs in each column:")
colSums(is.na(egizio_visitors_df))

rows_egizio_with_na <- egizio_visitors_df[which(rowSums(is.na(egizio_visitors_df)) > 0), ]
egizio_visitors_df[is.na(egizio_visitors_df)] <- 0

print("Rows with NAs:")
print(rows_egizio_with_na)

print("Count of NAs in each column after modification:")
print(colSums(is.na(egizio_visitors_df)))

egizio_visitors_df$visitors <- as.integer(egizio_visitors_df$visitors)
str(egizio_visitors_df)

ggplot(egizio_visitors_df, aes(x = date, y = visitors)) +
  geom_line() +
  labs(title = "Egyptian Museum Visitors", x = "Date", y = "Visitors") +
  theme_minimal()

# ---------------------------------------------------------------------------- #
# Read Google Trends data
egizio_googletrends_df <- read.csv("../../data/museo_egizio_google_trends.csv")

egizio_googletrends_df$date <- as.Date(as.yearmon(egizio_googletrends_df$date))
egizio_googletrends_df$trends <- as.integer(egizio_googletrends_df$trends)

print("Count of NAs in each column:")
print(colSums(is.na(egizio_googletrends_df)))

# ---------------------------------------------------------------------------- #
# Read Turin weather data
turin_weather_df <- read.csv("../../data/turin_weather.csv")

turin_weather_df$date <- as.Date(as.yearmon(paste(turin_weather_df$year,
                                                  turin_weather_df$month,
                                                  sep = "-")))

# turin_weather_df$raining_days <- factor(turin_weather_df$raining_days, ordered = TRUE, levels = 0:31)

# Drop 'year' and 'month' and reorganize columns
turin_weather_df <- turin_weather_df[, c('date', 'average_temperature', 'raining_days')]

str(turin_weather_df)

print("Count of NAs in each column:")
print(colSums(is.na(turin_weather_df)))

# ---------------------------------------------------------------------------- #
# Read Italy holiday data
italy_holidays_df <- read.csv("../../data/italy_holidays.csv")

italy_holidays_df$date <- as.Date(as.yearmon(paste(italy_holidays_df$year,
                                                   italy_holidays_df$month,
                                                   sep = "-")))
# Drop 'year' and 'month' and reorganize columns
italy_holidays_df <- italy_holidays_df[, c('date', 'hl_sch')]
names(italy_holidays_df)[names(italy_holidays_df) == "hl_sch"] <- "school_holidays" # Rename

# italy_holidays_df$school_holidays <- factor(italy_holidays_df$school_holidays, ordered = TRUE, levels = 0:31)

str(italy_holidays_df)

print("Count of NAs in each column:")
print(colSums(is.na(italy_holidays_df)))

# ---------------------------------------------------------------------------- #
# Read Turin arrivals data 
turin_arrivals_df <- read.csv("../../data/turin_arrivals.csv")
str(turin_arrivals_df)

turin_arrivals_df$date <- as.Date(turin_arrivals_df$date, format = "%m/%d/%Y")

str(turin_arrivals_df)

print("Count of NAs in each column:")
print(colSums(is.na(turin_arrivals_df)))

# ---------------------------------------------------------------------------- #
# Read Covid-19_closures data
covid_closures_df <- read.csv("../../data/Covid19_closures.csv")

covid_closures_df$date <- as.Date(as.yearmon(paste(covid_closures_df$year,
                                                   covid_closures_df$month,
                                                  sep = "-")))

# Do we need to convert the binary variables to factor?
# Probably no, but it can be useful for standardization, to avoid standardizing it.
covid_closures_df$Covid_closures <- as.factor(covid_closures_df$Covid_closures)

str(covid_closures_df)

covid_closures_df <- covid_closures_df[, c('date', 'Covid_closures')]

print("Count of NAs in each column:")
print(colSums(is.na(covid_closures_df)))

# ---------------------------------------------------------------------------- #
# Read Museum renovation
egizio_renovation_df <- read.csv("../../data/museo_egizio_renovation.csv")

egizio_renovation_df$date <- as.Date(as.yearmon(paste(egizio_renovation_df$year,
                                                      egizio_renovation_df$month,
                                                   sep = "-")))

# Same as above, done for simplifying standardization.
egizio_renovation_df$renovation <- as.factor(egizio_renovation_df$renovation)

egizio_renovation_df <- egizio_renovation_df[, c('date', 'renovation')]

str(egizio_renovation_df)

print("Count of NAs in each column:")
print(colSums(is.na(egizio_renovation_df)))

# ---------------------------------------------------------------------------- #
# Concatenate dataframes

egizio_df <- merge(egizio_visitors_df, egizio_googletrends_df, by = "date", all.x = TRUE)
egizio_df <- merge(egizio_df, turin_weather_df, by = "date", all.x = TRUE)
egizio_df <- merge(egizio_df, italy_holidays_df, by = "date", all.x = TRUE)
egizio_df <- merge(egizio_df, turin_arrivals_df, by = "date", all.x = TRUE)
egizio_df <- merge(egizio_df, covid_closures_df, by = "date", all.x = TRUE)
egizio_df <- merge(egizio_df, egizio_renovation_df, by = "date", all.x = TRUE)

str(egizio_df)

print("Count of NAs in each column:")
print(colSums(is.na(egizio_df)))

# ---------------------------------------------------------------------------- #
# Lagged regressors

# We first check the correlation at different lags between our response variable
# and all the other variables. The lag k value returned by ccf(x, y) estimates
# the correlation between x[t+k] and y[t].

# Let's extend egizio_df with the lagged variables, i.e., we put inside 
# all the features that show correlation of type:
# visitors[t] ~ feature[t-k]
# We will get some NAs, but we will replace them with the current value.

str(egizio_df)

attach(egizio_df)

LAG_MAX <- 12

# Google Trends
ccf_trends <- ccf(visitors, trends, lag.max = LAG_MAX, plot = T)
# The lag 0 that has the maximum positive correlation, which is obvious.
abs_acf_values_trends <- abs(ccf_trends$acf)[1:12]
top_lags_trends <- order(abs_acf_values_trends, decreasing = TRUE)[1:3]
top_lags_trends <- top_lags_trends - LAG_MAX - 1
print(top_lags_trends)
# -11 -12  -1
# We have maximum correlation at:
# visitors[t] ~ google_trends[t-11]

# Create lagged trends
lag_trends <- abs(top_lags_trends[1])
print(lag_trends)
trends_lagged <- egizio_df$trends # copy
trends_lagged[(lag_trends+1):nrow(egizio_df)] <- egizio_df$trends[1:(nrow(egizio_df)-lag_trends)] # replace with lagged values
egizio_df$lagged_trends <- trends_lagged


# Average_temperature
ccf_avg_temp <- ccf(visitors, average_temperature, lag.max = LAG_MAX, plot = T)
abs_acf_values_avg_temp <- abs(ccf_avg_temp$acf)[1:12]
top_lags_avg_temp <- order(abs_acf_values_avg_temp, decreasing = TRUE)[1:3]
top_lags_avg_temp <- top_lags_avg_temp - LAG_MAX - 1
print(top_lags_avg_temp)
# -3 -4 -9
# We have maximum correlation at:
# visitors[t] ~ average_temperature[t-3]

# Create lagged average_temperature
lag_average_temperature <- abs(top_lags_avg_temp[1])
print(lag_average_temperature)
average_temperature_lagged <- egizio_df$average_temperature # copy
average_temperature_lagged[(lag_average_temperature+1):nrow(egizio_df)] <- egizio_df$average_temperature[1:(nrow(egizio_df)-lag_average_temperature)] # replace with lagged values
egizio_df$lagged_average_temperature <- average_temperature_lagged


# Raining_days
ccf_rain_days <- ccf(visitors, raining_days, lag.max = LAG_MAX, plot = T)
abs_acf_values_rain_days <- abs(ccf_rain_days$acf)[1:12]
top_lags_rain_days <- order(abs_acf_values_rain_days, decreasing = TRUE)[1:3]
top_lags_rain_days <- top_lags_rain_days - LAG_MAX - 1
print(top_lags_rain_days)
# -1 -8 -9
# We have maximum correlation at:
# visitors[t] ~ raining_days[t-1]

# Create lagged raining_days
lag_raining_days <- abs(top_lags_rain_days[1])
print(lag_raining_days)
raining_days_lagged <- egizio_df$raining_days # copy
raining_days_lagged[(lag_raining_days+1):nrow(egizio_df)] <- egizio_df$raining_days[1:(nrow(egizio_df)-lag_raining_days)] # replace with lagged values
egizio_df$lagged_raining_days <- raining_days_lagged


# School_holidays
ccf_school_holidays <- ccf(visitors, school_holidays, lag.max = LAG_MAX, plot = T)
abs_acf_values_school_holidays <- abs(ccf_school_holidays$acf)[1:12]
top_lags_school_holidays <- order(abs_acf_values_school_holidays, decreasing = TRUE)[1:3]
top_lags_school_holidays <- top_lags_school_holidays - LAG_MAX - 1
print(top_lags_school_holidays)
# -3  -4 -12
# We have maximum correlation at:
# visitors[t] ~ school_holidays[t-3]

# Create lagged school_holidays
lag_school_holidays <- abs(top_lags_school_holidays[1])
print(lag_school_holidays)
school_holidays_lagged <- egizio_df$school_holidays # copy
school_holidays_lagged[(lag_school_holidays+1):nrow(egizio_df)] <- egizio_df$school_holidays[1:(nrow(egizio_df)-lag_school_holidays)] # replace with lagged values
egizio_df$lagged_school_holidays <- school_holidays_lagged


# Arrivals
ccf_arrivals <- ccf(visitors, arrivals, lag.max = LAG_MAX, plot = T)
abs_acf_values_arrivals <- abs(ccf_arrivals$acf)[1:12]
top_lags_arrivals <- order(abs_acf_values_arrivals, decreasing = TRUE)[1:3]
top_lags_arrivals <- top_lags_arrivals - LAG_MAX - 1
print(top_lags_arrivals)
# -6 -1 -5
# We have maximum correlation at:
# visitors[t] ~ arrivals[t-6]

# Create lagged arrivals
lag_arrivals <- abs(top_lags_arrivals[1])
print(lag_arrivals)
arrivals_lagged <- egizio_df$arrivals # copy
arrivals_lagged[(lag_arrivals+1):nrow(egizio_df)] <- egizio_df$arrivals[1:(nrow(egizio_df)-lag_arrivals)] # replace with lagged values
egizio_df$lagged_arrivals <- arrivals_lagged


# Covid
ccf_covid <- ccf(visitors, Covid_closures, lag.max = LAG_MAX, plot = T)
# There are no significant lags.


# Renovation (this is a binary variable )
ccf_renovation <- ccf(visitors,renovation, lag.max = LAG_MAX, plot = T)
abs_acf_values_renovation <- abs(ccf_renovation$acf)[1:12]
top_lags_renovation <- order(abs_acf_values_renovation, decreasing = TRUE)[1:3]
top_lags_renovation <- top_lags_renovation - LAG_MAX - 1
print(top_lags_renovation)
# -1 -2 -3
# We have maximum correlation at:
# visitors[t] ~ renovation[t-1]

# Create lagged renovation
lag_renovation <- abs(top_lags_renovation[1])
print(lag_renovation)
renovation_lagged <- egizio_df$renovation # copy
renovation_lagged[(lag_renovation+1):nrow(egizio_df)] <- egizio_df$renovation[1:(nrow(egizio_df)-lag_renovation)] # replace with lagged values
egizio_df$lagged_renovation <- renovation_lagged

detach(egizio_df)

# ---------------------------------------------------------------------------- #
# Save the final dataset
# write.csv(egizio_df, "../../data/egizio_final.csv", row.names = FALSE)
# csv doesn't store information about the changed variable types.
# So, we use RDS (R Data Serialization) file instead.

saveRDS(egizio_df, file = "../../data/egizio_final.rds")
