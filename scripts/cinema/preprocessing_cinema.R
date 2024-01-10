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
# Cinema museum visitors

cinema_visitors_df <- read.csv("../../data/museo_cinema_visitors.csv")

str(cinema_visitors_df)

cinema_visitors_df$date <- as.Date(as.yearmon(paste(cinema_visitors_df$year,
                                                    cinema_visitors_df$month,
                                                    sep = "-")))

cinema_visitors_df$month <- as.ordered(cinema_visitors_df$month)

# We can't use the year as a categorical variable, because if in the training
# set we have years 2006-2021, and we want to predict for the year 2022, this
# year will have no variable associated to it in the model, and predictions
# won't be possible.
# Therefore, we will keep year as integer.

# ToDo: Create lagged variables.

# Reorder columns
cinema_visitors_df <- cinema_visitors_df[, c('date', 'year', 'month', 'visitors')]

cinema_visitors_df$date_numeric <- as.numeric(cinema_visitors_df$date)

str(cinema_visitors_df)

print("Count of NAs in each column:")
print(colSums(is.na(cinema_visitors_df)))

cinema_visitors_df$visitors <- as.integer(cinema_visitors_df$visitors)
# NAs are created for COVID months, when the museum was closed.

print("Count of NAs in each column:")
colSums(is.na(cinema_visitors_df))

rows_cinema_with_na <- cinema_visitors_df[which(rowSums(is.na(cinema_visitors_df)) > 0), ]
cinema_visitors_df[is.na(cinema_visitors_df)] <- 0

print("Rows with NAs:")
print(rows_cinema_with_na)

print("Count of NAs in each column after modification:")
print(colSums(is.na(cinema_visitors_df)))

cinema_visitors_df$visitors <- as.integer(cinema_visitors_df$visitors)
str(cinema_visitors_df)

ggplot(cinema_visitors_df, aes(x = date, y = visitors)) +
  geom_line() +
  labs(title = "Cinema Visitors", x = "Date", y = "Visitors") +
  theme_minimal()

# ---------------------------------------------------------------------------- #
# Read Google Trends data
cinema_googletrends_df <- read.csv("../../data/museo_cinema_google_trends.csv")

cinema_googletrends_df$date <- as.Date(as.yearmon(cinema_googletrends_df$date))
cinema_googletrends_df$trends <- as.integer(cinema_googletrends_df$trends)

print("Count of NAs in each column:")
print(colSums(is.na(cinema_googletrends_df)))

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
# Concatenate dataframes

cinema_df <- merge(cinema_visitors_df, cinema_googletrends_df, by = "date", all.x = TRUE)
cinema_df <- merge(cinema_df, turin_weather_df, by = "date", all.x = TRUE)
cinema_df <- merge(cinema_df, italy_holidays_df, by = "date", all.x = TRUE)
cinema_df <- merge(cinema_df, turin_arrivals_df, by = "date", all.x = TRUE)
cinema_df <- merge(cinema_df, covid_closures_df, by = "date", all.x = TRUE)

str(cinema_df)

print("Count of NAs in each column:")
print(colSums(is.na(cinema_df)))

# ---------------------------------------------------------------------------- #
# Lagged regressors

# We first check the correlation at different lags between our response variable
# and all the other variables. The lag k value returned by ccf(x, y) estimates
# the correlation between x[t+k] and y[t].

# Let's extend cinema_df with the lagged variables, i.e., we put inside 
# all the features that show correlation of type:
# visitors[t] ~ feature[t-k]
# We will get some NAs, but we will replace them with the current value.

str(cinema_df)

attach(cinema_df)

LAG_MAX <- 12

# Google Trends
ccf_trends <- ccf(visitors, trends, lag.max = LAG_MAX, plot = T)
# The lag 0 that has the maximum positive correlation, which is obvious.
abs_acf_values_trends <- abs(ccf_trends$acf)[1:12]
top_lags_trends <- order(abs_acf_values_trends, decreasing = TRUE)[1:3]
top_lags_trends <- top_lags_trends - LAG_MAX - 1
print(top_lags_trends)
# -3 -2 -8
# We have maximum correlation at:
# visitors[t] ~ google_trends[t-3]

# Create lagged trends
lag_trends <- abs(top_lags_trends[1])
print(lag_trends)
trends_lagged <- cinema_df$trends # copy
trends_lagged[(lag_trends+1):nrow(cinema_df)] <- cinema_df$trends[1:(nrow(cinema_df)-lag_trends)] # replace with lagged values
cinema_df$lagged_trends <- trends_lagged


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
average_temperature_lagged <- cinema_df$average_temperature # copy
average_temperature_lagged[(lag_average_temperature+1):nrow(cinema_df)] <- cinema_df$average_temperature[1:(nrow(cinema_df)-lag_average_temperature)] # replace with lagged values
cinema_df$lagged_average_temperature <- average_temperature_lagged


# Raining_days
ccf_rain_days <- ccf(visitors, raining_days, lag.max = LAG_MAX, plot = T)
abs_acf_values_rain_days <- abs(ccf_rain_days$acf)[1:12]
top_lags_rain_days <- order(abs_acf_values_rain_days, decreasing = TRUE)[1:3]
top_lags_rain_days <- top_lags_rain_days - LAG_MAX - 1
print(top_lags_rain_days)
# -1  -9 -12
# We have maximum correlation at:
# visitors[t] ~ raining_days[t-1]

# Create lagged raining_days
lag_raining_days <- abs(top_lags_rain_days[1])
print(lag_raining_days)
raining_days_lagged <- cinema_df$raining_days # copy
raining_days_lagged[(lag_raining_days+1):nrow(cinema_df)] <- cinema_df$raining_days[1:(nrow(cinema_df)-lag_raining_days)] # replace with lagged values
cinema_df$lagged_raining_days <- raining_days_lagged


# School_holidays
ccf_school_holidays <- ccf(visitors, school_holidays, lag.max = LAG_MAX, plot = T)
abs_acf_values_school_holidays <- abs(ccf_school_holidays$acf)[1:12]
top_lags_school_holidays <- order(abs_acf_values_school_holidays, decreasing = TRUE)[1:3]
top_lags_school_holidays <- top_lags_school_holidays - LAG_MAX - 1
print(top_lags_school_holidays)
# -3  -4 -6
# We have maximum correlation at:
# visitors[t] ~ school_holidays[t-3]

# Create lagged school_holidays
lag_school_holidays <- abs(top_lags_school_holidays[1])
print(lag_school_holidays)
school_holidays_lagged <- cinema_df$school_holidays # copy
school_holidays_lagged[(lag_school_holidays+1):nrow(cinema_df)] <- cinema_df$school_holidays[1:(nrow(cinema_df)-lag_school_holidays)] # replace with lagged values
cinema_df$lagged_school_holidays <- school_holidays_lagged


# Arrivals
ccf_arrivals <- ccf(visitors, arrivals, lag.max = LAG_MAX, plot = T)
abs_acf_values_arrivals <- abs(ccf_arrivals$acf)[1:12]
top_lags_arrivals <- order(abs_acf_values_arrivals, decreasing = TRUE)[1:3]
top_lags_arrivals <- top_lags_arrivals - LAG_MAX - 1
print(top_lags_arrivals)
# -6 -12  -1
# We have maximum correlation at:
# visitors[t] ~ arrivals[t-6]

# Create lagged arrivals
lag_arrivals <- abs(top_lags_arrivals[1])
print(lag_arrivals)
arrivals_lagged <- cinema_df$arrivals # copy
arrivals_lagged[(lag_arrivals+1):nrow(cinema_df)] <- cinema_df$arrivals[1:(nrow(cinema_df)-lag_arrivals)] # replace with lagged values
cinema_df$lagged_arrivals <- arrivals_lagged


# Covid (this is a binary variable )
ccf_covid <- ccf(visitors, Covid_closures, lag.max = LAG_MAX, plot = T)
# Here, there are significant lags (for Egizio there aren't).
abs_acf_values_covid <- abs(ccf_covid$acf)[1:12]
top_lags_covid <- order(abs_acf_values_covid, decreasing = TRUE)[1:3]
top_lags_covid <- top_lags_covid - LAG_MAX - 1
print(top_lags_covid)
# -1  -9 -10
# We have maximum correlation at:
# visitors[t] ~ Covid_closures[t-1]

# Create lagged covid_closures
lag_covid <- abs(top_lags_covid[1])
print(lag_covid)
covid_lagged <- cinema_df$Covid_closures # copy
covid_lagged[(lag_covid+1):nrow(cinema_df)] <- cinema_df$Covid_closures[1:(nrow(cinema_df)-lag_covid)] # replace with lagged values
cinema_df$lagged_covid_closures <- covid_lagged

detach(cinema_df)

# ---------------------------------------------------------------------------- #
# Save the final dataset
# write.csv(cinema_df, "../../data/cinema_final.csv", row.names = FALSE)
# csv doesn't store information about the changed variable types.
# So, we use RDS (R Data Serialization) file instead.

saveRDS(cinema_df, file = "../../data/cinema_final.rds")
