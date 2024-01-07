# ---------------------------------------------------------------------------- #
#                              Preprocessing
# ---------------------------------------------------------------------------- #

# Imports
library(zoo)
library(lubridate)
library(ggplot2)

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
turin_arrivals_df <- read.csv("../../data/Turin_arrivals.csv")
str(turin_arrivals_df)

turin_arrivals_df$date <- as.Date(as.yearmon(turin_arrivals_df$date))

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

# write.csv(cinema_df, "../../data/cinema_final.csv", row.names = FALSE)
# csv doesn't store information about the changed variable types.
# So, we use RDS (R Data Serialization) file instead.

saveRDS(cinema_df, file = "../../data/cinema_final.rds")
