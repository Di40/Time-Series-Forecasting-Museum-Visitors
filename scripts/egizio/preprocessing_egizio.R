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

# write.csv(egizio_df, "../../data/egizio_final.csv", row.names = FALSE)
# csv doesn't store information about the changed variable types.
# So, we use RDS (R Data Serialization) file instead.

saveRDS(egizio_df, file = "../../data/egizio_final.rds")