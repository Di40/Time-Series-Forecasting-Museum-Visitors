# ---------------------------------------------------------------------------- #
#                     Cinema - Exploratory Data Analysis
# ---------------------------------------------------------------------------- #

rm(list=ls())

# Imports
library(ggplot2)
library(lubridate)
library(reshape2)
library(forecast)
library(dplyr)

# Change working directory
script_path <- rstudioapi::getSourceEditorContext()$path
script_dir <- dirname(script_path)
setwd(script_dir)

# To use English for the dates (instead of Macedonian/Italian)
Sys.setlocale("LC_TIME", "English")

# Load dataset

if (!file.exists("../../data/cinema_final.rds")) {
  source("preprocessing_cinema.R")
}
cinema_df <- readRDS("../../data/cinema_final.rds")

# We don't need month and year for visualization purposes, so we drop them.
columns_to_exclude <- c("month", "year")
cinema_df <- cinema_df[, !names(cinema_df) %in% columns_to_exclude]

str(cinema_df)

# Exclude the lagged regressors:
cinema_df <- cinema_df %>% select(-starts_with("lagged_"))

# Convert back to integer just for visualization purposes.
cinema_df$Covid_closures <- as.integer(cinema_df$Covid_closures)

str(cinema_df)

tsdisplay(cinema_df$visitors)

# ToDo: Check whether better visualization techniques can be used.
# ToDo: Plot the visitors of each year on the same graph, with different colors.

# ---------------------------------------------------------------------------- #
# Smart plotting

# Melt the dataframe for easier plotting
cinema_melted_df <- melt(cinema_df, id.vars = "date")

# Create a single plot for all variables
png("../../plots/cinema/time_series.png", width=1500, height=1000)
ggplot(cinema_melted_df, aes(x = date, y = value, color = variable)) +
  geom_line() +
  labs(title = "Cinema", x = "Date", y = "Values") +
  facet_wrap(~ variable, scales = "free_y", ncol = 1)
dev.off()

# ---------------------------------------------------------------------------- #
# Plot just for one year
select_year <- "2018" # Modify this as needed

columns_to_exclude <- c("Covid_closures")
cinema_reduced_df <- cinema_df[, !names(cinema_df) %in% columns_to_exclude]

cinema_one_year_df <- subset(cinema_reduced_df, year(date) == select_year)
cinema_one_year_melted_df <- melt(cinema_one_year_df, id.vars = "date")

png(paste0("../../plots/cinema/", select_year, "_time_series.png"), width=1500, height=1000)
ggplot(cinema_one_year_melted_df, aes(x = date, y = value, color = variable)) +
  geom_line() +
  labs(title = paste("Cinema -", select_year), x = "Date", y = "Values") +
  facet_wrap(~ variable, scales = "free_y", ncol = 1)
dev.off()

# ---------------------------------------------------------------------------- #
# Boxplots

cinema_reduced_melted_df <- melt(cinema_reduced_df, id.vars = "date")

png("../../plots/cinema/boxplots.png", width=1200, height=1000)
ggplot(cinema_reduced_melted_df, aes(x = date, y = value, fill = variable)) +
  geom_boxplot() +
  labs(title = "Cinema - Boxplots", x = "Date", y = "Values") +
  theme(axis.title.x = element_blank(), axis.text.x = element_blank()) + # blank x-axis
  facet_wrap(~ variable, scales = "free_y", ncol = length(unique(cinema_melted_df$variable)))
dev.off()

# ---------------------------------------------------------------------------- #
# Violin plots

png("../../plots/cinema/violin_plots.png", width=1200, height=1000)
ggplot(cinema_reduced_melted_df, aes(x = date, y = value, fill = variable)) +
  geom_violin() +
  labs(title = "Cinema - Violinplots", x = "Date", y = "Values") +
  theme(axis.title.x = element_blank(), axis.text.x = element_blank()) +
  facet_wrap(~ variable, scales = "free_y", ncol = length(unique(cinema_melted_df$variable)))
dev.off()

# ---------------------------------------------------------------------------- #
# Monthly and yearly boxplots

# Function to create monthly/yearly boxplots
create_boxplots <- function(data, group_var, title, x_label, y_label) {
  melted_df <- data
  
  if (group_var == "month") {
    melted_df$group_col <- month(melted_df$date)
    x_facet_label <- "Month"
  } else if (group_var == "year") {
    melted_df$group_col <- year(melted_df$date)
    x_facet_label <- "Year"
  } else {
    stop("Invalid group_var. Use 'month' or 'year'.")
  }
  
  melted_df <- melt(melted_df, id.vars = c("date", "group_col"))
  df_melted_visitors <- subset(melted_df, variable == "visitors")
  
  ggplot(df_melted_visitors, aes(x = factor(group_col), y = value, fill = variable)) +
    geom_boxplot() +
    labs(title = title, x = x_facet_label, y = y_label) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    theme(axis.title.x = element_blank(), axis.text.x = element_blank()) +
    facet_wrap(~ group_col, scales = "free_x", ncol = length(unique(df_melted_visitors$group_col)))
}

# Create boxplots for each month
png("../../plots/cinema/monthly_boxplots.png", width=1500, height=1000)
create_boxplots(cinema_df, "month", "Cinema - Monthly Boxplots for Visitors", "Month", "Visitors")
dev.off()

png("../../plots/cinema/yearly_boxplots.png", width=1500, height=1000)
# Create boxplots for each year
create_boxplots(cinema_df, "year", "Cinema - Yearly Boxplots for Visitors", "Year", "Visitors")
dev.off()
# ---------------------------------------------------------------------------- #