# ---------------------------------------------------------------------------- #
#                       Exploratory Data Analysis
# ---------------------------------------------------------------------------- #

# Imports
library(ggplot2)
library(lubridate)
library(reshape2)

# Change working directory
script_path <- rstudioapi::getSourceEditorContext()$path
script_dir <- dirname(script_path)
setwd(script_dir)

# To use English for the dates (instead of Macedonian/Italian)
Sys.setlocale("LC_TIME", "English")

# Plotting

# ---------------------------------------------------------------------------- #
#                                  Egizio                                      #
# ---------------------------------------------------------------------------- #

egizio_df <- readRDS("../data/egizio_final.rds")

# We don't need month and year for visualization purposes, so we drop them.
columns_to_exclude <- c("month", "year")
egizio_df <- egizio_df[, !names(egizio_df) %in% columns_to_exclude]

str(egizio_df)

# Convert back to integer just for visualization purposes.
egizio_df$raining_days <- as.integer(egizio_df$raining_days)
egizio_df$school_holidays <- as.integer(egizio_df$school_holidays)

str(egizio_df)

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
egizio_melted_df <- reshape2::melt(egizio_df, id.vars = "date")

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
  theme(axis.title.x = element_blank(), axis.text.x = element_blank()) + # blank x-axis
  facet_wrap(~ variable, scales = "free_y", ncol = length(unique(egizio_melted_df$variable)))

# ---------------------------------------------------------------------------- #
# Violin plots

ggplot(egizio_melted_df, aes(x = date, y = value, fill = variable)) +
  geom_violin() +
  labs(title = "Egizio - Violinplots", x = "Date", y = "Values") +
  theme(axis.title.x = element_blank(), axis.text.x = element_blank()) +
  facet_wrap(~ variable, scales = "free_y", ncol = length(unique(egizio_melted_df$variable)))

# ---------------------------------------------------------------------------- #
# Plot just for one year
select_year <- "2018" # Modify this as needed
egizio_one_year_df <- subset(egizio_df, year(date) == select_year)
egizio_one_year_melted_df <- reshape2::melt(egizio_one_year_df, id.vars = "date")

ggplot(egizio_one_year_melted_df, aes(x = date, y = value, color = variable)) +
  geom_line() +
  labs(title = paste("Egizio -", select_year), x = "Date", y = "Values") +
  facet_wrap(~ variable, scales = "free_y", ncol = 1)

# ---------------------------------------------------------------------------- #
# Monthly and yearly boxplots
  
# egizio_melted_df <- egizio_df
# egizio_melted_df$month <- month(egizio_df$date)
# egizio_melted_df <- melt(egizio_melted_df, id.vars = c("date", "month"))
# egizio_df_melted_visitors <- subset(egizio_melted_df, variable == "visitors")
# 
# ggplot(egizio_df_melted_visitors, aes(x = factor(month), y = value, fill = variable)) +
#   geom_boxplot() +
#   labs(title = "Egizio - Monthly Boxplots for Visitors", x = "Month", y = "Visitors") +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1)) +  # Rotate x-axis labels
#   theme(axis.title.x = element_blank(), axis.text.x = element_blank()) +
#   facet_wrap(~ month, scales = "free_x", ncol = 12)
  
# egizio_melted_df <- egizio_df
# egizio_melted_df$year <- year(egizio_df$date)
# egizio_melted_df <- melt(egizio_melted_df, id.vars = c("date", "year"))
# egizio_df_melted_visitors <- subset(egizio_melted_df, variable == "visitors")
# 
# ggplot(egizio_df_melted_visitors, aes(x = factor(year), y = value, fill = variable)) +
#   geom_boxplot() +
#   labs(title = "Egizio - Boxplots for Visitors", x = "Year", y = "Visitors") +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1)) +  # Rotate x-axis labels
#   theme(axis.title.x = element_blank(), axis.text.x = element_blank()) +
#   facet_wrap(~ year, scales = "free_x", ncol = length(unique(egizio_df_melted_visitors$year)))
  
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
create_boxplots(egizio_df, "month", "Egizio - Monthly Boxplots for Visitors", "Month", "Visitors")

# Create boxplots for each year
create_boxplots(egizio_df, "year", "Egizio - Yearly Boxplots for Visitors", "Year", "Visitors")

# ToDo: Create visualizations for the other museum(s).