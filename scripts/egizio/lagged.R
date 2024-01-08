# WE TRY NOW TO USE THE SAME MODEL BUT LAGGING THE REGRESSORS

# We first have to check the correlation at different lag between
# our response variable and all the other variables:

# The lag k value returned by ccf(x, y) estimates the correlation 
# between x[t+k] and y[t].
LAG_MAX = 12
attach(egizio_df)
ccf_trends = ccf(visitors,trends,
              lag.max = LAG_MAX, plot = T)


lag_max_trends =0 
# we have maximum correlation at:
# visitors[t] ~ trends[t]

# We could consider the lag 0 that has tha maximum positive
# correlation
#we can try to consider the correlation at 1 
cor(egizio_df$visitors[2:216], egizio_df$trends[1:215])
#0.5479857
cor(egizio_df$visitors,egizio_df$trends)
#0.5763311


ccf_year= ccf(visitors,year,
                   lag.max = LAG_MAX, plot = T)

lag_max_years=0 
# we have maximum correlation at:
# visitors[t] ~ years[t]


#we can try to consider the correlation at 1 
cor(egizio_df$visitors[2:216], egizio_df$year[1:215])
#0.2850102
cor(egizio_df$visitors,egizio_df$year)
#0.287492


ccf_temp= ccf(visitors,average_temperature,
              lag.max = LAG_MAX, plot = T)

lag_max_temp=-3 # or 3?
# we have maximum correlation at:
# visitors[t-3] ~ average_temp[t]


#we can try to consider the correlation at 3
cor(egizio_df$visitors[4:216], egizio_df$average_temperature[1:213])
#-0.2802583
cor(egizio_df$visitors,egizio_df$average_temperature)
#-0.007764211

ccf_days= ccf(visitors,raining_days,
              lag.max = LAG_MAX, plot = T)

lag_max_days=4 #or -1?
# we have maximum correlation at:
# visitors[t+4] ~ days[t]


#we can try to consider the correlation at 3
cor(egizio_df$visitors[4:216], egizio_df$raining_days[1:213])
#-0.2105932
cor(egizio_df$visitors,egizio_df$raining_days)
#0.1711197


ccf_school= ccf(visitors,school_holidays,
              lag.max = LAG_MAX, plot = T)

lag_max_school=-3

# we have maximum correlation at:
# visitors[t-3] ~ days[t]
ccf_arrivals= ccf(visitors,arrivals,
                lag.max = LAG_MAX, plot = T)

lag_max_arrivals=0
# we have maximum correlation at:
# visitors[t] ~ arrivals[t]
cor(egizio_df$visitors,egizio_df$arrivals)
#0.5046152

# we have maximum correlation at:
ccf_covid= ccf(visitors,Covid_closures,
                  lag.max = LAG_MAX, plot = T)


# seems no correlations

# we have maximum correlation at:
ccf_renn= ccf(visitors,renovation,
                  lag.max = LAG_MAX, plot = T)

lag_max_renovation=0
##it seems positively correlated with precence of renovation, but this is a binary variable 
##we can't do other analysis


library(openxlsx)
path_file_excel <- "C:/Users/grazy/Desktop/Graziana/università Padova/Business/progetto/git/Time-Series-Forecasting-Museum-Visitors/egizio_df.xlsx"

# Salva il dataframe come file Excel
write.xlsx(egizio_df, file = path_file_excel)

detach(egizio_df)

## LAGGED DATASET

# Now we can create a "lagged dataset" where we put inside all the
# fetaures that shows correlation of type:
# necar[t+k] ~ feature[t]


# (we did it externally using excel)
lagged_df = read.csv("egizio_df.csv", header = T, sep =";",
                     dec = ",")

# Then, we look at the dataset, and in order to obtain the wanted
# temporal corrispondence between the variable, we have some
# "NAN" value to deal with:
# --> we consider only the complete rows:
head(lagged_df)
tail(lagged_df)


sum(is.na(lagged_df))
# 0

# Now we try to create our model with the lagged variables.
colnames(lagged_df) = c("date", "visitors", "trends",
                        "average_temperature", "raining_days",
                        "school_holidays", "arrivals",
                        "renovation")
summary(lagged_df)
class(lagged_df$date)
lagged_df$date = lubridate::dmy(lagged_df$date)
class(lagged_df$date)


# We start again with the full model:

necar_lagged = ts(lagged_df$visitors, frequency = 12)

tslm_l_full = tslm(necar_lagged ~ trend+season+OilPrice_3+
                     ElectricityPrice_1+GDP_4+Inflation_2+UnRate+
                     CovidCases_1+ECarSales_4, data = lagged_df)
summary(tslm_l_full)
