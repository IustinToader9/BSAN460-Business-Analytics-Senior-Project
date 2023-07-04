rm(list=ls())
setwd("/Users/iustintoader/Desktop/Drexel 22:23/Fall/BSAN460/SmartWater/acea-water-prediction")

library(DescTools)
library(caret)
library(data.table)
library(ggplot2)
library(Amelia)
library(forecast)
library(zoo)
library(lubridate)
library(dplyr)
library(imputeTS)
library(vars)
library(Metrics)
library(tensorflow)
# might need to execute following line as well
#install_tensorflow()
library(keras)
library(RSNNS)
library(reshape2)
library(tidyverse)
library(glue)
library(forcats)
library(timetk)
library(tidyquant)
library(tibbletime)
library(recipes)
library(mlr)

#Function to compute accuracy. The installation of packages for LSTM interfered
#with the default accuracy function and we had to write our own
getPerformance = function(pred, val) {
  res = pred - val
  MAE = sum(abs(res))/length(val)
  RSS = sum(res^2)
  MSE = RSS/length(val)
  RMSE = sqrt(MSE)
  perf = data.frame(MAE, RSS, MSE, RMSE)
}


madonna_original <- read.csv(file="Water_Spring_Madonna_di_Canneto.csv")
madonna_original <- madonna_original[madonna_original$Date!="",]

str(madonna_original)

Desc(x=madonna_original, plotit=FALSE)


madonna_original$Date <- as.Date(madonna_original$Date, format="%d/%m/%Y")

#Computes the NA rate for each variable by year
madonna_original$year <- as.numeric(substr(madonna_original$Date, 1, 4))
madonna_NA <- setDT(madonna_original)[,lapply(.SD, function(x) mean(is.na(x))
), by=year, .SDcols=colnames(madonna_original)[!colnames(madonna_original) %in% c("Date", "year")]]

madonna_NA <- melt(madonna_NA, id.vars="year")

#Plots the NA rate for each variable by year for Madonna
ggplot(madonna_NA, 
       aes(x=year, y=variable, fill=value))+
  geom_tile(color="black")+
  scale_fill_viridis_c(direction=-1)+
  theme_bw()+
  labs(fill="NA rate", x="Year", y="")+
  theme(legend.position="right", text=element_text(size=6, face="bold"))

# Since we encounter a 100% rate of missing values in the flowrate of Madonna di Canneto 
# for years: 2012, 2013, and 2014, we will exclude these years from our training datasets.
# Similarly, we can see a 100% rate of missing values in the temperature and rainfall
# data for Settefrati from 2019 to 2020, although flowrate data is present. Since we are dealing with 
# 2 years worth of missing data here, we decided that applying imputation is not suitable
# since it will most likely not capture the defining features over such a long time horizon
# Instead, we will exclude these years from our training dataset as well. 

madonna_drop_years <- madonna_original[!(madonna_original$year %in% c(2012:2014, 2019:2020))]

# Since all remaining years have 0 missing values for their explanatory variables, we must
# only impute the values for flowrate. This means that we do not have any empty rows, so
# we can use the Amelia package for our imputation strategy. We will be using 
# the newly created Madonna dataset, set Date as the time-series variable, 
# create lag and lead features for both explanatory variables, and polynomial
# time equal to 2 in order to better capture patterns within variables across time

madonna_imputations <- amelia(madonna_drop_years, m=1, ts="Date", 
                              polytime=2, lags=c("Flow_Rate_Madonna_di_Canneto","Rainfall_Settefrati", "Temperature_Settefrati"), 
                              leads=c("Flow_Rate_Madonna_di_Canneto","Rainfall_Settefrati", "Temperature_Settefrati"))

madonna_imputed=madonna_imputations[["imputations"]][[1]]

# Plotting difference before and after imputation

ggplot(madonna_drop_years, aes(x=Date)) +
  geom_line(aes(y=Flow_Rate_Madonna_di_Canneto), color="blue", shape="Original") +
  geom_line(data=madonna_imputed[is.na(madonna_drop_years$Flow_Rate_Madonna_di_Canneto),],
            aes(y=Flow_Rate_Madonna_di_Canneto), linetype="dashed", color="red", shape="Imputed")+
  theme_bw()


# We will be splitting the preprocessed data for Madonna into training and testing
# datasets based on 90-day forecast (for daily). Ratio is preserved for weekly and monthly

madonna_pp <- subset(madonna_imputed, select=-year)
madonna_times <- madonna_pp %>%
  mutate(year = year(Date),
         month = month(Date),
         week = week(Date))

### Daily data

# Create time-series object
madonna_daily_ts <- ts(madonna_times[,2:4], start=2015, frequency=365)

madonna_daily_training <- madonna_daily_ts[1:(nrow(madonna_daily_ts)-90), 3]
madonna_daily_testing <- madonna_daily_ts[-c(1:(nrow(madonna_daily_ts)-90)), 3]

#Seasonality plots
ggseasonplot(madonna_daily_ts[,3])
ggseasonplot(madonna_daily_ts[,2])
ggseasonplot(madonna_daily_ts[,1])

# Univariate ARIMA

madonna_daily_uni = auto.arima(madonna_daily_training, test="adf")
summary(madonna_daily_uni)

madonna_daily_forecasted_uni <- forecast(madonna_daily_uni, h=length(madonna_daily_testing))
plot(madonna_daily_forecasted_uni)

# Compute accuracy with pre-defined function
madonna_daily_forecasted_uni_df <- as.data.frame(madonna_daily_forecasted_uni)
madonna_daily_accuracy_uni <- getPerformance(as.vector(madonna_daily_forecasted_uni_df$`Point Forecast`),
                                             as.vector(madonna_daily_testing))
madonna_daily_accuracy_uni

# Create data frame to keep track of results for each model specification
summary_results <- data.frame(matrix(ncol=4, nrow=0))
colnames(summary_results) <- c("Water Spring", "Type", "Model", "RMSE")
summary_results[nrow(summary_results) + 1,] = c("Madonna",
                      "Daily", "Univariate ARIMA", madonna_daily_accuracy_uni$RMSE)

#Multivariate Arima

#Prepare multivariate regressors
madonna_xreg_daily_training_multi <- madonna_daily_ts[1:(nrow(madonna_daily_ts)-90), 1:2]
madonna_xreg_daily_testing_multi <- madonna_daily_ts[-c(1:(nrow(madonna_daily_ts)-90)), 1:2]

madonna_daily_multi = auto.arima(madonna_daily_training, xreg = madonna_xreg_daily_training_multi, test="adf")
summary(madonna_daily_multi)

madonna_daily_forecasted_multi <- forecast(madonna_daily_multi, xreg=madonna_xreg_daily_testing_multi)
plot(madonna_daily_forecasted_multi)

madonna_daily_forecasted_multi_df <- as.data.frame(madonna_daily_forecasted_multi)
madonna_daily_accuracy_multi <- getPerformance(as.vector(madonna_daily_forecasted_multi_df$`Point Forecast`),
                                               as.vector(madonna_daily_testing))
madonna_daily_accuracy_multi

summary_results[nrow(summary_results) + 1,] = c("Madonna", "Daily",
                  "Multivariate ARIMA", madonna_daily_accuracy_multi$RMSE)


### LSTM

#Prepare time-series data for LSTM analysis
madonna_target_daily <- zoo(madonna_times$Flow_Rate_Madonna_di_Canneto, seq(from=as.Date("2015-01-01"), to=as.Date("2018-12-31"), by=1))
madonna_input_daily <- madonna_times
madonna_input_daily$day <- lubridate::day(madonna_input_daily$Date)
madonna_input_daily$Date <- NULL

# delta computation for avoiding negative sqrt
delta_madonna_daily <- abs(min(madonna_target_daily))
madonna_target_daily <- madonna_target_daily + delta_madonna_daily 

# Create target object as time tibble
tbltarget_madonna_daily <- madonna_target_daily %>%
  tk_tbl() %>%
  mutate(index = as_date(index)) %>%
  as_tbl_time(index=index)

# Compute desired LSTM specifications to ensure required divisibility maintaining
# recommended 1-to-3 ratio of batch size to test length
batch_size_madonna_daily <- 30
tot_length_madonna_daily <- nrow(tbltarget_madonna_daily)
tot_length_5_madonna_daily <- as.integer(tot_length_madonna_daily/batch_size_madonna_daily)*batch_size_madonna_daily
start_train_madonna_daily <- tot_length_madonna_daily - tot_length_5_madonna_daily
test_length_madonna_daily <- 90
train_length_madonna_daily <- tot_length_5_madonna_daily - test_length_madonna_daily
df_train_madonna_daily <- tbltarget_madonna_daily[(start_train_madonna_daily+1):(start_train_madonna_daily+train_length_madonna_daily), ]
df_test_madonna_daily <- tbltarget_madonna_daily[(train_length_madonna_daily+1):tot_length_5_madonna_daily, ]

# Put together training and tasting data with keys
df_madonna_daily <- bind_rows(
  df_train_madonna_daily %>% add_column(key="training"),
  df_test_madonna_daily %>% add_column(key="testing")
) %>%
  as_tbl_time(index=index)

# Normalize and standardize data as needed for LSTM
rec_madonna_daily <- recipe(value ~ ., df_madonna_daily) %>%
  step_sqrt(value) %>%
  step_center(value) %>%
  step_scale(value) %>%
  prep()

pp_madonna_daily <- bake(rec_madonna_daily, df_madonna_daily)

# Save pre-processing data to reinvert for comparison
center_inv_madonna_daily <- rec_madonna_daily$steps[[2]]$means["value"]
scale_inv_madonna_daily <- rec_madonna_daily$steps[[3]]$sds["value"]

# Create lag feature as predictor variable
lag_setting_madonna_daily <- nrow(df_test_madonna_daily)
train_len_madonna_daily <- nrow(df_train_madonna_daily)
tsteps_madonna_daily <- 1

lag_train_madonna_daily <- pp_madonna_daily %>%
  mutate(value_lag = lag(value, n=lag_setting_madonna_daily)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "training") %>%
  tail(train_len_madonna_daily)

# Get training and testing data in required format
x_train_vec_madonna_daily <- lag_train_madonna_daily$value_lag
x_train_arr_madonna_daily <- array(data=x_train_vec_madonna_daily, dim=c(length(
  x_train_vec_madonna_daily), 1, 1))

y_train_vec_madonna_daily <- lag_train_madonna_daily$value
y_train_arr_madonna_daily <- array(data=y_train_vec_madonna_daily, dim=c(length(
  y_train_vec_madonna_daily), 1))

lag_test_madonna_daily <- pp_madonna_daily %>%
  mutate(value_lag = lag(value, n=lag_setting_madonna_daily)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "testing")

x_test_vec_madonna_daily <- lag_test_madonna_daily$value_lag
x_test_arr_madonna_daily <- array(data=x_test_vec_madonna_daily, dim = c(length(
  x_test_vec_madonna_daily), 1, 1))

y_test_vec_madonna_daily <- lag_test_madonna_daily$value
y_test_arr_madonna_daily <- array(data=y_test_vec_madonna_daily, dim = c(length(
  y_test_vec_madonna_daily), 1))

# Initiate model
model <- keras_model_sequential()

# Model specification
model %>%
  layer_lstm(units = 50,
             input_shape = c(tsteps_madonna_daily, 1),
             batch_size = batch_size_madonna_daily,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dense(units=1)

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam',
  metrics = c('mean_absolute_error', 'mean_squared_error')
)

# Fit model
history <- model %>% fit(x = x_train_arr_madonna_daily,
              y = y_train_arr_madonna_daily,
              batch_size = batch_size_madonna_daily,
              epochs = 90,
              verbose = 0,
              shuffle = FALSE,
              validation_data = list(x_test_arr_madonna_daily, y_test_arr_madonna_daily)
)

plot(history)

# Compute training accuracy
predLSTM_madonna_daily_training <- model %>% 
  predict(x_train_arr_madonna_daily, batch_size=batch_size_madonna_daily)
predLSTM_madonna_daily_training <- as.vector(predLSTM_madonna_daily_training)

# Reinvert prediction for comparison
predLSTM_madonna_daily_training_tbl <- tibble(
  index = lag_train_madonna_daily$index,
  value = (predLSTM_madonna_daily_training*scale_inv_madonna_daily + center_inv_madonna_daily)^2
)

madonna_daily_accuracy_LSTM_training <- getPerformance(as.vector(predLSTM_madonna_daily_training_tbl$value-delta_madonna_daily),
                                              as.vector((y_train_vec_madonna_daily*scale_inv_madonna_daily+center_inv_madonna_daily)^2-delta_madonna_daily))
madonna_daily_accuracy_LSTM_training

# Plot training accuracy
plot_act_train = as.ts((y_train_vec_madonna_daily*scale_inv_madonna_daily+center_inv_madonna_daily)^2-delta_madonna_daily, frequency=365)
plot_pred_train = as.ts(predLSTM_madonna_daily_training_tbl$value-delta_madonna_daily, frequency=365)

autoplot(plot_act_train, xlab="days", ylab="Flow Rate", series="Train set") +
  autolayer(plot_pred_train, series="Prediction")

# Compute 90-day forecast
predLSTM_madonna_daily_testing <- model %>% 
  predict(x_test_arr_madonna_daily, batch_size=batch_size_madonna_daily)
predLSTM_madonna_daily_testing <- as.vector(predLSTM_madonna_daily_testing)

# Reinvert for comparison
predLSTM_madonna_daily_testing_tbl <- tibble(
  index = lag_test_madonna_daily$index,
  value = (predLSTM_madonna_daily_testing*scale_inv_madonna_daily + center_inv_madonna_daily)^2
)

madonna_daily_accuracy_LSTM_testing <- getPerformance(as.vector(predLSTM_madonna_daily_testing_tbl$value-delta_madonna_daily),
                                               as.vector(df_test_madonna_daily$value-delta_madonna_daily))
madonna_daily_accuracy_LSTM_testing

# Plot testing accuracy
plot_act_test = as.ts(df_test_madonna_daily$value-delta_madonna_daily, frequency=365)
plot_pred_test = as.ts(predLSTM_madonna_daily_testing_tbl$value-delta_madonna_daily, frequency=365)

autoplot(plot_act_test, xlab="days", ylab="Flow Rate", series="Test set") +
  autolayer(plot_pred_test, series="Prediction")

# Append to testing results dataframe
summary_results[nrow(summary_results) + 1,] = c("Madonna", "Daily",
                                                "LSTM", madonna_daily_accuracy_LSTM_testing$RMSE)

# Rest of code for Madonna not commented for redundancy, process is replicated
# to reflect the different levels of aggregation

### Weekly

# Aggregates data on a weekly basis
madonna_weekly <- madonna_times %>%
  group_by(year, week) %>%
  summarize(rainfall = mean(Rainfall_Settefrati),
            temperature = mean(Temperature_Settefrati),
            flowrate = mean(Flow_Rate_Madonna_di_Canneto))
madonna_weekly_ts <- ts(madonna_weekly[,3:5], start=2015, frequency=52)

ggseasonplot(madonna_weekly_ts[,3])
ggseasonplot(madonna_weekly_ts[,2])
ggseasonplot(madonna_weekly_ts[,1])

madonna_weekly_training <- madonna_weekly_ts[1:(nrow(madonna_weekly_ts)-12), 3]
madonna_weekly_testing <- madonna_weekly_ts[-c(1:(nrow(madonna_weekly_ts)-12)), 3]

# Univariate ARIMA

madonna_weekly_uni = auto.arima(madonna_weekly_training, test="adf")
summary(madonna_weekly_uni)

madonna_weekly_forecasted_uni <- forecast(madonna_weekly_uni, h=length(madonna_weekly_testing))
plot(madonna_weekly_forecasted_uni)

madonna_weekly_forecasted_uni_df <- as.data.frame(madonna_weekly_forecasted_uni)
madonna_weekly_accuracy_uni <- getPerformance(as.vector(madonna_weekly_forecasted_uni_df$`Point Forecast`),
                                              as.vector(madonna_weekly_testing))
madonna_weekly_accuracy_uni

summary_results[nrow(summary_results) + 1,] = c("Madonna", "Weekly",
                                                "Univariate ARIMA", madonna_weekly_accuracy_uni$RMSE)

#Multivariate Arima

madonna_xreg_weekly_training_multi <- madonna_weekly_ts[1:(nrow(madonna_weekly_ts)-12), 1:2]
madonna_xreg_weekly_testing_multi <- madonna_weekly_ts[-c(1:(nrow(madonna_weekly_ts)-12)), 1:2]

madonna_weekly_multi = auto.arima(madonna_weekly_training, xreg = madonna_xreg_weekly_training_multi, test="adf")
summary(madonna_weekly_multi)

madonna_weekly_forecasted_multi <- forecast(madonna_weekly_multi, xreg=madonna_xreg_weekly_testing_multi)
plot(madonna_weekly_forecasted_multi)

madonna_weekly_forecasted_multi_df <- as.data.frame(madonna_weekly_forecasted_multi)
madonna_weekly_accuracy_multi <- getPerformance(as.vector(madonna_weekly_forecasted_multi_df$`Point Forecast`),
                                                as.vector(madonna_weekly_testing))
madonna_weekly_accuracy_multi

summary_results[nrow(summary_results) + 1,] = c("Madonna", "Weekly",
                                                "Multivariate ARIMA", madonna_weekly_accuracy_multi$RMSE)

### LSTM

madonna_target_weekly <- zoo(madonna_weekly_ts[,3], seq(from=as.Date("2015-01-01"), to=as.Date("2018-12-31"), by=7))
madonna_input_weekly <- madonna_times
madonna_input_weekly$day <- lubridate::day(madonna_input_weekly$Date)
madonna_input_weekly$Date <- NULL

delta_madonna_weekly <- abs(min(madonna_target_weekly))
madonna_target_weekly <- madonna_target_weekly + delta_madonna_weekly # avoid negative sqrt

tbltarget_madonna_weekly <- madonna_target_weekly %>%
  tk_tbl() %>%
  mutate(index = as_date(index)) %>%
  as_tbl_time(index=index)

batch_size_madonna_weekly <- 4
tot_length_madonna_weekly <- nrow(tbltarget_madonna_weekly)
tot_length_5_madonna_weekly <- as.integer(tot_length_madonna_weekly/batch_size_madonna_weekly)*batch_size_madonna_weekly
start_train_madonna_weekly <- tot_length_madonna_weekly - tot_length_5_madonna_weekly
test_length_madonna_weekly <- 12
train_length_madonna_weekly <- tot_length_5_madonna_weekly - test_length_madonna_weekly
df_train_madonna_weekly <- tbltarget_madonna_weekly[(start_train_madonna_weekly+1):(start_train_madonna_weekly+train_length_madonna_weekly), ]
df_test_madonna_weekly <- tbltarget_madonna_weekly[(train_length_madonna_weekly+1):tot_length_5_madonna_weekly, ]

df_madonna_weekly <- bind_rows(
  df_train_madonna_weekly %>% add_column(key="training"),
  df_test_madonna_weekly %>% add_column(key="testing")
) %>%
  as_tbl_time(index=index)

rec_madonna_weekly <- recipe(value ~ ., df_madonna_weekly) %>%
  step_sqrt(value) %>%
  step_center(value) %>%
  step_scale(value) %>%
  prep()

pp_madonna_weekly <- bake(rec_madonna_weekly, df_madonna_weekly)

center_inv_madonna_weekly <- rec_madonna_weekly$steps[[2]]$means["value"]
scale_inv_madonna_weekly <- rec_madonna_weekly$steps[[3]]$sds["value"]

lag_setting_madonna_weekly <- nrow(df_test_madonna_weekly)
train_len_madonna_weekly <- nrow(df_train_madonna_weekly)
tsteps_madonna_weekly <- 1

lag_train_madonna_weekly <- pp_madonna_weekly %>%
  mutate(value_lag = lag(value, n=lag_setting_madonna_weekly)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "training") %>%
  tail(train_len_madonna_weekly)

x_train_vec_madonna_weekly <- lag_train_madonna_weekly$value_lag
x_train_arr_madonna_weekly <- array(data=x_train_vec_madonna_weekly, dim=c(length(
  x_train_vec_madonna_weekly), 1, 1))

y_train_vec_madonna_weekly <- lag_train_madonna_weekly$value
y_train_arr_madonna_weekly <- array(data=y_train_vec_madonna_weekly, dim=c(length(
  y_train_vec_madonna_weekly), 1))

lag_test_madonna_weekly <- pp_madonna_weekly %>%
  mutate(value_lag = lag(value, n=lag_setting_madonna_weekly)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "testing")

x_test_vec_madonna_weekly <- lag_test_madonna_weekly$value_lag
x_test_arr_madonna_weekly <- array(data=x_test_vec_madonna_weekly, dim = c(length(
  x_test_vec_madonna_weekly), 1, 1))

y_test_vec_madonna_weekly <- lag_test_madonna_weekly$value
y_test_arr_madonna_weekly <- array(data=y_test_vec_madonna_weekly, dim = c(length(
  y_test_vec_madonna_weekly), 1))

model <- keras_model_sequential()

model %>%
  layer_lstm(units = 50,
             input_shape = c(tsteps_madonna_weekly, 1),
             batch_size = batch_size_madonna_weekly,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dense(units=1)

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam',
  metrics = c('mean_absolute_error', 'mean_squared_error')
)

history <- model %>% fit(x = x_train_arr_madonna_weekly,
                         y = y_train_arr_madonna_weekly,
                         batch_size = batch_size_madonna_weekly,
                         epochs = 12,
                         verbose = 0,
                         shuffle = FALSE,
                         validation_data = list(x_test_arr_madonna_weekly, y_test_arr_madonna_weekly)
)

plot(history)


predLSTM_madonna_weekly_training <- model %>% 
  predict(x_train_arr_madonna_weekly, batch_size=batch_size_madonna_weekly)
predLSTM_madonna_weekly_training <- as.vector(predLSTM_madonna_weekly_training)

predLSTM_madonna_weekly_training_tbl <- tibble(
  index = lag_train_madonna_weekly$index,
  value = (predLSTM_madonna_weekly_training*scale_inv_madonna_weekly + center_inv_madonna_weekly)^2
)

madonna_weekly_accuracy_LSTM_training <- getPerformance(as.vector(predLSTM_madonna_weekly_training_tbl$value-delta_madonna_weekly),
                                                       as.vector((y_train_vec_madonna_weekly*scale_inv_madonna_weekly+center_inv_madonna_weekly)^2-delta_madonna_weekly))
madonna_weekly_accuracy_LSTM_training

plot_act_train_weekly = as.ts((y_train_vec_madonna_weekly*scale_inv_madonna_weekly+center_inv_madonna_weekly)^2-delta_madonna_weekly, frequency=52)
plot_pred_train_weekly = as.ts(predLSTM_madonna_weekly_training_tbl$value-delta_madonna_weekly, frequency=52)

autoplot(plot_act_train_weekly, xlab="weeks", ylab="Flow Rate", series="Train set") +
  autolayer(plot_pred_train_weekly, series="Prediction")

predLSTM_madonna_weekly_testing <- model %>% 
  predict(x_test_arr_madonna_weekly, batch_size=batch_size_madonna_weekly)
predLSTM_madonna_weekly_testing <- as.vector(predLSTM_madonna_weekly_testing)

predLSTM_madonna_weekly_testing_tbl <- tibble(
  index = lag_test_madonna_weekly$index,
  value = (predLSTM_madonna_weekly_testing*scale_inv_madonna_weekly + center_inv_madonna_weekly)^2
)

madonna_weekly_accuracy_LSTM_testing <- getPerformance(as.vector(predLSTM_madonna_weekly_testing_tbl$value-delta_madonna_weekly),
                                              as.vector(df_test_madonna_weekly$value-delta_madonna_weekly))
madonna_weekly_accuracy_LSTM_testing

plot_act_test_weekly = as.ts(df_test_madonna_weekly$value-delta_madonna_weekly, frequency=52)
plot_pred_test_weekly = as.ts(predLSTM_madonna_weekly_testing_tbl$value-delta_madonna_weekly, frequency=52)

autoplot(plot_act_test_weekly, xlab="weeks", ylab="Flow Rate", series="Test set") +
  autolayer(plot_pred_test_weekly, series="Prediction")

summary_results[nrow(summary_results) + 1,] = c("Madonna", "Weekly",
                                                "LSTM", madonna_weekly_accuracy_LSTM_testing$RMSE)

### Monthly

# Aggregates data on a monthly basis
madonna_monthly <- madonna_times %>%
  group_by(year, month) %>%
  summarize(rainfall = mean(Rainfall_Settefrati),
            temperature = mean(Temperature_Settefrati),
            flowrate = mean(Flow_Rate_Madonna_di_Canneto))
madonna_monthly_ts <- ts(madonna_monthly[,3:5], start=2015, frequency=12)


ggseasonplot(madonna_monthly_ts[,3])
ggseasonplot(madonna_monthly_ts[,2])
ggseasonplot(madonna_monthly_ts[,1])

madonna_monthly_training <- madonna_monthly_ts[1:(nrow(madonna_monthly_ts)-3), 3]
madonna_monthly_testing <- madonna_monthly_ts[-c(1:(nrow(madonna_monthly_ts)-3)), 3]

# Univariate ARIMA

madonna_monthly_uni = auto.arima(madonna_monthly_training, test="adf")
summary(madonna_monthly_uni)

madonna_monthly_forecasted_uni <- forecast(madonna_monthly_uni, h=length(madonna_monthly_testing))
plot(madonna_monthly_forecasted_uni)

madonna_monthly_forecasted_uni_df <- as.data.frame(madonna_monthly_forecasted_uni)
madonna_monthly_accuracy_uni <- getPerformance(as.vector(madonna_monthly_forecasted_uni_df$`Point Forecast`),
                                               as.vector(madonna_monthly_testing))
madonna_monthly_accuracy_uni

summary_results[nrow(summary_results) + 1,] = c("Madonna", "Monthly",
                                                "Univariate ARIMA", madonna_monthly_accuracy_uni$RMSE)


#Multivariate Arima

madonna_xreg_monthly_training_multi <- madonna_monthly_ts[1:(nrow(madonna_monthly_ts)-3), 1:2]
madonna_xreg_monthly_testing_multi <- madonna_monthly_ts[-c(1:(nrow(madonna_monthly_ts)-3)), 1:2]

madonna_monthly_multi = auto.arima(madonna_monthly_training, xreg = madonna_xreg_monthly_training_multi, test="adf")
summary(madonna_monthly_multi)

madonna_monthly_forecasted_multi <- forecast(madonna_monthly_multi, xreg=madonna_xreg_monthly_testing_multi)
plot(madonna_monthly_forecasted_multi)

madonna_monthly_forecasted_multi_df <- as.data.frame(madonna_monthly_forecasted_multi)
madonna_monthly_accuracy_multi <- getPerformance(as.vector(madonna_monthly_forecasted_multi_df$`Point Forecast`),
                                                 as.vector(madonna_monthly_testing))
madonna_monthly_accuracy_multi

summary_results[nrow(summary_results) + 1,] = c("Madonna", "Monthly",
                                                "Multivariate ARIMA", madonna_monthly_accuracy_multi$RMSE)


### LUPA

lupa_original <- read.csv(file="Water_Spring_Lupa.csv")
lupa_original <- lupa_original[lupa_original$Date!="",]
lupa_original$Date <- as.Date(lupa_original$Date, format="%d/%m/%Y")
lupa_original$Flow_Rate_Lupa <- abs(lupa_original$Flow_Rate_Lupa)

#Computes the NA rate for each variable by year for Lupa
lupa_original$year <- as.numeric(substr(lupa_original$Date, 1, 4))
lupa_NA <- setDT(lupa_original)[,lapply(.SD, function(x) mean(is.na(x))
), by=year, .SDcols=colnames(lupa_original)[!colnames(lupa_original) %in% c("Date", "year")]]

lupa_NA <- melt(lupa_NA, id.vars="year")

#Plots the NA rate for each variable by year
ggplot(lupa_NA, 
       aes(x=year, y=variable, fill=value))+
  geom_tile(color="black")+
  scale_fill_viridis_c(direction=-1)+
  theme_bw()+
  labs(fill="NA rate", x="Year", y="")+
  theme(legend.position="right", text=element_text(size=6, face="bold"))

# The missing value rate for 2009 is high, so we choose to drop that year
lupa_drop_years <- lupa_original[!(lupa_original$year %in% c(2009))]

# Since Lupa's flow rate follows a linear trend, we decide to use time-series
# linear interpolation to impute missing values
lupa_imputed <- na_interpolation(lupa_drop_years)

ggplot(lupa_drop_years, aes(x=Date)) +
  geom_line(aes(y=Flow_Rate_Lupa), color="blue", shape="Original") +
  geom_line(data=lupa_imputed[is.na(lupa_drop_years$Flow_Rate_Lupa),],
            aes(y=Flow_Rate_Lupa), linetype="dashed", color="red", shape="Imputed")+
  theme_bw()

lupa_pp <- subset(lupa_imputed, select=-year)
lupa_times <- lupa_pp %>%
  mutate(year = year(Date),
         month = month(Date),
         week = week(Date))

### Daily data

lupa_daily_ts <- ts(lupa_times[,2:3], start=2010, frequency=365)


ggseasonplot(lupa_daily_ts[,2])
ggseasonplot(lupa_daily_ts[,1])

# Given the sudden change in the flow rate following a largely entirely linear trend
# up until 2020, we decided to only try and forecast 30-days ahead. Same ratios for
# all aggregations and model specifications are maintained

lupa_daily_training <- lupa_daily_ts[1:(nrow(lupa_daily_ts)-30), 2]
lupa_daily_testing <- lupa_daily_ts[-c(1:(nrow(lupa_daily_ts)-30)), 2]

# Univariate ARIMA

lupa_daily_uni = auto.arima(lupa_daily_training, test="adf")
summary(lupa_daily_uni)

lupa_daily_forecasted_uni <- forecast(lupa_daily_uni, h=length(lupa_daily_testing))
plot(lupa_daily_forecasted_uni)

lupa_daily_forecasted_uni_df <- as.data.frame(lupa_daily_forecasted_uni)
lupa_daily_accuracy_uni <- getPerformance(as.vector(lupa_daily_forecasted_uni_df$`Point Forecast`),
                                          as.vector(lupa_daily_testing))
lupa_daily_accuracy_uni

summary_results_lupa <- data.frame(matrix(ncol=4, nrow=0))
colnames(summary_results_lupa) <- c("Water Spring", "Type", "Model", "RMSE")
summary_results_lupa[nrow(summary_results_lupa) + 1,] = c("Lupa",
                                                "Daily", "Univariate ARIMA", lupa_daily_accuracy_uni$RMSE)

#Multivariate Arima

lupa_xreg_daily_training_multi <- lupa_daily_ts[1:(nrow(lupa_daily_ts)-30), 1]
lupa_xreg_daily_testing_multi <- lupa_daily_ts[-c(1:(nrow(lupa_daily_ts)-30)), 1]

lupa_daily_multi = auto.arima(lupa_daily_training, xreg = lupa_xreg_daily_training_multi, test="adf")
summary(lupa_daily_multi)

lupa_daily_forecasted_multi <- forecast(lupa_daily_multi, xreg=lupa_xreg_daily_testing_multi)
plot(lupa_daily_forecasted_multi)

lupa_daily_forecasted_multi_df <- as.data.frame(lupa_daily_forecasted_multi)
lupa_daily_accuracy_multi <- getPerformance(as.vector(lupa_daily_forecasted_multi_df$`Point Forecast`),
                                            as.vector(lupa_daily_testing))
lupa_daily_accuracy_multi

summary_results_lupa[nrow(summary_results_lupa) + 1,] = c("Lupa",
                                                          "Daily", "Multivariate ARIMA", lupa_daily_accuracy_multi$RMSE)


### LSTM

lupa_target_daily <- zoo(lupa_times$Flow_Rate_Lupa, seq(from=as.Date("2010-01-01"), to=as.Date("2020-06-30"), by=1))
lupa_input_daily <- lupa_times
lupa_input_daily$day <- lubridate::day(lupa_input_daily$Date)
lupa_input_daily$Date <- NULL

delta_lupa_daily <- abs(min(lupa_target_daily))
lupa_target_daily <- lupa_target_daily + delta_lupa_daily # avoid negative sqrt

tbltarget_lupa_daily <-lupa_target_daily %>%
  tk_tbl() %>%
  mutate(index = as_date(index)) %>%
  as_tbl_time(index=index)

batch_size_lupa_daily <- 30
tot_length_lupa_daily <- nrow(tbltarget_lupa_daily)
tot_length_5_lupa_daily <- as.integer(tot_length_lupa_daily/batch_size_lupa_daily)*batch_size_lupa_daily
start_train_lupa_daily <- tot_length_lupa_daily - tot_length_5_lupa_daily
test_length_lupa_daily <- 90
train_length_lupa_daily <- tot_length_5_lupa_daily - test_length_lupa_daily
df_train_lupa_daily <- tbltarget_lupa_daily[(start_train_lupa_daily+1):(start_train_lupa_daily+train_length_lupa_daily), ]
df_test_lupa_daily <- tbltarget_lupa_daily[(train_length_lupa_daily+1):tot_length_5_lupa_daily, ]

df_lupa_daily <- bind_rows(
  df_train_lupa_daily %>% add_column(key="training"),
  df_test_lupa_daily %>% add_column(key="testing")
) %>%
  as_tbl_time(index=index)

rec_lupa_daily <- recipe(value ~ ., df_lupa_daily) %>%
  step_sqrt(value) %>%
  step_center(value) %>%
  step_scale(value) %>%
  prep()

pp_lupa_daily <- bake(rec_lupa_daily, df_lupa_daily)

center_inv_lupa_daily <- rec_lupa_daily$steps[[2]]$means["value"]
scale_inv_lupa_daily <- rec_lupa_daily$steps[[3]]$sds["value"]

lag_setting_lupa_daily <- nrow(df_test_lupa_daily)
train_len_lupa_daily <- nrow(df_train_lupa_daily)
tsteps_lupa_daily <- 1

lag_train_lupa_daily <- pp_lupa_daily %>%
  mutate(value_lag = lag(value, n=lag_setting_lupa_daily)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "training") %>%
  tail(train_len_lupa_daily)

x_train_vec_lupa_daily <- lag_train_lupa_daily$value_lag
x_train_arr_lupa_daily <- array(data=x_train_vec_lupa_daily, dim=c(length(
  x_train_vec_lupa_daily), 1, 1))

y_train_vec_lupa_daily <- lag_train_lupa_daily$value
y_train_arr_lupa_daily <- array(data=y_train_vec_lupa_daily, dim=c(length(
  y_train_vec_lupa_daily), 1))

lag_test_lupa_daily <- pp_lupa_daily %>%
  mutate(value_lag = lag(value, n=lag_setting_lupa_daily)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "testing")

x_test_vec_lupa_daily <- lag_test_lupa_daily$value_lag
x_test_arr_lupa_daily <- array(data=x_test_vec_lupa_daily, dim = c(length(
  x_test_vec_lupa_daily), 1, 1))

y_test_vec_lupa_daily <- lag_test_lupa_daily$value
y_test_arr_lupa_daily <- array(data=y_test_vec_lupa_daily, dim = c(length(
  y_test_vec_lupa_daily), 1))

model <- keras_model_sequential()

model %>%
  layer_lstm(units = 50,
             input_shape = c(tsteps_lupa_daily, 1),
             batch_size = batch_size_lupa_daily,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dense(units=1)

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam',
  metrics = c('mean_absolute_error', 'mean_squared_error')
)

history <- model %>% fit(x = x_train_arr_lupa_daily,
                         y = y_train_arr_lupa_daily,
                         batch_size = batch_size_lupa_daily,
                         epochs = 90,
                         verbose = 0,
                         shuffle = FALSE,
                         validation_data = list(x_test_arr_lupa_daily, y_test_arr_lupa_daily)
)

plot(history)


predLSTM_lupa_daily_training <- model %>% 
  predict(x_train_arr_lupa_daily, batch_size=batch_size_lupa_daily)
predLSTM_lupa_daily_training <- as.vector(predLSTM_lupa_daily_training)

predLSTM_lupa_daily_training_tbl <- tibble(
  index = lag_train_lupa_daily$index,
  value = (predLSTM_lupa_daily_training*scale_inv_lupa_daily + center_inv_lupa_daily)^2
)

lupa_daily_accuracy_LSTM_training <- getPerformance(as.vector(predLSTM_lupa_daily_training_tbl$value-delta_lupa_daily),
                                                       as.vector((y_train_vec_lupa_daily*scale_inv_lupa_daily+center_inv_lupa_daily)^2-delta_lupa_daily))
lupa_daily_accuracy_LSTM_training

lupa_plot_act_train_daily = as.ts((y_train_vec_lupa_daily*scale_inv_lupa_daily+center_inv_lupa_daily)^2-delta_lupa_daily, frequency=365)
lupa_plot_pred_train_daily = as.ts(predLSTM_lupa_daily_training_tbl$value-delta_lupa_daily, frequency=365)

autoplot(lupa_plot_act_train_daily, xlab="days", ylab="Flow Rate", series="Train set") +
  autolayer(lupa_plot_pred_train_daily, series="Prediction")

predLSTM_lupa_daily_testing <- model %>% 
  predict(x_test_arr_lupa_daily, batch_size=batch_size_lupa_daily)
predLSTM_lupa_daily_testing <- as.vector(predLSTM_lupa_daily_testing)

predLSTM_lupa_daily_testing_tbl <- tibble(
  index = lag_test_lupa_daily$index,
  value = (predLSTM_lupa_daily_testing*scale_inv_lupa_daily + center_inv_lupa_daily)^2
)

lupa_daily_accuracy_LSTM_testing <- getPerformance(as.vector(predLSTM_lupa_daily_testing_tbl$value-delta_lupa_daily),
                                                      as.vector(df_test_lupa_daily$value-delta_lupa_daily))
lupa_daily_accuracy_LSTM_testing

lupa_plot_act_test_daily = as.ts(df_test_lupa_daily$value-delta_lupa_daily, frequency=365)
lupa_plot_pred_test_daily = as.ts(predLSTM_lupa_daily_testing_tbl$value-delta_lupa_daily, frequency=365)

autoplot(lupa_plot_act_test_daily, xlab="days", ylab="Flow Rate", series="Test set") +
  autolayer(lupa_plot_pred_test_daily, series="Prediction")

summary_results_lupa[nrow(summary_results_lupa) + 1,] = c("Lupa", "Daily",
                                                "LSTM", lupa_daily_accuracy_LSTM_testing$RMSE)


### Weekly data
lupa_weekly <- lupa_times %>%
  group_by(year, week) %>%
  summarize(rainfall = mean(Rainfall_Terni),
            flowrate = mean(Flow_Rate_Lupa))

lupa_weekly_ts <- ts(lupa_weekly[,3:4], start=2010, frequency=52)

autoplot(decompose(lupa_weekly_ts[,2], type="multiplicative"))
autoplot(decompose(lupa_weekly_ts[,1], type="multiplicative"))

ggseasonplot(lupa_weekly_ts[,2])
ggseasonplot(lupa_weekly_ts[,1])

lupa_weekly_training <- lupa_weekly_ts[1:(nrow(lupa_weekly_ts)-12), 2]
lupa_weekly_testing <- lupa_weekly_ts[-c(1:(nrow(lupa_weekly_ts)-12)), 2]

# Univariate ARIMA

lupa_weekly_uni = auto.arima(lupa_weekly_training, test="adf")
summary(lupa_weekly_uni)

lupa_weekly_forecasted_uni <- forecast(lupa_weekly_uni, h=length(lupa_weekly_testing))
plot(lupa_weekly_forecasted_uni)

lupa_weekly_forecasted_uni_df <- as.data.frame(lupa_weekly_forecasted_uni)
lupa_weekly_accuracy_uni <- getPerformance(as.vector(lupa_weekly_forecasted_uni_df$`Point Forecast`),
                                           as.vector(lupa_weekly_testing))
lupa_weekly_accuracy_uni

summary_results_lupa[nrow(summary_results_lupa) + 1,] = c("Lupa", "Weekly",
                                                          "Univariate ARIMA", lupa_weekly_accuracy_uni$RMSE)


#Multivariate Arima

lupa_xreg_weekly_training_multi <- lupa_weekly_ts[1:(nrow(lupa_weekly_ts)-12), 1]
lupa_xreg_weekly_testing_multi <- lupa_weekly_ts[-c(1:(nrow(lupa_weekly_ts)-12)), 1]

lupa_weekly_multi = auto.arima(lupa_weekly_training, xreg = lupa_xreg_weekly_training_multi, test="adf")
summary(lupa_weekly_multi)

lupa_weekly_forecasted_multi <- forecast(lupa_weekly_multi, xreg=lupa_xreg_weekly_testing_multi)
plot(lupa_weekly_forecasted_multi)

lupa_weekly_forecasted_multi_df <- as.data.frame(lupa_weekly_forecasted_multi)
lupa_weekly_accuracy_multi <- getPerformance(as.vector(lupa_weekly_forecasted_multi_df$`Point Forecast`),
                                             as.vector(lupa_weekly_testing))
lupa_weekly_accuracy_multi

summary_results_lupa[nrow(summary_results_lupa) + 1,] = c("Lupa", "Weekly",
                                                          "Multivariate ARIMA", lupa_weekly_accuracy_multi$RMSE)

### LSTM

lupa_target_weekly <- zoo(lupa_weekly_ts[,2], seq(from=as.Date("2010-01-01"), to=as.Date("2020-06-30"), by=7))
lupa_input_weekly <- lupa_times
lupa_input_weekly$day <- lubridate::day(lupa_input_weekly$Date)
lupa_input_weekly$Date <- NULL

delta_lupa_weekly <- abs(min(lupa_target_weekly))
lupa_target_weekly <- lupa_target_weekly + delta_lupa_weekly # avoid negative sqrt

tbltarget_lupa_weekly <- lupa_target_weekly %>%
  tk_tbl() %>%
  mutate(index = as_date(index)) %>%
  as_tbl_time(index=index)

batch_size_lupa_weekly <- 4
tot_length_lupa_weekly <- nrow(tbltarget_lupa_weekly)
tot_length_5_lupa_weekly <- as.integer(tot_length_lupa_weekly/batch_size_lupa_weekly)*batch_size_lupa_weekly
start_train_lupa_weekly <- tot_length_lupa_weekly - tot_length_5_lupa_weekly
test_length_lupa_weekly <- 12
train_length_lupa_weekly <- tot_length_5_lupa_weekly - test_length_lupa_weekly
df_train_lupa_weekly <- tbltarget_lupa_weekly[(start_train_lupa_weekly+1):(start_train_lupa_weekly+train_length_lupa_weekly), ]
df_test_lupa_weekly <- tbltarget_lupa_weekly[(train_length_lupa_weekly+1):tot_length_5_lupa_weekly, ]

df_lupa_weekly <- bind_rows(
  df_train_lupa_weekly %>% add_column(key="training"),
  df_test_lupa_weekly %>% add_column(key="testing")
) %>%
  as_tbl_time(index=index)

rec_lupa_weekly <- recipe(value ~ ., df_lupa_weekly) %>%
  step_sqrt(value) %>%
  step_center(value) %>%
  step_scale(value) %>%
  prep()

pp_lupa_weekly <- bake(rec_lupa_weekly, df_lupa_weekly)

center_inv_lupa_weekly <- rec_lupa_weekly$steps[[2]]$means["value"]
scale_inv_lupa_weekly <- rec_lupa_weekly$steps[[3]]$sds["value"]

lag_setting_lupa_weekly <- nrow(df_test_lupa_weekly)
train_len_lupa_weekly <- nrow(df_train_lupa_weekly)
tsteps_lupa_weekly <- 1

lag_train_lupa_weekly <- pp_lupa_weekly %>%
  mutate(value_lag = lag(value, n=lag_setting_lupa_weekly)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "training") %>%
  tail(train_len_lupa_weekly)

x_train_vec_lupa_weekly <- lag_train_lupa_weekly$value_lag
x_train_arr_lupa_weekly <- array(data=x_train_vec_lupa_weekly, dim=c(length(
  x_train_vec_lupa_weekly), 1, 1))

y_train_vec_lupa_weekly <- lag_train_lupa_weekly$value
y_train_arr_lupa_weekly <- array(data=y_train_vec_lupa_weekly, dim=c(length(
  y_train_vec_lupa_weekly), 1))

lag_test_lupa_weekly <- pp_lupa_weekly %>%
  mutate(value_lag = lag(value, n=lag_setting_lupa_weekly)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "testing")

x_test_vec_lupa_weekly <- lag_test_lupa_weekly$value_lag
x_test_arr_lupa_weekly <- array(data=x_test_vec_lupa_weekly, dim = c(length(
  x_test_vec_lupa_weekly), 1, 1))

y_test_vec_lupa_weekly <- lag_test_lupa_weekly$value
y_test_arr_lupa_weekly <- array(data=y_test_vec_lupa_weekly, dim = c(length(
  y_test_vec_lupa_weekly), 1))

model <- keras_model_sequential()

model %>%
  layer_lstm(units = 50,
             input_shape = c(tsteps_lupa_weekly, 1),
             batch_size = batch_size_lupa_weekly,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dense(units=1)

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam',
  metrics = c('mean_absolute_error', 'mean_squared_error')
)

history <- model %>% fit(x = x_train_arr_lupa_weekly,
                         y = y_train_arr_lupa_weekly,
                         batch_size = batch_size_lupa_weekly,
                         epochs = 12,
                         verbose = 0,
                         shuffle = FALSE,
                         validation_data = list(x_test_arr_lupa_weekly, y_test_arr_lupa_weekly)
)

plot(history)


predLSTM_lupa_weekly_training <- model %>% 
  predict(x_train_arr_lupa_weekly, batch_size=batch_size_lupa_weekly)
predLSTM_lupa_weekly_training <- as.vector(predLSTM_lupa_weekly_training)

predLSTM_lupa_weekly_training_tbl <- tibble(
  index = lag_train_lupa_weekly$index,
  value = (predLSTM_lupa_weekly_training*scale_inv_lupa_weekly + center_inv_lupa_weekly)^2
)

lupa_weekly_accuracy_LSTM_training <- getPerformance(as.vector(predLSTM_lupa_weekly_training_tbl$value-delta_lupa_weekly),
                                                        as.vector((y_train_vec_lupa_weekly*scale_inv_lupa_weekly+center_inv_lupa_weekly)^2-delta_lupa_weekly))
lupa_weekly_accuracy_LSTM_training

lupa_plot_act_train_weekly = as.ts((y_train_vec_lupa_weekly*scale_inv_lupa_weekly+center_inv_lupa_weekly)^2-delta_lupa_weekly, frequency=52)
lupa_plot_pred_train_weekly = as.ts(predLSTM_lupa_weekly_training_tbl$value-delta_lupa_weekly, frequency=52)

autoplot(lupa_plot_act_train_weekly, xlab="weeks", ylab="Flow Rate", series="Train set") +
  autolayer(lupa_plot_pred_train_weekly, series="Prediction")

predLSTM_lupa_weekly_testing <- model %>% 
  predict(x_test_arr_lupa_weekly, batch_size=batch_size_lupa_weekly)
predLSTM_lupa_weekly_testing <- as.vector(predLSTM_lupa_weekly_testing)

predLSTM_lupa_weekly_testing_tbl <- tibble(
  index = lag_test_lupa_weekly$index,
  value = (predLSTM_lupa_weekly_testing*scale_inv_lupa_weekly + center_inv_lupa_weekly)^2
)

lupa_weekly_accuracy_LSTM_testing <- getPerformance(as.vector(predLSTM_lupa_weekly_testing_tbl$value-delta_lupa_weekly),
                                                       as.vector(df_test_lupa_weekly$value-delta_lupa_weekly))
lupa_weekly_accuracy_LSTM_testing

lupa_plot_act_test_weekly = as.ts(df_test_lupa_weekly$value-delta_lupa_weekly, frequency=52)
lupa_plot_pred_test_weekly = as.ts(predLSTM_lupa_weekly_testing_tbl$value-delta_lupa_weekly, frequency=52)

autoplot(lupa_plot_act_test_weekly, xlab="weeks", ylab="Flow Rate", series="Test set") +
  autolayer(lupa_plot_pred_test_weekly, series="Prediction")

summary_results_lupa[nrow(summary_results_lupa) + 1,] = c("Lupa", "Weekly",
                                                "LSTM", lupa_weekly_accuracy_LSTM_testing$RMSE)


