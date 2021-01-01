
'''

Time series is a collection of data points that are collected at constant time 
intervals. It is a dynamic or time dependent problem with or without increasing 
or decreasing trend, seasonality. Time series modeling is a powerful method to 
describe and extract information from time-based data and help to make informed 
decisions about future outcomes.

Components of time series
- Trend (increasing or decreasing with time)
- Seasonality (outdoor temperature increases during summer months) 

Difference between time series and regression
- Time series is time dependent, so the basic assumption of linear regression that observations are independent does not hold
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels as sm

# load data
train = pd.read_csv('/Users/vivekparashar/OneDrive/GitHub-OneDrive/Learning-Data-Science/Data/Train_SU63ISt.csv')
test = pd.read_csv('/Users/vivekparashar/OneDrive/GitHub-OneDrive/Learning-Data-Science/Data/Test_0qrQsBZ.csv')

# understand data structure
train.info()
train.columns
train.dtypes
train.shape

for i in (train, test):
    i.Datetime = pd.to_datetime(i.Datetime, format = '%d-%m-%Y %H:%M')

for i in (train, test):
    i['year'] = i.Datetime.dt.year
    i['month'] = i.Datetime.dt.month
    i['day'] = i.Datetime.dt.day
    i['Hour'] = i.Datetime.dt.hour

# create a column called weekend
# we will use dt.dayofweek (this is 5 or 6 for weekend)
def wkend (row):
    if row.dayofweek == 5 or row.dayofweek ==6:
        return 1
    else:
        return 0

train['day_of_week'] = train['Datetime'].apply(wkend)
train = train.set_index('Datetime')
train.plot(y='Count')

# Exploratory data analysis - lets test our hypothesis
# traffic will increase as years pass by
train.groupby('year').mean()['Count'].plot.bar()
# traffic will be high from may-oct
train.groupby('month').mean()['Count'].plot.bar()
# traffic on weekends will be more
train.groupby('day_of_week').mean()['Count'].plot.bar() # 0 is for weekdays and 1 for weekend
# traffic during peak hours will be high
train.groupby('Hour').mean()['Count'].plot.bar()

# there is a lot of noise at hourly level
# lets see if we can reduce the noise by aggregating at daily, weekly and monthly level
fig, axes = plt.subplots(4,1)
train.resample('H').mean()['Count'].plot(ax=axes[0])
train.resample('D').mean()['Count'].plot(ax=axes[1])
train.resample('W').mean()['Count'].plot(ax=axes[2])
train.resample('M').mean()['Count'].plot(ax=axes[3])
plt.show()

# lets go with the daily timeseries
train1 = train.resample('D').mean()

# lets split train dataset into train and valid (for validation / this is not same as test)
train2 = train1.loc['2012-08-25':'2014-06-24']
valid2 = train1.loc['2014-06-25':'2014-09-25']

train2.Count.plot()
valid2.Count.plot()
# Here the blue part represents the trianing data and the orange part represents the validation data

# TIME SERIES FORECASTING TECHNIQUES
# Naive approach - here we assume that the next expected point is same as the last observed point. So we can expect a straight horizontal line as the prediction
dd = np.asarray(train2.Count)
y_hat = valid2.copy()
y_hat['naive'] = dd [ len(dd) - 1 ]
plt.figure(figsize=(12,8))
train2.Count.plot()
valid2.Count.plot()
y_hat.naive.plot()

# use RMSE - root mean square error (this is standard deviaiton of the residuals) test
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(valid2.Count, y_hat.naive))
print(rms)
# we can infer that naive method is not suitable for datasets with high variability

# moving average - here the predictions are made based on the average of last few points instead of taking all the previously known values
# lets try the rolling average for last 10, 20, 50 days
# rolling average for last 10 days
y_hat = valid2.copy()
y_hat['moving_avg'] = train2['Count'].rolling(10)\
    .mean().iloc[-1] # avg of last 10 observations
plt.figure(figsize=(12,8))
train2.Count.plot()
valid2.Count.plot()
y_hat.moving_avg.plot()

# simple exponential smoothing - we assign larger weights to more recent observations 
# the weights decrease exponentially as observations come from further in the past
# the smallest weights are associated with the oldest observations
# if we give entire weight to the last value then this technique becomes same as naive approach
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
y_hat = valid2.copy()
y_hat['SES'] = SimpleExpSmoothing(np.array(train2.Count))\
    .fit(smoothing_level=0.6, optimized=False)\
    .forecast(len(valid2))
plt.figure(figsize=(12,8))
train2.Count.plot()
valid2.Count.plot()
y_hat.SES.plot()

# Holt linear trend model
# first lets decompose the time series into its four parts - 
# observed (this is the original time series), trend, seasonal and 
# residual (remainder after removing trend and seasonality from time series)
import statsmodels.api as sm_api
sm_api.tsa.seasonal_decompose(train2.Count).plot();
result = sm.tsa.stattools.adfuller(train2.Count)
# holt implementation
# it is an extenstion of ses to allow forecasting of data with a trend
# the forecasting function in this method is a function of level and trend
y_hat = valid2.copy()
y_hat['holt_linear'] = Holt(np.array(train2.Count))\
    .fit(smoothing_level=0.3, smoothing_slope=0.1)\
    .forecast(len(valid2))
plt.figure(figsize=(12,8))
train2.Count.plot()
valid2.Count.plot()
y_hat.holt_linear.plot()

# Holt winters model - takes into account both trend and seasonality
y_hat = valid2.copy()
y_hat['holt_winter'] = ExponentialSmoothing(np.array(train2.Count)\
    , seasonal_periods=7, trend='add', seasonal='add').fit()\
    .forecast(len(valid2))
plt.figure(figsize=(12,8))
train2.Count.plot()
valid2.Count.plot()
y_hat.holt_winter.plot()

# ARIMA - Auto Regression Integrated Moving Average
# we need to make timeseries stationary for ARIMA
# stationary time series - mean and variance of time series are not funciton of time, co-var of i'th and (i+m)'th terms are not function of time
# so ARIMA forecast for a stationary time series is nothing but a linear equaiton
#
# parameters for ARIMA model: (p,d,q)
# p: order of the autoregressive model (number of time lags)
# d: degree of differencing (number of times the data has had past values subtracted)
# q: order of moving average model
#
# First we check if the series is stationary, if not, then we make it stationary
# we use dicky fuller test to test if the series is stationary
# the intuition behind this test is that it determines how strongly a time series is defined by a trend
# null hypothesis si that time series is not stationary (has some dependent structure)
# alternate hypothesis (rejecting the null hypothesis) is that the time series is stationary
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
#
def test_stationary(timeseries):
    # determining rolling statistics
    rolmean = timeseries.rolling(24).mean() # 24 hours on each day
    rolstd = timeseries.rolling(24).std() # 24 hours on each day
    # plot rolling statistics
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='blue', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)
    # perform dicky fuller test
    print('Results of Dicky Fuller test')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic',\
        'p-value', '#Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
#
test_stationary(train.Count) #train original is used here
# as t-statistic is less than the critical value but we can see an increasing trend in data, we assume that the series is stationary
# still we will try to make the data more stationary by removing trend and seasonality from data
#
# removing trend
# we see increasing trend in data, so we can apply a transform like log, which penalizes higher values
# we will take rolling average here to remove the trend. we will take window size of 24 as there are 24 hours in a day
train_log = np.log(train2.Count)
valid_log = np.log(valid2.Count)
# moving_avg = pd.rolling_mean(train_log,24) # this is depricated
moving_avg = train_log.rolling(24).mean()
plt.plot(train_log)
plt.plot(moving_avg, color='red')
plt.show()
# now we will remove thsi increasing trend to make out time series stationary
train_log_moving_avg_diff = train_log - moving_avg
# since we took the average of 24 values, rolling mean is not defined for the first 23 values. so lets drop those null values. 
train_log_moving_avg_diff = train_log_moving_avg_diff.dropna()
test_stationary(train_log_moving_avg_diff)
# we can see that the test statistic is very small compared to critical value. So we can assume that the trend has been removed. 
#
# lets now stabalize the mean which is also a requirement for a stationary time series
# differencing can help to make the timeseries stable and eliminate the trend
train_log_diff = train_log - train_log.shift(1)
test_stationary(train_log_diff.dropna())
#
# removing seasonality 
# we will use seasonal decompose to decompose the time series into trend, seasonality and residuals
from statsmodels.tsa. seasonal import seasonal_decompose
decomposition = seasonal_decompose(pd.DataFrame(train_log).Count.values, freq=24)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
#
plt.subplot(411)
plt.plot(train_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residual')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
#
#
# lets check if residulas are stationary using dicky fuller test
train_log_decompose = pd.DataFrame(residual)
train_log_decompose['date'] = train_log.index
train_log_decompose.set_index('date', inplace=True)
train_log_decompose.dropna(inplace=True)
test_stationary(train_log_decompose[0])
# we can see that the test statistic is very small compared to critical value. So we can assume that residuals are stationary
#
# now we will forecast using arima
# first we need to find optimal values for p, d, q 
# we will use ACF (auto correlation function) and PACF (partial auto correlation) graph for this
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(train_log_diff.dropna(), nlags=25)
lag_pacf = pacf(train_log_diff.dropna(), nlags=25, method = 'ols')
# plot acf
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())), linestyle = '--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())), linestyle = '--', color='gray')
plt.title('Autocorrelation Function')
plt.show()
# plot pacf
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())), linestyle = '--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())), linestyle = '--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.show()
# p value is the lag value where the PACF chart crosses the upper confidence interval for the first time. it can be noticed that in this case p=1
# q value is the lag value where the ACF chart crosses the upper confidence interval for the first time. it can be noticed that in this case p=1
#
# now we will make the AR and MA model separately and join them
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(train_log, order=(2,1,0)) # here the q value is 0 since this is just an AR model
results_AR = model.fit(disp=-1)
plt.plot(train_log_diff.dropna(), label='Original')
plt.plot(results_AR.fittedvalues, color='red', label='Predicitons')
plt.legend(loc='best')
plt.show()
#
# lets plot the validation curve for AR model
# we have to change the scale of the mdoel to original scale
# first step would be to store the predicted results as a separate seires and observe it
AR_predict = results_AR.predict(start='2014-06-25', end='2014-09-25')
AR_predict = AR_predict.cumsum().shift().fillna(0)
AR_predict1 = pd.Series(np.ones(valid2.shape[0])\
    * np.log(valid2.Count)[0], index=valid2.index)
AR_predict = AR_predict1.add(AR_predict, fill_value=0)
AR_predict = np.exp(AR_predict1)
plt.plot(valid2.Count, label='Valid')
plt.plot(AR_predict, color='red', label='Predict')
plt.legend(loc='best')
plt.title('RMSE: %.4f'% (np.sqrt(np.dot(AR_predict, valid2.Count))/valid2.shape[0]))
plt.show()




<this is where i am>

# understanding data

# hypothesis generation - list out all possible factors that can affect the outcome
# do this without looking at the data to avoid bias
# in this case we say 1. there will be an increase in traffic as years pass by
# 2. traffic will be higher from may-oct due to tourists
# 3. traffic on weekdays will be more than weekends




##########################################################################
######################### co2 emission case study ########################
##########################################################################

'''
CO2 Emission Forecast with Python (Seasonal ARIMA)
https://www.kaggle.com/berhag/co2-emission-forecast-with-python-seasonal-arima
'''

##########################################################################
############################# j&j case study #############################
##########################################################################

df = pd.read_csv('https://github.com/marcopeix/time-series-analysis/raw/master/data/jj.csv')
df.info()

df.date = pd.to_datetime(df.date, format='%Y-%m-%d', errors = 'coerce')
# df.data = pd.to_numeric(df.data, errors='coerce')
# df.dropna(inplace = True)
# df = df.set_index('date')

# display a plot of the time series
df.set_index('date').plot()

# Clearly, the time series is not stationary, as its mean is not constant through time,
# and we see an increasing variance in the data, a sign of heteroscedasticity.
# To make sure, let’s plot the PACF and ACF
sm.graphics.tsaplots.plot_acf(df.data);
sm.graphics.tsaplots.plot_pacf(df.data);

# no information can be deduced from those plots. 
# You can further test for stationarity with the Augmented Dickey-Fuller test
ad_fuller_result = sm.tsa.stattools.adfuller(df.data)
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')
# Since the p-value is large, we cannot reject the null hypothesis 
# and must assume that the time series is non-stationary.

# Now, let’s take the log difference in an effort to make it stationary:
df.data = np.log(df.data).diff()
df = df.drop(df.index[0])
# Log Difference of Quarterly EPS for Johnson & Johnson
df.plot(y='data')

# we still see the seasonality in the plot above. 
# Since we are dealing with quarterly data, our period is 4. 
# Therefore, we will take the difference over a period of 4
df.data = df.data.diff(4)
df = df.drop([1, 2, 3, 4], axis=0).reset_index(drop=True)
df.plot(y='data')

# let’s run the Augmented Dickey-Fuller test again to see if we have a stationary time series
ad_fuller_result = sm.tsa.stattools.adfuller(df.data)
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')
# Indeed, the p-value is small enough for us to reject the null hypothesis, 
# and we can consider that the time series is stationary.


#############################################################################
################################# statsmodels ###############################
#############################################################################

'''
1. Select Time Series Forecast Model
This is where the bulk of the effort will be in preparing the data, performing analysis, and ultimately selecting a model and model hyperparameters that best capture the relationships in the data.
In this case, we can arbitrarily select an autoregression model (AR) with a lag of 6 on the differenced dataset.
'''

'''
https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
11 Classical Time Series Forecasting Methods in Python (Cheat Sheet)

These methods are suitable for univariate time series 
without trend and seasonal components.
1. Autoregression (AR)
2. Moving Average (MA)
3. Autoregressive Moving Average (ARMA)
10. Simple Exponential Smoothing (SES)

with trend and without seasonal components.
4. Autoregressive Integrated Moving Average (ARIMA)

with trend and/or seasonal components.
5. Seasonal Autoregressive Integrated Moving-Average (SARIMA)
11. Holt Winter’s Exponential Smoothing (HWES)

with trend and/or seasonal components and exogenous variables.
6. Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)

These methods are suitable for multivariate time series 
without trend and seasonal components.
7. Vector Autoregression (VAR)
8. Vector Autoregression Moving-Average (VARMA)

without trend and seasonal components with exogenous variables.
9. Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX)


SUMMARY:
time series type ->                         univariate              multivariate
-trend, -seasonal                           AR, MA, ARMA, SES       VAR, VARMA
+trend, -seaosnal                           ARIMA
+trend, +seasonal                           SARIMA, HWES
-trend, 1seasonal, +exogenous variables                             VARMAX
+trend, +seasonal, +exogenous variables     SARIMAX
'''

# The method is suitable for univariate time series
# The method is suitable for univariate time series without trend and seasonal components.

# AR method models the next step in the sequence as a linear function of the observations at prior time steps.
# The notation for the model involves specifying the order of the model p as a parameter to the AR function, e.g. AR(p). For example, AR(1) is a first-order autoregression model.
from statsmodels.tsa.ar_model import AutoReg
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = AutoReg(data, lags=1)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)

# MA method models the next step in the sequence as a linear function of the residual errors from a mean process at prior time steps.
# The notation for the model involves specifying the order of the model q as a parameter to the MA function, e.g. MA(q). For example, MA(1) is a first-order moving average model.
# A moving average model is different from calculating the moving average of the time series.
# MA example
from statsmodels.tsa.arima.model import ARIMA
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = ARIMA(data, order=(0, 0, 1))
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)

# Autoregressive Moving Average (ARMA) method models the next step in the sequence as a linear function of the observations and residual errors at prior time steps.
# It combines both Autoregression (AR) and Moving Average (MA) models.
# The notation for the model involves specifying the order for the AR(p) and MA(q) models as parameters to an ARMA function, e.g. ARMA(p, q). An ARIMA model can be used to develop AR or MA models.
# ARMA example
from statsmodels.tsa.arima.model import ARIMA
from random import random
# contrived dataset
data = [random() for x in range(1, 100)]
# fit model
model = ARIMA(data, order=(2, 0, 1))
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)


# The method is suitable for univariate time series with trend and without seasonal components.

# Autoregressive Integrated Moving Average (ARIMA) method models the next step in the sequence as a linear function of the differenced observations and residual errors at prior time steps.
# It combines both Autoregression (AR) and Moving Average (MA) models as well as a differencing pre-processing step of the sequence to make the sequence stationary, called integration (I).
# The notation for the model involves specifying the order for the AR(p), I(d), and MA(q) models as parameters to an ARIMA function, e.g. ARIMA(p, d, q). An ARIMA model can also be used to develop AR, MA, and ARMA models.
# ARIMA example
from statsmodels.tsa.arima.model import ARIMA
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = ARIMA(data, order=(1, 1, 1))
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data), typ='levels')
print(yhat)


# The method is suitable for univariate time series with trend and/or seasonal components.

# Seasonal Autoregressive Integrated Moving-Average (SARIMA) method models the next step in the sequence as a linear function of the differenced observations, errors, differenced seasonal observations, and seasonal errors at prior time steps.
# It combines the ARIMA model with the ability to perform the same autoregression, differencing, and moving average modeling at the seasonal level.
# The notation for the model involves specifying the order for the AR(p), I(d), and MA(q) models as parameters to an ARIMA function and AR(P), I(D), MA(Q) and m parameters at the seasonal level, e.g. SARIMA(p, d, q)(P, D, Q)m where “m” is the number of time steps in each season (the seasonal period). A SARIMA model can be used to develop AR, MA, ARMA and ARIMA models.
# SARIMA example
from statsmodels.tsa.statespace.sarimax import SARIMAX
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)


# The method is suitable for univariate time series with trend and/or seasonal components and exogenous variables.

# Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX) is an extension of the SARIMA model that also includes the modeling of exogenous variables.
# Exogenous variables are also called covariates and can be thought of as parallel input sequences that have observations at the same time steps as the original series. The primary series may be referred to as endogenous data to contrast it from the exogenous sequence(s). The observations for exogenous variables are included in the model directly at each time step and are not modeled in the same way as the primary endogenous sequence (e.g. as an AR, MA, etc. process).
# The SARIMAX method can also be used to model the subsumed models with exogenous variables, such as ARX, MAX, ARMAX, and ARIMAX.
# SARIMAX example
from statsmodels.tsa.statespace.sarimax import SARIMAX
from random import random
# contrived dataset
data1 = [x + random() for x in range(1, 100)]
data2 = [x + random() for x in range(101, 200)]
# fit model
model = SARIMAX(data1, exog=data2, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit(disp=False)
# make prediction
exog2 = [200 + random()]
yhat = model_fit.predict(len(data1), len(data1), exog=[exog2])
print(yhat)


# The method is suitable for multivariate time series
# The method is suitable for multivariate time series without trend and seasonal components.

# Vector Autoregression (VAR) method models the next step in each time series using an AR model. It is the generalization of AR to multiple parallel time series, e.g. multivariate time series.
# The notation for the model involves specifying the order for the AR(p) model as parameters to a VAR function, e.g. VAR(p).
# VAR example
from statsmodels.tsa.vector_ar.var_model import VAR
from random import random
# contrived dataset with dependency
data = list()
for i in range(100):
    v1 = i + random()
    v2 = v1 + random()
    row = [v1, v2]
    data.append(row)
# fit model
model = VAR(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.forecast(model_fit.y, steps=1)
print(yhat)

# Vector Autoregression Moving-Average (VARMA) method models the next step in each time series using an ARMA model. It is the generalization of ARMA to multiple parallel time series, e.g. multivariate time series.
# The notation for the model involves specifying the order for the AR(p) and MA(q) models as parameters to a VARMA function, e.g. VARMA(p, q). A VARMA model can also be used to develop VAR or VMA models.
# VARMA example
from statsmodels.tsa.statespace.varmax import VARMAX
from random import random
# contrived dataset with dependency
data = list()
for i in range(100):
    v1 = random()
    v2 = v1 + random()
    row = [v1, v2]
    data.append(row)
# fit model
model = VARMAX(data, order=(1, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.forecast()
print(yhat)


# The method is suitable for multivariate time series without trend and seasonal components with exogenous variables.

# Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX) is an extension of the VARMA model that also includes the modeling of exogenous variables. It is a multivariate version of the ARMAX method.
# Exogenous variables are also called covariates and can be thought of as parallel input sequences that have observations at the same time steps as the original series. The primary series(es) are referred to as endogenous data to contrast it from the exogenous sequence(s). The observations for exogenous variables are included in the model directly at each time step and are not modeled in the same way as the primary endogenous sequence (e.g. as an AR, MA, etc. process).
# The VARMAX method can also be used to model the subsumed models with exogenous variables, such as VARX and VMAX.
# VARMAX example
from statsmodels.tsa.statespace.varmax import VARMAX
from random import random
# contrived dataset with dependency
data = list()
for i in range(100):
    v1 = random()
    v2 = v1 + random()
    row = [v1, v2]
    data.append(row)
data_exog = [x + random() for x in range(100)]
# fit model
model = VARMAX(data, exog=data_exog, order=(1, 1))
model_fit = model.fit(disp=False)
# make prediction
data_exog2 = [[100]]
yhat = model_fit.forecast(exog=data_exog2)
print(yhat)

# The method is suitable for univariate time series without trend and seasonal components.

# Simple Exponential Smoothing (SES) method models the next time step as an exponentially weighted linear function of observations at prior time steps.
# SES example
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = SimpleExpSmoothing(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)

# The method is suitable for univariate time series with trend and/or seasonal components.

# Holt Winter’s Exponential Smoothing (HWES) also called the Triple Exponential Smoothing method models the next time step as an exponentially weighted linear function of observations at prior time steps, taking trends and seasonality into account.
# HWES example
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = ExponentialSmoothing(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)


#############################################################################
################################# fbprophet #################################
#############################################################################

'''
https://facebook.github.io/prophet/docs/quick_start.html#python-api
Facebook Prophet
The input to Prophet is always a dataframe with two columns: ds and y. The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement we wish to forecast.
'''

import pandas as pd
from fbprophet import Prophet

