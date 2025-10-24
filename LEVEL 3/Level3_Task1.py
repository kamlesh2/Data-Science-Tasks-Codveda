import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv("yahoo_stock.csv", parse_dates=['Date'], index_col='Date')
df = df.sort_index()
ts = df['Close']

plt.figure(figsize=(10, 4))
plt.plot(ts, label='Close Price')
plt.title("Stock Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()

decomp = seasonal_decompose(ts, model='additive', period=30)
decomp.plot()
plt.suptitle("Time Series Decomposition")
plt.tight_layout()
plt.show()

ts_ma = ts.rolling(window=30).mean()
plt.figure(figsize=(10, 4))
plt.plot(ts, label='Original')
plt.plot(ts_ma, label='30-day Moving Average', color='red')
plt.title("Moving Average Smoothing")
plt.legend()
plt.tight_layout()
plt.show()

exp_model = ExponentialSmoothing(ts, trend='add', seasonal=None)
exp_fit = exp_model.fit()
ts_exp = exp_fit.fittedvalues

plt.figure(figsize=(10, 4))
plt.plot(ts, label='Original')
plt.plot(ts_exp, label='Exponential Smoothing', color='green')
plt.title("Exponential Smoothing")
plt.legend()
plt.tight_layout()
plt.show()

train_size = int(len(ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]

arima_model = ARIMA(train, order=(5, 1, 0))
arima_fit = arima_model.fit()
forecast = arima_fit.forecast(steps=len(test))

rmse = np.sqrt(mean_squared_error(test, forecast))
print("ARIMA RMSE:", round(rmse, 2))

plt.figure(figsize=(10, 4))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, forecast, label='ARIMA Forecast', linestyle='--')
plt.title("ARIMA Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()